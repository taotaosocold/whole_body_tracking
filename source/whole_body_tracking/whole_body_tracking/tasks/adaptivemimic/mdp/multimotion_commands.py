"""Multi-motion command term — two-level adaptive sampling across multiple motion files."""

from __future__ import annotations

import glob as _glob
import math
import os
import torch
import numpy as np
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .commands import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FlatMotionLoader:
    """Load multiple .npz motion files into flat concatenated tensors.

    All motions are concatenated along the time axis → ``[sum(T_i), …]``.
    A per-env ``current_frame = motion_start[idx] + time_steps`` indexes
    directly into the flat tensors with a single integer gather.
    """

    def __init__(self, motion_files: list[str], device: str = "cpu"):
        self.device = device
        self.motion_names = [os.path.splitext(os.path.basename(f))[0] for f in motion_files]
        self.num_motions = len(motion_files)

        all_joint_pos = []
        all_joint_vel = []
        all_body_pos_w = []
        all_body_quat_w = []
        all_body_lin_vel_w = []
        all_body_ang_vel_w = []
        motion_frames = []

        for f in motion_files:
            data = np.load(f)
            all_joint_pos.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
            all_joint_vel.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
            all_body_pos_w.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
            all_body_quat_w.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
            all_body_lin_vel_w.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
            all_body_ang_vel_w.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
            motion_frames.append(data["joint_pos"].shape[0])

        self.joint_pos = torch.cat(all_joint_pos, dim=0)  # [total_frames, D]
        self.joint_vel = torch.cat(all_joint_vel, dim=0)
        self.body_pos_w = torch.cat(all_body_pos_w, dim=0)  # [total_frames, B_all, 3]
        self.body_quat_w = torch.cat(all_body_quat_w, dim=0)
        self.body_lin_vel_w = torch.cat(all_body_lin_vel_w, dim=0)
        self.body_ang_vel_w = torch.cat(all_body_ang_vel_w, dim=0)

        self.motion_frames = torch.tensor(motion_frames, dtype=torch.long, device=device)
        self.motion_start = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), self.motion_frames.cumsum(dim=0)[:-1]]
        )

        # for compatibility with code that reads .time_step_total
        self.time_step_total = self.joint_pos.shape[0]


class MultiMotionCommand(MotionCommand):
    """Motion command for simultaneous training on multiple motion sequences.

    Two-level adaptive sampling:

    * **Inter-motion**: softmax over per-motion failure counts decides which
      ``.npz`` file an env tracks next.
    * **Intra-motion**: within the selected clip, bin-level adaptive sampling.

    Uses flat concatenation (not padded stacking) for O(1) single-index lookups.
    """

    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        CommandTerm.__init__(self, cfg, env)

        self.robot = env.scene[cfg.asset_name]
        print("IsaacLab 中的所有 Body 名称 (self.robot.body_names):", self.robot.body_names)
        print("IsaacLab中的所有关节名称:", self.robot.data.joint_names)
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        # ── load all motions into flat tensors ────────────────────────
        motion_files = sorted(_glob.glob(os.path.join(cfg.motion_folder, "*.npz")))
        if not motion_files:
            raise FileNotFoundError(f"No .npz files found in: {cfg.motion_folder}")

        self.lafan_motion = FlatMotionLoader(motion_files, device=self.device)
        self.num_motions = self.lafan_motion.num_motions
        self.motion_names = self.lafan_motion.motion_names
        print(f"[INFO] MultiMotionCommand: loaded {self.num_motions} motions from {cfg.motion_folder}")
        for i, name in enumerate(self.motion_names):
            print(f"  [{i}] {name} — {self.lafan_motion.motion_frames[i].item()} frames")

        # compat: self.motion → flat loader (used by exporter, etc.)
        self.motion = self.lafan_motion

        # ── per-env state ─────────────────────────────────────────────
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ── bin tracking (uniform bin_count across all motions) ────────
        self.bin_count = self.cfg.bin_count
        self.bin_failed_count = torch.zeros(self.num_motions, self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)

        # compat pointers
        self.bin_failed_counts = [self.bin_failed_count[i] for i in range(self.num_motions)]
        self._current_bin_faileds = [self._current_bin_failed[i] for i in range(self.num_motions)]
        self.bin_failed_count = self.bin_failed_count  # keep reference
        self._current_bin_failed = self._current_bin_failed

        # ── inter-motion tracking ─────────────────────────────────────
        self.motion_failed_count = torch.zeros(self.num_motions, dtype=torch.float, device=self.device)
        self._current_motion_failed = torch.zeros(self.num_motions, dtype=torch.float, device=self.device)

        # ── bin-sampling kernel ───────────────────────────────────────
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        # ── relative body state buffers ───────────────────────────────
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # proprioceptive history buffer [num_envs, K, proprio_dim]
        self.prop_history_len = getattr(cfg, "prop_history_len", 10)
        self.prop_history = torch.zeros(self.num_envs, self.prop_history_len, 90, device=self.device)

        # periodic difficulty logging
        self._step_counter = 0

        # ── metrics ───────────────────────────────────────────────────
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["active_motion"] = torch.zeros(self.num_envs, device=self.device)

    # ── flat-index helper ──────────────────────────────────────────────

    @property
    def current_frame(self) -> torch.Tensor:
        """Global frame index into the flat tensors for each env."""
        return self.lafan_motion.motion_start[self.active_motion_idx] + self.time_steps

    # ── data properties ────────────────────────────────────────────────

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.lafan_motion.joint_pos[self.current_frame]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.lafan_motion.joint_vel[self.current_frame]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.lafan_motion.body_pos_w[self.current_frame][:, self.body_indexes]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.lafan_motion.body_quat_w[self.current_frame][:, self.body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.lafan_motion.body_lin_vel_w[self.current_frame][:, self.body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.lafan_motion.body_ang_vel_w[self.current_frame][:, self.body_indexes]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.lafan_motion.body_pos_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.lafan_motion.body_quat_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.lafan_motion.body_lin_vel_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.lafan_motion.body_ang_vel_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]

    # ── two-level adaptive sampling ───────────────────────────────────

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return

        # track per-(motion, bin) failures
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps[env_ids] * self.bin_count)
                // self.lafan_motion.motion_frames[self.active_motion_idx[env_ids]].clamp(min=1),
                0,
                self.bin_count - 1,
            )
            fail_bins = current_bin_index[episode_failed]
            fail_motions = self.active_motion_idx[env_ids][episode_failed]
            self._current_bin_failed[fail_motions, fail_bins] += 1.0

        bin_sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        bin_sampling_probabilities = torch.nn.functional.pad(
            bin_sampling_probabilities.unsqueeze(1),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        bin_sampling_probabilities = torch.nn.functional.conv1d(
            bin_sampling_probabilities, self.kernel.view(1, 1, -1)
        ).squeeze(1)

        bin_sampling_probabilities = bin_sampling_probabilities / bin_sampling_probabilities.sum(dim=1, keepdim=True)

        sampled_bins = torch.multinomial(
            bin_sampling_probabilities[self.active_motion_idx[env_ids]], 1, replacement=True
        ).squeeze(-1)

        motion_lengths = self.lafan_motion.motion_frames[self.active_motion_idx[env_ids]]
        bin_width = (motion_lengths.float() / self.bin_count).long().clamp(min=1)
        self.time_steps[env_ids] = (
            sampled_bins.float() / self.bin_count * motion_lengths.float()
            + torch.rand(len(env_ids), device=self.device) * bin_width.float()
        ).long()

        # metrics
        H = -(bin_sampling_probabilities * (bin_sampling_probabilities + 1e-12).log()).sum(dim=1)
        H_norm = H / math.log(self.bin_count)
        self.metrics["sampling_entropy"][:] = H_norm[self.active_motion_idx]
        pmax, imax = bin_sampling_probabilities.max(dim=1)
        self.metrics["sampling_top1_prob"][:] = pmax[self.active_motion_idx]
        self.metrics["sampling_top1_bin"][:] = imax[self.active_motion_idx].float() / self.bin_count
        self.metrics["active_motion"][:] = self.active_motion_idx.float()

    def _resample_motion(self, env_ids: torch.Tensor):
        """Inter-motion sampling — select which clip each env tracks next."""
        n = len(env_ids)
        probs = self.motion_failed_count + self.cfg.motion_adaptive_uniform_ratio / float(self.num_motions)
        probs = probs / probs.sum()
        self.active_motion_idx[env_ids] = torch.multinomial(probs, n, replacement=True)

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return

        # Track inter-motion failures BEFORE resampling
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed_env_ids = env_ids[episode_failed]
            failed_motions = self.active_motion_idx[failed_env_ids]
            self._current_motion_failed.zero_()
            self._current_motion_failed.scatter_add_(
                0, failed_motions, torch.ones_like(failed_motions, dtype=torch.float)
            )
            self.motion_failed_count = (
                self.cfg.motion_adaptive_alpha * self._current_motion_failed
                + (1 - self.cfg.motion_adaptive_alpha) * self.motion_failed_count
            )
            self._current_motion_failed.zero_()

        # Level 1: inter-motion
        self._resample_motion(env_ids)

        # Level 2: intra-motion time-step
        self._adaptive_sampling(env_ids)

        # ── set robot state (uses properties → current_frame) ──────────
        root_pos = self.lafan_motion.body_pos_w[self.current_frame, 0].clone()
        root_ori = self.lafan_motion.body_quat_w[self.current_frame, 0].clone()
        root_lin_vel = self.lafan_motion.body_lin_vel_w[self.current_frame, 0].clone()
        root_ang_vel = self.lafan_motion.body_ang_vel_w[self.current_frame, 0].clone()

        range_list = [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=self.device)
        root_pos[env_ids] += rand[:, :3]
        orient_delta = quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5])
        root_ori[env_ids] = quat_mul(orient_delta, root_ori[env_ids])

        range_list = [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=self.device)
        root_lin_vel[env_ids] += rand[:, :3]
        root_ang_vel[env_ids] += rand[:, 3:]

        joint_pos = self.lafan_motion.joint_pos[self.current_frame].clone()
        joint_vel = self.lafan_motion.joint_vel[self.current_frame].clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, self.device)
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(joint_pos[env_ids], soft_limits[:, :, 0], soft_limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(
            self.time_steps >= self.lafan_motion.motion_frames[self.active_motion_idx]
        )[0]
        self._resample_command(env_ids)

        # update body-relative transforms
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # update per-motion intra bin EMA
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

        # periodic difficulty logging
        self._step_counter += 1
        if self._step_counter % self.cfg.log_interval_steps == 0:
            _, indices = torch.topk(self.motion_failed_count, min(self.cfg.log_topk, self.num_motions))
            print(
                f"\n[MultiMotion] Step {self._step_counter} | "
                f"Motion failure counts (top {min(self.cfg.log_topk, self.num_motions)}):"
            )
            for rank, midx in enumerate(indices.tolist()):
                print(
                    f"  {rank + 1}. [{midx}] {self.motion_names[midx]}"
                    f" — fail_count={self.motion_failed_count[midx].item():.4f}"
                )
            print()


# ── config ──────────────────────────────────────────────────────────────


@configclass
class MultiMotionCommandCfg(MotionCommandCfg):
    """Configuration for the multi-motion command."""

    class_type: type = MultiMotionCommand

    motion_folder: str = MISSING
    motion_file: str = ""  # not used; kept to pass parent validation

    # uniform bin count for intra-motion sampling
    bin_count: int = 5

    # inter-motion adaptive sampling parameters
    motion_adaptive_alpha: float = 0.001  # EMA update rate for per-motion failure counts
    motion_adaptive_uniform_ratio: float = 0.1  # uniform mixing for inter-motion sampling

    # difficulty logging
    log_interval_steps: int = 240  # print top-k hardest motions every N physics steps
    log_topk: int = 10  # number of hardest motions to print
