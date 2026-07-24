"""Multi-motion command term — unified bin-level adaptive sampling across multiple motion files.

Based on the GR00T-WholeBodyControl adaptive sampling approach:
* All motions are divided into fixed-size bins (``bin_size`` frames).
* Sampling a bin simultaneously selects motion + time segment in one step.
* Tracks per-bin failure rate (num_failures / num_episodes) instead of EMA counts.
"""

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


class MotionLoader:
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

    Unified bin-level adaptive sampling (GR00T-style):

    * All motions are divided into fixed-size bins (``bin_size`` frames each).
    * Each bin belongs to a specific motion — sampling a bin simultaneously
      selects the motion AND the time segment in one ``multinomial`` call.
    * Tracks per-bin failure rate (``num_failures / num_episodes``), blends
      with a uniform baseline, and applies optional max-probability constraints
      to prevent over-concentration on outlier motions.
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

        self.motion = MotionLoader(motion_files, device=self.device)
        self.num_motions = self.motion.num_motions
        self.motion_names = self.motion.motion_names
        print(f"[INFO] MultiMotionCommand: loaded {self.num_motions} motions from {cfg.motion_folder}")
        for i, name in enumerate(self.motion_names):
            print(f"  [{i}] {name} — {self.motion.motion_frames[i].item()} frames")

        # ── per-env state ─────────────────────────────────────────────
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ── build bin structures ──────────────────────────────────────
        self._build_bins()

        # ── adaptive sampling statistics (per-bin cumulative counts) ──
        init_num_failures = 1
        self.adp_samp_num_failures = (
            torch.ones(self.num_bins, device=self.device, dtype=torch.float32) * init_num_failures
        )
        self.adp_samp_num_episodes = (
            torch.ones(self.num_bins, device=self.device, dtype=torch.float32) * init_num_failures
        )
        self.adp_samp_failure_rate = torch.ones(self.num_bins, device=self.device)
        self.adp_sampling_prob = torch.ones(self.num_bins, device=self.device) / self.num_bins

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

    # ── bin construction ──────────────────────────────────────────────

    def _build_bins(self):
        """Divide every motion into fixed-size bins and build global index structures.

        Creates:
          - ``self.bins``: (num_bins, 3) tensor of (motion_id, frame_start, frame_end)
            where frame_start/end are LOCAL offsets within the motion.
          - ``self.frame_to_bin``: maps every global flat frame → bin index.
          - ``self.bin_motion_length``: number of frames in each bin.
          - ``self.bin_new_motion_mask``: True for the first bin of each motion.
          - ``self.bin_weights``: per-bin length-normalized weights.
        """
        motion_frames = self.motion.motion_frames
        motion_start = self.motion.motion_start
        num_motions = self.motion.num_motions
        total_frames = self.motion.time_step_total
        bin_size = self.cfg.bin_size

        all_bins = []
        bin_motion_lengths = []
        bin_new_motion_masks = []
        num_peer_bins = []

        self.frame_to_bin = torch.zeros(total_frames, dtype=torch.long, device=self.device)

        cur_bin_idx = 0
        for motion_idx in range(num_motions):
            nf = motion_frames[motion_idx].item()
            f_start = motion_start[motion_idx].item()

            bin_starts = list(range(0, nf, bin_size))
            num_bins = len(bin_starts)

            for j, bs in enumerate(bin_starts):
                be = min(bs + bin_size, nf)
                all_bins.append((motion_idx, bs, be))
                bin_motion_lengths.append(float(be - bs))

            # map frames to bins
            for j in range(num_bins):
                bs = bin_starts[j]
                be = min(bs + bin_size, nf)
                self.frame_to_bin[f_start + bs : f_start + be] = cur_bin_idx + j

            bin_new_motion_masks.extend([True] + [False] * (num_bins - 1))
            num_peer_bins.extend([num_bins] * num_bins)
            cur_bin_idx += num_bins

        self.num_bins = len(all_bins)
        self.bins = torch.tensor(all_bins, dtype=torch.long, device=self.device)  # (N, 3)
        self.bin_motion_length = torch.tensor(bin_motion_lengths, dtype=torch.float32, device=self.device)
        self.bin_new_motion_mask = torch.tensor(bin_new_motion_masks, dtype=torch.bool, device=self.device)
        self.num_peer_bins = torch.tensor(num_peer_bins, dtype=torch.long, device=self.device)

        # bin weights: length normalization, optionally sequence_length_agnostic
        self.bin_weights = self.bin_motion_length / self.bin_motion_length.mean()
        if self.cfg.sequence_length_agnostic:
            self.bin_weights = self.bin_weights / self.num_peer_bins.float()

    # ── flat-index helper ──────────────────────────────────────────────

    @property
    def current_frame(self) -> torch.Tensor:
        """Global frame index into the flat tensors for each env."""
        return self.motion.motion_start[self.active_motion_idx] + self.time_steps

    # ── data properties ────────────────────────────────────────────────

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.current_frame]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.current_frame]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.current_frame][:, self.body_indexes]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.current_frame][:, self.body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.current_frame][:, self.body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.current_frame][:, self.body_indexes]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]

    # ── adaptive sampling probability update ──────────────────────────

    def _update_sampling_probabilities(self):
        """Recompute per-bin sampling probabilities from failure rates.

        Steps:
        1. Compute failure_rate = num_failures / num_episodes.
        2. Clip at mean * failure_rate_max_over_mean.
        3. Blend with uniform baseline.
        4. Apply bin weights (length / peer normalization).
        5. Apply max_prob_per_bin and max_prob_per_motion constraints (if configured).
        """
        failure_rate = self.adp_samp_num_failures / self.adp_samp_num_episodes.clamp(min=1)
        self.adp_samp_failure_rate = failure_rate

        # clip outliers
        upper_bound = failure_rate.mean() * self.cfg.failure_rate_max_over_mean
        failure_rate_clipped = torch.clamp(failure_rate, 0.0, upper_bound)

        # blend with uniform
        prob = failure_rate_clipped / failure_rate_clipped.sum()
        uniform_prob = torch.ones(self.num_bins, device=self.device) / self.num_bins
        prob = prob * (1.0 - self.cfg.uniform_sampling_rate) + uniform_prob * self.cfg.uniform_sampling_rate

        # apply bin weights (length normalization)
        prob = prob * self.bin_weights
        prob = prob / prob.sum()

        # ── max probability constraints ──
        # "auto" resolves to failure_rate_max_over_mean / num_items at call site
        if self.cfg.max_prob_per_bin is not None or self.cfg.max_prob_per_motion is not None:
            # per-bin cap
            if self.cfg.max_prob_per_bin is not None:
                max_pb = self.cfg.failure_rate_max_over_mean / self.num_bins if self.cfg.max_prob_per_bin == "auto" else float(self.cfg.max_prob_per_bin)
                if max_pb > 0 and self.num_bins > 1.0 / max_pb:
                    prob = torch.clamp(prob, max=max_pb)
                    prob = prob / prob.sum()

            # per-motion cap
            if self.cfg.max_prob_per_motion is not None:
                max_pm = self.cfg.failure_rate_max_over_mean / self.motion.num_motions if self.cfg.max_prob_per_motion == "auto" else float(self.cfg.max_prob_per_motion)
                if max_pm > 0 and self.motion.num_motions > 1.0 / max_pm:
                    for motion_idx in range(self.motion.num_motions):
                        mask = self.bins[:, 0] == motion_idx
                        motion_total = prob[mask].sum()
                        if motion_total > max_pm:
                            prob[mask] *= max_pm / motion_total
                    prob = prob / prob.sum()

        self.adp_sampling_prob = prob

    # ── sampling ──────────────────────────────────────────────────────

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample motion and time-step for the given environments.

        Unified bin-level sampling: draws bins from the global distribution,
        which simultaneously selects the motion and the starting time segment.
        """
        n = len(env_ids)
        if n == 0:
            return

        # ── track failures & update stats ────────────────────────────
        episode_failed = self._env.termination_manager.terminated[env_ids]

        global_frames = self.motion.motion_start[self.active_motion_idx[env_ids]] + self.time_steps[env_ids]
        bin_ids = self.frame_to_bin[global_frames]

        # increment episode counts for ALL resampled envs (normalized by bin length)
        episode_counts = torch.bincount(bin_ids, minlength=self.num_bins).float()
        episode_counts = episode_counts / self.bin_motion_length
        self.adp_samp_num_episodes += episode_counts

        if episode_failed.any():
            failed_bin_ids = bin_ids[episode_failed]
            failure_counts = torch.bincount(failed_bin_ids, minlength=self.num_bins).float()
            self.adp_samp_num_failures += failure_counts * self.cfg.failure_counts_multiplier

        self._update_sampling_probabilities()

        # ── unified bin sampling ─────────────────────────────────────
        sampled_bin_ids = torch.multinomial(self.adp_sampling_prob, n, replacement=True)
        bins = self.bins[sampled_bin_ids]  # (n, 3): motion_id, frame_start, frame_end
        self.active_motion_idx[env_ids] = bins[:, 0]

        bin_width = (bins[:, 2] - bins[:, 1]).float()
        self.time_steps[env_ids] = (torch.rand(n, device=self.device) * bin_width).long() + bins[:, 1]

        # pre-failure window: shift backward so policy practices before hard segments
        if self.cfg.pre_failure_sample_window > 0:
            offset = torch.randint(self.cfg.pre_failure_sample_window, (n,), device=self.device)
            self.time_steps[env_ids] = torch.clamp(self.time_steps[env_ids] - offset, min=0)

        # ── metrics ──────────────────────────────────────────────────
        H = -(self.adp_sampling_prob * (self.adp_sampling_prob + 1e-12).log()).sum()
        H_norm = H / math.log(self.num_bins) if self.num_bins > 1 else 0.0
        self.metrics["sampling_entropy"][:] = H_norm
        pmax, imax = self.adp_sampling_prob.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = self.bins[imax, 0].float() / max(self.motion.num_motions, 1)
        self.metrics["active_motion"][:] = self.active_motion_idx.float()

        # ── set robot state ──────────────────────────────────────────
        root_pos = self.motion.body_pos_w[self.current_frame, 0].clone()
        root_ori = self.motion.body_quat_w[self.current_frame, 0].clone()
        root_lin_vel = self.motion.body_lin_vel_w[self.current_frame, 0].clone()
        root_ang_vel = self.motion.body_ang_vel_w[self.current_frame, 0].clone()

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

        joint_pos = self.motion.joint_pos[self.current_frame].clone()
        joint_vel = self.motion.joint_vel[self.current_frame].clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, self.device)
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(joint_pos[env_ids], soft_limits[:, :, 0], soft_limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    # ── per-step update ───────────────────────────────────────────────

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(
            self.time_steps >= self.motion.motion_frames[self.active_motion_idx]
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

        # periodic difficulty logging
        self._step_counter += 1
        if self._step_counter % self.cfg.log_interval_steps == 0:
            # aggregate per-motion failure rates from bins
            motion_failure_rates = torch.zeros(self.motion.num_motions, device=self.device)
            for i in range(self.motion.num_motions):
                mask = self.bins[:, 0] == i
                if mask.any():
                    motion_failure_rates[i] = self.adp_samp_failure_rate[mask].mean()

            _, indices = torch.topk(motion_failure_rates, min(self.cfg.log_topk, self.num_motions))
            print(
                f"\n[MultiMotion] Step {self._step_counter} | "
                f"Failure rates (top {min(self.cfg.log_topk, self.num_motions)}):"
            )
            for rank, midx in enumerate(indices.tolist()):
                print(
                    f"  {rank + 1}. [{midx}] {self.motion_names[midx]}"
                    f" — failure_rate={motion_failure_rates[midx].item():.4f}"
                )
            print()


# ── config ──────────────────────────────────────────────────────────────


@configclass
class MultiMotionCommandCfg(MotionCommandCfg):
    """Configuration for the multi-motion command with unified bin-level adaptive sampling."""

    class_type: type = MultiMotionCommand

    motion_folder: str = MISSING
    motion_file: str = ""  # not used; kept to pass parent validation

    # bin construction
    bin_size: int = 50  # frames per bin (absolute, GR00T-style)

    # adaptive sampling
    uniform_sampling_rate: float = 0.1  # blend with uniform baseline
    failure_rate_max_over_mean: float = 50.0  # clip failure rate outliers
    failure_counts_multiplier: int = 1  # multiply failure counts (>= 1)

    # max probability constraints: "auto" | float | None
    # "auto" = failure_rate_max_over_mean / num_items
    max_prob_per_bin: str | float | None = None
    max_prob_per_motion: str | float | None = None

    # pre-failure window: shift sampled frame backward so policy practices
    # before difficult segments (0 = disabled)
    pre_failure_sample_window: int = 0

    # if True, normalize bin weights by peer bin count so each motion
    # gets equal total sampling weight regardless of duration
    sequence_length_agnostic: bool = True

    # difficulty logging
    log_interval_steps: int = 240  # print top-k hardest motions every N physics steps
    log_topk: int = 10  # number of hardest motions to print
