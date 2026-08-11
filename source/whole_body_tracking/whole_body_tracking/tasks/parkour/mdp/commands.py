"""Motion-to-terrain command for perceptive parkour tracking."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ParkourMotionLoader:
    """Load the explicitly configured parkour motions into flat tensors."""

    def __init__(self, motion_files: list[str], device: str):
        if not motion_files:
            raise ValueError("Parkour requires at least one configured motion file.")

        missing_files = [path for path in motion_files if not os.path.isfile(path)]
        if missing_files:
            raise FileNotFoundError(f"Configured parkour motions do not exist: {missing_files}")

        self.motion_names = [os.path.splitext(os.path.basename(path))[0] for path in motion_files]
        self.num_motions = len(motion_files)

        joint_pos = []
        joint_vel = []
        body_pos_w = []
        body_quat_w = []
        body_lin_vel_w = []
        body_ang_vel_w = []
        motion_frames = []
        motion_fps = []

        for path in motion_files:
            data = np.load(path)
            joint_pos.append(torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=device))
            joint_vel.append(torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=device))
            body_pos_w.append(torch.as_tensor(data["body_pos_w"], dtype=torch.float32, device=device))
            body_quat_w.append(torch.as_tensor(data["body_quat_w"], dtype=torch.float32, device=device))
            body_lin_vel_w.append(
                torch.as_tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            )
            body_ang_vel_w.append(
                torch.as_tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
            )
            motion_frames.append(data["joint_pos"].shape[0])
            motion_fps.append(float(np.asarray(data["fps"]).reshape(-1)[0]))

        self.joint_pos = torch.cat(joint_pos, dim=0)
        self.joint_vel = torch.cat(joint_vel, dim=0)
        self.body_pos_w = torch.cat(body_pos_w, dim=0)
        self.body_quat_w = torch.cat(body_quat_w, dim=0)
        self.body_lin_vel_w = torch.cat(body_lin_vel_w, dim=0)
        self.body_ang_vel_w = torch.cat(body_ang_vel_w, dim=0)
        self.motion_frames = torch.tensor(motion_frames, dtype=torch.long, device=device)
        self.motion_fps = torch.tensor(motion_fps, dtype=torch.float32, device=device)
        self.motion_start = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                self.motion_frames.cumsum(dim=0)[:-1],
            ]
        )
        self.time_step_total = self.joint_pos.shape[0]


class ParkourMotionCommand(CommandTerm):
    """Sample InstinctLab-style concatenated motion bins on paired terrains."""

    cfg: ParkourMotionCommandCfg

    def __init__(self, cfg: ParkourMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        motion_paths = [os.path.join(cfg.motion_folder, name) for name in cfg.motion_files]
        self.motion = ParkourMotionLoader(motion_paths, device=self.device)
        self.num_motions = self.motion.num_motions
        self.motion_names = self.motion.motion_names
        self._validate_terrain_bindings(env)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        self._build_motion_bins()
        self.motion_bin_fail_counter = torch.zeros(
            self.num_bins, dtype=torch.float32, device=self.device
        )
        self.current_motion_bin_fail_counter = torch.zeros_like(self.motion_bin_fail_counter)
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
            dtype=torch.float32,
            device=self.device,
        )
        self.kernel /= self.kernel.sum()
        self._update_sampling_probabilities()

        self._step_counter = 0
        for name in [
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
            "active_motion",
        ]:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

        print(f"[INFO] ParkourMotionCommand loaded {self.num_motions} configured motions:")
        for motion_idx, name in enumerate(self.motion_names):
            print(
                f"  [{motion_idx}] {name} — "
                f"{self.motion.motion_frames[motion_idx].item()} frames"
            )

    def _validate_terrain_bindings(self, env: ManagerBasedRLEnv):
        configured_names = set(self.cfg.motion_terrain_columns)
        loaded_names = set(self.motion_names)
        if configured_names != loaded_names:
            missing = sorted(loaded_names - configured_names)
            extra = sorted(configured_names - loaded_names)
            raise ValueError(
                "Every parkour motion must have exactly one terrain binding. "
                f"Missing bindings: {missing}; unknown bindings: {extra}"
            )

        terrain_origins = env.scene.terrain.terrain_origins
        if terrain_origins is None or terrain_origins.ndim != 3:
            raise ValueError("Parkour requires generated terrain with row/column origins.")

        columns = [self.cfg.motion_terrain_columns[name] for name in self.motion_names]
        if min(columns) < 0 or max(columns) >= terrain_origins.shape[1]:
            raise ValueError(
                f"Terrain column binding {columns} exceeds generated terrain columns "
                f"[0, {terrain_origins.shape[1] - 1}]."
            )
        self._terrain_origins_by_motion = torch.stack(
            [terrain_origins[:, column, :] for column in columns], dim=0
        )

    def _build_motion_bins(self):
        """Split each motion into one-second bins and concatenate their indices."""
        bins = []
        bins_per_motion = []
        self.frame_to_bin = torch.zeros(
            self.motion.time_step_total, dtype=torch.long, device=self.device
        )

        global_bin_idx = 0
        for motion_idx in range(self.num_motions):
            num_frames = int(self.motion.motion_frames[motion_idx].item())
            bin_size = max(
                1,
                int(
                    round(
                        self.motion.motion_fps[motion_idx].item()
                        * self.cfg.motion_bin_length_s
                    )
                ),
            )
            motion_bins = 0
            for frame_start in range(0, num_frames, bin_size):
                frame_end = min(frame_start + bin_size, num_frames)
                bins.append((motion_idx, frame_start, frame_end))
                global_start = int(self.motion.motion_start[motion_idx].item()) + frame_start
                self.frame_to_bin[global_start : global_start + frame_end - frame_start] = global_bin_idx
                global_bin_idx += 1
                motion_bins += 1
            bins_per_motion.append(motion_bins)

        self.bins = torch.tensor(bins, dtype=torch.long, device=self.device)
        self.num_bins = len(bins)
        self.bins_per_motion = torch.tensor(bins_per_motion, dtype=torch.long, device=self.device)
        self.bin_motion_starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=self.device),
                self.bins_per_motion.cumsum(dim=0)[:-1],
            ]
        )
        self.num_peer_bins = torch.repeat_interleave(self.bins_per_motion, self.bins_per_motion)

    def _update_sampling_probabilities(self):
        """Implement InstinctLab's BeyondConcatMotionAdaptiveWeighting."""
        probability = self.motion_bin_fail_counter + (
            self.cfg.adaptive_uniform_ratio / self.num_peer_bins.float()
        )
        smoothed = torch.empty_like(probability)

        for motion_idx in range(self.num_motions):
            start = int(self.bin_motion_starts[motion_idx].item())
            count = int(self.bins_per_motion[motion_idx].item())
            for local_bin_idx in range(count):
                value = torch.zeros((), dtype=torch.float32, device=self.device)
                for kernel_idx in range(self.cfg.adaptive_kernel_size):
                    source_idx = start + min(local_bin_idx + kernel_idx, count - 1)
                    value += self.kernel[kernel_idx] * probability[source_idx]
                smoothed[start + local_bin_idx] = value

        self.sampling_probabilities = smoothed / smoothed.sum().clamp_min(1.0e-12)

    def _smooth_failure_counter(self):
        self.motion_bin_fail_counter.mul_(1.0 - self.cfg.adaptive_alpha)
        self.motion_bin_fail_counter.add_(
            self.cfg.adaptive_alpha * self.current_motion_bin_fail_counter
        )
        self.current_motion_bin_fail_counter.zero_()
        self._update_sampling_probabilities()

    @property
    def current_frame(self) -> torch.Tensor:
        return self.motion.motion_start[self.active_motion_idx] + self.time_steps

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
        return self.motion.body_pos_w[self.current_frame][:, self.body_indexes] + self.motion_origins[:, None, :]

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
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def command_window(self) -> torch.Tensor:
        half_size = self.cfg.command_window_half_size
        local_indices = self.time_steps[:, None] + torch.arange(
            -half_size, half_size + 1, device=self.device
        )
        last_frames = self.motion.motion_frames[self.active_motion_idx, None] - 1
        local_indices = torch.minimum(torch.clamp(local_indices, min=0), last_frames)
        indices = local_indices + self.motion.motion_start[self.active_motion_idx, None]
        return torch.cat([self.motion.joint_pos[indices], self.motion.joint_vel[indices]], dim=-1)

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def _record_failures(self, env_ids: torch.Tensor):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if not episode_failed.any():
            return

        last_valid_frames = self.motion.motion_frames[self.active_motion_idx[env_ids]] - 1
        old_time_steps = torch.minimum(self.time_steps[env_ids], last_valid_frames)
        old_global_frames = self.motion.motion_start[self.active_motion_idx[env_ids]] + old_time_steps
        failed_bin_ids = self.frame_to_bin[old_global_frames[episode_failed]]
        self.current_motion_bin_fail_counter += torch.bincount(
            failed_bin_ids, minlength=self.num_bins
        ).float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._record_failures(env_ids)

        sampled_bin_ids = torch.multinomial(
            self.sampling_probabilities, len(env_ids), replacement=True
        )
        bins = self.bins[sampled_bin_ids]
        self.active_motion_idx[env_ids] = bins[:, 0]
        bin_width = (bins[:, 2] - bins[:, 1]).float()
        self.time_steps[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * bin_width
        ).long() + bins[:, 1]

        motion_ids = self.active_motion_idx[env_ids]
        terrain_copy_ids = torch.randint(
            self._terrain_origins_by_motion.shape[1], (len(env_ids),), device=self.device
        )
        origins = self._terrain_origins_by_motion[motion_ids, terrain_copy_ids]
        self.motion_origins[env_ids] = origins

        root_pos = self.motion.body_pos_w[self.current_frame, 0].clone()
        root_ori = self.motion.body_quat_w[self.current_frame, 0].clone()
        root_lin_vel = self.motion.body_lin_vel_w[self.current_frame, 0].clone()
        root_ang_vel = self.motion.body_ang_vel_w[self.current_frame, 0].clone()

        pose_ranges = torch.tensor(
            [
                self.cfg.pose_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ],
            device=self.device,
        )
        pose_noise = sample_uniform(
            pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += pose_noise[:, :3]
        orientation_delta = quat_from_euler_xyz(
            pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientation_delta, root_ori[env_ids])

        velocity_ranges = torch.tensor(
            [
                self.cfg.velocity_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ],
            device=self.device,
        )
        velocity_noise = sample_uniform(
            velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += velocity_noise[:, :3]
        root_ang_vel[env_ids] += velocity_noise[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(
            *self.cfg.joint_position_range, joint_pos.shape, device=self.device
        )
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_limits[:, :, 0], soft_limits[:, :, 1]
        )

        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        root_pos[env_ids] += origins
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_ori[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

        entropy = -(
            self.sampling_probabilities * (self.sampling_probabilities + 1.0e-12).log()
        ).sum()
        normalized_entropy = entropy / math.log(self.num_bins) if self.num_bins > 1 else 0.0
        top_probability, top_bin = self.sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = normalized_entropy
        self.metrics["sampling_top1_prob"][:] = top_probability
        self.metrics["sampling_top1_bin"][:] = top_bin.float() / max(self.num_bins, 1)
        self.metrics["active_motion"][:] = self.active_motion_idx.float()

    def _update_command(self):
        self._smooth_failure_counter()
        self.time_steps += 1
        ended_env_ids = torch.where(
            self.time_steps >= self.motion.motion_frames[self.active_motion_idx]
        )[0]
        self._resample_command(ended_env_ids)

        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        delta_pos = robot_anchor_pos.clone()
        delta_pos[..., 2] = anchor_pos[..., 2]
        delta_quat = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_quat, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(
            delta_quat, self.body_pos_w - anchor_pos
        )

        self._step_counter += 1
        if self.cfg.log_interval_steps > 0 and self._step_counter % self.cfg.log_interval_steps == 0:
            hardest_probability, hardest_bin = self.sampling_probabilities.max(dim=0)
            print(
                f"[ParkourSampling] step={self._step_counter} "
                f"entropy={self.metrics['sampling_entropy'][0].item():.4f} "
                f"top_bin={hardest_bin.item()} probability={hardest_probability.item():.6f}"
            )
            top_count = min(5, self.num_bins)
            top_probabilities, top_bins = torch.topk(
                self.sampling_probabilities, k=top_count
            )
            for rank, (probability, global_bin) in enumerate(
                zip(top_probabilities.tolist(), top_bins.tolist()), start=1
            ):
                motion_idx, frame_start, frame_end = self.bins[global_bin].tolist()
                local_bin = global_bin - int(self.bin_motion_starts[motion_idx].item())
                fps = float(self.motion.motion_fps[motion_idx].item())
                print(
                    f"  Top {rank}: bin={global_bin} "
                    f"motion={self.motion_names[motion_idx]} "
                    f"local_bin={local_bin} "
                    f"frames=[{frame_start},{frame_end}) "
                    f"time=[{frame_start / fps:.2f},{frame_end / fps:.2f})s "
                    f"probability={probability:.6f}"
                )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, "current_anchor_visualizer"):
            self.current_anchor_visualizer = VisualizationMarkers(
                self.cfg.anchor_visualizer_cfg.replace(
                    prim_path="/Visuals/Command/current/anchor"
                )
            )
            self.goal_anchor_visualizer = VisualizationMarkers(
                self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
            )
            self.current_body_visualizers = [
                VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/current/{name}"
                    )
                )
                for name in self.cfg.body_names
            ]
            self.goal_body_visualizers = [
                VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/goal/{name}"
                    )
                )
                for name in self.cfg.body_names
            ]

        if hasattr(self, "current_anchor_visualizer"):
            self.current_anchor_visualizer.set_visibility(debug_vis)
            self.goal_anchor_visualizer.set_visibility(debug_vis)
            for visualizer in self.current_body_visualizers + self.goal_body_visualizers:
                visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.current_anchor_visualizer.visualize(
            self.robot_anchor_pos_w, self.robot_anchor_quat_w
        )
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)
        for body_idx in range(len(self.cfg.body_names)):
            self.current_body_visualizers[body_idx].visualize(
                self.robot_body_pos_w[:, body_idx], self.robot_body_quat_w[:, body_idx]
            )
            self.goal_body_visualizers[body_idx].visualize(
                self.body_pos_relative_w[:, body_idx], self.body_quat_relative_w[:, body_idx]
            )


@configclass
class ParkourMotionCommandCfg(CommandTermCfg):
    """Standalone parkour motion command configuration."""

    class_type: type = ParkourMotionCommand

    asset_name: str = MISSING
    motion_folder: str = MISSING
    motion_files: list[str] = MISSING
    motion_terrain_columns: dict[str, int] = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    motion_bin_length_s: float = 1.0
    adaptive_uniform_ratio: float = 0.1
    adaptive_kernel_size: int = 3
    adaptive_alpha: float = 0.001
    adaptive_lambda: float = 0.8
    command_window_half_size: int = 10
    log_interval_steps: int = 240

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
