"""Online diffusion-generated motion commands for CASBOT parkour."""

from __future__ import annotations

import importlib.util
import time
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import torch
import numpy as np

from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    yaw_quat,
)

from .commands import ParkourMotionCommand, ParkourMotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _load_python_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_numpy_pickle_compatibility() -> None:
    """Let NumPy-1.x load checkpoints pickled by NumPy-2.x."""
    import sys

    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def _quat_from_rpy(rpy: tuple[float, float, float], device: torch.device) -> torch.Tensor:
    values = [torch.tensor([value], dtype=torch.float32, device=device) for value in rpy]
    return quat_from_euler_xyz(*values)[0]


def _rot6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
    matrix = matrix_from_quat(quat)
    return torch.cat((matrix[..., :, 0], matrix[..., :, 2]), dim=-1)


def _quat_from_rot6d(d6: torch.Tensor) -> torch.Tensor:
    col0 = torch.nn.functional.normalize(d6[..., :3], dim=-1)
    col2 = d6[..., 3:6]
    col2 = col2 - (col0 * col2).sum(dim=-1, keepdim=True) * col0
    col2 = torch.nn.functional.normalize(col2, dim=-1)
    col1 = torch.cross(col2, col0, dim=-1)
    matrix = torch.stack((col0, col1, col2), dim=-1)

    # Stable batched matrix-to-quaternion conversion, returning wxyz.
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1.0e-8))
    qx = torch.copysign(
        0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1.0e-8)),
        matrix[..., 2, 1] - matrix[..., 1, 2],
    )
    qy = torch.copysign(
        0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1.0e-8)),
        matrix[..., 0, 2] - matrix[..., 2, 0],
    )
    qz = torch.copysign(
        0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1.0e-8)),
        matrix[..., 1, 0] - matrix[..., 0, 1],
    )
    return torch.nn.functional.normalize(torch.stack((qw, qx, qy, qz), dim=-1), dim=-1)


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Batched shortest-path SLERP for wxyz quaternions."""
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    slerp = (
        torch.sin((1.0 - alpha) * theta) / sin_theta.clamp_min(1.0e-8) * q0
        + torch.sin(alpha * theta) / sin_theta.clamp_min(1.0e-8) * q1
    )
    lerp = torch.nn.functional.normalize(q0 + alpha * (q1 - q0), dim=-1)
    return torch.where(dot > 0.9995, lerp, slerp)


class _UrdfForwardKinematics:
    """Small batched FK tree rooted at the diffusion root link."""

    def __init__(
        self,
        urdf_path: str,
        root_link: str,
        joint_names: Sequence[str],
        device: torch.device,
    ):
        self.device = device
        self.joint_names = tuple(joint_names)
        xml_root = ET.parse(urdf_path).getroot()
        adjacency: dict[str, list[tuple[str, ET.Element, bool]]] = {}
        for joint in xml_root.findall("joint"):
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            adjacency.setdefault(parent, []).append((child, joint, True))
            adjacency.setdefault(child, []).append((parent, joint, False))

        self.steps: list[tuple[str, str, str | None, bool, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        visited = {root_link}
        queue = deque([root_link])
        while queue:
            source = queue.popleft()
            for target, joint, forward in adjacency.get(source, []):
                if target in visited:
                    continue
                visited.add(target)
                queue.append(target)
                origin = joint.find("origin")
                xyz = tuple(float(v) for v in (origin.attrib.get("xyz", "0 0 0").split()))
                rpy = tuple(float(v) for v in (origin.attrib.get("rpy", "0 0 0").split()))
                axis_node = joint.find("axis")
                axis = tuple(
                    float(v)
                    for v in (
                        axis_node.attrib.get("xyz", "1 0 0").split()
                        if axis_node is not None
                        else ("1", "0", "0")
                    )
                )
                joint_name = joint.attrib["name"] if joint.attrib.get("type") != "fixed" else None
                self.steps.append(
                    (
                        source,
                        target,
                        joint_name,
                        forward,
                        torch.tensor(xyz, dtype=torch.float32, device=device),
                        _quat_from_rpy(rpy, device),
                        torch.tensor(axis, dtype=torch.float32, device=device),
                    )
                )

        self.root_link = root_link
        self.links = visited

    def __call__(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        body_names: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = root_pos.shape[0]
        joint_values = {
            name: joint_pos[:, index] for index, name in enumerate(self.joint_names)
        }
        positions = {self.root_link: root_pos}
        quaternions = {self.root_link: root_quat}

        for source, target, joint_name, forward, origin_pos, origin_quat, axis in self.steps:
            source_pos = positions[source]
            source_quat = quaternions[source]
            if joint_name is None:
                joint_quat = torch.zeros((batch, 4), device=root_pos.device)
                joint_quat[:, 0] = 1.0
            else:
                angle = joint_values[joint_name]
                half = 0.5 * angle
                joint_quat = torch.cat(
                    (torch.cos(half)[:, None], axis[None, :] * torch.sin(half)[:, None]), dim=-1
                )
            edge_quat = quat_mul(origin_quat[None, :].expand(batch, 4), joint_quat)
            edge_pos = origin_pos[None, :].expand(batch, 3)
            if not forward:
                edge_quat = quat_inv(edge_quat)
                edge_pos = quat_apply(edge_quat, -edge_pos)
            positions[target] = source_pos + quat_apply(source_quat, edge_pos)
            quaternions[target] = quat_mul(source_quat, edge_quat)

        missing = [name for name in body_names if name not in positions]
        if missing:
            raise KeyError(f"URDF FK cannot resolve configured bodies: {missing}")
        return (
            torch.stack([positions[name] for name in body_names], dim=1),
            torch.stack([quaternions[name] for name in body_names], dim=1),
        )


class DiffusionParkourMotionCommand(ParkourMotionCommand):
    """Generate sparse Flow-Matching keyframes and track their dense interpolation."""

    cfg: "DiffusionParkourMotionCommandCfg"

    def __init__(self, cfg: "DiffusionParkourMotionCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        checkpoint_path = Path(cfg.diffusion_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Diffusion checkpoint does not exist: {checkpoint_path}")
        diffusion_root = Path(cfg.diffusion_root)
        model_module = _load_python_module(
            "wbt_parkour_diffusion_model", diffusion_root / "source/common/model.py"
        )
        _install_numpy_pickle_compatibility()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_cfg = checkpoint["cfg"]
        required_semantics = {
            "diffusion_format_version": 7,
            "root_body": "waist_yaw_link",
            "terrain_layout": "root_z_minus_terrain_z",
            "motion_layout": "future_h0_heading_root_xyz_offset",
            "proprio_layout": "joint_pos,velocity_command_local",
            "joint_layout": "isaaclab_articulation",
        }
        mismatched = {
            key: model_cfg.get(key) for key, expected in required_semantics.items()
            if model_cfg.get(key) != expected
        }
        if mismatched:
            raise ValueError(
                "Diffusion checkpoint does not use the required velocity-command format-v7 "
                f"semantics: {mismatched}. Rebuild the conditional NPZ dataset and retrain."
            )
        self.diffusion_model = model_module.DiffusionDenoiser(
            feature_dim=int(model_cfg["feature_dim"]),
            window_size=int(model_cfg["window_size"]),
            d_model=int(model_cfg.get("d_model", 256)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            head_dim=model_cfg.get("head_dim"),
            terrain_dim=int(model_cfg["terrain_dim"]),
            terrain_height=int(model_cfg.get("terrain_height", 21)),
            terrain_width=int(model_cfg.get("terrain_width", 33)),
            terrain_feature_dim=int(model_cfg.get("terrain_feature_dim", 48)),
            proprio_dim=int(model_cfg.get("proprio_dim", 28)),
        ).to(self.device)
        state = checkpoint.get("model_ema", checkpoint["model"])
        self.diffusion_model.load_state_dict(state, strict=True)
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)
        self.generative_method = str(model_cfg.get("generative_method", "ddpm"))
        if self.generative_method == "ddpm":
            scheduler_module = _load_python_module(
                "wbt_parkour_diffusion_scheduler", diffusion_root / "source/ddpm/scheduler.py"
            )
            self.diffusion_scheduler = scheduler_module.DDPMScheduler(
                num_timesteps=int(model_cfg.get("num_timesteps", 50))
            ).to(self.device)
            self.diffusion_flow_sampler = None
            self.diffusion_steps = self.diffusion_scheduler.num_timesteps
        elif self.generative_method == "flow_matching":
            flow_module = _load_python_module(
                "wbt_parkour_flow_sampler", diffusion_root / "source/flow_matching/sampler.py"
            )
            self.diffusion_scheduler = None
            self.diffusion_flow_sampler = flow_module.sample_flow
            self.diffusion_steps = int(model_cfg.get("sampling_steps", 10))
            self.flow_sampler_method = str(model_cfg.get("sampler", "euler"))
            self.flow_time_embedding_scale = float(model_cfg.get("time_embedding_scale", 1000.0))
        else:
            raise ValueError(f"Unknown generative_method={self.generative_method!r}")
        self.future_size = int(model_cfg.get("future_size", model_cfg["window_size"]))
        # ``future_frame_stride`` describes the spacing used when the training
        # windows were built.  It is not the stride of the 50-Hz simulator
        # control loop.  The current model was trained from 10-Hz data with
        # stride=1, so each predicted keyframe is 5 simulator frames apart.
        self.future_frame_stride = int(model_cfg.get("future_frame_stride", 1))
        self.history_size = int(model_cfg.get("history_size", 2))
        if (self.future_size, self.history_size) != (10, 2):
            raise ValueError(
                f"Expected diffusion future/history=(10,2), got "
                f"({self.future_size},{self.history_size})"
            )
        if self.future_frame_stride <= 0:
            raise ValueError(
                f"Invalid future_frame_stride={self.future_frame_stride} in checkpoint"
            )

        self.diffusion_fps = float(cfg.diffusion_fps)
        motion_fps = self.motion.motion_fps
        if len(motion_fps) == 0 or torch.any(
            torch.abs(motion_fps - motion_fps[0]) > 1.0e-3
        ):
            raise ValueError(
                "Diffusion parkour requires all reset motions to have the same fps; "
                f"got {motion_fps.detach().cpu().tolist()}"
            )
        self.control_fps = float(motion_fps[0].item())
        history_stride_float = self.control_fps / self.diffusion_fps
        self.history_sample_stride = int(round(history_stride_float))
        if self.history_sample_stride <= 0 or abs(
            history_stride_float - self.history_sample_stride
        ) > 1.0e-4:
            raise ValueError(
                f"Motion fps ({self.control_fps}) must be an integer multiple of "
                f"diffusion_fps ({self.diffusion_fps})"
            )
        # A checkpoint stride>1 means that adjacent model outputs are separated
        # by multiple 10-Hz samples.  Convert that semantic spacing to control
        # frames before interpolation.
        self.runtime_frame_stride = self.history_sample_stride * self.future_frame_stride
        self.dense_future_size = self.future_size * self.runtime_frame_stride
        self.history_span = (self.history_size - 1) * self.history_sample_stride
        self.tracking_horizon_steps = int(cfg.tracking_horizon_steps)
        if not 1 <= self.tracking_horizon_steps <= self.dense_future_size:
            raise ValueError(
                f"tracking_horizon_steps must be in [1,{self.dense_future_size}], got "
                f"{self.tracking_horizon_steps}"
            )

        def _stat(name: str) -> torch.Tensor:
            return torch.as_tensor(checkpoint[name], dtype=torch.float32, device=self.device)

        self.q_low, self.q_high = _stat("q_low"), _stat("q_high")
        self.t_q_low, self.t_q_high = _stat("t_q_low"), _stat("t_q_high")
        self.p_q_low, self.p_q_high = _stat("p_q_low"), _stat("p_q_high")

        urdf_path = Path(cfg.robot_urdf)
        self._fk = _UrdfForwardKinematics(
            str(urdf_path), cfg.anchor_body_name, self.robot.joint_names, self.device
        )
        self._robot_root_fk = _UrdfForwardKinematics(
            str(urdf_path), self.robot.body_names[0], self.robot.joint_names, self.device
        )
        self._validate_fk_against_motion()
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor_name]
        root_body_name = self.robot.body_names[0]
        if root_body_name not in cfg.body_names:
            raise ValueError(
                f"Generated-F0 initialization requires root body {root_body_name!r} "
                "to be present in body_names."
            )
        self._root_reference_body_index = cfg.body_names.index(root_body_name)

        body_count = len(cfg.body_names)
        self._joint_pos = torch.zeros((self.num_envs, 25), device=self.device)
        self._joint_vel = torch.zeros_like(self._joint_pos)
        self._body_pos_w = torch.zeros((self.num_envs, body_count, 3), device=self.device)
        self._body_quat_w = torch.zeros((self.num_envs, body_count, 4), device=self.device)
        self._body_quat_w[..., 0] = 1.0
        self._body_lin_vel_w = torch.zeros_like(self._body_pos_w)
        self._body_ang_vel_w = torch.zeros_like(self._body_pos_w)
        self._previous_body_pos_w = self._body_pos_w.clone()
        self._previous_body_quat_w = self._body_quat_w.clone()

        self._joint_history = torch.zeros((self.num_envs, self.history_size, 25), device=self.device)
        self._velocity_command = torch.zeros((self.num_envs, 3), device=self.device)
        self._command_age_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._generated_motion = torch.zeros(
            (self.num_envs, self.dense_future_size, 80), device=self.device
        )
        self._generated_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._generated_motion_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._generated_anchor_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._generated_anchor_yaw = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self._generated_anchor_yaw[:, 0] = 1.0
        self._terrain_history = torch.zeros((self.num_envs, self.history_size, 693), device=self.device)
        self._root_pos_history = torch.zeros((self.num_envs, self.history_size, 3), device=self.device)
        self._yaw_history = torch.zeros((self.num_envs, self.history_size, 4), device=self.device)
        self._yaw_history[..., 0] = 1.0
        self._history_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._skip_history_append = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._history_update_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._hold_initialized_reference = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.motion.elevation_map_xyz is None:
            raise ValueError(
                "Diffusion reset motions must contain elevation_map_xyz so the "
                f"H0-H{self.history_size - 1} terrain history can be initialized."
            )
        self._active_terrain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._terrain_origins = env.scene.terrain.terrain_origins
        if self._terrain_origins is None or self._terrain_origins.shape[1] != len(cfg.reset_motion_groups):
            raise ValueError(
                "Diffusion terrain columns must match reset_motion_groups: "
                f"origins={None if self._terrain_origins is None else self._terrain_origins.shape}, "
                f"groups={len(cfg.reset_motion_groups)}"
            )

        name_to_index = {name: index for index, name in enumerate(self.motion_names)}
        self._reset_motion_groups = []
        for group in cfg.reset_motion_groups:
            unknown = [name for name in group if name not in name_to_index]
            if unknown:
                raise ValueError(f"Unknown reset motions: {unknown}")
            self._reset_motion_groups.append(
                torch.tensor([name_to_index[name] for name in group], device=self.device)
            )
        self._reset_motion_frames = self.motion.motion_frames.clone()
        for motion_index in range(self.num_motions):
            start = int(self.motion.motion_start[motion_index])
            count = int(self.motion.motion_frames[motion_index])
            root_xy = self.motion.body_pos_w[start : start + count, 0, :2]
            jumps = torch.where(
                torch.linalg.vector_norm(torch.diff(root_xy, dim=0), dim=-1)
                > cfg.reset_max_root_step
            )[0]
            if len(jumps) > 0:
                self._reset_motion_frames[motion_index] = jumps[0] + 1
                print(
                    f"[DiffusionReset] truncating {self.motion_names[motion_index]} "
                    f"from {count} to {int(jumps[0]) + 1} frames at a root discontinuity"
                )

        # Commands are sampled directly from the configured ranges. They are
        # intentionally independent of the selected motion sequence; the
        # command conditions Flow Matching, while the generated motion remains
        # the RL tracking target.
        command_low = torch.tensor(
            [
                self.cfg.command_vx_range[0],
                self.cfg.command_vy_range[0],
                self.cfg.command_wz_range[0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        command_high = torch.tensor(
            [
                self.cfg.command_vx_range[1],
                self.cfg.command_vy_range[1],
                self.cfg.command_wz_range[1],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._command_low = command_low
        self._command_high = command_high

        self._sample_counter = 0
        self._last_sample_ms = 0.0
        self.metrics["diffusion_sample_ms"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["diffusion_env_steps_per_s"] = torch.zeros(self.num_envs, device=self.device)
        print(
            f"[INFO] Frozen diffusion model: {checkpoint_path} "
            f"(method={self.generative_method}, steps={self.diffusion_steps}, "
            f"history={self.history_size}, keyframes={self.future_size}, "
            f"training_stride={self.future_frame_stride}, "
            f"control_fps={self.control_fps:g}, diffusion_fps={self.diffusion_fps:g}, "
            f"runtime_keyframe_stride={self.runtime_frame_stride}, "
            f"dense_future={self.dense_future_size}, "
            f"tracking_horizon={self.tracking_horizon_steps})"
        )

    @torch.inference_mode()
    def _validate_fk_against_motion(self) -> None:
        """Catch URDF/body/joint-order mismatches before training starts."""
        local_frames = torch.tensor(
            [0, max(int(self.motion.motion_frames[0]) // 2, 0)],
            dtype=torch.long,
            device=self.device,
        )
        frames = self.motion.motion_start[0] + local_frames
        anchor_body_id = int(self.body_indexes[self.motion_anchor_body_index])
        root_pos = self.motion.body_pos_w[frames, anchor_body_id]
        root_quat = self.motion.body_quat_w[frames, anchor_body_id]
        joint_lab = self.motion.joint_pos[frames]
        predicted_pos, _ = self._fk(
            root_pos, root_quat, joint_lab, self.cfg.body_names
        )
        expected_pos = self.motion.body_pos_w[frames][:, self.body_indexes]
        error = torch.linalg.vector_norm(predicted_pos - expected_pos, dim=-1)
        print(
            f"[DiffusionFK] validation mean_pos_error={error.mean().item():.6f}m "
            f"max_pos_error={error.max().item():.6f}m"
        )
        if error.max() > 0.03:
            raise ValueError(
                "Diffusion FK does not match reset motion body states; "
                f"maximum position error is {error.max().item():.6f} m"
            )

    @property
    def command(self) -> torch.Tensor:
        # Give the low-level policy the generated global-root target in a
        # placement-invariant robot-local form, followed by joint targets.
        # The policy therefore has direct inputs corresponding to the global
        # root position/orientation rewards instead of having to infer them
        # from joint targets alone.
        root_pos_error_b = quat_apply(
            quat_inv(self.robot_anchor_quat_w),
            self.anchor_pos_w - self.robot_anchor_pos_w,
        )
        root_quat_error_b = quat_mul(
            quat_inv(self.robot_anchor_quat_w), self.anchor_quat_w
        )
        target_lin_vel_b = quat_apply(
            quat_inv(self.robot_anchor_quat_w), self.anchor_lin_vel_w
        )
        target_ang_vel_b = quat_apply(
            quat_inv(self.robot_anchor_quat_w), self.anchor_ang_vel_w
        )
        return torch.cat(
            (
                root_pos_error_b,
                _rot6d_from_quat(root_quat_error_b),
                target_lin_vel_b,
                target_ang_vel_b,
                self._joint_pos,
                self._joint_vel,
            ),
            dim=-1,
        )

    @property
    def velocity_command(self) -> torch.Tensor:
        """Episode command [forward vx, left vy, yaw wz] in the local heading frame."""
        return self._velocity_command

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w

    @property
    def command_window(self) -> torch.Tensor:
        return self.command[:, None, :].expand(-1, 2 * self.cfg.command_window_half_size + 1, -1)

    def _current_terrain_clearance(self) -> tuple[torch.Tensor, torch.Tensor]:
        ray_hits_w = self._height_sensor.data.ray_hits_w
        waist_z = self.robot_anchor_pos_w[:, 2:3]
        hit_z = ray_hits_w[..., 2]
        finite = torch.isfinite(hit_z)
        fallback_z = waist_z - self.cfg.default_root_clearance
        hit_z = torch.where(finite, hit_z, fallback_z)
        clearance = torch.clamp(waist_z - hit_z, min=-20.0, max=20.0)
        return clearance, hit_z

    def _append_actual_history(self, env_ids: torch.Tensor | None = None, repeat: bool = False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        joint_lab = self.robot_joint_pos[env_ids]
        clearance, _ = self._current_terrain_clearance()
        actual_yaw = yaw_quat(self.robot_anchor_quat_w[env_ids])
        actual_root_pos = self.robot_anchor_pos_w[env_ids]
        if repeat:
            self._joint_history[env_ids] = joint_lab[:, None, :]
            self._terrain_history[env_ids] = clearance[env_ids, None, :]
            self._root_pos_history[env_ids] = actual_root_pos[:, None, :]
            self._yaw_history[env_ids] = actual_yaw[:, None, :]
        else:
            self._joint_history[env_ids] = torch.roll(self._joint_history[env_ids], -1, dims=1)
            self._terrain_history[env_ids] = torch.roll(self._terrain_history[env_ids], -1, dims=1)
            self._root_pos_history[env_ids] = torch.roll(self._root_pos_history[env_ids], -1, dims=1)
            self._yaw_history[env_ids] = torch.roll(self._yaw_history[env_ids], -1, dims=1)
            self._joint_history[env_ids, -1] = joint_lab
            self._terrain_history[env_ids, -1] = clearance[env_ids]
            self._root_pos_history[env_ids, -1] = actual_root_pos
            self._yaw_history[env_ids, -1] = actual_yaw
        self._history_ready[env_ids] = True

    def _normalized_conditions(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        repeated_command = self._velocity_command[env_ids, None, :].expand(
            -1, self.history_size, -1
        )
        proprio = torch.cat(
            (
                self._joint_history[env_ids],
                repeated_command,
            ),
            dim=-1,
        )
        terrain = 2.0 * (self._terrain_history[env_ids] - self.t_q_low) / (
            self.t_q_high - self.t_q_low
        ) - 1.0
        proprio = 2.0 * (proprio - self.p_q_low) / (self.p_q_high - self.p_q_low) - 1.0
        return terrain, proprio

    @torch.inference_mode()
    def _sample_motion(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        batch_size = len(env_ids)
        terrain, proprio = self._normalized_conditions(env_ids)
        x_t = torch.randn((batch_size, self.future_size, 80), device=self.device)
        should_time = self._sample_counter == 0 or (
            self.cfg.timing_interval_steps > 0
            and self._sample_counter % self.cfg.timing_interval_steps == 0
        )
        if should_time and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        if self.generative_method == "ddpm":
            for step in reversed(range(self.diffusion_steps)):
                timestep = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
                predicted_noise = self.diffusion_model(
                    x_t, timestep, terrain=terrain, proprio=proprio
                )
                x_t = self.diffusion_scheduler.step(predicted_noise, x_t, step)
        else:
            x_t = self.diffusion_flow_sampler(
                self.diffusion_model, x_t, terrain, proprio,
                self.diffusion_steps, self.flow_sampler_method,
                self.flow_time_embedding_scale,
            )
        if should_time and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
            self._last_sample_ms = 1000.0 * (time.perf_counter() - start)
            throughput = batch_size / max(self._last_sample_ms * 1.0e-3, 1.0e-9)
            print(
                f"[DiffusionTiming] envs={batch_size} steps={self.diffusion_steps} "
                f"sample_ms={self._last_sample_ms:.3f} env_samples_per_s={throughput:.1f}"
            )
        self._sample_counter += 1
        self.metrics["diffusion_sample_ms"][env_ids] = self._last_sample_ms
        self.metrics["diffusion_env_steps_per_s"][env_ids] = batch_size / max(
            self._last_sample_ms * 1.0e-3, 1.0e-9
        )
        motion = (x_t + 1.0) * 0.5 * (self.q_high - self.q_low) + self.q_low
        if not torch.isfinite(motion).all():
            raise RuntimeError("Diffusion sampler produced non-finite motion reference")
        return motion

    def _densify_motion(
        self, sparse_motion: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate the last-history pose to sparse keyframes.

        The checkpoint predicts ten keyframes at the configured diffusion rate.
        Starting from the robot's actual state at this block boundary and
        interpolating each keyframe to the simulator rate makes the first
        generated reference continuous even when the low-level policy has
        tracking error.
        """
        batch_size = len(env_ids)
        dense = torch.empty(
            (batch_size, self.dense_future_size, sparse_motion.shape[-1]),
            dtype=sparse_motion.dtype,
            device=self.device,
        )
        anchor_pos = self._generated_anchor_pos[env_ids]
        anchor_yaw = self._generated_anchor_yaw[env_ids]
        previous_pos = quat_apply(
            quat_inv(anchor_yaw), self.robot_anchor_pos_w[env_ids] - anchor_pos
        )
        previous_quat = quat_mul(
            quat_inv(anchor_yaw), self.robot_anchor_quat_w[env_ids]
        )
        previous_joint_pos = self.robot_joint_pos[env_ids]
        previous_joint_vel = self.robot_joint_vel[env_ids]
        # These auxiliary output fields are not consumed by the RL reference
        # decoder, but interpolate them as well to keep the tensor coherent.
        previous_aux = sparse_motion[:, 0, 59:]

        for key_index in range(self.future_size):
            target = sparse_motion[:, key_index]
            target_pos = target[:, :3]
            target_quat = _quat_from_rot6d(target[:, 3:9])
            target_joint_pos = target[:, 9:34]
            target_joint_vel = target[:, 34:59]
            target_aux = target[:, 59:]
            segment_start = key_index * self.runtime_frame_stride
            for substep in range(1, self.runtime_frame_stride + 1):
                alpha = substep / float(self.runtime_frame_stride)
                dense_index = segment_start + substep - 1
                dense[:, dense_index, :3] = torch.lerp(previous_pos, target_pos, alpha)
                dense[:, dense_index, 3:9] = _rot6d_from_quat(
                    _quat_slerp(previous_quat, target_quat, alpha)
                )
                dense[:, dense_index, 9:34] = torch.lerp(
                    previous_joint_pos, target_joint_pos, alpha
                )
                dense[:, dense_index, 34:59] = torch.lerp(
                    previous_joint_vel, target_joint_vel, alpha
                )
                dense[:, dense_index, 59:] = torch.lerp(
                    previous_aux, target_aux, alpha
                )
            previous_pos = target_pos
            previous_quat = target_quat
            previous_joint_pos = target_joint_pos
            previous_joint_vel = target_joint_vel
            previous_aux = target_aux
        return dense

    def _terrain_height_at(self, local_xy: torch.Tensor) -> torch.Tensor:
        ray_hits = self._height_sensor.data.ray_hits_w
        current_pos = self.robot_anchor_pos_w
        current_yaw = yaw_quat(self.robot_anchor_quat_w)
        local_hits = quat_apply(
            quat_inv(current_yaw)[:, None, :].expand(-1, ray_hits.shape[1], -1).reshape(-1, 4),
            (ray_hits - current_pos[:, None, :]).reshape(-1, 3),
        ).reshape(self.num_envs, ray_hits.shape[1], 3)
        valid_hits = torch.isfinite(ray_hits).all(dim=-1)
        distance_sq = (local_hits[..., :2] - local_xy[:, None, :]).square().sum(dim=-1)
        distance_sq = torch.where(valid_hits, distance_sq, torch.full_like(distance_sq, 1.0e12))
        nearest = distance_sq.argmin(dim=-1)
        terrain_z = ray_hits[..., 2].gather(1, nearest[:, None]).squeeze(1)
        fallback_z = self.robot_anchor_pos_w[:, 2] - self.cfg.default_root_clearance
        return torch.where(torch.isfinite(terrain_z), terrain_z, fallback_z)

    def _decode_root_and_bodies(
        self,
        frame: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        anchor_pos: torch.Tensor | None = None,
        anchor_yaw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        root_pos_local = frame[:, :3]
        root_rot_local = _quat_from_rot6d(frame[:, 3:9])
        joint_pos_diffusion = frame[:, 9:34]

        # Every generated future spatial feature is expressed in H0's yaw-only
        # heading frame. H0 is the oldest entry of the configured history.
        h0_pos = self._root_pos_history[env_ids, 0] if anchor_pos is None else anchor_pos
        h0_yaw = self._yaw_history[env_ids, 0] if anchor_yaw is None else anchor_yaw
        root_pos = h0_pos + quat_apply(h0_yaw, root_pos_local)
        root_quat = quat_mul(h0_yaw, root_rot_local)

        body_pos, body_quat = self._fk(root_pos, root_quat, joint_pos_diffusion, self.cfg.body_names)
        return root_pos, root_quat, joint_pos_diffusion, body_pos, body_quat

    def _set_generated_reference(
        self,
        motion: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        frame_indices: torch.Tensor | None = None,
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if frame_indices is None:
            frame_indices = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        rows = torch.arange(len(env_ids), device=self.device)
        frame = motion[rows, frame_indices]
        anchor_pos = self._generated_anchor_pos[env_ids]
        anchor_yaw = self._generated_anchor_yaw[env_ids]
        _, _, joint_pos_diffusion, body_pos, body_quat = self._decode_root_and_bodies(
            frame, env_ids, anchor_pos, anchor_yaw
        )
        next_indices = torch.clamp(frame_indices + 1, max=self.dense_future_size - 1)
        previous_indices = torch.clamp(frame_indices - 1, min=0)
        _, _, _, next_body_pos, next_body_quat = self._decode_root_and_bodies(
            motion[rows, next_indices], env_ids, anchor_pos, anchor_yaw
        )
        _, _, _, previous_body_pos, previous_body_quat = self._decode_root_and_bodies(
            motion[rows, previous_indices], env_ids, anchor_pos, anchor_yaw
        )
        joint_vel_diffusion = frame[:, 34:59]

        dt = float(self._env.step_dt)
        is_last = frame_indices == self.dense_future_size - 1
        forward_lin_vel = (next_body_pos - body_pos) / dt
        backward_lin_vel = (body_pos - previous_body_pos) / dt
        body_lin_vel = torch.where(
            is_last[:, None, None], backward_lin_vel, forward_lin_vel
        )
        forward_delta_quat = quat_mul(next_body_quat, quat_inv(body_quat))
        backward_delta_quat = quat_mul(body_quat, quat_inv(previous_body_quat))
        delta_quat = torch.where(
            is_last[:, None, None], backward_delta_quat, forward_delta_quat
        )
        delta_quat = torch.where(delta_quat[..., :1] < 0.0, -delta_quat, delta_quat)
        vector = delta_quat[..., 1:]
        vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(vector_norm, delta_quat[..., :1].clamp_min(1.0e-8))
        body_ang_vel = vector / vector_norm.clamp_min(1.0e-8) * angle / dt

        self._joint_pos[env_ids] = joint_pos_diffusion
        self._joint_vel[env_ids] = joint_vel_diffusion
        self._body_pos_w[env_ids] = body_pos
        self._body_quat_w[env_ids] = body_quat
        self._body_lin_vel_w[env_ids] = body_lin_vel
        self._body_ang_vel_w[env_ids] = body_ang_vel
        self._previous_body_pos_w[env_ids] = body_pos
        self._previous_body_quat_w[env_ids] = body_quat

    def _sample_velocity_commands(self, env_ids: torch.Tensor) -> None:
        """Sample [vx, vy, wz] uniformly from the configured ranges."""
        if len(env_ids) == 0:
            return
        self._velocity_command[env_ids] = self._command_low + torch.rand(
            (len(env_ids), 3), device=self.device
        ) * (self._command_high - self._command_low)
        self._command_age_steps[env_ids] = 0

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self.cfg.initialize_from_default_pose:
            self._reset_from_default_pose(env_ids)
            return
        count = len(env_ids)
        terrain_ids = torch.randint(
            0, len(self._reset_motion_groups), (count,), device=self.device
        )
        self._active_terrain[env_ids] = terrain_ids

        motion_ids = torch.empty(count, dtype=torch.long, device=self.device)
        for terrain_index, group in enumerate(self._reset_motion_groups):
            mask = terrain_ids == terrain_index
            if mask.any():
                motion_ids[mask] = group[
                    torch.randint(0, len(group), (int(mask.sum()),), device=self.device)
                ]
        self.active_motion_idx[env_ids] = motion_ids
        frame_count = self._reset_motion_frames[motion_ids]
        required_frames = self.history_span + self.dense_future_size + 1
        if torch.any(frame_count < required_frames):
            raise ValueError(
                f"Diffusion reset motions need at least {required_frames} frames"
            )
        # The sampled reset state is the last history frame. Keep enough
        # preceding frames to initialize the exact history condition used in
        # pretraining.
        available_last_history = frame_count - self.history_span - self.dense_future_size
        local_frames = (
            torch.rand(count, device=self.device) * available_last_history.float()
        ).long() + self.history_span
        self.time_steps[env_ids] = local_frames
        global_frames = self.motion.motion_start[motion_ids] + local_frames

        history_offsets = self.history_sample_stride * torch.arange(
            self.history_size - 1, -1, -1, device=self.device
        )
        history_global_frames = global_frames[:, None] - history_offsets[None, :]

        terrain_rows = torch.randint(self._terrain_origins.shape[0], (count,), device=self.device)
        origins = self._terrain_origins[terrain_rows, terrain_ids]
        self.motion_origins[env_ids] = origins

        root_pos = self.motion.body_pos_w[global_frames, 0].clone() + origins
        root_quat = self.motion.body_quat_w[global_frames, 0].clone()
        root_lin_vel = self.motion.body_lin_vel_w[global_frames, 0].clone()
        root_ang_vel = self.motion.body_ang_vel_w[global_frames, 0].clone()
        reset_joint_pos = self.motion.joint_pos[global_frames].clone()
        reset_joint_vel = self.motion.joint_vel[global_frames].clone()
        self.robot.write_joint_state_to_sim(reset_joint_pos, reset_joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat((root_pos, root_quat, root_lin_vel, root_ang_vel), dim=-1), env_ids=env_ids
        )

        selected_body_pos = self.motion.body_pos_w[global_frames][:, self.body_indexes] + origins[:, None, :]
        selected_body_quat = self.motion.body_quat_w[global_frames][:, self.body_indexes]
        selected_body_lin = self.motion.body_lin_vel_w[global_frames][:, self.body_indexes]
        selected_body_ang = self.motion.body_ang_vel_w[global_frames][:, self.body_indexes]
        self._joint_pos[env_ids] = reset_joint_pos
        self._joint_vel[env_ids] = reset_joint_vel
        self._body_pos_w[env_ids] = selected_body_pos
        self._body_quat_w[env_ids] = selected_body_quat
        self._body_lin_vel_w[env_ids] = selected_body_lin
        self._body_ang_vel_w[env_ids] = selected_body_ang
        self._previous_body_pos_w[env_ids] = selected_body_pos
        self._previous_body_quat_w[env_ids] = selected_body_quat

        # Prime the diffusion condition with the real history trajectory. Raw
        # motion, diffusion features and IsaacLab all share articulation order.
        history_joint_robot = self.motion.joint_pos[history_global_frames]
        self._joint_history[env_ids] = history_joint_robot
        motion_anchor_body_id = int(self.body_indexes[self.motion_anchor_body_index])
        history_anchor_pos = self.motion.body_pos_w[
            history_global_frames, motion_anchor_body_id
        ]
        self._root_pos_history[env_ids] = history_anchor_pos + origins[:, None, :]
        history_anchor_quat = self.motion.body_quat_w[
            history_global_frames, motion_anchor_body_id
        ]
        self._yaw_history[env_ids] = yaw_quat(history_anchor_quat.reshape(-1, 4)).reshape(
            count, self.history_size, 4
        )
        history_terrain_z = self.motion.elevation_map_xyz[history_global_frames, :, 2]
        self._terrain_history[env_ids] = (
            history_anchor_pos[..., 2:3] - history_terrain_z
        ).clamp(-20.0, 20.0)
        self._history_ready[env_ids] = True
        self._history_update_counter[env_ids] = 0
        # CommandManager may call _update_command immediately after reset.  Do
        # not shift out H0 before the first generated reference is sampled.
        self._skip_history_append[env_ids] = True

        # Sample a command independently of the selected last-history frame, while keeping the
        # complete vector inside this motion's demonstrated distribution.
        self._sample_velocity_commands(env_ids)
        self._generated_phase[env_ids] = 0
        # A reset environment must never continue an old generated block.  Its
        # last-history remains the temporary reference until the next command
        # update generates a fresh block from the newly initialized history.
        self._generated_motion_valid[env_ids] = False

        if self.cfg.initialize_from_generated_f0:
            # Generate from the exact real history condition, densify it, and
            # initialize the simulator from the first interpolated frame.
            # difference velocity of the floating base.
            sparse_motion = self._sample_motion(env_ids)
            initial_motion = self._densify_motion(sparse_motion, env_ids)
            self._generated_motion[env_ids] = initial_motion
            self._generated_motion_valid[env_ids] = True
            self._generated_anchor_pos[env_ids] = self._root_pos_history[env_ids, 0]
            self._generated_anchor_yaw[env_ids] = self._yaw_history[env_ids, 0]
            self._set_generated_reference(initial_motion, env_ids)
            root_index = self._root_reference_body_index
            generated_root_state = torch.cat(
                (
                    self._body_pos_w[env_ids, root_index],
                    self._body_quat_w[env_ids, root_index],
                    self._body_lin_vel_w[env_ids, root_index],
                    self._body_ang_vel_w[env_ids, root_index],
                ),
                dim=-1,
            )
            self.robot.write_joint_state_to_sim(
                self._joint_pos[env_ids], self._joint_vel[env_ids], env_ids=env_ids
            )
            self.robot.write_root_state_to_sim(generated_root_state, env_ids=env_ids)
            # IsaacLab calls command.compute() immediately after resetting an
            # environment. Keep this exact F0 for the first action/reward,
            # instead of replacing it with another stochastic sample.
            self._hold_initialized_reference[env_ids] = True
        else:
            self._hold_initialized_reference[env_ids] = False

    def _reset_from_default_pose(self, env_ids: torch.Tensor) -> None:
        """Initialize play environments from the articulation's nominal pose."""
        count = len(env_ids)
        terrain_rows = torch.zeros(count, dtype=torch.long, device=self.device)
        terrain_ids = torch.zeros_like(terrain_rows)
        origins = self._terrain_origins[terrain_rows, terrain_ids]
        self._active_terrain[env_ids] = terrain_ids
        self.motion_origins[env_ids] = origins

        root_state = self.robot.data.default_root_state[env_ids].clone()
        configured_position = torch.tensor(
            self.cfg.default_pose_position, dtype=root_state.dtype, device=self.device
        )
        configured_rpy = torch.tensor(
            self.cfg.default_pose_rpy, dtype=root_state.dtype, device=self.device
        )
        root_state[:, :3] = origins + configured_position
        root_state[:, 3:7] = quat_from_euler_xyz(
            configured_rpy[0].expand(count),
            configured_rpy[1].expand(count),
            configured_rpy[2].expand(count),
        )
        root_state[:, 7:] = 0.0
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        body_pos, body_quat = self._robot_root_fk(
            root_state[:, :3], root_state[:, 3:7], joint_pos, self.cfg.body_names
        )
        anchor_index = self.motion_anchor_body_index
        anchor_pos = body_pos[:, anchor_index]
        anchor_quat = body_quat[:, anchor_index]
        self._joint_pos[env_ids] = joint_pos
        self._joint_vel[env_ids] = joint_vel
        self._body_pos_w[env_ids] = body_pos
        self._body_quat_w[env_ids] = body_quat
        self._body_lin_vel_w[env_ids] = 0.0
        self._body_ang_vel_w[env_ids] = 0.0
        self._previous_body_pos_w[env_ids] = body_pos
        self._previous_body_quat_w[env_ids] = body_quat

        self._joint_history[env_ids] = joint_pos[:, None, :].expand(-1, self.history_size, -1)
        self._root_pos_history[env_ids] = anchor_pos[:, None, :].expand(-1, self.history_size, -1)
        anchor_yaw = yaw_quat(anchor_quat)
        self._yaw_history[env_ids] = anchor_yaw[:, None, :].expand(-1, self.history_size, -1)
        clearance, _ = self._current_terrain_clearance()
        self._terrain_history[env_ids] = clearance[env_ids, None, :].expand(
            -1, self.history_size, -1
        )
        self._history_ready[env_ids] = True
        self._history_update_counter[env_ids] = 0
        self._skip_history_append[env_ids] = True
        self._sample_velocity_commands(env_ids)
        self._generated_phase[env_ids] = 0
        self._generated_motion_valid[env_ids] = False
        self._hold_initialized_reference[env_ids] = False

    def _update_command(self):
        missing_history = torch.where(~self._history_ready)[0]
        if len(missing_history) > 0:
            self._append_actual_history(missing_history, repeat=True)
            self._history_update_counter[missing_history] = 0
        # All normal environments append one actual frame.  Environments just
        # reset already contain the exact history and must preserve H0 for this
        # sample.  The simulator runs at 50 Hz, while the model was trained at
        # 10 Hz, so only append an actual state every five control steps.
        ready_mask = self._history_ready & ~self._skip_history_append
        ready_mask[missing_history] = False
        self._history_update_counter[ready_mask] += 1
        append_mask = ready_mask & (
            self._history_update_counter >= self.history_sample_stride
        )
        append_ids = torch.where(append_mask)[0]
        if len(append_ids) > 0:
            self._append_actual_history(append_ids, repeat=False)
            self._history_update_counter[append_ids] = 0

        # Commands only change at generated-block boundaries. The default
        # interval is 100 policy steps (2 s at 50 Hz).
        block_start = self._generated_phase == 0
        command_due = block_start & (
            self._command_age_steps >= self.cfg.command_resampling_steps
        )
        command_ids = torch.where(command_due)[0]
        self._sample_velocity_commands(command_ids)

        # Generate ten sparse 10-Hz keyframes, interpolate them into a 50-Hz
        # one-second block, then track only the first 0.4 s (20 control frames)
        # before invoking Flow Matching again.
        needs_generation = block_start | ~self._generated_motion_valid
        sample_ids = torch.where(
            needs_generation & ~self._hold_initialized_reference
        )[0]
        if len(sample_ids) > 0:
            self._generated_anchor_pos[sample_ids] = self._root_pos_history[sample_ids, 0]
            self._generated_anchor_yaw[sample_ids] = self._yaw_history[sample_ids, 0]
            sparse_motion = self._sample_motion(sample_ids)
            self._generated_motion[sample_ids] = self._densify_motion(
                sparse_motion, sample_ids
            )
            self._generated_motion_valid[sample_ids] = True
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._set_generated_reference(
            self._generated_motion,
            all_env_ids,
            self._generated_phase,
        )
        self._generated_phase = (
            self._generated_phase + 1
        ) % self.tracking_horizon_steps
        self._command_age_steps += 1
        self._skip_history_append[:] = False
        self._hold_initialized_reference[:] = False

        anchor_pos = self.anchor_pos_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        anchor_quat = self.anchor_quat_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].expand_as(anchor_pos)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].expand_as(anchor_quat)
        delta_pos = robot_anchor_pos.clone()
        delta_pos[..., 2] = anchor_pos[..., 2]
        delta_quat = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_quat, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(delta_quat, self.body_pos_w - anchor_pos)
        if self._sample_counter == 1:
            body_error = torch.linalg.vector_norm(
                self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
            )
            actual_fk_pos, _ = self._fk(
                self.robot_anchor_pos_w,
                self.robot_anchor_quat_w,
                self.robot_joint_pos,
                self.cfg.body_names,
            )
            robot_fk_error = torch.linalg.vector_norm(
                actual_fk_pos - self.robot_body_pos_w, dim=-1
            )
            reference_fk_error = torch.linalg.vector_norm(
                self.body_pos_relative_w - actual_fk_pos, dim=-1
            )
            raw_reference_error = torch.linalg.vector_norm(
                self.body_pos_w - actual_fk_pos, dim=-1
            )
            anchor_error = torch.linalg.vector_norm(
                self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
            )
            joint_error = torch.linalg.vector_norm(
                self.joint_pos - self.robot_joint_pos, dim=-1
            )
            print(
                f"[DiffusionReference] anchor_z=[{self.anchor_pos_w[:, 2].min().item():.3f},"
                f"{self.anchor_pos_w[:, 2].max().item():.3f}] "
                f"body_error_mean={body_error.mean().item():.6f}m "
                f"body_error_max={body_error.max().item():.6f}m "
                f"robot_fk_error={robot_fk_error.mean().item():.6f}m "
                f"reference_fk_error={reference_fk_error.mean().item():.6f}m "
                f"raw_reference_error={raw_reference_error.mean().item():.6f}m "
                f"anchor_error={anchor_error.mean().item():.6f}m "
                f"joint_error={joint_error.mean().item():.6f}rad"
            )

    def _record_failures(self, env_ids: torch.Tensor):
        # Diffusion training has no fixed motion bins to adaptively reweight.
        del env_ids

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Show the robot, current target, and the complete tracked diffusion block."""
        super()._set_debug_vis_impl(debug_vis)
        if (
            debug_vis
            and self.cfg.visualize_generated_horizon
            and not hasattr(self, "generated_horizon_visualizers")
        ):
            self.generated_horizon_visualizers = [
                VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/generated_horizon/{name}"
                    )
                )
                for name in self.cfg.body_names
            ]

        if hasattr(self, "generated_horizon_visualizers"):
            for visualizer in self.generated_horizon_visualizers:
                visualizer.set_visibility(
                    debug_vis and self.cfg.visualize_generated_horizon
                )

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if (
            not self.cfg.visualize_generated_horizon
            or not hasattr(self, "generated_horizon_visualizers")
            or not self._generated_motion_valid.any()
        ):
            return

        # The model predicts ten sparse keyframes.  The stored motion has
        # already been interpolated to the control rate, so pick each segment's
        # endpoint to recover the original generated keyframes for display.
        keyframe_count = min(
            self.cfg.generated_keyframes_visualization_count,
            self.future_size,
        )
        keyframe_indices = (
            torch.arange(1, keyframe_count + 1, device=self.device)
            * self.runtime_frame_stride
            - 1
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        flat_env_ids = env_ids[:, None].expand(-1, keyframe_count).reshape(-1)
        flat_frames = self._generated_motion[:, keyframe_indices].reshape(-1, 80)
        anchor_pos = self._generated_anchor_pos[:, None, :].expand(-1, keyframe_count, -1).reshape(-1, 3)
        anchor_yaw = self._generated_anchor_yaw[:, None, :].expand(-1, keyframe_count, -1).reshape(-1, 4)
        _, _, _, body_pos, body_quat = self._decode_root_and_bodies(
            flat_frames, flat_env_ids, anchor_pos, anchor_yaw
        )

        # Apply the same current robot-to-reference yaw/XY alignment used by
        # the tracking target, so the ghosts are drawn where tracking starts.
        current_anchor_pos = self.anchor_pos_w
        current_anchor_quat = self.anchor_quat_w
        delta_pos = self.robot_anchor_pos_w.clone()
        delta_pos[:, 2] = current_anchor_pos[:, 2]
        delta_quat = yaw_quat(
            quat_mul(self.robot_anchor_quat_w, quat_inv(current_anchor_quat))
        )
        expanded_delta_pos = delta_pos[:, None, None, :]
        expanded_delta_quat = delta_quat[:, None, None, :]
        expanded_anchor_pos = current_anchor_pos[:, None, None, :]
        aligned_body_pos = expanded_delta_pos + quat_apply(
            expanded_delta_quat.expand(-1, keyframe_count, len(self.cfg.body_names), -1).reshape(-1, 4),
            (body_pos.reshape(self.num_envs, keyframe_count, -1, 3) - expanded_anchor_pos).reshape(-1, 3),
        ).reshape(self.num_envs, keyframe_count, -1, 3)
        aligned_body_quat = quat_mul(
            expanded_delta_quat.expand(-1, keyframe_count, len(self.cfg.body_names), -1).reshape(-1, 4),
            body_quat.reshape(-1, 4),
        ).reshape(self.num_envs, keyframe_count, -1, 4)

        valid = self._generated_motion_valid[:, None].expand(-1, keyframe_count).reshape(-1)
        for body_index, visualizer in enumerate(self.generated_horizon_visualizers):
            visualizer.visualize(
                aligned_body_pos[:, :, body_index].reshape(-1, 3)[valid],
                aligned_body_quat[:, :, body_index].reshape(-1, 4)[valid],
            )


@configclass
class DiffusionParkourMotionCommandCfg(ParkourMotionCommandCfg):
    """Configuration for online frozen-diffusion parkour references."""

    class_type: type = DiffusionParkourMotionCommand

    diffusion_checkpoint: str = MISSING
    diffusion_root: str = MISSING
    robot_urdf: str = MISSING
    height_sensor_name: str = "height_scanner"
    reset_motion_groups: list[list[str]] = MISSING
    # The conditional model is trained from the 10-Hz dataset; the simulator
    # and motion files used by IsaacLab run at 50 Hz.
    diffusion_fps: float = 10.0
    tracking_horizon_steps: int = 20
    visualize_generated_horizon: bool = True
    generated_keyframes_visualization_count: int = 5
    timing_interval_steps: int = 50
    reset_max_root_step: float = 2.0
    default_root_clearance: float = 0.95
    initialize_from_generated_f0: bool = False
    initialize_from_default_pose: bool = False
    default_pose_position: tuple[float, float, float] = (0.0, 0.0, 0.875)
    default_pose_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    command_resampling_steps: int = 100
    command_vx_range: tuple[float, float] = (0.0, 2.0)
    command_vy_range: tuple[float, float] = (-0.5, 0.5)
    command_wz_range: tuple[float, float] = (-1.5, 1.5)

    log_interval_steps: int = 0
