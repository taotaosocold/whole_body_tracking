"""Generate and visualize terrain/velocity-conditioned CASBOT motion.

The GUI selects a recorded condition window and exposes a continuous target
heading relative to the latest historical yaw.

Usage:
  python diffusion/scripts/casbot_generate_viz_terrain.py \
    --ckpt-path logs/ddpm/.../pretrained.pt \
    --data-dir diffusion/source/datasets
"""

from __future__ import annotations

import time
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import viser
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)
from mjlab.viewer.viser.scene import MjlabViserScene

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.common.model import DiffusionDenoiser
from source.ddpm.scheduler import DDPMScheduler
from source.flow_matching.sampler import sample_flow
from source.utils import detect_device

# ---------------------------------------------------------------------------
# CasBot constants
# ---------------------------------------------------------------------------

JOINT_NAMES: tuple[str, ...] = (
  "left_leg_pelvic_pitch_joint",
  "left_leg_pelvic_roll_joint",
  "left_leg_pelvic_yaw_joint",
  "left_leg_knee_pitch_joint",
  "left_leg_ankle_pitch_joint",
  "left_leg_ankle_roll_joint",
  "right_leg_pelvic_pitch_joint",
  "right_leg_pelvic_roll_joint",
  "right_leg_pelvic_yaw_joint",
  "right_leg_knee_pitch_joint",
  "right_leg_ankle_pitch_joint",
  "right_leg_ankle_roll_joint",
  "waist_yaw_joint",
  "head_yaw_joint",
  "head_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
)
# Diffusion joint features stay in IsaacLab articulation order.  Reorder only
# at this MuJoCo visualization boundary into the limb-grouped JOINT_NAMES above.
ISAACLAB_TO_XML_JOINT_ORDER = (
  0, 3, 8, 13, 17, 21,
  1, 4, 9, 14, 18, 22,
  2, 5, 10,
  6, 11, 15, 19, 23,
  7, 12, 16, 20, 24,
)
NUM_JOINTS = len(JOINT_NAMES)

EE_BODY_NAMES: tuple[str, ...] = (
  "left_leg_ankle_roll_link",
  "right_leg_ankle_roll_link",
  "head_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
NUM_EE = len(EE_BODY_NAMES)
GRID_X = 33
GRID_Y = 21
TERRAIN_DIM = GRID_X * GRID_Y
CENTER_IDX = (GRID_X // 2) * GRID_Y + GRID_Y // 2
TERRAIN_GRID_LAYOUT = "x_ascending_y_ascending"

_DEFAULT_MODEL = str(
  Path(__file__).resolve().parents[1]
  / "assets"
  / "casbot_skeleton"
  / "casbot_skeleton_25dof.xml"
)

# ---------------------------------------------------------------------------
# Feature-to-state (CasBot-adapted from smp.sampling.feature_to_state)
# ---------------------------------------------------------------------------


def _slice_features(frame: torch.Tensor) -> dict[str, torch.Tensor]:
  """Slice a CasBot feature vector into named components.

  Layout (matches height_map_to_npz.py):
    [0:3]                   root_pos
    [3:9]                   root_rot (6D tan-norm)
    [9:9+J]                 joint_pos (J=25)
    [9+J:9+2J]              joint_vel (J=25)
    [9+2J:9+2J+E*3]         ee_pos (E=5)
    [9+2J+E*3:12+2J+E*3]    root_lin_vel
    [12+2J+E*3:15+2J+E*3]   root_ang_vel
  """
  J = NUM_JOINTS
  E = NUM_EE
  expected = 3 + 6 + J + J + E * 3 + 3 + 3
  if (d := frame.shape[-1]) != expected:
    msg = f"expected feature_dim={expected}; got {d}"
    raise ValueError(msg)
  joint_pos_end = 9 + J
  joint_vel_end = joint_pos_end + J
  ee_pos_end = joint_vel_end + E * 3
  lin_vel_end = ee_pos_end + 3
  ang_vel_end = lin_vel_end + 3
  return {
    "root_pos": frame[..., 0:3],
    "root_rot": frame[..., 3:9],
    "joint_pos": frame[..., 9:joint_pos_end],
    "joint_vel": frame[..., joint_pos_end:joint_vel_end],
    "ee_pos": frame[..., joint_vel_end:ee_pos_end],
    "root_lin_vel": frame[..., ee_pos_end:lin_vel_end],
    "root_ang_vel": frame[..., lin_vel_end:ang_vel_end],
  }


def _rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
  """Convert 6D tan-norm [col0, col2] → 3×3 rotation matrix."""
  col0 = d6[..., :3]
  col2 = d6[..., 3:6]
  col0 = torch.nn.functional.normalize(col0, dim=-1)
  col2 = col2 - (col0 * col2).sum(dim=-1, keepdim=True) * col0
  col2 = torch.nn.functional.normalize(col2, dim=-1)
  col1 = torch.cross(col2, col0, dim=-1)
  return torch.stack([col0, col1, col2], dim=-1)


def _rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
  return quat_from_matrix(_rot6d_to_matrix(d6))


def _window_to_pelvis_trajectory(
  window: torch.Tensor,
  anchor_pelvis_pos_w: torch.Tensor,
  anchor_pelvis_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Reconstruct world-frame pelvis pose + joint pos for each frame."""
  parts = _slice_features(window)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  W = window.shape[0]

  anchor_pelvis_pos_w = anchor_pelvis_pos_w.to(window)
  anchor_pelvis_quat_w = anchor_pelvis_quat_w.to(window)
  yaw_h0 = yaw_quat(anchor_pelvis_quat_w[None]).squeeze(0)

  world_offset = quat_apply(yaw_h0[None].expand(W, 4), root_pos_local)
  pelvis_pos_w = world_offset + anchor_pelvis_pos_w[None, :3]

  root_rot_local_quat = _rot6d_to_quat(root_rot_6d)
  pelvis_quat_w = quat_mul(yaw_h0[None].expand(W, 4), root_rot_local_quat)

  return pelvis_pos_w, pelvis_quat_w, parts["joint_pos"]


def _window_to_ee_trajectories(
  window: torch.Tensor,
  pelvis_pos_w: torch.Tensor,
  anchor_pelvis_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Reconstruct world-frame EE positions across a window."""
  parts = _slice_features(window)
  W = window.shape[0]
  E = NUM_EE
  ee_pos_local = parts["ee_pos"].reshape(W, E, 3)

  yaw_h0 = yaw_quat(anchor_pelvis_quat_w.to(window)[None])
  yaw_h0_e = yaw_h0.expand(W, 4)[:, None, :].expand(W, E, 4).reshape(-1, 4)
  ee_offset_w = quat_apply(yaw_h0_e, ee_pos_local.reshape(-1, 3)).reshape(W, E, 3)
  return ee_offset_w + pelvis_pos_w[:, None, :]


def _decode_history_trajectory(
  history_root: np.ndarray,
  history_joint: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Decode the visualization-only H0...H_last records.

  H0 stores an absolute waist pose with x/y set to zero. The remaining history
  frames store root xyz offsets and root orientations in H0's yaw-only heading
  frame.
  """
  root = torch.from_numpy(history_root.astype(np.float32, copy=False))
  joints = torch.from_numpy(history_joint.astype(np.float32, copy=False))
  h0_pos = root[0, :3]
  h0_quat = _rot6d_to_quat(root[0, 3:9])
  h0_yaw = yaw_quat(h0_quat[None]).squeeze(0)
  positions = [h0_pos]
  rotations = [h0_quat]
  for index in range(1, root.shape[0]):
    positions.append(h0_pos + quat_apply(h0_yaw, root[index, :3]))
    rotations.append(quat_mul(h0_yaw, _rot6d_to_quat(root[index, 3:9])))
  return torch.stack(positions), torch.stack(rotations), joints


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
  """Shortest-path SLERP for one wxyz quaternion pair."""
  q0 = torch.nn.functional.normalize(q0, dim=-1)
  q1 = torch.nn.functional.normalize(q1, dim=-1)
  dot = torch.sum(q0 * q1)
  if dot < 0.0:
    q1 = -q1
    dot = -dot
  if dot > 0.9995:
    return torch.nn.functional.normalize(q0 + alpha * (q1 - q0), dim=-1)
  theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
  sin_theta = torch.sin(theta)
  return (
    torch.sin((1.0 - alpha) * theta) / sin_theta * q0
    + torch.sin(alpha * theta) / sin_theta * q1
  )


def _interpolate_sparse_trajectory(
  h3_pos: torch.Tensor,
  h3_quat: torch.Tensor,
  h3_joint: torch.Tensor,
  key_pos: torch.Tensor,
  key_quat: torch.Tensor,
  key_joint: torch.Tensor,
  key_ee: torch.Tensor,
  frame_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Expand sparse future keyframes into one trajectory for visualization."""
  dense_pos, dense_quat, dense_joint, dense_ee = [], [], [], []
  previous_pos, previous_quat, previous_joint = h3_pos, h3_quat, h3_joint
  # EE markers are diagnostic only; the exact robot pose comes from root/joints.
  previous_ee = key_ee[0]
  for key_index in range(key_pos.shape[0]):
    for substep in range(1, frame_stride + 1):
      alpha = substep / float(frame_stride)
      dense_pos.append(torch.lerp(previous_pos, key_pos[key_index], alpha))
      dense_quat.append(_quat_slerp(previous_quat, key_quat[key_index], alpha))
      dense_joint.append(torch.lerp(previous_joint, key_joint[key_index], alpha))
      dense_ee.append(torch.lerp(previous_ee, key_ee[key_index], alpha))
    previous_pos = key_pos[key_index]
    previous_quat = key_quat[key_index]
    previous_joint = key_joint[key_index]
    previous_ee = key_ee[key_index]
  return tuple(torch.stack(values) for values in (dense_pos, dense_quat, dense_joint, dense_ee))


# ---------------------------------------------------------------------------
# Mujoco CasBot helpers
# ---------------------------------------------------------------------------


class _CasbotSim:
  """Minimal mujoco sim for CasBot (replaces mjlab Entity/Simulation)."""

  def __init__(self, model_path: str) -> None:
    self.model = mujoco.MjModel.from_xml_path(model_path)
    if self.model.nq < 7 or self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
      raise ValueError(
        f"{model_path} does not contain the required floating-base freejoint. "
        "Use the CASBOT MuJoCo XML, not the URDF directly."
      )
    self.data = mujoco.MjData(self.model)
    self._waist_body_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_BODY, "waist_yaw_link"
    )
    # Joint qpos addresses for the 25 actuated joints.
    self._joint_qposadr = np.array(
      [
        self.model.jnt_qposadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in JOINT_NAMES
      ],
      dtype=np.int32,
    )
    # Resolve the XML qpos0 free-root/joint state to the actual diffusion
    # anchor link.  qpos0[0:3] belongs to base_link and must not be mistaken
    # for the waist_yaw_link position.
    mujoco.mj_forward(self.model, self.data)

  def default_waist_pose(self) -> tuple[np.ndarray, np.ndarray]:
    """Return qpos0's waist pose, with H0 horizontal origin forced to (0, 0)."""
    pos = self.data.xpos[self._waist_body_id].astype(np.float32).copy()
    pos[:2] = 0.0
    quat = self.data.xquat[self._waist_body_id].astype(np.float32).copy()
    return pos, quat

  def write_pose(
    self,
    pelvis_pos: np.ndarray,        # desired waist_yaw_link position, (3,)
    pelvis_quat_wxyz: np.ndarray,  # desired waist_yaw_link orientation, (4,)
    joint_pos: np.ndarray,     # (25,)
  ) -> None:
    """Write a pose whose generated root is ``waist_yaw_link``.

    MuJoCo's freejoint belongs to ``base_link``.  Align that freejoint through
    FK so the resulting waist pose, rather than the base pose, equals the
    diffusion root target.
    """
    self.data.qpos[0:3] = pelvis_pos
    self.data.qpos[3:7] = pelvis_quat_wxyz
    self.data.qpos[self._joint_qposadr] = joint_pos[
      np.asarray(ISAACLAB_TO_XML_JOINT_ORDER, dtype=np.int32)
    ]
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

    current_waist_quat = self.data.xquat[self._waist_body_id].copy()
    current_waist_quat[1:] *= -1.0
    delta_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(delta_quat, pelvis_quat_wxyz, current_waist_quat)
    corrected_base_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(corrected_base_quat, delta_quat, self.data.qpos[3:7])
    self.data.qpos[3:7] = corrected_base_quat
    mujoco.mj_forward(self.model, self.data)

    self.data.qpos[0:3] += pelvis_pos - self.data.xpos[self._waist_body_id]
    mujoco.mj_forward(self.model, self.data)


# ---------------------------------------------------------------------------
# Config & main
# ---------------------------------------------------------------------------


@dataclass
class Cfg:
  ckpt_path: str = ""
  """Path to a local SMP diffusion checkpoint .pt file."""
  data_dir: str = "diffusion/source/datasets"
  """Directory containing windowed terrain/proprio NPZ files."""
  model_path: str = _DEFAULT_MODEL
  """Path to the floating-base CASBOT MuJoCo XML."""
  device: str = ""
  """Compute device. Empty = auto."""
  fps: float = 50.0
  """Playback frame rate."""
  command_vx: float = 0.5
  """Desired forward velocity in the current root frame (m/s)."""
  command_vy: float = 0.0
  """Desired left velocity in the current root frame (m/s)."""
  command_wz: float = 0.0
  """Desired yaw velocity in the current root frame (rad/s)."""


def _build_model_and_sampler(
  ckpt: dict, device: torch.device
) -> tuple[DiffusionDenoiser, DDPMScheduler | None, dict]:
  cfg = ckpt["cfg"]
  model = DiffusionDenoiser(
    feature_dim=cfg["feature_dim"],
    window_size=cfg["window_size"],
    d_model=cfg.get("d_model", 256),
    nhead=cfg.get("nhead", 8),
    num_layers=cfg.get("num_layers", 2),
    dropout=cfg.get("dropout", 0.0),
    terrain_dim=cfg["terrain_dim"],
    terrain_height=cfg.get("terrain_height", GRID_Y),
    terrain_width=cfg.get("terrain_width", GRID_X),
    terrain_feature_dim=cfg.get("terrain_feature_dim", 48),
    proprio_dim=cfg.get("proprio_dim", 28),
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()
  method = str(cfg.get("generative_method", "ddpm"))
  if method not in ("ddpm", "flow_matching"):
    raise ValueError(f"Unknown checkpoint generative_method={method!r}")
  scheduler = (
    DDPMScheduler(num_timesteps=cfg.get("num_timesteps", 50)).to(device)
    if method == "ddpm" else None
  )
  return model, scheduler, ckpt


def _quantile_denormalize(
  x: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor
) -> torch.Tensor:
  return (x + 1.0) / 2.0 * (q_high - q_low) + q_low


def _quantile_normalize(
  x: torch.Tensor, q_low: np.ndarray, q_high: np.ndarray
) -> torch.Tensor:
  low = torch.as_tensor(q_low, device=x.device, dtype=x.dtype)
  high = torch.as_tensor(q_high, device=x.device, dtype=x.dtype)
  return 2.0 * (x - low) / (high - low) - 1.0


def _load_condition_windows(
  data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
  motions, terrains, proprios, history_roots, history_joints = [], [], [], [], []
  frame_strides: set[int] = set()
  history_sizes: set[int] = set()
  for path in sorted(Path(data_dir).glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
      if "motion_windows" in data and "terrain" in data and "proprio" in data:
        if (
          "history_root" not in data
          or "history_joint" not in data
          or "history_root_layout" not in data
          or "terrain_grid_layout" not in data
          or "future_frame_stride" not in data
        ):
          raise ValueError(
            f"{path.name} predates sparse-future visualization metadata; "
            "re-run diffusion/scripts/height_map_to_npz.py"
          )
        history_layout = str(np.asarray(data["history_root_layout"]).item())
        if history_layout != "h0_abs_waist_pose_history_h0_heading_relative":
          raise ValueError(
            f"{path.name}: unsupported history_root_layout={history_layout!r}"
          )
        terrain_grid_layout = str(np.asarray(data["terrain_grid_layout"]).item())
        if terrain_grid_layout != TERRAIN_GRID_LAYOUT:
          raise ValueError(
            f"{path.name}: unsupported terrain_grid_layout={terrain_grid_layout!r}; "
            "re-run diffusion/scripts/height_map_to_npz.py"
          )
        motions.append(data["motion_windows"].astype(np.float32))
        terrains.append(data["terrain"].astype(np.float32))
        proprios.append(data["proprio"].astype(np.float32))
        motion = data["motion_windows"]
        terrain = data["terrain"]
        proprio = data["proprio"]
        history_root = data["history_root"]
        history_joint = data["history_joint"]
        if (
          history_root.ndim != 3
          or history_root.shape[0] != motion.shape[0]
          or history_root.shape[2] != 9
        ):
          raise ValueError(
            f"{path.name}: history_root has shape {history_root.shape}; "
            "expected (N,H,9) with the same N as motion_windows"
          )
        history_size = int(history_root.shape[1])
        if (
          history_joint.ndim != 3
          or history_joint.shape[:2] != (motion.shape[0], history_size)
          or history_joint.shape[2] != NUM_JOINTS
        ):
          raise ValueError(
            f"{path.name}: history_joint has shape {history_joint.shape}; "
            f"expected (N,{history_size},{NUM_JOINTS})"
          )
        if terrain.ndim != 3 or terrain.shape[:2] != (motion.shape[0], history_size):
          raise ValueError(
            f"{path.name}: terrain has shape {terrain.shape}; "
            f"expected (N,{history_size},H)"
          )
        if proprio.ndim != 3 or proprio.shape[:2] != (motion.shape[0], history_size):
          raise ValueError(
            f"{path.name}: proprio has shape {proprio.shape}; "
            f"expected (N,{history_size},P)"
          )
        history_roots.append(history_root.astype(np.float32))
        history_joints.append(history_joint.astype(np.float32))
        history_sizes.add(history_size)
        frame_strides.add(int(np.asarray(data["future_frame_stride"]).reshape(-1)[0]))
  if not terrains:
    raise FileNotFoundError(f"No conditional NPZ files found in {data_dir}")
  if len(frame_strides) != 1:
    raise ValueError(f"Mixed future_frame_stride values in {data_dir}: {sorted(frame_strides)}")
  if len(history_sizes) != 1:
    raise ValueError(f"Mixed history_size values in {data_dir}: {sorted(history_sizes)}")
  return (
    np.concatenate(motions),
    np.concatenate(terrains),
    np.concatenate(proprios),
    np.concatenate(history_roots),
    np.concatenate(history_joints),
    frame_strides.pop(),
  )


def _set_velocity_command(
  proprio: np.ndarray, vx: float, vy: float, wz: float
) -> np.ndarray:
  """Replace the repeated joystick-style command; preserve measured history."""
  result = proprio.copy()
  result[:, 25:28] = np.asarray([vx, vy, wz], dtype=np.float32)
  return result


@torch.no_grad()
def _run_generate(
  model: DiffusionDenoiser,
  scheduler: DDPMScheduler | None,
  sampling_cfg: dict,
  q_low: np.ndarray,
  q_high: np.ndarray,
  t_q_low: np.ndarray,
  t_q_high: np.ndarray,
  p_q_low: np.ndarray,
  p_q_high: np.ndarray,
  terrain_raw: torch.Tensor,
  proprio_raw: torch.Tensor,
  window_size: int,
  feature_dim: int,
  device: torch.device,
) -> torch.Tensor:
  """Conditional DDPM or Flow Matching sampling."""
  terrain = _quantile_normalize(terrain_raw, t_q_low, t_q_high)
  proprio = _quantile_normalize(proprio_raw, p_q_low, p_q_high)
  x_t = torch.randn(1, window_size, feature_dim, device=device)
  method = str(sampling_cfg.get("generative_method", "ddpm"))
  if method == "ddpm":
    if scheduler is None:
      raise RuntimeError("DDPM checkpoint has no scheduler")
    for t in reversed(range(scheduler.num_timesteps)):
      t_batch = torch.full((1,), t, dtype=torch.long, device=device)
      eps = model(x_t, t_batch, terrain=terrain, proprio=proprio)
      x_t = scheduler.step(eps, x_t, t)
  else:
    x_t = sample_flow(
      model, x_t, terrain, proprio,
      int(sampling_cfg.get("sampling_steps", 10)),
      str(sampling_cfg.get("sampler", "euler")),
      float(sampling_cfg.get("time_embedding_scale", 1000.0)),
    )
  q_low_t = torch.from_numpy(q_low).float().to(device)
  q_high_t = torch.from_numpy(q_high).float().to(device)
  return _quantile_denormalize(x_t.squeeze(0), q_low_t, q_high_t).cpu()


def main(cfg: Cfg) -> None:
  device_str = cfg.device or detect_device()
  device = torch.device(device_str)
  print(f"Device: {device_str}")

  ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
  model, scheduler, ckpt = _build_model_and_sampler(ckpt, device)
  print(f"Loaded checkpoint epoch={ckpt.get('epoch')} from {cfg.ckpt_path}")

  feature_dim = int(ckpt["cfg"]["feature_dim"])
  window_size = int(ckpt["cfg"]["window_size"])
  if feature_dim != 80 or window_size != 10:
    raise ValueError(
      f"This visualizer expects the CASBOT (Q=10,F=80) checkpoint; "
      f"got W={window_size}, F={feature_dim}"
    )
  if ckpt["cfg"].get("diffusion_format_version") != 7:
    raise ValueError(
      "This visualizer expects a format-v7 checkpoint; retrain after "
      "removing historical root velocity."
    )
  if ckpt["cfg"].get("proprio_layout") != "joint_pos,velocity_command_local":
    raise ValueError(
      "This visualizer expects the new proprio layout "
      "'joint_pos,velocity_command_local'; retrain the checkpoint after "
      "removing historical root velocity."
    )
  if int(ckpt["cfg"].get("proprio_dim", 28)) != 28:
    raise ValueError(
      f"This visualizer expects proprio_dim=28, got "
      f"{ckpt['cfg'].get('proprio_dim')}"
    )
  print(
    f"  feature_dim={feature_dim}  window_size={window_size}  "
    f"method={ckpt['cfg'].get('generative_method', 'ddpm')}  "
    f"steps={scheduler.num_timesteps if scheduler is not None else ckpt['cfg'].get('sampling_steps', 10)}"
  )

  # Set up CasBot via mujoco.
  sim = _CasbotSim(cfg.model_path)
  print(f"Loaded floating-base CASBOT model: {cfg.model_path}")

  # Diffusion root xyz is an H0-relative waist_yaw_link offset.  Visualize H0
  # at x=y=0 and use the XML qpos0 waist height/orientation.  The freejoint's
  # qpos0 position belongs to base_link, so using qpos0[0:3] directly here
  # would introduce a constant base-to-waist height error.
  anchor_pelvis_pos, anchor_pelvis_quat = sim.default_waist_pose()
  print(
    "Visualization H0 waist anchor: "
    f"xyz={np.array2string(anchor_pelvis_pos, precision=5)} "
    f"quat_wxyz={np.array2string(anchor_pelvis_quat, precision=5)}"
  )

  (
    all_motions,
    all_terrains,
    all_proprios,
    all_history_roots,
    all_history_joints,
    future_frame_stride,
  ) = (
    _load_condition_windows(cfg.data_dir)
  )
  # The checkpoint and dataset must use the same temporal target spacing.
  checkpoint_stride = int(ckpt["cfg"].get("future_frame_stride", 1))
  if checkpoint_stride != future_frame_stride:
    raise ValueError(
      f"Checkpoint future_frame_stride={checkpoint_stride} but dataset uses "
      f"{future_frame_stride}"
    )
  checkpoint_grid_layout = ckpt["cfg"].get("terrain_grid_layout")
  if checkpoint_grid_layout != TERRAIN_GRID_LAYOUT:
    print(
      "[WARN] This checkpoint predates canonical terrain-grid ordering; "
      "retrain Flow Matching/DDPM on the regenerated datasets for correct "
      f"terrain conditioning (checkpoint={checkpoint_grid_layout!r}, "
      f"required={TERRAIN_GRID_LAYOUT!r})."
    )
  terrain_index = {"value": 0}
  print(f"Loaded {len(all_terrains)} condition windows from {cfg.data_dir}")

  def run(
    terrain_raw: np.ndarray,
    proprio_raw: np.ndarray,
    actual_motion: np.ndarray,
    history_root: np.ndarray,
    history_joint: np.ndarray,
  ) -> tuple:
    terrain_tensor = torch.from_numpy(terrain_raw[None]).to(device)
    proprio_tensor = torch.from_numpy(proprio_raw[None]).to(device)
    history_pos, history_quat, history_joint_tensor = _decode_history_trajectory(
      history_root, history_joint
    )
    h0_pos, h0_quat = history_pos[0], history_quat[0]
    pred_denorm = _run_generate(
      model,
      scheduler,
      ckpt["cfg"],
      ckpt["q_low"],
      ckpt["q_high"],
      ckpt["t_q_low"],
      ckpt["t_q_high"],
      ckpt["p_q_low"],
      ckpt["p_q_high"],
      terrain_tensor,
      proprio_tensor,
      window_size,
      feature_dim,
      device,
    )
    generated_f0 = pred_denorm[0].numpy()
    actual_f0 = np.asarray(actual_motion[0], dtype=np.float32)
    command = np.asarray(proprio_raw[-1, 25:28], dtype=np.float32)
    # Motion layout: root lin vel is [74:77], root ang vel is [77:80].
    generated_velocity = generated_f0[[74, 75, 79]]
    actual_velocity = actual_f0[[74, 75, 79]]
    history_last = history_root.shape[0] - 1
    print(
      "[FlowSample] "
      f"command_H{history_last}_local[vx,vy,wz]={np.array2string(command, precision=4)} | "
      f"generated_H{history_last}+{future_frame_stride}_H0_local="
      f"{np.array2string(generated_velocity, precision=4)} | "
      f"actual_H{history_last}+{future_frame_stride}_H0_local="
      f"{np.array2string(actual_velocity, precision=4)}",
      flush=True,
    )
    p_pos, p_quat, p_joint = _window_to_pelvis_trajectory(
      pred_denorm,
      h0_pos,
      h0_quat,
    )
    key_ee_pos = _window_to_ee_trajectories(
      pred_denorm, p_pos, h0_quat
    )
    # Interpolate from the last recorded history pose to the first generated
    # keyframe, then through all generated keyframes. With the new 10 Hz
    # dataset and stride=1 this produces the ten contiguous future frames;
    # with the older stride=5 dataset it produces the dense F0...F49 view.
    h3_pos, h3_quat, h3_joint = (
      history_pos[-1], history_quat[-1], history_joint_tensor[-1]
    )
    terrain_center_h3 = h3_pos[2] - torch.as_tensor(
      terrain_raw[-1, CENTER_IDX], dtype=h3_pos.dtype, device=h3_pos.device
    )
    print(
      "[VizFrame] "
      f"H0_root_xyz={np.array2string(h0_pos.numpy(), precision=4)} "
      f"H0_root_quat_wxyz={np.array2string(h0_quat.numpy(), precision=4)} "
      f"H{history_pos.shape[0] - 1}_terrain_center_z={float(terrain_center_h3):.4f}",
      flush=True,
    )
    p_pos, p_quat, p_joint, _dense_ee = _interpolate_sparse_trajectory(
      h3_pos, h3_quat, h3_joint,
      p_pos, p_quat, p_joint, key_ee_pos, future_frame_stride,
    )
    all_pos = torch.cat((history_pos, p_pos), dim=0)
    all_quat = torch.cat((history_quat, p_quat), dim=0)
    all_joint = torch.cat((history_joint_tensor, p_joint), dim=0)
    return (
      all_pos.numpy(),
      all_quat.numpy(),
      all_joint.numpy(),
      terrain_raw,
      h3_pos.cpu().numpy(),
      h3_quat.cpu().numpy(),
    )

  initial_proprio = _set_velocity_command(
    all_proprios[0], cfg.command_vx, cfg.command_vy, cfg.command_wz
  )
  state: dict = {
    "pred": run(
      all_terrains[0], initial_proprio, all_motions[0],
      all_history_roots[0], all_history_joints[0]
    )
  }

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, sim.model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  terrain_points = server.scene.add_point_cloud(
    name="/terrain/height_map",
    points=np.zeros((TERRAIN_DIM, 3), dtype=np.float32),
    colors=np.tile(np.array([40, 220, 80], dtype=np.uint8), (TERRAIN_DIM, 1)),
    point_size=0.012,
  )

  # The visualizer shows all history frames followed by the interpolated future
  # frames.  ``history_root`` belongs to the nested ``run`` callback, so it
  # is not available in this scope; derive the frame count from the loaded
  # auxiliary metadata instead.
  history_size = int(all_history_roots.shape[1])
  total_visual_frames = history_size + window_size * future_frame_stride

  with server.gui.add_folder("Generate"):
    frame_slider = server.gui.add_slider(
      f"Frame (H0-H{history_size - 1}, future)",
      min=0,
      max=total_visual_frames - 1,
      step=1,
      initial_value=0,
    )
    play_btn = server.gui.add_button("Play / Pause")
    resample_btn = server.gui.add_button("Resample")
    reset_velocity_btn = server.gui.add_button("Reset Velocity")
    vx_slider = server.gui.add_slider(
      "command vx (m/s)", min=-1.5, max=2.0, step=0.05, initial_value=cfg.command_vx
    )
    vy_slider = server.gui.add_slider(
      "command vy (m/s)", min=-1.0, max=1.0, step=0.05, initial_value=cfg.command_vy
    )
    wz_slider = server.gui.add_slider(
      "command wz (rad/s)", min=-2.0, max=2.0, step=0.05, initial_value=cfg.command_wz
    )

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  @resample_btn.on_click
  def _(_evt) -> None:
    terrain_index["value"] = int(np.random.randint(0, len(all_terrains)))
    idx = terrain_index["value"]
    proprio = _set_velocity_command(
      all_proprios[idx], vx_slider.value, vy_slider.value, wz_slider.value
    )
    state["pred"] = run(
      all_terrains[idx], proprio, all_motions[idx],
      all_history_roots[idx], all_history_joints[idx]
    )

  @reset_velocity_btn.on_click
  def _(_evt) -> None:
    # Keep the currently selected history/terrain window and regenerate only
    # with the command currently shown by the velocity sliders.
    idx = terrain_index["value"]
    proprio = _set_velocity_command(
      all_proprios[idx], vx_slider.value, vy_slider.value, wz_slider.value
    )
    state["pred"] = run(
      all_terrains[idx], proprio, all_motions[idx],
      all_history_roots[idx], all_history_joints[idx]
    )

  def render(frame: int) -> None:
    p_pos, p_quat, p_joint, terrain, terrain_anchor_pos, terrain_anchor_quat = state["pred"]
    sim.write_pose(p_pos[frame], p_quat[frame], p_joint[frame])
    viser_scene.update_from_arrays(
	      body_xpos=np.asarray(sim.data.xpos).reshape(1, -1, 3),
	      body_xmat=np.asarray(sim.data.xmat).reshape(1, -1, 3, 3),
	      qpos=np.asarray(sim.data.qpos)[None],
      env_idx=0,
    )
    # Training stores clearance = waist_z - terrain_z in each history scan frame.
    gx, gy = np.meshgrid(
      np.linspace(-0.8, 0.8, GRID_X, dtype=np.float32),
      np.linspace(-0.5, 0.5, GRID_Y, dtype=np.float32),
      indexing="xy",
    )
    local_xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    # Historical scans belong to their corresponding history root pose. The
    # future scan is held fixed at the last history frame because no future
    # terrain is predicted.
    if frame < terrain.shape[0]:
      terrain_frame_pos = p_pos[frame]
      terrain_frame_quat = p_quat[frame]
      clearance = terrain[frame]
    else:
      terrain_frame_pos = terrain_anchor_pos
      terrain_frame_quat = terrain_anchor_quat
      clearance = terrain[-1]
    w, x, y, z = terrain_frame_quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy, sy = np.cos(yaw), np.sin(yaw)
    world_xy = np.empty_like(local_xy)
    world_xy[:, 0] = (
      local_xy[:, 0] * cy - local_xy[:, 1] * sy + terrain_frame_pos[0]
    )
    world_xy[:, 1] = (
      local_xy[:, 0] * sy + local_xy[:, 1] * cy + terrain_frame_pos[1]
    )
    # Each scan value is waist_z - terrain_z.  Recover world terrain heights
    # from the corresponding historical root or fixed H3 future root.
    terrain_z = terrain_frame_pos[2] - clearance
    terrain_world_points = np.column_stack([world_xy, terrain_z])
    # ``MjlabViserScene`` applies its camera-tracking offset to all MuJoCo
    # body meshes.  The point cloud is a separate Viser object, so it does not
    # inherit that transform automatically.  Apply the same offset here;
    # otherwise the robot follows the camera while the terrain remains in the
    # original world frame and appears artificially close to the waist.
    scene_offset = np.asarray(viser_scene._scene_offset, dtype=np.float32)
    terrain_points.points = terrain_world_points + scene_offset[None, :]
    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL in a browser.")
  dt_play = 1.0 / cfg.fps
  try:
    while True:
      render(int(frame_slider.value))
      if playing["v"]:
        nxt = (int(frame_slider.value) + 1) % total_visual_frames
        frame_slider.value = nxt
      time.sleep(dt_play)
  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
