"""Convert collected CASBOT height-map motions to conditional diffusion windows.

Each input NPZ contains per-frame G1 motion data + height_map (terrain).  This
script windows both the motion and terrain data, producing output NPZs with:

  motion_windows  (N, future_size, 80)  — future CASBOT motion targets
  terrain         (N, history_size, 693) — historical terrain
  proprio         (N, history_size, 31) — joint q25 + target-heading rot6d

Motion features are anchored to the LAST window frame's yaw-only local frame
(identical to csv_to_npz.py).  Terrain stays in each frame's own local frame.

Usage:
python diffusion/scripts/height_map_to_npz.py \
  --input-dir /home/casbot/Desktop/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/parkour/config/casbot_02/moton_with_height_map \
  --output-dir /home/casbot/Desktop/whole_body_tracking/diffusion/source/datasets 
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import tyro
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.utils import detect_device

# ── IsaacLab articulation body order (breadth-first URDF traversal) ──
# body_*_w is not in MuJoCo XML depth-first order.
ROOT_BODY_IDX = 3  # waist_yaw_link
ROOT_BODY_NAME = "waist_yaw_link"
TERRAIN_LAYOUT = "root_z_minus_terrain_z"
EE_IDXS = (22, 23, 11, 24, 25)  # left/right ankle, head, left/right wrist
NUM_JOINTS = 25
NUM_EE = len(EE_IDXS)

# ── Feature dimension breakdown ──
FEATURE_DIMS = (3, 6, NUM_JOINTS, NUM_JOINTS, NUM_EE * 3, 3, 3)  # 80 total
PROPRIO_DIMS = (NUM_JOINTS, 6)

# ── Terrain grid ──
GRID_X = 33
GRID_Y = 21
HEIGHT_MAP_DIM = GRID_X * GRID_Y  # 693

# Isaac articulation order -> CASBOT limb-grouped order used by visualization.
CASBOT_JOINT_ORDER = (
    0, 3, 8, 13, 17, 21,
    1, 4, 9, 14, 18, 22,
    2, 5, 10,
    6, 11, 15, 19, 23,
    7, 12, 16, 20, 24,
)


@dataclass
class Cfg:
    input_dir: str = "datasets/walk_g1_height_map_npz"
    """Directory of input NPZ files (height-map motion data)."""
    output_dir: str = "datasets/conditional_npz"
    """Directory to write output windowed NPZ files."""
    history_size: int = 4
    """Number of past/current condition frames."""
    future_size: int = 10
    """Number of future motion frames."""
    stride: int = 1
    """Stride between consecutive windows."""
    fps: int = 50
    """Input fps (data is already at this frame rate; no resampling)."""
    max_root_step: float = 2.0
    """Truncate at a larger per-frame root jump (end-of-motion terrain resampling)."""
    device: str = ""
    """Compute device. Empty = auto."""
    shard_index: int = 0
    """Index of this shard (for parallel runs)."""
    num_shards: int = 1
    """Total number of shards (for parallel runs)."""


def _tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to 6D tan-norm: stacked [col0, col2] of rot matrix."""
    mat = matrix_from_quat(quat)
    col0 = mat[..., :, 0]
    col2 = mat[..., :, 2]
    return torch.cat([col0, col2], dim=-1)


def _compute_windows(
    base_pos: torch.Tensor,       # (T, 3)
    base_quat: torch.Tensor,      # (T, 4)
    base_lin_vel: torch.Tensor,   # (T, 3)
    base_ang_vel: torch.Tensor,   # (T, 3)
    ee_pos: torch.Tensor,         # (T, E, 3)
    joint_pos: torch.Tensor,      # (T, 25)
    joint_vel: torch.Tensor,      # (T, 25)
    height_map: torch.Tensor,     # (T, 693), world terrain z only
    history_size: int,
    future_size: int,
    stride: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Return future motion and historical terrain/joint-heading conditions.

    Motion features anchored to the LAST frame's yaw-only local frame.
    root_pos z is terrain-relative (height above terrain at pelvis xy).
    Terrain is stored as root height minus terrain height for every ray.
    """
    T = base_pos.shape[0]
    total_size = history_size + future_size
    if T < total_size:
        return None, None, None

    E = ee_pos.shape[1]
    J = joint_pos.shape[1]

    starts = torch.arange(0, T - total_size + 1, stride, device=base_pos.device, dtype=torch.long)
    offsets = torch.arange(total_size, device=base_pos.device, dtype=torch.long)
    all_idx = starts[:, None] + offsets[None, :]
    history_idx = all_idx[:, :history_size]
    future_idx = all_idx[:, history_size:]
    N, H, W = all_idx.shape[0], history_size, future_size
    flat_idx = future_idx.reshape(-1)

    # Gather motion data
    win_base_pos = base_pos.index_select(0, flat_idx).reshape(N, W, 3)
    win_base_quat = base_quat.index_select(0, flat_idx).reshape(N, W, 4)
    win_base_lin_vel = base_lin_vel.index_select(0, flat_idx).reshape(N, W, 3)
    win_base_ang_vel = base_ang_vel.index_select(0, flat_idx).reshape(N, W, 3)
    win_ee_pos = ee_pos.index_select(0, flat_idx).reshape(N, W, E, 3)
    win_joint = joint_pos.index_select(0, flat_idx).reshape(N, W, J)
    win_joint_vel = joint_vel.index_select(0, flat_idx).reshape(N, W, J)

    # Future targets are anchored to the latest available history frame.
    current_idx = history_idx[:, -1]
    anchor_pos_T = base_pos.index_select(0, current_idx)
    anchor_quat_T = base_quat.index_select(0, current_idx)
    yaw_T = yaw_quat(anchor_quat_T)                # (N, 4)
    heading_inv_T_WF = quat_conjugate(yaw_T)[:, None, :].expand(N, W, 4).reshape(-1, 4)
    yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

    # root_pos: xy in heading-invariant frame, z terrain-relative (height above terrain)
    root_offset = win_base_pos - anchor_pos_T[:, None, :]  # (N, W, 3)
    root_pos_local = quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(N, W, 3)
    root_pos_local = root_pos_local.clone()
    # Terrain height directly below the waist: center of the 33×21 grid.
    future_terrain = height_map.index_select(0, flat_idx).reshape(N, W, HEIGHT_MAP_DIM)
    terrain_z = future_terrain[:, :, GRID_X // 2 * GRID_Y + GRID_Y // 2]
    root_pos_local[..., 2] = win_base_pos[..., 2] - terrain_z

    # root_rot: heading_inv(T) ⊗ root_quat[t] → 6D tan-norm
    root_rot_local_quat = quat_mul(
        heading_inv_T_WF, win_base_quat.reshape(-1, 4)
    ).reshape(N, W, 4)
    root_rot_6d = _tan_norm_from_quat(root_rot_local_quat)

    # EE: (ee[t] - root[t]) rotated into last-frame heading-invariant frame
    ee_offset_w = win_ee_pos - win_base_pos[:, :, None, :]  # (N, W, E, 3)
    yaw_T_E = yaw_T[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
    ee_pos_local = quat_apply_inverse(yaw_T_E, ee_offset_w.reshape(-1, 3)).reshape(N, W, E * 3)

    # Velocities: rotated into last-frame heading-invariant frame
    lin_vel_local = quat_apply_inverse(yaw_T_W, win_base_lin_vel.reshape(-1, 3)).reshape(N, W, 3)
    ang_vel_local = quat_apply_inverse(yaw_T_W, win_base_ang_vel.reshape(-1, 3)).reshape(N, W, 3)

    # Historical joint configuration plus desired yaw.  The desired yaw is
    # the final future frame's yaw, expressed relative to every history frame.
    hist_flat = history_idx.reshape(-1)
    hist_quat = base_quat.index_select(0, hist_flat)
    hist_joint = joint_pos.index_select(0, hist_flat).reshape(N, H, J)
    hist_yaw = yaw_quat(hist_quat).reshape(N, H, 4)
    target_yaw = yaw_quat(win_base_quat[:, -1])[:, None, :].expand(N, H, 4)
    relative_target_yaw = quat_mul(
        quat_conjugate(hist_yaw.reshape(-1, 4)), target_yaw.reshape(-1, 4)
    ).reshape(N, H, 4)
    target_heading_6d = _tan_norm_from_quat(relative_target_yaw)
    proprio = torch.cat([hist_joint, target_heading_6d], dim=-1)

    motion = torch.cat(
        [
            root_pos_local,
            root_rot_6d,
            win_joint,
            win_joint_vel,
            ee_pos_local,
            lin_vel_local,
            ang_vel_local,
        ],
        dim=-1,
    )  # (N, W, 80)

    history_terrain_w = height_map.index_select(0, hist_flat).reshape(
        N, H, HEIGHT_MAP_DIM
    )
    history_root_z = base_pos.index_select(0, hist_flat)[..., 2].reshape(N, H, 1)
    history_terrain = history_root_z - history_terrain_w
    return motion, history_terrain, proprio


def main(cfg: Cfg) -> None:
    if not cfg.device:
        cfg.device = detect_device()
    print(f"Device: {cfg.device}")

    in_dir = Path(cfg.input_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {in_dir}")
    if cfg.num_shards > 1:
        npz_files = npz_files[cfg.shard_index :: cfg.num_shards]
        print(f"Shard {cfg.shard_index}/{cfg.num_shards}: {len(npz_files)} files")

    total_motion_dim = sum(FEATURE_DIMS)
    print(f"Files: {len(npz_files)} in {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Window: history={cfg.history_size} future={cfg.future_size} stride={cfg.stride}")
    print(f"Motion dim: {total_motion_dim} (= {' + '.join(str(d) for d in FEATURE_DIMS)})")
    print(f"Terrain dim: {HEIGHT_MAP_DIM} (= {GRID_X}×{GRID_Y} grid)")

    total_windows = 0

    for i, npz_path in enumerate(npz_files):
        print(f"\n[{i + 1}/{len(npz_files)}] {npz_path.name}")

        data = np.load(npz_path, allow_pickle=False)

        # Validate
        T = data["joint_pos"].shape[0]
        root_xy = data["body_pos_w"][:, ROOT_BODY_IDX, :2]
        jump_indices = np.flatnonzero(
            np.linalg.norm(np.diff(root_xy, axis=0), axis=-1) > cfg.max_root_step
        )
        if len(jump_indices) > 0:
            original_T = T
            T = int(jump_indices[0] + 1)
            print(
                f"  [WARN] root discontinuity after frame {T - 1}; "
                f"truncating {original_T} -> {T} frames"
            )
        total_size = cfg.history_size + cfg.future_size
        if T < total_size:
            print(f"  [SKIP] too short ({T} < {total_size})")
            continue

        # Extract motion data → torch tensors on device
        device = torch.device(cfg.device)
        base_pos = torch.from_numpy(data["body_pos_w"][:T, ROOT_BODY_IDX, :].astype(np.float32)).to(device)
        base_quat = torch.from_numpy(data["body_quat_w"][:T, ROOT_BODY_IDX, :].astype(np.float32)).to(device)
        base_lin_vel = torch.from_numpy(data["body_lin_vel_w"][:T, ROOT_BODY_IDX, :].astype(np.float32)).to(device)
        base_ang_vel = torch.from_numpy(data["body_ang_vel_w"][:T, ROOT_BODY_IDX, :].astype(np.float32)).to(device)

        ee_pos_list = [data["body_pos_w"][:T, idx, :].astype(np.float32) for idx in EE_IDXS]
        ee_pos = torch.from_numpy(np.stack(ee_pos_list, axis=1)).to(device)  # (T, E, 3)

        joint_pos = torch.from_numpy(
            data["joint_pos"][:T].astype(np.float32)[:, CASBOT_JOINT_ORDER]
        ).to(device)
        joint_vel = torch.from_numpy(
            data["joint_vel"][:T].astype(np.float32)[:, CASBOT_JOINT_ORDER]
        ).to(device)

        elevation_xyz = data["elevation_map_xyz"][:T].astype(np.float32)
        if elevation_xyz.shape[1:] != (HEIGHT_MAP_DIM, 3):
            raise ValueError(
                f"{npz_path.name}: elevation_map_xyz has shape {elevation_xyz.shape}, "
                f"expected (T,{HEIGHT_MAP_DIM},3)"
            )
        height_map = torch.from_numpy(elevation_xyz[..., 2]).to(device)

        motion, terrain, proprio = _compute_windows(
            base_pos, base_quat, base_lin_vel, base_ang_vel,
            ee_pos, joint_pos, joint_vel, height_map,
            cfg.history_size, cfg.future_size, cfg.stride,
        )

        if (
            motion is None
            or terrain is None
            or proprio is None
        ):
            print(f"  [SKIP] too short for history+future={total_size}")
            continue

        n_windows = motion.shape[0]
        total_windows += n_windows

        out_path = out_dir / f"{npz_path.stem}.npz"
        np.savez_compressed(
            out_path,
            motion_windows=motion.cpu().numpy().astype(np.float32),
            terrain=terrain.cpu().numpy().astype(np.float32),
            proprio=proprio.cpu().numpy().astype(np.float32),
            fps=np.array([cfg.fps], dtype=np.float32),
            history_size=np.array([cfg.history_size], dtype=np.int32),
            future_size=np.array([cfg.future_size], dtype=np.int32),
            stride=np.array([cfg.stride], dtype=np.int32),
            feature_dims=np.array(FEATURE_DIMS, dtype=np.int32),
            proprio_dims=np.array(PROPRIO_DIMS, dtype=np.int32),
            diffusion_format_version=np.array(3, dtype=np.int32),
            root_body=np.array(ROOT_BODY_NAME),
            terrain_layout=np.array(TERRAIN_LAYOUT),
            # Actual flattened array layout for IsaacLab ordering="xy": Y x X.
            terrain_shape=np.array([GRID_Y, GRID_X], dtype=np.int32),
        )
        print(
            f"  saved {out_path.name}: motion={tuple(motion.shape)} "
            f"terrain={tuple(terrain.shape)} proprio={tuple(proprio.shape)}"
        )

    print(f"\nDone. Total windows: {total_windows}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
