"""Convert collected CASBOT height-map motions to conditional diffusion windows.

Each input NPZ contains per-frame G1 motion data + height_map (terrain).  This
script windows both the motion and terrain data, producing output NPZs with:

  motion_windows  (N, window_size, 80) — CASBOT motion features
  terrain         (N, window_size, 693) — z-only 33×21 terrain grid
  command         (N, window_size, 3)   — root-local [vx, vy, wz]

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

# ── Body indices into body_*_w arrays (world excluded, so MJCF idx - 1) ──
PELVIS_IDX = 0
EE_IDXS = (6, 12, 13, 19, 25)  # ankles, waist proxy, wrists
NUM_JOINTS = 25
NUM_EE = len(EE_IDXS)

# ── Feature dimension breakdown ──
FEATURE_DIMS = (3, 6, NUM_JOINTS, NUM_JOINTS, NUM_EE * 3, 3, 3)  # 80 total

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
    window_size: int = 4
    """Number of frames per window."""
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
    window_size: int,
    stride: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Return motion, terrain-z and root-local velocity-command windows.

    Motion features anchored to the LAST frame's yaw-only local frame.
    root_pos z is terrain-relative (height above terrain at pelvis xy).
    Terrain stays in each frame's own local frame.
    """
    T = base_pos.shape[0]
    if T < window_size:
        return None, None, None

    E = ee_pos.shape[1]
    J = joint_pos.shape[1]

    starts = torch.arange(0, T - window_size + 1, stride, device=base_pos.device, dtype=torch.long)
    offsets = torch.arange(window_size, device=base_pos.device, dtype=torch.long)
    win_idx = starts[:, None] + offsets[None, :]  # (N, W)
    N, W = win_idx.shape[0], window_size

    flat_idx = win_idx.reshape(-1)

    # Gather motion data
    win_base_pos = base_pos.index_select(0, flat_idx).reshape(N, W, 3)
    win_base_quat = base_quat.index_select(0, flat_idx).reshape(N, W, 4)
    win_base_lin_vel = base_lin_vel.index_select(0, flat_idx).reshape(N, W, 3)
    win_base_ang_vel = base_ang_vel.index_select(0, flat_idx).reshape(N, W, 3)
    win_ee_pos = ee_pos.index_select(0, flat_idx).reshape(N, W, E, 3)
    win_joint = joint_pos.index_select(0, flat_idx).reshape(N, W, J)
    win_joint_vel = joint_vel.index_select(0, flat_idx).reshape(N, W, J)

    # ── Anchor to last frame's yaw-only local frame ──
    anchor_pos_T = win_base_pos[:, -1, :]          # (N, 3)
    anchor_quat_T = win_base_quat[:, -1, :]        # (N, 4)
    yaw_T = yaw_quat(anchor_quat_T)                # (N, 4)
    heading_inv_T_WF = quat_conjugate(yaw_T)[:, None, :].expand(N, W, 4).reshape(-1, 4)
    yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

    # root_pos: xy in heading-invariant frame, z terrain-relative (height above terrain)
    root_offset = win_base_pos - anchor_pos_T[:, None, :]  # (N, W, 3)
    root_pos_local = quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(N, W, 3)
    root_pos_local = root_pos_local.clone()
    # Terrain height at pelvis xy = center of the 17×11 grid (directly below pelvis).
    # Grid is pelvis-centered, yaw-aligned → center index (8, 5) = 8*11+5 = 93.
    win_terrain = height_map.index_select(0, flat_idx).reshape(N, W, HEIGHT_MAP_DIM)
    terrain_z = win_terrain[:, :, GRID_X // 2 * GRID_Y + GRID_Y // 2]  # (N, W)
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

    # Hand-controller condition uses each frame's own heading frame, rather
    # than the last-frame anchor used by the motion representation.
    frame_yaw = yaw_quat(win_base_quat.reshape(-1, 4))
    command_lin = quat_apply_inverse(
        frame_yaw, win_base_lin_vel.reshape(-1, 3)
    ).reshape(N, W, 3)
    command_ang = quat_apply_inverse(
        frame_yaw, win_base_ang_vel.reshape(-1, 3)
    ).reshape(N, W, 3)
    command = torch.stack(
        [command_lin[..., 0], command_lin[..., 1], command_ang[..., 2]], dim=-1
    )

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

    return motion, win_terrain, command


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
    print(f"Window: size={cfg.window_size} stride={cfg.stride}")
    print(f"Motion dim: {total_motion_dim} (= {' + '.join(str(d) for d in FEATURE_DIMS)})")
    print(f"Terrain dim: {HEIGHT_MAP_DIM} (= {GRID_X}×{GRID_Y} grid)")

    total_windows = 0

    for i, npz_path in enumerate(npz_files):
        print(f"\n[{i + 1}/{len(npz_files)}] {npz_path.name}")

        data = np.load(npz_path, allow_pickle=False)

        # Validate
        T = data["joint_pos"].shape[0]
        root_xy = data["body_pos_w"][:, PELVIS_IDX, :2]
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
        if T < cfg.window_size:
            print(f"  [SKIP] too short ({T} < {cfg.window_size})")
            continue

        # Extract motion data → torch tensors on device
        device = torch.device(cfg.device)
        base_pos = torch.from_numpy(data["body_pos_w"][:T, PELVIS_IDX, :].astype(np.float32)).to(device)
        base_quat = torch.from_numpy(data["body_quat_w"][:T, PELVIS_IDX, :].astype(np.float32)).to(device)
        base_lin_vel = torch.from_numpy(data["body_lin_vel_w"][:T, PELVIS_IDX, :].astype(np.float32)).to(device)
        base_ang_vel = torch.from_numpy(data["body_ang_vel_w"][:T, PELVIS_IDX, :].astype(np.float32)).to(device)

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

        motion, terrain, command = _compute_windows(
            base_pos, base_quat, base_lin_vel, base_ang_vel,
            ee_pos, joint_pos, joint_vel, height_map,
            cfg.window_size, cfg.stride,
        )

        if motion is None or terrain is None or command is None:
            print(f"  [SKIP] too short for window_size={cfg.window_size}")
            continue

        n_windows = motion.shape[0]
        total_windows += n_windows

        out_path = out_dir / f"{npz_path.stem}.npz"
        np.savez_compressed(
            out_path,
            motion_windows=motion.cpu().numpy().astype(np.float32),
            terrain=terrain.cpu().numpy().astype(np.float32),
            command=command.cpu().numpy().astype(np.float32),
            fps=np.array([cfg.fps], dtype=np.float32),
            window_size=np.array([cfg.window_size], dtype=np.int32),
            stride=np.array([cfg.stride], dtype=np.int32),
            feature_dims=np.array(FEATURE_DIMS, dtype=np.int32),
            # Actual flattened array layout for IsaacLab ordering="xy": Y x X.
            terrain_shape=np.array([GRID_Y, GRID_X], dtype=np.int32),
        )
        print(
            f"  saved {out_path.name}: motion={tuple(motion.shape)} "
            f"terrain={tuple(terrain.shape)} command={tuple(command.shape)}"
        )

    print(f"\nDone. Total windows: {total_windows}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
