"""Convert collected CASBOT height-map motions to conditional diffusion windows.

Each input NPZ contains per-frame G1 motion data + height_map (terrain).  This
script windows both the motion and terrain data, producing output NPZs with:

  motion_windows  (N, future_size, 80)  — sparse future CASBOT keyframes
  terrain         (N, history_size, 693) — historical terrain, in the
                   canonical IsaacLab grid ordering (x inner, y outer)
  proprio         (N, history_size, 28) — joint q25 + command3
  history_root    (N, history_size, 9)  — visualization-only root metadata
  history_joint   (N, history_size, 25) — visualization-only history joints

Future spatial features are expressed in the yaw-only heading frame of H0,
the first known history frame.  Future root xyz is an offset from
H0; end-effector positions remain offsets from each future frame's own root.
Terrain stays in each history frame's own local sampling grid.

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
MOTION_LAYOUT = "future_h0_heading_root_xyz_offset"
PROPRIO_LAYOUT = "joint_pos,velocity_command_local"
JOINT_LAYOUT = "isaaclab_articulation"
EE_IDXS = (22, 23, 11, 24, 25)  # left/right ankle, head, left/right wrist
NUM_JOINTS = 25
NUM_EE = len(EE_IDXS)

# ── Feature dimension breakdown ──
FEATURE_DIMS = (3, 6, NUM_JOINTS, NUM_JOINTS, NUM_EE * 3, 3, 3)  # 80 total
PROPRIO_DIMS = (NUM_JOINTS, 3)

# ── Terrain grid ──
GRID_X = 33
GRID_Y = 21
HEIGHT_MAP_DIM = GRID_X * GRID_Y  # 693
GRID_RESOLUTION = 0.05
GRID_X_MIN = -0.8
GRID_Y_MIN = -0.5
TERRAIN_GRID_LAYOUT = "x_ascending_y_ascending"


def _canonicalize_elevation_map(
    elevation_xyz: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    height_map_frame: str,
) -> np.ndarray:
    """Reorder XYZ ray samples into IsaacLab's fixed grid order.

    The recorded XYZ points are world points when ``--center_xy`` was used,
    but their flattened order is not guaranteed to be identical across all
    motion files (mirrored files can reverse the Y traversal).  The online
    RayCaster uses ``GridPatternCfg(ordering="xy")``: X is the inner index
    and Y is the outer index, with both coordinates increasing.  Recover the
    local X/Y grid index from the recorded points and place each terrain Z at
    that canonical index before computing the root-relative clearance.
    """
    if elevation_xyz.ndim != 3 or elevation_xyz.shape[1:] != (HEIGHT_MAP_DIM, 3):
        raise ValueError(
            "elevation_map_xyz must have shape "
            f"(T,{HEIGHT_MAP_DIM},3), got {elevation_xyz.shape}"
        )
    if root_pos_w.shape != (elevation_xyz.shape[0], 3):
        raise ValueError(
            f"root_pos_w must have shape {(elevation_xyz.shape[0], 3)}, "
            f"got {root_pos_w.shape}"
        )
    if root_quat_w.shape != (elevation_xyz.shape[0], 4):
        raise ValueError(
            f"root_quat_w must have shape {(elevation_xyz.shape[0], 4)}, "
            f"got {root_quat_w.shape}"
        )

    if height_map_frame in {"world", "world_centered"}:
        # Convert world XY samples into the current waist yaw frame.  The
        # scanner is mounted on waist_yaw_link, which is ROOT_BODY_IDX.
        w = root_quat_w[:, 0]
        qx = root_quat_w[:, 1]
        qy = root_quat_w[:, 2]
        qz = root_quat_w[:, 3]
        yaw = np.arctan2(
            2.0 * (w * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        cos_yaw = np.cos(yaw)[:, None]
        sin_yaw = np.sin(yaw)[:, None]
        relative_xy = elevation_xyz[..., :2] - root_pos_w[:, None, :2]
        local_x = relative_xy[..., 0] * cos_yaw + relative_xy[..., 1] * sin_yaw
        local_y = -relative_xy[..., 0] * sin_yaw + relative_xy[..., 1] * cos_yaw
        terrain_z_w = elevation_xyz[..., 2]
    elif height_map_frame == "yaw_aligned_sensor":
        # This is the representation returned before collect_motion's
        # --center_xy conversion.  Parkour's scanner offset is +20 m in Z.
        # The current data uses world coordinates, but handling this format
        # explicitly prevents silently treating sensor-relative Z as world Z.
        local_x = elevation_xyz[..., 0]
        local_y = elevation_xyz[..., 1]
        terrain_z_w = root_pos_w[:, None, 2] + 20.0 + elevation_xyz[..., 2]
    else:
        raise ValueError(
            f"Unsupported height_map_frame={height_map_frame!r}; expected "
            "'world', 'world_centered', or 'yaw_aligned_sensor'."
        )

    x_index = np.rint((local_x - GRID_X_MIN) / GRID_RESOLUTION).astype(np.int64)
    y_index = np.rint((local_y - GRID_Y_MIN) / GRID_RESOLUTION).astype(np.int64)
    flat_index = y_index * GRID_X + x_index
    valid = (
        (x_index >= 0)
        & (x_index < GRID_X)
        & (y_index >= 0)
        & (y_index < GRID_Y)
    )
    if not np.all(valid):
        bad = np.argwhere(~valid)[0]
        raise ValueError(
            "elevation_map_xyz contains a point outside the expected grid: "
            f"frame={int(bad[0])}, point={int(bad[1])}, "
            f"local_xy=({local_x[tuple(bad)]:.5f}, {local_y[tuple(bad)]:.5f})"
        )

    # Every frame must contain exactly one ray at every canonical grid index.
    expected = np.arange(HEIGHT_MAP_DIM, dtype=np.int64)
    sorted_indices = np.sort(flat_index, axis=1)
    if not np.all(sorted_indices == expected[None, :]):
        bad_frame = int(np.flatnonzero(np.any(sorted_indices != expected[None, :], axis=1))[0])
        raise ValueError(
            "elevation_map_xyz does not contain a complete regular grid at "
            f"frame {bad_frame}; cannot canonicalize terrain ordering."
        )

    canonical_z = np.empty((elevation_xyz.shape[0], HEIGHT_MAP_DIM), dtype=np.float32)
    frame_index = np.arange(elevation_xyz.shape[0], dtype=np.int64)[:, None]
    canonical_z[frame_index, flat_index] = terrain_z_w.astype(np.float32)
    return canonical_z

@dataclass
class Cfg:
    input_dir: str = "data"
    """Directory of input NPZ files (height-map motion data)."""
    output_dir: str = "datasets"
    """Directory to write output windowed NPZ files."""
    history_size: int = 2
    """Number of past/current condition frames."""
    future_size: int = 10
    """Number of sparse future motion keyframes."""
    future_frame_stride: int = 1
    """Source-frame interval between future keyframes."""
    stride: int = 1
    """Stride between consecutive windows."""
    fps: int = 10
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
    future_frame_stride: int,
    stride: int,
    fps: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Return future motion and historical terrain/proprio conditions.

    Future spatial features use H0's yaw-only heading frame.
    root_pos xyz is relative to H0's world root position.
    Terrain is stored as root height minus terrain height for every ray.
    """
    T = base_pos.shape[0]
    # History frames are consecutive. The future targets are sampled at the
    # configured source-frame interval after the last history frame.
    total_size = history_size + future_size * future_frame_stride
    if T < total_size:
        return None, None, None, None, None

    E = ee_pos.shape[1]
    J = joint_pos.shape[1]

    starts = torch.arange(0, T - total_size + 1, stride, device=base_pos.device, dtype=torch.long)
    history_offsets = torch.arange(history_size, device=base_pos.device, dtype=torch.long)
    history_idx = starts[:, None] + history_offsets[None, :]
    future_offsets = history_size - 1 + future_frame_stride * torch.arange(
        1, future_size + 1, device=base_pos.device, dtype=torch.long
    )
    future_idx = starts[:, None] + future_offsets[None, :]
    N, H, W = starts.shape[0], history_size, future_size
    flat_idx = future_idx.reshape(-1)

    # Gather motion data
    win_base_pos = base_pos.index_select(0, flat_idx).reshape(N, W, 3)
    win_base_quat = base_quat.index_select(0, flat_idx).reshape(N, W, 4)
    win_base_lin_vel = base_lin_vel.index_select(0, flat_idx).reshape(N, W, 3)
    win_base_ang_vel = base_ang_vel.index_select(0, flat_idx).reshape(N, W, 3)
    win_ee_pos = ee_pos.index_select(0, flat_idx).reshape(N, W, E, 3)
    win_joint = joint_pos.index_select(0, flat_idx).reshape(N, W, J)
    win_joint_vel = joint_vel.index_select(0, flat_idx).reshape(N, W, J)

    # H0 is known at inference and uniquely anchors the generated world trajectory.
    h0_idx = history_idx[:, 0]
    anchor_pos_h0 = base_pos.index_select(0, h0_idx)
    anchor_quat_h0 = base_quat.index_select(0, h0_idx)
    yaw_h0 = yaw_quat(anchor_quat_h0)                # (N, 4)
    heading_inv_h0_wf = quat_conjugate(yaw_h0)[:, None, :].expand(N, W, 4).reshape(-1, 4)
    yaw_h0_w = yaw_h0[:, None, :].expand(N, W, 4).reshape(-1, 4)

    # Exact history root poses are not model conditions. They are saved only
    # for visualizing the history and interpolating to generated keyframes.
    history_base_pos = base_pos.index_select(0, history_idx.reshape(-1)).reshape(N, H, 3)
    history_base_quat = base_quat.index_select(0, history_idx.reshape(-1)).reshape(N, H, 4)
    history_root_offset = history_base_pos - anchor_pos_h0[:, None, :]
    yaw_h0_h = yaw_h0[:, None, :].expand(N, H, 4).reshape(-1, 4)
    history_root_pos_local = quat_apply_inverse(
        yaw_h0_h, history_root_offset.reshape(-1, 3)
    ).reshape(N, H, 3)
    history_heading_inv = quat_conjugate(yaw_h0_h)
    history_root_rot_local_6d = _tan_norm_from_quat(
        quat_mul(history_heading_inv, history_base_quat.reshape(-1, 4))
    ).reshape(N, H, 6)
    # H0 is stored as an absolute waist pose (x/y are deliberately zero).
    # Remaining history frames retain H0-relative position and heading-frame
    # orientation.
    history_root_pos_local[:, 0] = torch.stack(
        (
            torch.zeros(N, device=base_pos.device, dtype=base_pos.dtype),
            torch.zeros(N, device=base_pos.device, dtype=base_pos.dtype),
            anchor_pos_h0[:, 2],
        ),
        dim=-1,
    )
    history_root_rot_6d = history_root_rot_local_6d.clone()
    history_root_rot_6d[:, 0] = _tan_norm_from_quat(anchor_quat_h0)
    history_root = torch.cat((history_root_pos_local, history_root_rot_6d), dim=-1)

    # root_pos: full xyz offset from H0, expressed in H0's heading frame.
    root_offset = win_base_pos - anchor_pos_h0[:, None, :]  # (N, W, 3)
    root_pos_local = quat_apply_inverse(yaw_h0_w, root_offset.reshape(-1, 3)).reshape(N, W, 3)

    # root_rot: heading_inv(H0) ⊗ root_quat[t] → 6D tan-norm
    root_rot_local_quat = quat_mul(
        heading_inv_h0_wf, win_base_quat.reshape(-1, 4)
    ).reshape(N, W, 4)
    root_rot_6d = _tan_norm_from_quat(root_rot_local_quat)

    # EE: per-frame root offset, expressed in H0's heading frame.
    ee_offset_w = win_ee_pos - win_base_pos[:, :, None, :]  # (N, W, E, 3)
    yaw_h0_e = yaw_h0[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
    ee_pos_local = quat_apply_inverse(yaw_h0_e, ee_offset_w.reshape(-1, 3)).reshape(N, W, E * 3)

    # Root velocities: world vectors expressed in H0's heading frame.
    lin_vel_local = quat_apply_inverse(yaw_h0_w, win_base_lin_vel.reshape(-1, 3)).reshape(N, W, 3)
    ang_vel_local = quat_apply_inverse(yaw_h0_w, win_base_ang_vel.reshape(-1, 3)).reshape(N, W, 3)

    hist_flat = history_idx.reshape(-1)
    hist_joint = joint_pos.index_select(0, hist_flat).reshape(N, H, J)

    # Desired joystick-style command.  For supervised data it is estimated
    # from the last history frame to the final future keyframe, expressed in
    # the last history frame's yaw frame, and repeated for every history token.
    last_history_idx = history_idx[:, -1]
    last_history_pos = base_pos.index_select(0, last_history_idx)
    last_history_yaw = yaw_quat(base_quat.index_select(0, last_history_idx))
    command_duration = float(future_size * future_frame_stride) / float(fps)
    command_disp_b = quat_apply_inverse(
        last_history_yaw, win_base_pos[:, -1] - last_history_pos
    )
    relative_target_yaw = quat_mul(
        quat_conjugate(last_history_yaw), yaw_quat(win_base_quat[:, -1])
    )
    w, x, y, z = relative_target_yaw.unbind(-1)
    command_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    velocity_command = torch.cat(
        [command_disp_b[:, :2] / command_duration, command_yaw[:, None] / command_duration], dim=-1
    )
    velocity_command = velocity_command[:, None, :].expand(N, H, 3)
    proprio = torch.cat([hist_joint, velocity_command], dim=-1)

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
    return motion, history_terrain, proprio, history_root, hist_joint


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
    print(
        f"Window: history={cfg.history_size} future_keyframes={cfg.future_size} "
        f"future_frame_stride={cfg.future_frame_stride} window_stride={cfg.stride}"
    )
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
        total_size = cfg.history_size + cfg.future_size * cfg.future_frame_stride
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

        # collect_motion.py stores robot.data.joint_{pos,vel} directly.  Keep
        # that IsaacLab articulation order throughout preprocessing/training.
        joint_pos = torch.from_numpy(
            data["joint_pos"][:T].astype(np.float32)
        ).to(device)
        joint_vel = torch.from_numpy(
            data["joint_vel"][:T].astype(np.float32)
        ).to(device)

        elevation_xyz = data["elevation_map_xyz"][:T].astype(np.float32)
        height_map_frame = str(
            np.asarray(data.get("height_map_frame", "yaw_aligned_sensor")).item()
        )
        canonical_height_map = _canonicalize_elevation_map(
            elevation_xyz,
            base_pos.detach().cpu().numpy(),
            base_quat.detach().cpu().numpy(),
            height_map_frame,
        )
        height_map = torch.from_numpy(canonical_height_map).to(device)

        motion, terrain, proprio, history_root, history_joint = _compute_windows(
            base_pos, base_quat, base_lin_vel, base_ang_vel,
            ee_pos, joint_pos, joint_vel, height_map,
            cfg.history_size, cfg.future_size, cfg.future_frame_stride,
            cfg.stride, cfg.fps,
        )

        if (
            motion is None
            or terrain is None
            or proprio is None
            or history_root is None
            or history_joint is None
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
            history_root=history_root.cpu().numpy().astype(np.float32),
            history_joint=history_joint.cpu().numpy().astype(np.float32),
            history_root_layout=np.array(
                "h0_abs_waist_pose_history_h0_heading_relative"
            ),
            history_joint_layout=np.array("isaaclab_articulation"),
            fps=np.array([cfg.fps], dtype=np.float32),
            history_size=np.array([cfg.history_size], dtype=np.int32),
            future_size=np.array([cfg.future_size], dtype=np.int32),
            future_frame_stride=np.array([cfg.future_frame_stride], dtype=np.int32),
            stride=np.array([cfg.stride], dtype=np.int32),
            feature_dims=np.array(FEATURE_DIMS, dtype=np.int32),
            proprio_dims=np.array(PROPRIO_DIMS, dtype=np.int32),
            diffusion_format_version=np.array(7, dtype=np.int32),
            root_body=np.array(ROOT_BODY_NAME),
            terrain_layout=np.array(TERRAIN_LAYOUT),
            terrain_grid_layout=np.array(TERRAIN_GRID_LAYOUT),
            motion_layout=np.array(MOTION_LAYOUT),
            proprio_layout=np.array(PROPRIO_LAYOUT),
            joint_layout=np.array(JOINT_LAYOUT),
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
