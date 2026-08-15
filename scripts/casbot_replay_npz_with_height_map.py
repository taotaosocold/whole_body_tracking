"""Replay a CASBOT NPZ kinematically and append a noise-free terrain height map.

Unlike ``collect_motion.py``, this script does not load or run an RL policy.  It
writes every source motion frame directly to the articulation, updates the
RayCaster, and saves the original NPZ fields unchanged together with the same
height-map metadata used by ``collect_motion.py``.

Example:

.. code-block:: bash

    python scripts/casbot_replay_npz_with_height_map.py \
        --motion_file /home/casbot/Desktop/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/parkour/config/casbot_02/motion_walk/walking.npz \
        --terrain_file /home/casbot/Desktop/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/parkour/config/casbot_02/STL_motion/flat_placeholder.stl \
        --output /home/casbot/Desktop/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/parkour/config/casbot_02/motion_walk/walking_with_height_map.npz \
        --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Replay a CASBOT motion without a policy and collect height maps."
)
parser.add_argument("--motion_file", type=str, required=True, help="Input CASBOT motion NPZ.")
parser.add_argument("--terrain_file", type=str, required=True, help="STL bound to the motion.")
parser.add_argument("--output", type=str, required=True, help="Output NPZ with elevation_map_xyz.")
parser.add_argument(
    "--max_frames",
    type=int,
    default=None,
    help="Optional number of frames to replay; defaults to the complete motion.",
)
parser.add_argument(
    "--height_map_frame",
    choices=("world", "yaw_aligned_sensor"),
    default="world",
    help="Coordinate frame used to save elevation_map_xyz (default: world).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, yaw_quat

from whole_body_tracking.robots.casbot_02 import CASBOT_02_25DOF_CYLINDER_CFG
from whole_body_tracking.tasks.locomotion.velocity.mdp import elevation_map_xyz
from whole_body_tracking.tasks.parkour.terrains import StaticObstacleTerrainCfg


MOTION_PATH = Path(args_cli.motion_file).expanduser().resolve()
TERRAIN_PATH = Path(args_cli.terrain_file).expanduser().resolve()
OUTPUT_PATH = Path(args_cli.output).expanduser().resolve()


@configclass
class CasbotReplayHeightMapSceneCfg(InteractiveSceneCfg):
    """One CASBOT, one authored-coordinate STL terrain and one 33x21 scanner."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(9.0, 12.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            curriculum=False,
            use_cache=False,
            sub_terrains={
                "selected_motion": StaticObstacleTerrainCfg(
                    proportion=1.0,
                    obstacle_file=str(TERRAIN_PATH),
                )
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = CASBOT_02_25DOF_CYLINDER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/waist_yaw_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def _validate_inputs(source: dict[str, np.ndarray]) -> int:
    if not MOTION_PATH.is_file():
        raise FileNotFoundError(f"Motion file not found: {MOTION_PATH}")
    if not TERRAIN_PATH.is_file():
        raise FileNotFoundError(f"Terrain STL not found: {TERRAIN_PATH}")
    required = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"Motion NPZ is missing fields: {missing}")
    frame_count = int(source["joint_pos"].shape[0])
    for key in required[1:]:
        if source[key].shape[0] != frame_count:
            raise ValueError(
                f"{key} contains {source[key].shape[0]} frames, expected {frame_count}"
            )
    if source["joint_pos"].shape[1] != 25 or source["joint_vel"].shape[1] != 25:
        raise ValueError("This script expects a CASBOT 25-DOF motion")
    return frame_count


def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
    source: dict[str, np.ndarray],
    frame_count: int,
) -> np.ndarray:
    """Write exact NPZ states and collect one noise-free scan per frame."""

    robot: Articulation = scene["robot"]
    sensor = scene["height_scanner"]
    sensor_cfg = SceneEntityCfg("height_scanner")
    observation_env = SimpleNamespace(scene=scene)
    device = sim.device
    motion_fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    motion_dt = 1.0 / motion_fps
    scans: list[np.ndarray] = []

    terrain_origin = scene.env_origins[0]
    if not torch.allclose(terrain_origin[:2], torch.zeros_like(terrain_origin[:2]), atol=1.0e-6):
        raise RuntimeError(
            "Expected the single terrain at world XY=(0,0), got "
            f"{terrain_origin.tolist()}"
        )

    print(
        f"[INFO] Replaying {frame_count} frames at {motion_fps:.3f} fps; "
        f"height rays={sensor.data.ray_hits_w.shape[1]}"
    )
    for frame in range(frame_count):
        root_state = np.concatenate(
            (
                source["body_pos_w"][frame, 0],
                source["body_quat_w"][frame, 0],
                source["body_lin_vel_w"][frame, 0],
                source["body_ang_vel_w"][frame, 0],
            )
        ).astype(np.float32, copy=False)
        robot.write_root_state_to_sim(torch.from_numpy(root_state).to(device).unsqueeze(0))
        robot.write_joint_state_to_sim(
            torch.from_numpy(source["joint_pos"][frame]).to(device).unsqueeze(0),
            torch.from_numpy(source["joint_vel"][frame]).to(device).unsqueeze(0),
        )
        scene.write_data_to_sim()

        # Kinematic update only: do not integrate gravity/contact or run a policy.
        sim.forward()
        scene.update(motion_dt)

        height_xyz = elevation_map_xyz(observation_env, sensor_cfg, noise=False)[0].reshape(-1, 3)
        if args_cli.height_map_frame == "world":
            sensor_yaw = yaw_quat(sensor.data.quat_w[0:1]).expand(height_xyz.shape[0], -1)
            height_xyz = quat_apply(sensor_yaw, height_xyz)
            height_xyz += sensor.data.pos_w[0]
        scans.append(height_xyz.cpu().numpy().astype(np.float32, copy=False))

        if not args_cli.headless and frame % 2 == 0:
            sim.render()
            lookat = source["body_pos_w"][frame, 0]
            sim.set_camera_view(lookat + np.array([2.0, 2.0, 0.5]), lookat)
        if frame % 100 == 0 or frame == frame_count - 1:
            print(f"[INFO] Frame {frame + 1}/{frame_count}")

    return np.stack(scans, axis=0)


def main() -> None:
    if not MOTION_PATH.is_file():
        raise FileNotFoundError(f"Motion file not found: {MOTION_PATH}")
    if not TERRAIN_PATH.is_file():
        raise FileNotFoundError(f"Terrain STL not found: {TERRAIN_PATH}")

    with np.load(MOTION_PATH, allow_pickle=False) as data:
        source = {key: data[key].copy() for key in data.files}
    total_frames = _validate_inputs(source)
    frame_count = total_frames
    if args_cli.max_frames is not None:
        if args_cli.max_frames <= 0:
            raise ValueError("--max_frames must be positive")
        frame_count = min(frame_count, args_cli.max_frames)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / float(source["fps"][0]))
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(CasbotReplayHeightMapSceneCfg(num_envs=1, env_spacing=1.0))
    sim.reset()

    elevation_map_xyz_data = run_simulator(sim, scene, source, frame_count)
    output = {
        key: (value[:frame_count] if value.ndim > 0 and value.shape[0] == total_frames else value)
        for key, value in source.items()
    }
    sensor = scene["height_scanner"]
    output.update(
        elevation_map_xyz=elevation_map_xyz_data,
        height_map_grid_size=np.asarray(sensor.cfg.pattern_cfg.size, dtype=np.float32),
        height_map_resolution=np.asarray([sensor.cfg.pattern_cfg.resolution], dtype=np.float32),
        height_map_frame=np.asarray(args_cli.height_map_frame),
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH, **output)

    print(f"[INFO] Saved: {OUTPUT_PATH}")
    print(f"  original motion fields preserved: {len(source)}")
    print(f"  elevation_map_xyz: {elevation_map_xyz_data.shape}")
    print(f"  height_map_grid_size: {output['height_map_grid_size'].tolist()}")
    print(f"  height_map_frame: {args_cli.height_map_frame}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
