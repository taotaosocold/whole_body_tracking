"""CASBOT configuration for motion-matched static-terrain tracking."""

import copy
from pathlib import Path

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from whole_body_tracking.robots.casbot_02 import CASBOT_02_25DOF_ACTION_SCALE, CASBOT_02_25DOF_CYLINDER_CFG
from whole_body_tracking.tasks.parkour.mdp import (
    DiffusionParkourMotionCommandCfg,
    ParkourMotionCommandCfg,
)
from whole_body_tracking.tasks.parkour.parkour_env_cfg import ParkourEnvCfg, VELOCITY_RANGE
from whole_body_tracking.tasks.parkour.terrains import StaticObstacleTerrainCfg
from whole_body_tracking.tasks.tracking import mdp as tracking_mdp


CONFIG_DIR = Path(__file__).resolve().parent

MOTION_TERRAIN_PAIRS = [
    ("106_07_poses_with_height_map.npz", "stairs_106_07.stl"),
    ("106_07_poses_M_with_height_map.npz", "stairs_106_07_M.stl"),
    ("141_07_poses_with_height_map.npz", "stairs_141_07.stl"),
    ("141_07_poses_M_with_height_map.npz", "stairs_141_07_M.stl"),
    ("141_08_poses_with_height_map.npz", "stairs_141_08.stl"),
    ("141_08_poses_M_with_height_map.npz", "stairs_141_08_M.stl"),
    ("36_07_poses_with_height_map.npz", "stair_bridge_36_07.stl"),
    ("36_07_poses_M_with_height_map.npz", "stair_bridge_36_07_M.stl"),
    ("13_36_poses_with_height_map.npz", "stairs_13_36.stl"),
    ("13_36_poses_M_with_height_map.npz", "stairs_13_36_M.stl"),
    ("obstacles1_subject1_clip1_with_height_map.npz", "stairs_obstacles1_subject1_clip1.stl"),
    ("obstacles1_subject1_clip1_M_with_height_map.npz", "stairs_obstacles1_subject1_clip1_M.stl"),
    ("stairs_climbing_jog_up_start_with_height_map.npz", "stairs_climbing_jog_up_start.stl"),
    ("stairs_climbing_jog_up_start_M_with_height_map.npz", "stairs_climbing_jog_up_start_M.stl"),
    ("stairs_climbing_up_start_with_height_map.npz", "stairs_climbing_up_start.stl"),
    ("stairs_climbing_up_start_M_with_height_map.npz", "stairs_climbing_up_start_M.stl"),
    ("stairs_up_slow1_with_height_map.npz", "stairs_up_slow1.stl"),
    ("stairs_up_slow1_M_with_height_map.npz", "stairs_up_slow1_M.stl"),
    ("stairs_up_slow2_with_height_map.npz", "stairs_up_slow2.stl"),
    ("stairs_up_slow2_M_with_height_map.npz", "stairs_up_slow2_M.stl"),
    ("stop_forward_walk_with_height_map.npz", "flat_placeholder.stl"),
    ("stop_forward_walk_M_with_height_map.npz", "flat_placeholder.stl"),
    ("Turn_Start_Walk_with_height_map.npz", "flat_placeholder.stl"),
    ("Turn_Start_Walk_M_with_height_map.npz", "flat_placeholder.stl"),
    ("arc_walk_left_loop_with_height_map.npz", "flat_placeholder.stl"),
    ("arc_walk_left_loop_M_with_height_map.npz", "flat_placeholder.stl"),
    ("ide_turn_000_with_height_map.npz", "flat_placeholder.stl"),
    ("ide_turn_000_M_with_height_map.npz", "flat_placeholder.stl"),
    ("ide_turn_045_with_height_map.npz", "flat_placeholder.stl"),
    ("ide_turn_045_M_with_height_map.npz", "flat_placeholder.stl"),
    ("ide_turn_090_with_height_map.npz", "flat_placeholder.stl"),
    ("ide_turn_090_M_with_height_map.npz", "flat_placeholder.stl"),
    ("ide_turn_135_with_height_map.npz", "flat_placeholder.stl"),
    ("ide_turn_135_M_with_height_map.npz", "flat_placeholder.stl"),
    ("turn_start_walk_with_height_map.npz", "flat_placeholder.stl"),
    ("turn_start_walk_M_with_height_map.npz", "flat_placeholder.stl"),
    ("turn_walk_270_with_height_map.npz", "flat_placeholder.stl"),
    ("turn_walk_270_M_with_height_map.npz", "flat_placeholder.stl"),
    ("turn_walk_360_with_height_map.npz", "flat_placeholder.stl"),
    ("turn_walk_360_M_with_height_map.npz", "flat_placeholder.stl"),
    ("walking_with_height_map.npz", "flat_placeholder.stl"),
    ("walking_M_with_height_map.npz", "flat_placeholder.stl"),
]


@configclass
class CASBOTParkourEnvCfg(ParkourEnvCfg):
    """CASBOT tracks multiple motions on their statically bound terrains."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CASBOT_02_25DOF_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = CASBOT_02_25DOF_ACTION_SCALE

        body_names = [
            "base_link",
            "left_leg_pelvic_roll_link",
            "left_leg_knee_pitch_link",
            "left_leg_ankle_roll_link",
            "right_leg_pelvic_roll_link",
            "right_leg_knee_pitch_link",
            "right_leg_ankle_roll_link",
            "waist_yaw_link",
            "left_shoulder_roll_link",
            "left_elbow_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_pitch_link",
            "right_wrist_yaw_link",
        ]

        terrain_cfgs = {}
        motion_terrain_columns = {}
        for column, (motion_name, terrain_name) in enumerate(MOTION_TERRAIN_PAIRS):
            terrain_key = f"motion_{column:03d}"
            terrain_cfgs[terrain_key] = StaticObstacleTerrainCfg(
                proportion=1.0 / len(MOTION_TERRAIN_PAIRS),
                obstacle_file=str(CONFIG_DIR / "STL_motion" / terrain_name),
            )
            motion_terrain_columns[Path(motion_name).stem] = column

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(9.0, 12.0),
            border_width=0.0,
            num_rows=7,
            num_cols=len(MOTION_TERRAIN_PAIRS),
            curriculum=True,
            use_cache=False,
            sub_terrains=terrain_cfgs,
        )
        self.commands.motion = ParkourMotionCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            motion_folder=str(CONFIG_DIR / "motion_with_height_map"),
            motion_files=[motion_name for motion_name, _ in MOTION_TERRAIN_PAIRS],
            motion_terrain_columns=motion_terrain_columns,
            anchor_body_name="waist_yaw_link",
            body_names=body_names,
            pose_range={
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            velocity_range={key: (0.0, 0.0) for key in VELOCITY_RANGE},
            joint_position_range=(-0.1, 0.1),
        )

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"
        self.events.base_com.params["asset_cfg"].body_names = "waist_yaw_link"
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            "waist_yaw_link",
            ".*ankle.*",
            ".*wrist.*",
        ]


@configclass
class CASBOTParkourNoDREnvCfg(CASBOTParkourEnvCfg):
    """CASBOT parkour without domain randomization or observation noise."""

    def __post_init__(self):
        super().__post_init__()

        # Disable all startup domain randomization.
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.randomize_ray_offsets = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_rigid_body_mass = None

        # Disable motion-reset pose, velocity and joint perturbations.
        self.commands.motion.pose_range = {
            key: (0.0, 0.0) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        }
        self.commands.motion.velocity_range = {
            key: (0.0, 0.0) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        }
        self.commands.motion.joint_position_range = (0.0, 0.0)

        # Use the noise-free privileged observation for both Actor and Critic.
        self.observations.policy = copy.deepcopy(self.observations.critic)


@configclass
class CASBOTDiffusionParkourEnvCfg(CASBOTParkourEnvCfg):
    """CASBOT parkour with frozen diffusion-generated online references."""

    def __post_init__(self):
        super().__post_init__()

        # Flow-generated episodes have no motion-file endpoint. Truncate each
        # environment after exactly 240 policy steps (240 * 0.02 s = 4.8 s).
        self.episode_length_s = 240 * self.sim.dt * self.decimation

        reset_pairs = [
            (motion_name, terrain_name, column)
            for column, (motion_name, terrain_name) in enumerate(MOTION_TERRAIN_PAIRS)
        ]
        terrain_cfgs = {
            f"motion_{column:03d}": StaticObstacleTerrainCfg(
                proportion=1.0 / len(reset_pairs),
                obstacle_file=str(CONFIG_DIR / "STL_motion" / terrain_name),
            )
            for motion_name, terrain_name, column in reset_pairs
        }
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(9.0, 12.0),
            border_width=0.0,
            num_rows=7,
            num_cols=len(reset_pairs),
            # Keep one motion/STL pair in each fixed terrain column.  With
            # curriculum disabled IsaacLab samples a random sub-terrain for
            # every tile, which breaks the terrain_ids -> column binding used
            # by DiffusionParkourMotionCommand.
            curriculum=True,
            use_cache=False,
            sub_terrains=terrain_cfgs,
        )

        body_names = self.commands.motion.body_names
        motion_terrain_columns = {
            Path(motion_name).stem: terrain_column
            for motion_name, _, terrain_column in reset_pairs
        }
        self.commands.motion = DiffusionParkourMotionCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            motion_folder=str(CONFIG_DIR / "motion_with_height_map"),
            motion_files=[motion_name for motion_name, _, _ in reset_pairs],
            motion_terrain_columns=motion_terrain_columns,
            anchor_body_name="waist_yaw_link",
            body_names=body_names,
            diffusion_checkpoint=(
                "/home/casbot/Desktop/whole_body_tracking/logs/flow_matching/"
                "casbot_flow_matching/20260824_145301/pretrained.pt"
            ),
            diffusion_root="/home/casbot/Desktop/whole_body_tracking/diffusion",
            robot_urdf=str(
                Path(
                    "/home/casbot/Desktop/whole_body_tracking/source/whole_body_tracking/"
                    "whole_body_tracking/assets/casbot_skeleton_description/urdf/"
                    "casbot_skeleton_25dof_rev_1_0.urdf"
                )
            ),
            reset_motion_groups=[
                [Path(motion_name).stem]
                for motion_name, _, _ in reset_pairs
            ],
            # The Flow-Matching checkpoint predicts 10-Hz keyframes. Interpolate
            # to the 50-Hz simulator and resample after 20 control frames (0.4 s).
            diffusion_fps=10.0,
            tracking_horizon_steps=20,
            # Initialize the physical robot from the terrain-matched real H3.
            # Flow Matching still generates F0 as the first tracking target,
            # but a possibly penetrating generated pose is never written into
            # the simulator during reset.
            initialize_from_generated_f0=False,
            # Sample a new independent velocity command every 2 s.
            command_resampling_steps=100,
            command_vx_range=(0.0, 2.0),
            command_vy_range=(-0.5, 0.5),
            command_wz_range=(-1.5, 1.5),
            timing_interval_steps=50,
        )
        # Diffusion Parkour relies on the generated trajectory's motion rather
        # than a direct joystick-velocity reward.  Strengthen both linear and
        # angular reference-velocity tracking only for this environment.
        self.rewards.motion_body_lin_vel.weight = 2.0
        self.rewards.motion_body_ang_vel.weight = 2.0
        # Generated root xyz is now uniquely lifted into world coordinates from
        # H0, so horizontal tracking failure is meaningful as well as z failure.
        self.terminations.anchor_xy = DoneTerm(
            func=tracking_mdp.bad_anchor_pos_xy_only,
            params={"command_name": "motion", "threshold": 0.5},
        )


@configclass
class CASBOTDiffusionParkourPlayEnvCfg(CASBOTDiffusionParkourEnvCfg):
    """Deterministic diffusion demo with one selectable obstacle and command."""

    # These are the intended play-time controls.  A relative STL name is
    # resolved under config/casbot_02/STL; an absolute path is also accepted.
    play_stl_file: str = "stair_bridge_36_07.stl"
    play_obstacle_distance: float = 1.0
    play_command_velocity: tuple[float, float, float] = (0.5, 0.0, 0.0)
    play_robot_position: tuple[float, float, float] = (-2.0, -2.0, 0.875)
    play_robot_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()

        stl_path = Path(self.play_stl_file).expanduser()
        if not stl_path.is_absolute():
            stl_path = CONFIG_DIR / "STL" / stl_path
        if self.play_obstacle_distance < 0.0:
            raise ValueError("play_obstacle_distance must be non-negative")
        if len(self.play_command_velocity) != 3:
            raise ValueError("play_command_velocity must be (vx, vy, wz)")
        if len(self.play_robot_position) != 3:
            raise ValueError("play_robot_position must be (x, y, z)")
        if len(self.play_robot_rpy) != 3:
            raise ValueError("play_robot_rpy must be (roll, pitch, yaw)")

        self.scene.num_envs = 1
        self.scene.env_spacing = 12.0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(9.0, 12.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            curriculum=True,
            use_cache=False,
            sub_terrains={
                "play_obstacle": StaticObstacleTerrainCfg(
                    proportion=1.0,
                    obstacle_file=str(stl_path),
                    place_in_front_distance=self.play_obstacle_distance,
                )
            },
        )

        # The loader still needs one metadata motion, but it is never used to
        # initialize the robot or diffusion history in this play branch.
        reset_motion = MOTION_TERRAIN_PAIRS[0][0]
        reset_motion_stem = Path(reset_motion).stem
        self.commands.motion.motion_files = [reset_motion]
        self.commands.motion.motion_terrain_columns = {reset_motion_stem: 0}
        self.commands.motion.reset_motion_groups = [[reset_motion_stem]]
        self.commands.motion.initialize_from_default_pose = True
        self.commands.motion.initialize_from_generated_f0 = False
        self.commands.motion.default_pose_position = self.play_robot_position
        self.commands.motion.default_pose_rpy = self.play_robot_rpy

        vx, vy, wz = (float(value) for value in self.play_command_velocity)
        self.commands.motion.command_vx_range = (vx, vx)
        self.commands.motion.command_vy_range = (vy, vy)
        self.commands.motion.command_wz_range = (wz, wz)
        self.commands.motion.command_resampling_steps = 10**9

        # Remove training-time randomization while retaining failure resets, so
        # every retry starts from the exact same nominal standing condition.
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.randomize_ray_offsets = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_rigid_body_mass = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.episode_length_s = 60.0
