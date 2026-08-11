"""CASBOT configuration for motion-matched static-terrain tracking."""

import copy
from pathlib import Path

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.casbot_02 import CASBOT_02_25DOF_ACTION_SCALE, CASBOT_02_25DOF_CYLINDER_CFG
from whole_body_tracking.tasks.parkour.mdp import ParkourMotionCommandCfg
from whole_body_tracking.tasks.parkour.parkour_env_cfg import ParkourEnvCfg, VELOCITY_RANGE
from whole_body_tracking.tasks.parkour.terrains import StaticObstacleTerrainCfg


CONFIG_DIR = Path(__file__).resolve().parent

MOTION_TERRAIN_PAIRS = [
    (
        "come_up_50cm_box_R_001__A300_with_height_map.npz",
        "box_come_up_50cm.stl",
    ),
    (
        "come_up_50cm_box_R_001__A300_M_with_height_map.npz",
        "box_come_up_50cm_M.stl",
    ),
    (
        "obstacles1_subject1_clip1_with_height_map.npz",
        "stairs_obstacles1_subject1_clip1.stl",
    ),
    (
        "obstacles1_subject1_clip1_M_with_height_map.npz",
        "stairs_obstacles1_subject1_clip1_M.stl",
    ),
    (
        "obstacles1_subject3_clip1_with_height_map.npz",
        "stairs_obstacles1_subject3_clip1.stl",
    ),
    (
        "obstacles1_subject3_clip1_M_with_height_map.npz",
        "stairs_obstacles1_subject3_clip1_M.stl",
    ),
    (
        "36_07_poses_with_height_map.npz",
        "stair_bridge_36_07.stl",
    ),
    (
        "36_07_poses_M_with_height_map.npz",
        "stair_bridge_36_07_M.stl",
    ),
    (
        "106_07_poses_with_height_map.npz",
        "stairs_106_07.stl",
    ),
    (
        "106_07_poses_M_with_height_map.npz",
        "stairs_106_07_M.stl",
    ),
    (
        "114_09_poses_clip1_with_height_map.npz",
        "stair_bridge_114_09_clip1.stl",
    ),
    (
        "114_09_poses_clip1_M_with_height_map.npz",
        "stair_bridge_114_09_clip1_M.stl",
    ),
    (
        "13_35_poses_with_height_map.npz",
        "stairs_13_35.stl",
    ),
    (
        "13_35_poses_M_with_height_map.npz",
        "stairs_13_35_M.stl",
    ),
    (
        "13_36_poses_with_height_map.npz",
        "stairs_13_36.stl",
    ),
    (
        "13_36_poses_M_with_height_map.npz",
        "stairs_13_36_M.stl",
    ),
    (
        "141_07_poses_with_height_map.npz",
        "stairs_141_07.stl",
    ),
    (
        "141_07_poses_M_with_height_map.npz",
        "stairs_141_07_M.stl",
    ),
    (
        "141_08_poses_with_height_map.npz",
        "stairs_141_08.stl",
    ),
    (
        "141_08_poses_M_with_height_map.npz",
        "stairs_141_08_M.stl",
    ),
    (
        "143_17_poses_with_height_map.npz",
        "stairs_143_17.stl",
    ),
    (
        "143_17_poses_M_with_height_map.npz",
        "stairs_143_17_M.stl",
    ),
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
                obstacle_file=str(CONFIG_DIR / "STL" / terrain_name),
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
            motion_folder=str(CONFIG_DIR / "moton_with_height_map"),
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
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.randomize_ray_offsets = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_rigid_body_mass = None


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
