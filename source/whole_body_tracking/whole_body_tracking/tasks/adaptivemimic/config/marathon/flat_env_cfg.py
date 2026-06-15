from isaaclab.utils import configclass

from whole_body_tracking.robots.marathon import MARATHON_001_ACTION_SCALE, MARATHON_001_CYLINDER_CFG, MARATHON_ACTION_SCALE, MARATHON_CYLINDER_CFG
from whole_body_tracking.tasks.adaptivemimic.config.marathon.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.adaptivemimic.tracking_env_cfg import TrackingEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import whole_body_tracking.tasks.adaptivemimic.mdp as mdp

@configclass
class MARATHONFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MARATHON_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = MARATHON_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_leg_pelvic_roll_link",
            "left_leg_knee_pitch_link",
            "left_leg_ankle_roll_link",
            "right_leg_pelvic_roll_link",
            "right_leg_knee_pitch_link",
            "right_leg_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_elbow_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_pitch_link",
            "right_wrist_yaw_link",
        ]
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )
        self.terminations.ee_body_global_pos_z = DoneTerm(
            func=mdp.bad_motion_body_global_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            },
        )
        self.terminations.ee_body_pos = None
        self.commands.motion.adaptive_kernel_size = 3
        self.commands.motion.adaptive_lambda = 0.8

@configclass
class MARATHONFlatWoStateEstimationEnvCfg(MARATHONFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

@configclass
class MARATHONFlatWoStateEstimationResidualEnvCfg(MARATHONFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.MotionResidualActionCfg(asset_name="robot", joint_names=[".*"])
        self.actions.joint_pos.scale = MARATHON_ACTION_SCALE


@configclass
class MARATHONFlatNoDisturbanceEnvCfg(MARATHONFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = self.observations.critic
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None

@configclass
class MARATHONFlatResidualNoDisturbanceEnvCfg(MARATHONFlatNoDisturbanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.MotionResidualActionCfg(asset_name="robot", joint_names=[".*"])
        self.actions.joint_pos.scale = MARATHON_ACTION_SCALE

@configclass
class MARATHONFlatLowFreqEnvCfg(MARATHONFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
