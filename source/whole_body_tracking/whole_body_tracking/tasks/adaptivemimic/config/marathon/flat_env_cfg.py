from isaaclab.utils import configclass

from whole_body_tracking.robots.marathon import MARATHON_001_ACTION_SCALE, MARATHON_001_CYLINDER_CFG
from whole_body_tracking.tasks.adaptivemimic.config.marathon.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.adaptivemimic.tracking_env_cfg import TrackingEnvCfg


@configclass
class MARATHONFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MARATHON_001_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = MARATHON_001_ACTION_SCALE
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
            "right_shoulder_roll_link",
            "right_elbow_pitch_link",
        ]


@configclass
class MARATHONFlatWoStateEstimationEnvCfg(MARATHONFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class MARATHONFlatLowFreqEnvCfg(MARATHONFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
