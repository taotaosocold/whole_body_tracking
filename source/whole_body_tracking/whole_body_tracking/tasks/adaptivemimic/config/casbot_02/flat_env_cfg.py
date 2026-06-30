from isaaclab.utils import configclass

from whole_body_tracking.robots.casbot_02 import CASBOT_02_25DOF_ACTION_SCALE, CASBOT_02_25DOF_CYLINDER_CFG, CASBOT_02_25DOF_CYLINDER_WITH_HANDS_CFG
from whole_body_tracking.tasks.adaptivemimic.config.casbot_02.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.adaptivemimic.tracking_env_cfg import RGMTPolicyCfg, TrackingEnvCfg, VELOCITY_RANGE
from whole_body_tracking.tasks.adaptivemimic.mdp.multimotion_commands import MultiMotionCommandCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import whole_body_tracking.tasks.adaptivemimic.mdp as mdp

@configclass
class CASBOTFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CASBOT_02_25DOF_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = CASBOT_02_25DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.body_names = [
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


@configclass
class CASBOTFlatWoStateEstimationEnvCfg(CASBOTFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

@configclass
class CASBOTFlatWoStateEstimationAggressiveDomainEnvCfg(CASBOTFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (-0.3, 0.8),
                "operation": "add",
            },
    )

@configclass
class CASBOTFlatWoStateEstimationResidualEnvCfg(CASBOTFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.MotionResidualActionCfg(asset_name="robot", joint_names=[".*"])
        self.actions.joint_pos.scale = CASBOT_02_25DOF_ACTION_SCALE

@configclass
class CASBOTFlatNoDisturbanceEnvCfg(CASBOTFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = self.observations.critic
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None

@configclass
class CASBOTFlatResidualNoDisturbanceEnvCfg(CASBOTFlatNoDisturbanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.MotionResidualActionCfg(asset_name="robot", joint_names=[".*"])
        self.actions.joint_pos.scale = CASBOT_02_25DOF_ACTION_SCALE


@configclass
class CASBOTFlatResidualNoDisturbancePenRewardEnvCfg(CASBOTFlatResidualNoDisturbanceEnvCfg):
    """Residual, no disturbance, with foot-slip and joint-acceleration penalty rewards."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_slip = RewTerm(
            func=mdp.feet_slip_penalty,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_leg_ankle_roll_link", "right_leg_ankle_roll_link"],
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_leg_ankle_roll_link", "right_leg_ankle_roll_link"],
                ),
                "threshold": 1.0,
            },
        )


@configclass
class CASBOTFlatLowFreqEnvCfg(CASBOTFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


_MULTI_MOTION_COMMAND_KWARGS = dict(
    asset_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
    },
    velocity_range=VELOCITY_RANGE,
    joint_position_range=(-0.1, 0.1),
    motion_folder="",  # filled by CLI (--motion_folder)
)


@configclass
class CASBOTMultiNoDisturbanceEnvCfg(CASBOTFlatNoDisturbanceEnvCfg):
    """Multi-motion, no domain randomization, zero observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = MultiMotionCommandCfg(
            anchor_body_name="waist_yaw_link",
            body_names=self.commands.motion.body_names,
            **_MULTI_MOTION_COMMAND_KWARGS,
        )

@configclass
class CASBOTMultiResidualNoDisturbanceEnvCfg(CASBOTMultiNoDisturbanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.MotionResidualActionCfg(asset_name="robot", joint_names=[".*"])
        self.actions.joint_pos.scale = CASBOT_02_25DOF_ACTION_SCALE

@configclass
class CASBOTMultiEnvCfg(CASBOTFlatWoStateEstimationEnvCfg):
    """Multi-motion, with domain randomization and observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = MultiMotionCommandCfg(
            anchor_body_name="waist_yaw_link",
            body_names=self.commands.motion.body_names,
            **_MULTI_MOTION_COMMAND_KWARGS,
        )


@configclass
class CASBOTFlatRGMTNoDisturbanceEnvCfg(CASBOTFlatNoDisturbanceEnvCfg):
    """CASBOT no-disturbance env with RGMT attention policy observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = RGMTPolicyCfg()


@configclass
class CASBOTFlatRGMTWoStateEstimationEnvCfg(CASBOTFlatWoStateEstimationEnvCfg):
    """CASBOT wo-state-estimation env with RGMT attention policy observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = RGMTPolicyCfg()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
