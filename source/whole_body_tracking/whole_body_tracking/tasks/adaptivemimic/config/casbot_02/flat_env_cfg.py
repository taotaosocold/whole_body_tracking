from dataclasses import fields

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.casbot_02 import CASBOT_02_25DOF_ACTION_SCALE, CASBOT_02_25DOF_CYLINDER_CFG, CASBOT_02_25DOF_CYLINDER_WITH_HANDS_CFG, CASBOT_02_25DOF_CYLINDER_DIRECT_CFG, CASBOT_02_25DOF_DIRECT_ACTION_SCALE, CASBOT_02_25DOF_CYLINDER_G1_CFG, CASBOT_02_25DOF_G1_ACTION_SCALE 
from whole_body_tracking.tasks.adaptivemimic.config.casbot_02.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.adaptivemimic.tracking_env_cfg import TrackingEnvCfg, VELOCITY_RANGE
from whole_body_tracking.tasks.adaptivemimic.mdp.multimotion_commands import MultiMotionCommandCfg
from whole_body_tracking.robots.actuator import DelayedImplicitActuatorCfg
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

@configclass
class CASBOTFlatWoStateEstimationEnvCfg(CASBOTFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class CASBOTFlatWoStateEstimationAdaptivePPOEnvCfg(CASBOTFlatWoStateEstimationEnvCfg):
    """Unchanged wo-state-estimation environment registered with AdaptivePPO."""

    pass

@configclass
class CASBOTFlatWoStateEstimationAggressiveDomainEnvCfg(CASBOTFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Randomize motor communication latency independently for every environment
        # at reset. With sim.dt=5 ms, 0--2 physics steps correspond to 0--10 ms.
        for actuator_name, actuator_cfg in self.scene.robot.actuators.items():
            actuator_params = {
                field.name: getattr(actuator_cfg, field.name)
                for field in fields(actuator_cfg)
                if field.name != "class_type"
            }
            self.scene.robot.actuators[actuator_name] = DelayedImplicitActuatorCfg(
                **actuator_params,
                min_delay=0,
                max_delay=2,
            )

        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (-4.0, 4.0),
                "operation": "add",
            },
        )

        self.events.scale_link_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["^(?!base_link$).*"]),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.base_com.params["com_range"] = {
            "x": (-0.06, 0.06),
            "y": (-0.06, 0.06),
            "z": (-0.06, 0.06),
        }
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.035, 0.035)
        self.events.joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "friction_distribution_params": (0.01, 1.15),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "armature_distribution_params": (0.008, 0.06),
                "operation": "abs",
                "distribution": "uniform",
            },
        )

@configclass
class CASBOTFlatResidualNoDisturbanceEnvCfg(CASBOTFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.MotionResidualActionCfg(asset_name="robot", joint_names=[".*"])
        self.actions.joint_pos.scale = CASBOT_02_25DOF_ACTION_SCALE
        self.observations.policy = self.observations.critic
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None


@configclass
class CASBOTFlatNewRewardEnvCfg(CASBOTFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -2e-1


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
class CASBOTMultiResidualNoDisturbanceEnvCfg(CASBOTFlatEnvCfg):
    """Multi-motion residual control without domain randomization or observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.MotionResidualActionCfg(asset_name="robot", joint_names=[".*"])
        self.actions.joint_pos.scale = CASBOT_02_25DOF_ACTION_SCALE
        self.observations.policy = self.observations.critic
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None
        self.commands.motion = MultiMotionCommandCfg(
            anchor_body_name="waist_yaw_link",
            body_names=self.commands.motion.body_names,
            **_MULTI_MOTION_COMMAND_KWARGS,
        )


@configclass
class CASBOTMultiEnvCfg(CASBOTFlatWoStateEstimationEnvCfg):
    """Multi-motion with domain randomization and observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = MultiMotionCommandCfg(
            anchor_body_name="waist_yaw_link",
            body_names=self.commands.motion.body_names,
            **_MULTI_MOTION_COMMAND_KWARGS,
        )


@configclass
class CASBOTFlatWoStateEstimationStubbornEnvCfg(CASBOTFlatWoStateEstimationEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_yaw_aligned_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.motion_global_anchor_orientation_yaw_aligned_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        )
        self.terminations.anchor_ori = DoneTerm(
            func=mdp.bad_anchor_ori_probabilistic,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "motion",
                "threshold": 1.5708,  # π/2 rad
                "prob": 0.005,
            },
        )
        self.terminations.anchor_pos = DoneTerm(
            func=mdp.bad_anchor_pos_z_only_probabilistic,
            params={"command_name": "motion", "threshold": 0.25, "prob": 0.005},
        )
