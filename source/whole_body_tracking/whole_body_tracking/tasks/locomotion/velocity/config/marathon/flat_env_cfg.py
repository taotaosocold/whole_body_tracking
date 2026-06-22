from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.marathon import MARATHON_ACTION_SCALE, MARATHON_CYLINDER_CFG
from whole_body_tracking.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityFlatEnvCfg
import whole_body_tracking.tasks.locomotion.velocity.mdp as mdp


@configclass
class MARATHONLocomotionFlatEnvCfg(LocomotionVelocityFlatEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_ankle_roll_link"

    def __post_init__(self):
        super().__post_init__()

        # ── Scene ──
        self.scene.robot = MARATHON_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Actions ──
        self.actions.joint_pos.scale = MARATHON_ACTION_SCALE
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ── Observations ──
        # remove base_lin_vel from policy (for domain randomization robustness)
        self.observations.policy.base_lin_vel = None

        # ── Events ──
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ── Rewards ──
        # General
        self.rewards.is_terminated.weight = -200.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -0.2

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [
            ".*_leg_pelvic_.*", ".*_leg_knee_.*", ".*_leg_ankle_.*"
        ]
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [".*_leg_pelvic_.*", ".*_leg_knee_.*"]
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_hip_l1", -0.1, [".*leg_pelvic_yaw.*", ".*leg_pelvic_roll.*"]
        )
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_arms_l1", -0.1, [".*shoulder.*", ".*elbow.*"]
        )
        self.rewards.joint_pos_limits.weight = -0.5

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.005

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Feet rewards
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 1.0

        # Undesired contacts
        self.rewards.undesired_contacts.weight = 0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*{self.foot_link_name}).*"
        ]

        # ── Terminations ──
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ── Commands ──
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Clean zero-weight rewards
        if self.__class__.__name__ == "MARATHONLocomotionFlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class MARATHONLocomotionFlatNoDRCfg(MARATHONLocomotionFlatEnvCfg):
    """MARATHON flat-terrain locomotion without domain randomization or observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.disable_domain_randomization()
        if self.__class__.__name__ == "MARATHONLocomotionFlatNoDRCfg":
            self.disable_zero_weight_rewards()
