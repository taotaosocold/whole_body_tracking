"""Casbot-02 distillation BC environment configs — absolute and residual action modes.

Usage
-----
.. code-block:: bash

    python scripts/rsl_rl/train.py \\
        --task Distillation-BC-Flat-CASBOT-Abs-v0 \\
        --motion_file /path/to/motion.npz \\
        --num_envs 4096 \\
        --load_run /path/to/teacher_checkpoint_dir
"""

from isaaclab.utils import configclass

from whole_body_tracking.robots.casbot_02 import (
    CASBOT_02_25DOF_ACTION_SCALE,
    CASBOT_02_25DOF_CYLINDER_CFG,
)
from whole_body_tracking.tasks.distillation_bc.distillation_env_cfg import DistillationEnvCfg
import whole_body_tracking.tasks.distillation_bc.mdp as mdp


class CASBOTDistillationBCBaseEnvCfg(DistillationEnvCfg):
    """Shared Casbot-02 setup — not registered directly."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CASBOT_02_25DOF_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
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

        # Casbot uses "base_link" as root
        self.events.base_com.params["asset_cfg"].body_names = "base_link"


@configclass
class CASBOTDistillationBCFlatAbsEnvCfg(CASBOTDistillationBCBaseEnvCfg):
    """Absolute action mode — student outputs target joint positions directly."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=CASBOT_02_25DOF_ACTION_SCALE, use_default_offset=True
        )


@configclass
class CASBOTDistillationBCFlatResidualEnvCfg(CASBOTDistillationBCBaseEnvCfg):
    """Residual → Absolute mode — teacher was residual-trained, student learns absolute actions.

    The :class:`DistillationResidualToAbsolute` algorithm wraps the teacher to
    convert its residual output to absolute-equivalent before computing the BC loss.
    The student outputs absolute joint positions, same as Abs mode.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=CASBOT_02_25DOF_ACTION_SCALE, use_default_offset=True
        )


@configclass
class CASBOTDistillationBCFlatAbsWoStateEstimationEnvCfg(CASBOTDistillationBCFlatAbsEnvCfg):
    """Abs mode without anchor/base_vel observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class CASBOTDistillationBCFlatResidualWoStateEstimationEnvCfg(CASBOTDistillationBCFlatResidualEnvCfg):
    """Residual mode without anchor/base_vel observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
