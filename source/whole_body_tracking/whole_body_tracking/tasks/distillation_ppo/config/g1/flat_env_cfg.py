"""G1 distillation PPO environment configs — absolute and residual action modes.

Usage
-----
.. code-block:: bash

    # Absolute action mode
    python scripts/rsl_rl/train.py \\
        --task Distillation-PPO-Flat-G1-Abs-v0 \\
        --motion_file /path/to/motion.npz \\
        --teacher_onnx /path/to/teacher.onnx \\
        --num_envs 4096

    # Residual action mode
    python scripts/rsl_rl/train.py \\
        --task Distillation-PPO-Flat-G1-Residual-v0 \\
        --motion_file /path/to/motion.npz \\
        --teacher_onnx /path/to/teacher.onnx \\
        --num_envs 4096

How it works
------------
Standard PPO training with all adaptivemimic tracking rewards PLUS a
Gaussian-kernel BC reward that pulls the student towards the teacher ONNX
output.  The teacher is a frozen ONNX model exported from a previously
trained adaptivemimic policy.
"""

from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.distillation_ppo.distillation_ppo_env_cfg import (
    DistillationPPOEnvCfg,
)
import whole_body_tracking.tasks.distillation_ppo.mdp as mdp


class _G1DistillationPPOBaseEnvCfg(DistillationPPOEnvCfg):
    """Shared G1 setup — not registered directly."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        # G1 uses "torso_link" as root
        self.events.base_com.params["asset_cfg"].body_names = "torso_link"


@configclass
class G1DistillationPPOFlatAbsEnvCfg(_G1DistillationPPOBaseEnvCfg):
    """Absolute action mode — student outputs target joint positions directly."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=G1_ACTION_SCALE, use_default_offset=True
        )


@configclass
class G1DistillationPPOFlatResidualEnvCfg(_G1DistillationPPOBaseEnvCfg):
    """Residual → Absolute mode — teacher was residual-trained, student learns absolute actions.

    The reward function converts teacher ONNX residual output to absolute
    equivalent before computing the Gaussian kernel.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=G1_ACTION_SCALE, use_default_offset=True
        )
        self.rewards.distillation_action_l2.params["teacher_mode"] = "residual_to_abs"


@configclass
class G1DistillationPPOFlatAbsWoStateEstimationEnvCfg(G1DistillationPPOFlatAbsEnvCfg):
    """Abs mode without anchor/base_vel observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1DistillationPPOFlatResidualWoStateEstimationEnvCfg(G1DistillationPPOFlatResidualEnvCfg):
    """Residual mode without anchor/base_vel observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
