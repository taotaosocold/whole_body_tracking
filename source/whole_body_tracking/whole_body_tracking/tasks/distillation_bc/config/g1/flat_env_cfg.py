"""G1 distillation BC environment configs — absolute and residual action modes.

Usage
-----
.. code-block:: bash

    # Absolute action mode (teacher outputs target joint positions)
    python scripts/rsl_rl/train.py \\
        --task Distillation-BC-Flat-G1-Abs-v0 \\
        --motion_file /path/to/motion.npz \\
        --num_envs 4096 \\
        --load_run /path/to/teacher_checkpoint_dir

    # Residual action mode (teacher outputs residuals on top of motion reference)
    python scripts/rsl_rl/train.py \\
        --task Distillation-BC-Flat-G1-Residual-v0 \\
        --motion_file /path/to/motion.npz \\
        --num_envs 4096 \\
        --load_run /path/to/teacher_checkpoint_dir

How it works
------------
RSL-RL's :class:`~rsl_rl.algorithms.Distillation` runs pure behavior cloning
(no RL rewards). At each step:

1.  Student (policy obs) → actions → step the env.
2.  Teacher (privileged obs) → reference actions → stored alongside.
3.  Update: MSE(student_actions, teacher_actions).

Choosing the teacher
--------------------
Pass ``--load_run <path>`` at the command line. The runner will load the
latest checkpoint from that directory and use its actor weights as the
teacher.  The teacher MUST have been trained with the matching action mode
(absolute or residual).

Teacher architecture
--------------------
The teacher model config in the agent file MUST match the original actor
architecture (hidden_dims, activation, obs_normalization) exactly.
Check the original PPO config for these values.

Observation alignment
---------------------
- Student → ``observations.policy`` group (proprioceptive)
- Teacher → ``observations.policy``  group (same as student, matches actor checkpoint)
- ``obs_groups`` in the agent config routes them.
"""

from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.distillation_bc.distillation_env_cfg import DistillationEnvCfg
import whole_body_tracking.tasks.distillation_bc.mdp as mdp


class _G1DistillationBCBaseEnvCfg(DistillationEnvCfg):
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
class G1DistillationBCFlatAbsEnvCfg(_G1DistillationBCBaseEnvCfg):
    """Absolute action mode — student outputs target joint positions directly."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=G1_ACTION_SCALE, use_default_offset=True
        )


@configclass
class G1DistillationBCFlatResidualEnvCfg(_G1DistillationBCBaseEnvCfg):
    """Residual → Absolute mode — teacher was residual-trained, student learns absolute actions.

    The :class:`DistillationResidualToAbsolute` algorithm wraps the teacher to
    convert its residual output to absolute-equivalent before computing the BC loss.
    The student outputs absolute joint positions, same as Abs mode.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=G1_ACTION_SCALE, use_default_offset=True
        )


@configclass
class G1DistillationBCFlatAbsWoStateEstimationEnvCfg(G1DistillationBCFlatAbsEnvCfg):
    """Abs mode without anchor/base_vel observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1DistillationBCFlatResidualWoStateEstimationEnvCfg(G1DistillationBCFlatResidualEnvCfg):
    """Residual mode without anchor/base_vel observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
