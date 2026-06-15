"""Distillation PPO env config — adaptivemimic rewards + BC distillation reward."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from whole_body_tracking.tasks.adaptivemimic.tracking_env_cfg import TrackingEnvCfg

import whole_body_tracking.tasks.distillation_ppo.mdp as mdp


@configclass
class DistillationPPOEnvCfg(TrackingEnvCfg):
    """PPO env with teacher BC distillation reward added on top of tracking rewards.

    The teacher ONNX is loaded once and its action output serves as a soft
    target via a Gaussian-kernel reward.  Standard PPO handles the RL side.
    """

    teacher_onnx_path: str = ""
    """Path to the teacher ONNX model exported from a trained adaptivemimic policy."""

    teacher_mode: str = "abs"
    """Teacher action mode: "abs" or "residual_to_abs"."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.distillation_action_l2 = RewTerm(
            func=mdp.distillation_action_l2,
            weight=0.5,
            params={
                "sigma": 0.1,
                "teacher_onnx_path": self.teacher_onnx_path,
                "teacher_mode": self.teacher_mode,
            },
        )
