"""Custom Distillation algorithm for residual-trained teacher → absolute-action student.

When the teacher was trained with residuals (MotionResidualActionCfg) but
you want the student to output absolute joint positions directly
(JointPositionActionCfg), this algorithm wraps the teacher to convert its
residual output to the equivalent absolute action before the BC loss.

Conversion:
    teacher_target = residual * scale + motion_ref
    student_target = absolute * scale + default_pos
    → absolute = residual + (motion_ref - default_pos) / scale

The first ``num_joints`` elements of the command observation are assumed
to be the motion reference joint positions (see MotionCommand.command).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv


class _ResidualTeacherToAbsolute(nn.Module):
    """Wrap a residual-trained teacher so its forward() returns absolute-equivalent actions."""

    def __init__(self, teacher, default_joint_pos, action_scale, num_joints):
        super().__init__()
        self.teacher = teacher
        self.register_buffer("default_joint_pos", default_joint_pos.clone())
        self.register_buffer("action_scale", action_scale.clone())
        self.num_joints = num_joints

    def forward(self, obs: TensorDict) -> torch.Tensor:
        residual = self.teacher(obs)
        # First num_joints of "policy" command → motion reference joint positions
        joint_pos_cmd = obs["policy"][..., : self.num_joints]
        return residual + (joint_pos_cmd - self.default_joint_pos) / self.action_scale

    # Delegate all other calls to the inner teacher
    def reset(self, *args, **kwargs):
        return self.teacher.reset(*args, **kwargs)

    def update_normalization(self, *args, **kwargs):
        return self.teacher.update_normalization(*args, **kwargs)

    def detach_hidden_state(self, *args, **kwargs):
        return self.teacher.detach_hidden_state(*args, **kwargs)

    def get_hidden_state(self):
        return self.teacher.get_hidden_state()

    def train(self, *args, **kwargs):
        return self.teacher.train(*args, **kwargs)

    def eval(self):
        return self.teacher.eval()

    def state_dict(self, *args, **kwargs):
        return self.teacher.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.teacher.load_state_dict(*args, **kwargs)

    @property
    def obs_normalizer(self):
        return self.teacher.obs_normalizer


class DistillationResidualToAbsolute(Distillation):
    """Distillation where teacher outputs residuals but student learns absolute actions.

    Set ``algorithm.class_name`` to the fully-qualified path::

        whole_body_tracking.tasks.distillation_bc.mdp.distillation_bc:DistillationResidualToAbsolute
    """

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> DistillationResidualToAbsolute:
        alg = Distillation.construct_algorithm(obs, env, cfg, device)

        isaac_env = env.unwrapped
        data = isaac_env.scene["robot"].data
        default_joint_pos = getattr(data, "default_joint_pos_nominal", data.default_joint_pos[0])
        _scale = isaac_env.action_manager.get_term("joint_pos")._scale
        if isinstance(_scale, torch.Tensor):
            _scale = _scale[0]
        action_scale = _scale.clone() if isinstance(_scale, torch.Tensor) else torch.tensor(_scale)
        num_joints = isaac_env.action_manager.get_term("joint_pos").action_dim

        alg.teacher = _ResidualTeacherToAbsolute(
            alg.teacher, default_joint_pos, action_scale, num_joints
        )
        return alg
