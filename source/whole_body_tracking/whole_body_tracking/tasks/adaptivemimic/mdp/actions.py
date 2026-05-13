from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


class MotionResidualAction(JointPositionAction):
    def process_actions(self, actions: torch.Tensor):
        command: MotionCommand = self._env.command_manager.get_term(self.cfg.command_name)
        self._offset = command.joint_pos.clone()
        super().process_actions(actions)


@configclass
class MotionResidualActionCfg(JointPositionActionCfg):
    class_type = MotionResidualAction
    command_name: str = "motion"