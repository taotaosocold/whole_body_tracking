"""Distillation PPO MDP — adaptivemimic components + teacher reward."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from whole_body_tracking.tasks.adaptivemimic.mdp import *  # noqa: F401, F403
from whole_body_tracking.tasks.distillation_ppo.mdp.distillation_rewards import (  # noqa: F401
    distillation_action_l2,
)
