"""Distillation MDP — reuses adaptivemimic components for environment-side terms.

RSL-RL handles the BC loss (teacher→student) natively via
:class:`rsl_rl.algorithms.Distillation`. The env only needs to provide
observations, commands, terminations, and events.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from whole_body_tracking.tasks.adaptivemimic.mdp import *  # noqa: F401, F403
