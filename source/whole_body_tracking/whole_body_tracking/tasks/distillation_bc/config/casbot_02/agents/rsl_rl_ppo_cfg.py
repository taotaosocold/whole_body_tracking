"""Casbot-02 distillation runner config — uses RSL-RL's native Distillation algorithm."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class CASBOTDistillationPPORunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "casbot_distillation_bc"
    empirical_normalization = True  # will be ignored; use obs_normalization per model

    # --- observation routing ---
    obs_groups = {
        "student": ["policy"],
        "teacher": ["policy"],
    }

    # --- student model (near-deterministic — minimal noise for BC) ---
    student = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="GaussianDistribution",
            init_std=1e-6,
        ),
    )

    # --- teacher model (deterministic) ---
    # WARNING: architecture MUST match the original training config.
    teacher = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    # --- distillation algorithm (standard, for Abs-mode teacher) ---
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class CASBOTDistillationResidualToAbsPPORunnerCfg(CASBOTDistillationPPORunnerCfg):
    """BC Residual → Absolute: teacher was residual-trained, student learns absolute actions."""

    experiment_name = "casbot_distillation_bc_residual_to_abs"

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="whole_body_tracking.tasks.distillation_bc.mdp.distillation_bc:DistillationResidualToAbsolute",
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )
