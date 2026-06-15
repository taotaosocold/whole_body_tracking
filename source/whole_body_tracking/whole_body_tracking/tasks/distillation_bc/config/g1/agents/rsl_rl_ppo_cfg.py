"""G1 distillation runner config — uses RSL-RL's native Distillation algorithm."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class G1DistillationPPORunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_distillation_bc"
    empirical_normalization = True  # will be ignored; use obs_normalization per model

    # --- observation routing ---
    # student sees proprioceptive "policy" obs | teacher sees privileged "critic" obs
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

    # --- teacher model (deterministic — loaded from checkpoint) ---
    # WARNING: hidden_dims / activation / obs_normalization MUST match the
    # original actor the teacher checkpoint was trained with, because the
    # state_dict is loaded into this exact architecture.
    teacher = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        # no distribution_cfg → deterministic teacher
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
class G1DistillationResidualToAbsPPORunnerCfg(G1DistillationPPORunnerCfg):
    """BC Residual → Absolute: teacher was residual-trained, student learns absolute actions.

    Uses :class:`DistillationResidualToAbsolute` to convert teacher residual
    output to equivalent absolute actions before the BC loss.
    """

    experiment_name = "g1_distillation_bc_residual_to_abs"

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="whole_body_tracking.tasks.distillation_bc.mdp.distillation_bc:DistillationResidualToAbsolute",
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )
