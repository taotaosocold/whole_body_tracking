from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from whole_body_tracking.tasks.locomotion.velocity.terrain_encoder import RslRlTerrainEncoderModelCfg


@configclass
class G1LocomotionFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 50
    experiment_name = "g1_locomotion_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1LocomotionTerrainPPORunnerCfg(G1LocomotionFlatPPORunnerCfg):
    """AME-style CNN/cross-attention policy for terrain locomotion."""

    max_iterations = 10000
    experiment_name = "g1_locomotion_terrain"
    actor = RslRlTerrainEncoderModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )
    critic = RslRlTerrainEncoderModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=False,
        init_noise_std=0.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )

    def __post_init__(self):
        self.algorithm.share_cnn_encoders = False
