from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from whole_body_tracking.tasks.locomotion.velocity.terrain_encoder import RslRlTerrainEncoderModelCfg
from whole_body_tracking.tasks.parkour.cnn_encoder import RslRlParkourCnnModelCfg


@configclass
class CASBOTParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "CASBOT_parkour"
    actor = RslRlTerrainEncoderModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )
    critic = RslRlTerrainEncoderModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=0.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        self.algorithm.class_name = "PPO"
        self.algorithm.share_cnn_encoders = False


@configclass
class CASBOTParkourNoDRPPORunnerCfg(CASBOTParkourPPORunnerCfg):
    experiment_name = "CASBOT_parkour_no_dr"


@configclass
class CASBOTParkourCnnPPORunnerCfg(CASBOTParkourPPORunnerCfg):
    """Parkour PPO using independent CNN+GAP encoders for actor and critic."""

    experiment_name = "CASBOT_parkour_cnn"
    actor = RslRlParkourCnnModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )
    critic = RslRlParkourCnnModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=0.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.share_cnn_encoders = False


@configclass
class CASBOTDiffusionParkourPPORunnerCfg(CASBOTParkourCnnPPORunnerCfg):
    """CNN policy for H0-anchored online diffusion references."""

    experiment_name = "CASBOT_parkour_diffusion"
