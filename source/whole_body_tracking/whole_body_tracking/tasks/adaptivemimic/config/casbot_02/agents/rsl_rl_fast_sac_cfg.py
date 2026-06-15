from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlFastSacAlgorithmCfg, RslRlFastSacRunnerCfg


@configclass
class CASBOTFastSacRunnerCfg(RslRlFastSacRunnerCfg):
    num_steps_per_env = 1
    max_iterations = 100000
    save_interval = 500
    experiment_name = "CASBOT_flat_fast_sac"
    empirical_normalization = False  # FastSAC uses its own normalizer

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    algorithm = RslRlFastSacAlgorithmCfg(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        buffer_size=512,
        num_steps=1,
        gamma=0.97,
        tau=0.125,
        batch_size=4096,
        learning_starts=10,
        policy_frequency=4,
        num_updates=2,
        target_entropy_ratio=0.0,
        num_atoms=101,
        v_min=-50.0,
        v_max=50.0,
        actor_hidden_dim=512,
        critic_hidden_dim=768,
        num_q_networks=2,
        use_layer_norm=True,
        use_tanh=True,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_autotune=True,
        alpha_init=0.001,
        max_grad_norm=0.0,
        weight_decay=0.001,
        obs_normalization=True,
        compile=False,
        amp=True,
        save_interval=500,
        logging_interval=100,
    )
