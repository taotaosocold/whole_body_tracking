import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

# --- Absolute action mode ---

gym.register(
    id="Distillation-PPO-Flat-CASBOT-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTDistillationPPOFlatAbsEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTDistillationPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-PPO-Flat-CASBOT-Abs-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTDistillationPPOFlatAbsWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTDistillationPPORunnerCfg",
    },
)

# --- Residual action mode ---

gym.register(
    id="Distillation-PPO-Flat-CASBOT-Residual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTDistillationPPOFlatResidualEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTDistillationPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-PPO-Flat-CASBOT-Residual-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTDistillationPPOFlatResidualWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTDistillationPPORunnerCfg",
    },
)
