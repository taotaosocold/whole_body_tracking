import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

# --- Absolute action mode ---

gym.register(
    id="Distillation-PPO-Flat-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationPPOFlatAbsEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DistillationPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-PPO-Flat-G1-Abs-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationPPOFlatAbsWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DistillationPPORunnerCfg",
    },
)

# --- Residual action mode ---

gym.register(
    id="Distillation-PPO-Flat-G1-Residual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationPPOFlatResidualEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DistillationPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-PPO-Flat-G1-Residual-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationPPOFlatResidualWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DistillationPPORunnerCfg",
    },
)
