import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

# --- Absolute action mode (teacher was absolute-trained → standard Distillation) ---

gym.register(
    id="Distillation-BC-Flat-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationBCFlatAbsEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DistillationPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-BC-Flat-G1-Abs-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationBCFlatAbsWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DistillationPPORunnerCfg",
    },
)

# --- Residual action mode (teacher was residual-trained → custom conversion) ---

gym.register(
    id="Distillation-BC-Flat-G1-Residual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationBCFlatResidualEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DistillationResidualToAbsPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-BC-Flat-G1-Residual-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationBCFlatResidualWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DistillationResidualToAbsPPORunnerCfg",
    },
)
