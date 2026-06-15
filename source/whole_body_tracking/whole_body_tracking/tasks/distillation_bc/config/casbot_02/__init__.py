import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

# --- Absolute action mode (teacher was absolute-trained → standard Distillation) ---

gym.register(
    id="Distillation-BC-Flat-CASBOT-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTDistillationBCFlatAbsEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTDistillationPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-BC-Flat-CASBOT-Abs-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTDistillationBCFlatAbsWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTDistillationPPORunnerCfg",
    },
)

# --- Residual action mode (teacher was residual-trained → custom conversion) ---

gym.register(
    id="Distillation-BC-Flat-CASBOT-Residual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTDistillationBCFlatResidualEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTDistillationResidualToAbsPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-BC-Flat-CASBOT-Residual-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTDistillationBCFlatResidualWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTDistillationResidualToAbsPPORunnerCfg",
    },
)
