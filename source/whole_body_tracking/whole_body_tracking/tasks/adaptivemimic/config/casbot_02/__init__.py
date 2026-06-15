import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-CASBOT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-Wo-State-Estimation-Aggressive-Domain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatWoStateEstimationAggressiveDomainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-Wo-State-Estimation-Residual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatWoStateEstimationResidualEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-No-Disturbance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatNoDisturbanceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-Residual-No-Disturbance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatResidualNoDisturbanceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatLowFreqPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-Multi-No-Disturbance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTMultiNoDisturbanceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-Multi-Residual-No-Disturbance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTMultiResidualNoDisturbanceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-Multi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTMultiEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-FastSAC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_fast_sac_cfg:CASBOTFastSacRunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-RGMT-No-Disturbance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatRGMTNoDisturbanceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatRGMTPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CASBOT-RGMT-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTFlatRGMTWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTFlatRGMTPPORunnerCfg",
    },
)
