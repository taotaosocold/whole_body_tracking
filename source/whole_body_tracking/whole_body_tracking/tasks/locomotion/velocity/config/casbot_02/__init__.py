import gymnasium as gym

from . import agents, flat_env_cfg

gym.register(
    id="Locomotion-Flat-CASBOT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTLocomotionFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTLocomotionFlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Flat-CASBOT-NoDR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CASBOTLocomotionFlatNoDRCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTLocomotionFlatPPORunnerCfg",
    },
)
