import gymnasium as gym

from . import agents, flat_env_cfg

gym.register(
    id="Locomotion-Flat-MARATHON-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MARATHONLocomotionFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MARATHONLocomotionFlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Flat-MARATHON-NoDR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MARATHONLocomotionFlatNoDRCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MARATHONLocomotionFlatPPORunnerCfg",
    },
)
