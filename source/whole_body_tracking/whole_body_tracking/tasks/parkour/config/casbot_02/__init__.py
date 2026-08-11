import gymnasium as gym

from . import agents, parkour_env_cfg


gym.register(
    id="Tracking-Parkour-CASBOT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg.CASBOTParkourEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTParkourPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Parkour-NoDR-CASBOT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg.CASBOTParkourNoDREnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CASBOTParkourNoDRPPORunnerCfg"
        ),
    },
)
