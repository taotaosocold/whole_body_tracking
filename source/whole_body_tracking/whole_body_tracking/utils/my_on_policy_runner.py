import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class _PolicyCompat:
    """Adapter to make rsl_rl 5.x MLPModel look like old ActorCritic for ONNX export."""

    is_recurrent = False

    def __init__(self, actor_model):
        self.actor = actor_model.mlp


def _get_policy_and_normalizer(runner):
    """Return (policy_compat, normalizer) that work with both old and new rsl_rl."""
    if hasattr(runner.alg, "policy"):
        return runner.alg.policy, getattr(runner, "obs_normalizer", None)
    actor = runner.alg.actor
    return _PolicyCompat(actor), actor.obs_normalizer


def _is_wandb_logger(runner):
    logger_type = getattr(runner, "logger_type", None) or getattr(runner.logger, "logger_type", None)
    return logger_type == "wandb"


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        policy, normalizer = _get_policy_and_normalizer(self)
        export_policy_as_onnx(policy, normalizer=normalizer, path=policy_path, filename=filename)
        if _is_wandb_logger(self):
            import wandb

            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
        else:
            attach_onnx_metadata(self.env.unwrapped, "local", path=policy_path, filename=filename)


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        policy, normalizer = _get_policy_and_normalizer(self)
        export_motion_policy_as_onnx(
            self.env.unwrapped, policy, normalizer=normalizer, path=policy_path, filename=filename
        )
        if _is_wandb_logger(self):
            import wandb

            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
        else:
            attach_onnx_metadata(self.env.unwrapped, "local", path=policy_path, filename=filename)
