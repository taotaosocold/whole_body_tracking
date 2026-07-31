import numpy as np
import os
import time
import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners.distillation_runner import DistillationRunner
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan

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
    if hasattr(runner.alg, "actor"):
        actor = runner.alg.actor
        return _PolicyCompat(actor), actor.obs_normalizer
    # Distillation algorithm: student model is the policy
    if hasattr(runner.alg, "student"):
        student = runner.alg.student
        return _PolicyCompat(student), student.obs_normalizer
    raise RuntimeError(f"Unknown algorithm type: {type(runner.alg)}")


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


def _best_metrics_from_logger(logger):
    """Return (mean_reward, mean_ep_length) from the logger's recent episode buffers."""
    if not logger.rewbuffer or not logger.lenbuffer:
        return -float("inf"), -float("inf")
    return float(np.mean(logger.rewbuffer)), float(np.mean(logger.lenbuffer))


def _try_save_best_models(runner, it, log_dir, best_state):
    """Check current metrics and overwrite best-{reward,ep_length,combined}.pt if improved."""
    r, l = _best_metrics_from_logger(runner.logger)
    if r == -float("inf"):
        return

    c = r + l / 30.0
    entries = [
        ("best_reward.pt", r, "reward"),
        ("best_ep_length.pt", l, "ep_length"),
        ("best_combined.pt", c, "combined"),
    ]

    for filename, score, key in entries:
        if score > best_state[key]:
            best_state[key] = score
            OnPolicyRunner.save(runner, os.path.join(log_dir, filename))
            print(f"[Best] New best {key}: {score:.3f} at iteration {it}")


def _learn_with_best_tracking(runner, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    """Patched learn() that tracks best{reward,ep_length,combined} models in addition to periodic saves."""
    # -- copied from OnPolicyRunner.learn() with best-model tracking added --
    if init_at_random_ep_len:
        runner.env.episode_length_buf = torch.randint_like(
            runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
        )

    obs = runner.env.get_observations().to(runner.device)
    runner.alg.train_mode()

    if runner.is_distributed:
        print(f"Synchronizing parameters for rank {runner.gpu_global_rank}...")
        runner.alg.broadcast_parameters()

    runner.logger.init_logging_writer()

    log_dir = runner.logger.log_dir
    best_state = {"reward": -float("inf"), "ep_length": -float("inf"), "combined": -float("inf")}

    start_it = runner.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
        start = time.time()
        with torch.inference_mode():
            for _ in range(runner.cfg["num_steps_per_env"]):
                actions = runner.alg.act(obs)
                # Optional hook used by AdaptivePPO. The existing adaptive-sampling bin is captured
                # before env.step(), so it matches the observation/action stored
                # by alg.act() rather than the next reference frame.
                if (
                    hasattr(runner.alg, "record_motion_bin")
                    and "motion" in runner.env.unwrapped.command_manager.active_terms
                ):
                    motion_command = runner.env.unwrapped.command_manager.get_term("motion")
                    runner.alg.record_motion_command(motion_command)
                obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
                if runner.cfg.get("check_for_nan", True):
                    check_nan(obs, rewards, dones)
                obs, rewards, dones = (
                    obs.to(runner.device),
                    rewards.to(runner.device),
                    dones.to(runner.device),
                )
                runner.alg.process_env_step(obs, rewards, dones, extras)
                intrinsic_rewards = runner.alg.intrinsic_rewards if runner.cfg["algorithm"]["rnd_cfg"] else None
                runner.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

            stop = time.time()
            collect_time = stop - start
            start = stop

            runner.alg.compute_returns(obs)

        loss_dict = runner.alg.update()

        stop = time.time()
        learn_time = stop - start
        runner.current_learning_iteration = it

        runner.logger.log(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=runner.alg.learning_rate,
            action_std=runner.alg.get_policy().output_std,
            rnd_weight=runner.alg.rnd.weight if runner.cfg["algorithm"]["rnd_cfg"] else None,
        )

        # periodic save
        if runner.logger.writer is not None and it % runner.cfg["save_interval"] == 0:
            runner.save(os.path.join(log_dir, f"model_{it}.pt"))

        # --- best-model tracking (the only added lines) ---
        if runner.logger.writer is not None:
            _try_save_best_models(runner, it, log_dir, best_state)

    # final save
    if runner.logger.writer is not None:
        runner.save(os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt"))
        runner.logger.stop_logging_writer()


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def _is_motion_task(self) -> bool:
        """Check if the environment has a motion command (vs. locomotion velocity commands)."""
        return "motion" in self.env.unwrapped.command_manager.active_terms

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if not self._is_motion_task():
            return
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

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        _learn_with_best_tracking(self, num_learning_iterations, init_at_random_ep_len)


class MotionDistillationRunner(DistillationRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def _is_motion_task(self) -> bool:
        """Check if the environment has a motion command (vs. locomotion velocity commands)."""
        return "motion" in self.env.unwrapped.command_manager.active_terms

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if not self._is_motion_task():
            return
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

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # DistillationRunner's validation
        if self.alg.teacher is None:
            raise RuntimeError(
                "No teacher model loaded for distillation. "
                "Use --load_run or specify a checkpoint to load the teacher."
            )
        _learn_with_best_tracking(self, num_learning_iterations, init_at_random_ep_len)
