from __future__ import annotations

import math
import os
import time
import torch
import tqdm
from contextlib import contextmanager
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict

from .networks import Actor, Critic
from .normalizer import EmpiricalNormalization
from .replay_buffer import SimpleReplayBuffer
from .fast_sac_utils import cpu_state, save_params




class FastSacRunner:
    """Standalone FastSAC training/inference runner with replay buffer and distributional critic.

    Presents the same interface as OnPolicyRunner for compatibility with train.py/play.py:
    - __init__(env, train_cfg, log_dir, device)
    - learn(num_learning_iterations, init_at_random_ep_len=False)
    - save(path) / load(path)
    - get_inference_policy(device)
    - add_git_repo_to_log(path)
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str | None = None):
        self.env = env
        self.train_cfg = train_cfg
        self.log_dir = log_dir
        self.device = device
        self.registry_name = registry_name

        self.num_envs = env.num_envs
        self.n_act = env.num_actions

        algo = train_cfg["algorithm"]

        # resolve observation groups
        obs_groups = train_cfg.get("obs_groups", {"actor": ["policy"], "critic": ["critic"]})
        self.actor_obs_keys = obs_groups.get("actor", ["policy"])
        self.critic_obs_keys = obs_groups.get("critic", ["critic"])

        # determine observation dimensions from env
        sample_obs = env.get_observations()
        self.actor_obs_dim = sum(int(sample_obs[k].shape[-1]) for k in self.actor_obs_keys)
        self.critic_obs_dim = sum(int(sample_obs[k].shape[-1]) for k in self.critic_obs_keys)

        # action scaling (identity by default — env handles scaling)
        action_scale = torch.ones(self.n_act, device=device)
        action_bias = torch.zeros(self.n_act, device=device)

        # networks
        self.actor = Actor(
            n_obs=self.actor_obs_dim,
            n_act=self.n_act,
            hidden_dim=algo.get("actor_hidden_dim", 512),
            log_std_max=algo.get("log_std_max", 0.0),
            log_std_min=algo.get("log_std_min", -5.0),
            use_tanh=algo.get("use_tanh", True),
            use_layer_norm=algo.get("use_layer_norm", True),
            device=device,
            action_scale=action_scale,
            action_bias=action_bias,
        )
        self.qnet = Critic(
            n_obs=self.critic_obs_dim,
            n_act=self.n_act,
            num_atoms=algo.get("num_atoms", 101),
            v_min=algo.get("v_min", -20.0),
            v_max=algo.get("v_max", 20.0),
            hidden_dim=algo.get("critic_hidden_dim", 768),
            use_layer_norm=algo.get("use_layer_norm", True),
            num_q_networks=algo.get("num_q_networks", 2),
            device=device,
        )
        self.qnet_target = Critic(
            n_obs=self.critic_obs_dim,
            n_act=self.n_act,
            num_atoms=algo.get("num_atoms", 101),
            v_min=algo.get("v_min", -20.0),
            v_max=algo.get("v_max", 20.0),
            hidden_dim=algo.get("critic_hidden_dim", 768),
            use_layer_norm=algo.get("use_layer_norm", True),
            num_q_networks=algo.get("num_q_networks", 2),
            device=device,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.obs_normalization = algo.get("obs_normalization", True)
        if self.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=self.actor_obs_dim, device=device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=self.critic_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        self.log_alpha = nn.Parameter(torch.tensor([math.log(algo.get("alpha_init", 0.001))], device=device))
        self.target_entropy = -self.n_act * algo.get("target_entropy_ratio", 0.0)

        # optimizers
        self.actor_optimizer = torch.optim.AdamW(
            list(self.actor.parameters()),
            lr=algo.get("actor_learning_rate", 3e-4),
            weight_decay=algo.get("weight_decay", 0.001),
            fused=True if torch.cuda.is_available() else False,
            betas=(0.9, 0.95),
        )
        self.q_optimizer = torch.optim.AdamW(
            list(self.qnet.parameters()),
            lr=algo.get("critic_learning_rate", 3e-4),
            weight_decay=algo.get("weight_decay", 0.001),
            fused=True if torch.cuda.is_available() else False,
            betas=(0.9, 0.95),
        )
        self.alpha_optimizer = torch.optim.AdamW(
            [self.log_alpha],
            lr=algo.get("alpha_learning_rate", 3e-4),
            fused=True if torch.cuda.is_available() else False,
            betas=(0.9, 0.95),
        )

        self.amp_enabled = algo.get("amp", False)
        self.amp_dtype = algo.get("amp_dtype", "bf16")
        self.scaler = GradScaler(enabled=self.amp_enabled)

        # replay buffer
        self.rb = SimpleReplayBuffer(
            n_env=self.num_envs,
            buffer_size=algo.get("buffer_size", 1024),
            n_obs=self.actor_obs_dim,
            n_act=self.n_act,
            n_critic_obs=self.critic_obs_dim,
            n_steps=algo.get("num_steps", 1),
            gamma=algo.get("gamma", 0.97),
            device=device,
        )

        self.global_step = 0
        self.writer = None
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

        # episode tracking (accumulated across completed episodes)
        self._ep_rew_buf = []
        self._ep_len_buf = []
        self._ep_sum_rewards = torch.zeros(self.num_envs, device=device)
        self._ep_length = torch.zeros(self.num_envs, device=device)

        # per-term reward tracking
        self._ep_term_rewards: dict[str, list[float]] = {}
        self._ep_term_metrics: dict[str, list[float]] = {}
        self._ep_term_terminations: dict[str, list[float]] = {}
        self._reward_term_names: list[str] = []
        self._metric_term_names: list[str] = []
        self._termination_term_names: list[str] = []

    def _get_actor_obs(self, obs_td: TensorDict) -> torch.Tensor:
        return torch.cat([obs_td[k] for k in self.actor_obs_keys], dim=-1)

    def _get_critic_obs(self, obs_td: TensorDict) -> torch.Tensor:
        return torch.cat([obs_td[k] for k in self.critic_obs_keys], dim=-1)

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.amp_enabled):
            yield

    # ── update functions ──────────────────────────────────────────────────

    def _update_main(self, data: dict) -> dict:
        algo = self.train_cfg["algorithm"]
        scaler = self.scaler

        with self._maybe_amp():
            next_actor_obs = data["next"]["observations"]
            critic_obs = data["critic_observations"]
            next_critic_obs = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            bootstrap = (~dones).float()

            with torch.no_grad():
                next_actions, next_log_probs = self.actor.get_actions_and_log_probs(next_actor_obs)
                discount = algo["gamma"] ** data["next"]["effective_n_steps"]
                target_distributions = self.qnet_target.projection(
                    next_critic_obs,
                    next_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = self.qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = self.qnet(critic_obs, actions)
            critic_log_probs = torch.nn.functional.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        self.q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        scaler.unscale_(self.q_optimizer)
        max_grad_norm = algo.get("max_grad_norm", 0.0)
        if max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=max_grad_norm)
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(self.q_optimizer)
        scaler.update()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if algo.get("use_autotune", True):
            self.alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_log_probs.detach() + self.target_entropy)).mean()
            scaler.scale(alpha_loss).backward()
            scaler.unscale_(self.alpha_optimizer)
            scaler.step(self.alpha_optimizer)
            scaler.update()

        return {
            "qf_loss": qf_loss.detach(),
            "qf_max": target_value_max.detach(),
            "qf_min": target_value_min.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha_value": self.log_alpha.exp().detach(),
        }

    def _update_pol(self, data: dict) -> dict:
        algo = self.train_cfg["algorithm"]
        scaler = self.scaler

        with self._maybe_amp():
            critic_obs = data["critic_observations"]
            actions, log_probs = self.actor.get_actions_and_log_probs(data["observations"])
            with torch.no_grad():
                _, _, log_std = self.actor(data["observations"])
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()

            q_outputs = self.qnet(critic_obs, actions)
            q_probs = torch.nn.functional.softmax(q_outputs, dim=-1)
            q_values = self.qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        scaler.unscale_(self.actor_optimizer)
        max_grad_norm = algo.get("max_grad_norm", 0.0)
        if max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=max_grad_norm)
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(self.actor_optimizer)
        scaler.update()

        return {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "policy_entropy": policy_entropy.detach(),
            "action_std": action_std.detach(),
        }

    # ── training loop ─────────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        algo = self.train_cfg["algorithm"]
        device = self.device
        env = self.env
        rb = self.rb

        num_steps_per_env = self.train_cfg.get("num_steps_per_env", 1)

        # compile functions if requested
        normalize_fn = self.obs_normalizer.forward
        normalize_critic_fn = self.critic_obs_normalizer.forward
        policy = self.actor.explore
        update_main = self._update_main
        update_pol = self._update_pol

        if algo.get("compile", False):
            normalize_fn = torch.compile(normalize_fn)
            normalize_critic_fn = torch.compile(normalize_critic_fn)
            policy = torch.compile(policy)
            update_main = torch.compile(update_main)
            update_pol = torch.compile(update_pol)

        # get initial observations
        obs_td = env.get_observations()
        actor_obs = self._get_actor_obs(obs_td)
        critic_obs = self._get_critic_obs(obs_td)

        pbar = tqdm.tqdm(total=num_learning_iterations, initial=self.global_step)

        policy_frequency = algo.get("policy_frequency", 4)
        num_updates_per_step = algo.get("num_updates", 8)
        learning_starts = algo.get("learning_starts", 10)
        tau = algo.get("tau", 0.125)

        # accumulated metrics for smoother logging
        acc_metrics = {}

        while self.global_step <= num_learning_iterations:
            # ── collect ──────────────────────────────────────────────
            collect_start = time.time()
            with torch.no_grad(), self._maybe_amp():
                for _ in range(num_steps_per_env):
                    norm_actor_obs = normalize_fn(actor_obs, update=True)
                    actions = policy(norm_actor_obs)
                    next_obs_td, rewards, dones, extras = env.step(actions.float())
                    truncations = extras.get("time_outs", torch.zeros_like(dones))

                    next_actor_obs = self._get_actor_obs(next_obs_td)
                    next_critic_obs = self._get_critic_obs(next_obs_td)

                    transition = TensorDict(
                        {
                            "observations": actor_obs,
                            "actions": actions,
                            "next": {
                                "observations": next_actor_obs,
                                "rewards": rewards,
                                "truncations": truncations.long(),
                                "dones": dones.long(),
                            },
                        },
                        batch_size=(self.num_envs,),
                        device=device,
                    )
                    transition["critic_observations"] = critic_obs
                    transition["next"]["critic_observations"] = next_critic_obs

                    rb.extend(transition)
                    actor_obs = next_actor_obs
                    critic_obs = next_critic_obs

                    # track episode stats
                    self._ep_sum_rewards += rewards
                    self._ep_length += 1
                    reset_env_ids = dones.nonzero(as_tuple=False).flatten()
                    n_resets = len(reset_env_ids)
                    if n_resets > 0:
                        self._ep_rew_buf.extend(self._ep_sum_rewards[reset_env_ids].cpu().tolist())
                        self._ep_len_buf.extend(self._ep_length[reset_env_ids].cpu().tolist())
                        self._ep_length[reset_env_ids] = 0

                        # track per-term episodic rewards/metrics/terminations
                        # IsaacLab stores episodic info in extras["log"] with scalar values
                        # (means across all envs that reset in this step)
                        ep_info = extras.get("log", {})
                        for key, val in ep_info.items():
                            if isinstance(val, torch.Tensor):
                                if val.numel() == 1:
                                    val = val.item()
                                else:
                                    val = val.mean().item()
                            if key.startswith("Episode_Reward"):
                                if key not in self._ep_term_rewards:
                                    self._ep_term_rewards[key] = []
                                self._ep_term_rewards[key].append(val)
                                if key not in self._reward_term_names:
                                    self._reward_term_names.append(key)
                            elif key.startswith("Episode_Termination"):
                                if key not in self._ep_term_terminations:
                                    self._ep_term_terminations[key] = []
                                self._ep_term_terminations[key].append(val)
                                if key not in self._termination_term_names:
                                    self._termination_term_names.append(key)

                        self._ep_sum_rewards[reset_env_ids] = 0

            collect_time = time.time() - collect_start

            # ── learn ────────────────────────────────────────────────
            learn_start = time.time()
            if self.global_step > learning_starts:
                batch_size = max(algo["batch_size"] // self.num_envs, 1)

                # sample one large batch, normalize once, then split into updates
                samples_per_update = batch_size * self.num_envs
                large_data = rb.sample(batch_size * num_updates_per_step)
                large_data["observations"] = normalize_fn(large_data["observations"], update=False)
                large_data["next"]["observations"] = normalize_fn(large_data["next"]["observations"], update=False)
                large_data["critic_observations"] = normalize_critic_fn(large_data["critic_observations"], update=False)
                large_data["next"]["critic_observations"] = normalize_critic_fn(large_data["next"]["critic_observations"], update=False)

                for update_i in range(num_updates_per_step):
                    start_idx = update_i * samples_per_update
                    end_idx = (update_i + 1) * samples_per_update
                    data = {
                        "observations": large_data["observations"][start_idx:end_idx],
                        "actions": large_data["actions"][start_idx:end_idx],
                        "next": {
                            "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                            "dones": large_data["next"]["dones"][start_idx:end_idx],
                            "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                            "observations": large_data["next"]["observations"][start_idx:end_idx],
                            "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                        },
                        "critic_observations": large_data["critic_observations"][start_idx:end_idx],
                    }
                    data["next"]["critic_observations"] = large_data["next"]["critic_observations"][start_idx:end_idx]

                    q_result = update_main(data)

                    if num_updates_per_step > 1:
                        if update_i % policy_frequency == 1:
                            pi_result = update_pol(data)
                        else:
                            pi_result = {}
                    elif self.global_step % policy_frequency == 0:
                        pi_result = update_pol(data)
                    else:
                        pi_result = {}

                    # soft-update target
                    with torch.no_grad():
                        src_ps = [p.data for p in self.qnet.parameters()]
                        tgt_ps = [p.data for p in self.qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - tau)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

                    # accumulate metrics
                    for k, v in {**q_result, **pi_result}.items():
                        if k not in acc_metrics:
                            acc_metrics[k] = 0.0
                        acc_metrics[k] += v.item() if isinstance(v, torch.Tensor) else v

                # average accumulated metrics
                for k in acc_metrics:
                    acc_metrics[k] /= num_updates_per_step

            learn_time = time.time() - learn_start

            # ── logging ──────────────────────────────────────────────
            logging_interval = algo.get("logging_interval", 100)
            if self.global_step > 0 and self.global_step % logging_interval == 0 and self.writer is not None:
                it = self.global_step
                writer = self.writer

                # compute episode stats
                n_eps = len(self._ep_rew_buf)
                if n_eps > 0:
                    mean_rew = sum(self._ep_rew_buf) / n_eps
                    mean_len = sum(self._ep_len_buf) / n_eps
                    writer.add_scalar("Episode/mean_reward", mean_rew, it)
                    writer.add_scalar("Episode/mean_length", mean_len, it)
                else:
                    mean_rew = 0.0
                    mean_len = 0.0

                for k, v in acc_metrics.items():
                    writer.add_scalar(f"Loss/{k}", v, it)
                writer.add_scalar("Time/collect", collect_time, it)
                writer.add_scalar("Time/learn", learn_time, it)

                # ── header ──────────────────────────────────────────
                total_steps = it * self.num_envs * self.train_cfg.get("num_steps_per_env", 1)
                print(f"{'#' * 80}")
                print(f"  Learning iteration {it}/{num_learning_iterations}  ".center(80))
                print(f"{'#' * 80}")
                print(f"\n{'Total steps:':>28s} {total_steps}")
                print(f"{'Collection time:':>28s} {collect_time:.3f}s")
                print(f"{'Learning time:':>28s} {learn_time:.3f}s")

                # loss values
                for key in ("qf_loss", "actor_loss", "alpha_value", "critic_grad_norm",
                           "actor_grad_norm", "policy_entropy", "action_std"):
                    if key in acc_metrics:
                        label = key.replace("qf_loss", "Mean value loss") \
                                     .replace("actor_loss", "Mean surrogate loss") \
                                     .replace("alpha_value", "Mean alpha") \
                                     .replace("critic_grad_norm", "Critic grad norm") \
                                     .replace("actor_grad_norm", "Actor grad norm") \
                                     .replace("policy_entropy", "Mean entropy") \
                                     .replace("action_std", "Mean action std")
                        print(f"{label:>28s}: {acc_metrics[key]:.4f}")
                print(f"{'Mean reward:':>28s} {mean_rew:.2f}")
                print(f"{'Mean episode length:':>28s} {mean_len:.2f}")

                # per-term reward breakdowns
                if self._ep_term_rewards:
                    for name in sorted(self._ep_term_rewards):
                        vals = self._ep_term_rewards[name]
                        avg = sum(vals) / max(len(vals), 1)
                        print(f"{name:>28s}: {avg:.4f}")
                if self._ep_term_terminations:
                    for name in sorted(self._ep_term_terminations):
                        vals = self._ep_term_terminations[name]
                        avg = sum(vals) / max(len(vals), 1)
                        print(f"{name:>28s}: {avg:.4f}")

                print("-" * 80)
                print(f"{'Iteration time:':>28s} {collect_time + learn_time:.2f}s")
                print(f"\n")

                # clear buffers for next logging interval
                self._ep_rew_buf.clear()
                self._ep_len_buf.clear()
                for d in (self._ep_term_rewards, self._ep_term_metrics, self._ep_term_terminations):
                    for v in d.values():
                        v.clear()
                acc_metrics.clear()

            # ── saving ───────────────────────────────────────────────
            save_interval = algo.get("save_interval", 1000)
            if save_interval > 0 and self.global_step > 0 and self.global_step % save_interval == 0:
                if self.log_dir is not None:
                    self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))

            if self.global_step >= num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)

        # final save
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
        pbar.close()

    # ── checkpoint I/O ────────────────────────────────────────────────────

    def save(self, path: str, infos=None):
        save_params(
            global_step=self.global_step,
            actor=self.actor,
            qnet=self.qnet,
            qnet_target=self.qnet_target,
            log_alpha=self.log_alpha,
            obs_normalizer=self.obs_normalizer,
            critic_obs_normalizer=self.critic_obs_normalizer,
            actor_optimizer=self.actor_optimizer,
            q_optimizer=self.q_optimizer,
            alpha_optimizer=self.alpha_optimizer,
            scaler=self.scaler,
            config=self.train_cfg,
            save_path=path,
        )

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.qnet.load_state_dict(ckpt["qnet_state_dict"])
        self.qnet_target.load_state_dict(ckpt["qnet_target_state_dict"])
        if self.obs_normalization:
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])
            self.critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer_state"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(ckpt["q_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer_state_dict"])
        if ckpt.get("grad_scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.train_cfg = ckpt.get("config", self.train_cfg)

    def get_inference_policy(self, device: str | None = None):
        device = device or self.device
        actor = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        actor.eval()
        obs_normalizer.eval()

        def policy_fn(obs: torch.Tensor) -> torch.Tensor:
            if self.obs_normalization:
                norm_obs = obs_normalizer(obs, update=False)
            else:
                norm_obs = obs
            return actor.explore(norm_obs, deterministic=True)

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
        """Return a wrapper suitable for ONNX export."""
        import copy
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(self, actor, obs_normalizer):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    norm_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    norm_obs = actor_obs
                return self.actor(norm_obs)[0]

        return ActorWrapper(actor, obs_normalizer if self.obs_normalization else None)

    def add_git_repo_to_log(self, file):
        pass
