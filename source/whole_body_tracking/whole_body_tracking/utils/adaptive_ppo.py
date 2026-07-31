"""Strictly on-policy PPO with hard-bin elite mini-batch sampling.

Only transitions from the current rollout are used.  The PPO objective is not
changed: the complete rollout is shuffled and partitioned exactly as in normal
PPO, then elite transitions are additionally appended to every mini-batch.  No
cross-iteration replay buffer is maintained.
"""

from __future__ import annotations

import math
from collections.abc import Generator

import torch
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


@configclass
class AdaptivePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for :class:`AdaptivePPO`."""

    class_name: str = "whole_body_tracking.utils.adaptive_ppo:AdaptivePPO"

    hard_bin_fraction: float = 0.10
    """Maximum fraction of forgotten bins reinforced in one PPO update."""

    elite_fraction: float = 0.10
    """Top positive-advantage fraction selected inside each hard bin."""

    elite_min_advantage: float = 0.0
    """Minimum unnormalized GAE advantage for an elite candidate."""

    elite_min_samples_per_bin: int = 32
    """Minimum current-rollout samples required before a hard bin contributes elites."""

    elite_min_positive_samples_per_bin: int = 5
    """Minimum positive-advantage candidates required in a selected hard bin."""

    elite_max_fraction_of_rollout: float = 0.03
    """Safety cap on the final elite set as a fraction of the current rollout."""

    elite_warmup_iterations: int = 100
    """Number of ordinary PPO updates used to establish per-bin performance history."""

    forgetting_trigger_enabled: bool = True
    """Only repeat elites from bins whose recent performance fell from its historical best."""

    forgetting_ema_alpha: float = 0.05
    """EMA coefficient used to track the current mean reward of each motion bin."""

    forgetting_relative_margin: float = 0.05
    """Relative drop from a bin's best EMA that is tolerated before elite repetition starts."""


class AdaptivePPO(PPO):
    """PPO whose current-rollout elite transitions occur in every mini-batch."""

    def __init__(
        self,
        *args,
        hard_bin_fraction: float = 0.10,
        elite_fraction: float = 0.10,
        elite_min_advantage: float = 0.0,
        elite_min_samples_per_bin: int = 32,
        elite_min_positive_samples_per_bin: int = 5,
        elite_max_fraction_of_rollout: float = 0.03,
        elite_warmup_iterations: int = 100,
        forgetting_trigger_enabled: bool = True,
        forgetting_ema_alpha: float = 0.05,
        forgetting_relative_margin: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        for name, value in (
            ("hard_bin_fraction", hard_bin_fraction),
            ("elite_fraction", elite_fraction),
            ("elite_max_fraction_of_rollout", elite_max_fraction_of_rollout),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if elite_min_samples_per_bin < 1:
            raise ValueError("elite_min_samples_per_bin must be positive")
        if elite_min_positive_samples_per_bin < 1:
            raise ValueError("elite_min_positive_samples_per_bin must be positive")
        if elite_warmup_iterations < 0:
            raise ValueError("elite_warmup_iterations cannot be negative")
        if not 0.0 < forgetting_ema_alpha <= 1.0:
            raise ValueError("forgetting_ema_alpha must be in (0, 1]")
        if forgetting_relative_margin < 0.0:
            raise ValueError("forgetting_relative_margin cannot be negative")
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("AdaptivePPO elite sampling currently supports feed-forward models only")

        self.hard_bin_fraction = hard_bin_fraction
        self.elite_fraction = elite_fraction
        self.elite_min_advantage = elite_min_advantage
        self.elite_min_samples_per_bin = elite_min_samples_per_bin
        self.elite_min_positive_samples_per_bin = elite_min_positive_samples_per_bin
        self.elite_max_fraction_of_rollout = elite_max_fraction_of_rollout
        self.elite_warmup_iterations = elite_warmup_iterations
        self.forgetting_trigger_enabled = forgetting_trigger_enabled
        self.forgetting_ema_alpha = forgetting_ema_alpha
        self.forgetting_relative_margin = forgetting_relative_margin

        self._rollout_bins = torch.empty(
            self.storage.num_transitions_per_env,
            self.storage.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._bin_step = 0
        self._bin_count: int | None = None
        self._difficulty_scores: torch.Tensor | None = None
        self._bin_performance_ema: torch.Tensor | None = None
        self._bin_best_performance: torch.Tensor | None = None
        self._bin_performance_initialized: torch.Tensor | None = None
        self._forgetting_scores: torch.Tensor | None = None
        self._elite_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self._hard_bin_count = 0
        self._forgotten_bin_count = 0
        self._mean_forgetting_score = 0.0
        self._max_forgetting_score = 0.0
        self._elite_mean_advantage = 0.0
        self._update_count = 0

    def record_motion_command(self, motion_command: object) -> None:
        """Read adaptive-bin data without requiring changes to command classes."""
        if all(
            hasattr(motion_command, name)
            for name in ("num_bins", "frame_to_bin", "current_frame", "adp_samp_failure_rate")
        ):
            # MultiMotionCommand already maintains a global frame-to-bin map.
            bin_count = int(motion_command.num_bins)  # type: ignore[attr-defined]
            bin_indices = motion_command.frame_to_bin[motion_command.current_frame]  # type: ignore[attr-defined]
            difficulty_scores = motion_command.adp_samp_failure_rate  # type: ignore[attr-defined]
        elif all(
            hasattr(motion_command, name)
            for name in ("time_steps", "bin_count", "motion", "bin_failed_count")
        ):
            # MotionCommand's adaptive reset sampler uses this exact formula.
            bin_count = int(motion_command.bin_count)  # type: ignore[attr-defined]
            time_step_total = max(int(motion_command.motion.time_step_total), 1)  # type: ignore[attr-defined]
            bin_indices = torch.clamp(
                (motion_command.time_steps * bin_count) // time_step_total,  # type: ignore[attr-defined]
                0,
                bin_count - 1,
            )
            difficulty_scores = motion_command.bin_failed_count  # type: ignore[attr-defined]
        else:
            raise TypeError(
                f"Unsupported motion command type {type(motion_command).__name__}: "
                "adaptive bin state is unavailable"
            )

        self.record_motion_bin(bin_indices, bin_count, difficulty_scores)

    def record_motion_bin(
        self,
        bin_indices: torch.Tensor,
        bin_count: int,
        difficulty_scores: torch.Tensor,
    ) -> None:
        """Record current rollout bin IDs and the command's existing failure scores."""
        if self._bin_step >= self.storage.num_transitions_per_env:
            raise RuntimeError("Received more motion bins than rollout transitions")
        bin_count = int(bin_count)
        if self._bin_count is None:
            self._bin_count = bin_count
            self._bin_performance_ema = torch.zeros(bin_count, device=self.device)
            self._bin_best_performance = torch.zeros(bin_count, device=self.device)
            self._bin_performance_initialized = torch.zeros(
                bin_count, dtype=torch.bool, device=self.device
            )
            self._forgetting_scores = torch.zeros(bin_count, device=self.device)
        elif self._bin_count != bin_count:
            raise RuntimeError(f"Motion bin_count changed from {self._bin_count} to {bin_count}")
        if difficulty_scores.numel() != bin_count:
            raise ValueError(
                f"Expected {bin_count} difficulty scores, received {difficulty_scores.numel()}"
            )

        self._rollout_bins[self._bin_step].copy_(bin_indices.to(self.device).long())
        self._difficulty_scores = difficulty_scores.detach().to(self.device).float().clone()
        self._bin_step += 1

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute normal PPO targets, then select elites from this rollout only."""
        super().compute_returns(obs)
        self._update_bin_performance()
        self._select_rollout_elites()
        self._bin_step = 0

    @torch.no_grad()
    def _update_bin_performance(self) -> None:
        """Track recent and historical-best per-bin reward for forgetting detection."""
        if (
            self._bin_count is None
            or self._bin_performance_ema is None
            or self._bin_best_performance is None
            or self._bin_performance_initialized is None
            or self._forgetting_scores is None
            or self._bin_step != self.storage.num_transitions_per_env
        ):
            return

        rollout_bins = self._rollout_bins.flatten()
        rewards = self.storage.rewards.flatten().float()
        for bin_index in range(self._bin_count):
            in_bin = rollout_bins == bin_index
            if not torch.any(in_bin):
                continue
            rollout_mean = rewards[in_bin].mean()
            if not self._bin_performance_initialized[bin_index]:
                self._bin_performance_ema[bin_index] = rollout_mean
                self._bin_best_performance[bin_index] = rollout_mean
                self._bin_performance_initialized[bin_index] = True
                continue

            current = self._bin_performance_ema[bin_index]
            current = current.lerp(rollout_mean, self.forgetting_ema_alpha)
            self._bin_performance_ema[bin_index] = current
            self._bin_best_performance[bin_index] = torch.maximum(
                self._bin_best_performance[bin_index], current
            )

        initialized = self._bin_performance_initialized
        drop = self._bin_best_performance - self._bin_performance_ema
        scale = self._bin_best_performance.abs().clamp_min(1.0e-6)
        relative_drop = drop / scale
        self._forgetting_scores.zero_()
        self._forgetting_scores[initialized] = torch.clamp(
            relative_drop[initialized] - self.forgetting_relative_margin, min=0.0
        )

    @torch.no_grad()
    def _select_rollout_elites(self) -> None:
        self._elite_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self._hard_bin_count = 0
        self._forgotten_bin_count = 0
        self._mean_forgetting_score = 0.0
        self._max_forgetting_score = 0.0
        self._elite_mean_advantage = 0.0

        if (
            self._update_count < self.elite_warmup_iterations
            or self._bin_count is None
            or self._difficulty_scores is None
            or self._bin_step != self.storage.num_transitions_per_env
        ):
            return

        candidate_scores = self._difficulty_scores
        if self.forgetting_trigger_enabled:
            assert self._forgetting_scores is not None
            forgotten_bins = torch.where(self._forgetting_scores > 0.0)[0]
            self._forgotten_bin_count = int(forgotten_bins.numel())
            if forgotten_bins.numel() == 0:
                return
            active_scores = self._forgetting_scores[forgotten_bins]
            self._mean_forgetting_score = active_scores.mean().item()
            self._max_forgetting_score = active_scores.max().item()
            # Forgetting is the primary signal. Existing failure difficulty only
            # breaks near-ties without allowing a non-forgotten bin into replay.
            difficulty = self._difficulty_scores[forgotten_bins]
            difficulty = difficulty / difficulty.abs().max().clamp_min(1.0e-6)
            candidate_scores = self._forgetting_scores[forgotten_bins] + 1.0e-3 * difficulty

        num_hard_bins = max(1, math.ceil(self._bin_count * self.hard_bin_fraction))
        num_hard_bins = min(num_hard_bins, candidate_scores.numel())
        selected = torch.topk(candidate_scores, num_hard_bins, sorted=False).indices
        hard_bins = forgotten_bins[selected] if self.forgetting_trigger_enabled else selected
        self._hard_bin_count = int(hard_bins.numel())

        raw_advantages = (self.storage.returns - self.storage.values).flatten(0, 1).squeeze(-1)
        rollout_bins = self._rollout_bins.flatten()
        elite_chunks: list[torch.Tensor] = []
        for bin_index in hard_bins:
            in_bin = torch.where(rollout_bins == bin_index)[0]
            if in_bin.numel() < self.elite_min_samples_per_bin:
                continue
            positive = in_bin[raw_advantages[in_bin] > self.elite_min_advantage]
            if positive.numel() < self.elite_min_positive_samples_per_bin:
                continue
            keep_count = max(1, math.ceil(positive.numel() * self.elite_fraction))
            local_top = torch.topk(raw_advantages[positive], keep_count, sorted=False).indices
            elite_chunks.append(positive[local_top])

        if not elite_chunks:
            return

        elite_indices = torch.cat(elite_chunks)
        max_elites = math.floor(
            raw_advantages.numel() * self.elite_max_fraction_of_rollout
        )
        if max_elites == 0:
            return
        if elite_indices.numel() > max_elites:
            top = torch.topk(raw_advantages[elite_indices], max_elites, sorted=False).indices
            elite_indices = elite_indices[top]

        self._elite_indices = elite_indices
        self._elite_mean_advantage = raw_advantages[elite_indices].mean().item()

    def _adaptive_mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Partition the full rollout normally, then append all elites to every batch."""
        storage = self.storage
        batch_size = storage.num_envs * storage.num_transitions_per_env
        elite_indices = self._elite_indices
        all_indices = torch.arange(batch_size, device=self.device)

        observations = storage.observations.flatten(0, 1)
        actions = storage.actions.flatten(0, 1)
        values = storage.values.flatten(0, 1)
        returns = storage.returns.flatten(0, 1)
        old_actions_log_prob = storage.actions_log_prob.flatten(0, 1)
        advantages = storage.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in storage.distribution_params)

        for _ in range(num_epochs):
            shuffled = all_indices[torch.randperm(batch_size, device=self.device)]
            for rollout_chunk in torch.tensor_split(shuffled, num_mini_batches):
                batch_indices = torch.cat((rollout_chunk, elite_indices))
                batch_indices = batch_indices[
                    torch.randperm(batch_indices.numel(), device=self.device)
                ]
                yield RolloutStorage.Batch(
                    observations=observations[batch_indices],
                    actions=actions[batch_indices],
                    values=values[batch_indices],
                    advantages=advantages[batch_indices],
                    returns=returns[batch_indices],
                    old_actions_log_prob=old_actions_log_prob[batch_indices],
                    old_distribution_params=tuple(p[batch_indices] for p in old_distribution_params),
                )

    def update(self) -> dict[str, float]:
        """Run the unmodified PPO update with the adaptive current-rollout batches."""
        elite_count = int(self._elite_indices.numel())
        rollout_size = self.storage.num_envs * self.storage.num_transitions_per_env
        original_generator = self.storage.mini_batch_generator
        if elite_count:
            self.storage.mini_batch_generator = self._adaptive_mini_batch_generator  # type: ignore[method-assign]
        try:
            loss_dict = super().update()
        finally:
            self.storage.mini_batch_generator = original_generator  # type: ignore[method-assign]

        loss_dict["elite_count"] = float(elite_count)
        loss_dict["elite_fraction"] = elite_count / rollout_size
        loss_dict["hard_bin_count"] = float(self._hard_bin_count)
        loss_dict["elite_advantage"] = self._elite_mean_advantage
        loss_dict["forgotten_bin_count"] = float(self._forgotten_bin_count)
        loss_dict["forgetting_mean"] = self._mean_forgetting_score
        loss_dict["forgetting_max"] = self._max_forgetting_score

        self._elite_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self._update_count += 1
        return loss_dict
