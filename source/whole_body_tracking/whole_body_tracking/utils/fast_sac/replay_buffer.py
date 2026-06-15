from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn


class SimpleReplayBuffer(nn.Module):
    """Circular replay buffer storing transitions per environment.

    Supports n-step returns and asymmetric (actor/critic) observations.
    """

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        super().__init__()
        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.observations = torch.zeros((n_env, buffer_size, n_obs), device=device, dtype=torch.float)
        self.actions = torch.zeros((n_env, buffer_size, n_act), device=device, dtype=torch.float)
        self.rewards = torch.zeros((n_env, buffer_size), device=device, dtype=torch.float)
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.truncations = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.next_observations = torch.zeros((n_env, buffer_size, n_obs), device=device, dtype=torch.float)
        self.critic_observations = torch.zeros((n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float)
        self.next_critic_observations = torch.zeros((n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float)
        self.ptr = 0

    def extend(self, tensor_dict: TensorDict):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        self.critic_observations[:, ptr] = tensor_dict["critic_observations"]
        self.next_critic_observations[:, ptr] = tensor_dict["next"]["critic_observations"]
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size: int):
        if self.n_steps == 1:
            return self._sample_1step(batch_size)
        return self._sample_nstep(batch_size)

    def _sample_1step(self, batch_size: int):
        indices = torch.randint(
            0, min(self.buffer_size, self.ptr), (self.n_env, batch_size), device=self.device
        )

        obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
        act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
        critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)

        observations = torch.gather(self.observations, 1, obs_indices).reshape(self.n_env * batch_size, self.n_obs)
        next_observations = torch.gather(self.next_observations, 1, obs_indices).reshape(
            self.n_env * batch_size, self.n_obs
        )
        actions = torch.gather(self.actions, 1, act_indices).reshape(self.n_env * batch_size, self.n_act)
        rewards = torch.gather(self.rewards, 1, indices).reshape(self.n_env * batch_size)
        dones = torch.gather(self.dones, 1, indices).reshape(self.n_env * batch_size)
        truncations = torch.gather(self.truncations, 1, indices).reshape(self.n_env * batch_size)
        effective_n_steps = torch.ones_like(dones)

        critic_observations = torch.gather(self.critic_observations, 1, critic_obs_indices).reshape(
            self.n_env * batch_size, self.n_critic_obs
        )
        next_critic_observations = torch.gather(self.next_critic_observations, 1, critic_obs_indices).reshape(
            self.n_env * batch_size, self.n_critic_obs
        )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        out["critic_observations"] = critic_observations
        out["next"]["critic_observations"] = next_critic_observations
        return out

    def _sample_nstep(self, batch_size: int):
        if self.ptr >= self.buffer_size:
            current_pos = self.ptr % self.buffer_size
            curr_truncations = self.truncations[:, current_pos - 1].clone()
            self.truncations[:, current_pos - 1] = torch.logical_not(self.dones[:, current_pos - 1])
            indices = torch.randint(0, self.buffer_size, (self.n_env, batch_size), device=self.device)
        else:
            max_start_idx = max(1, self.ptr - self.n_steps + 1)
            indices = torch.randint(0, max_start_idx, (self.n_env, batch_size), device=self.device)

        obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
        act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
        critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)

        observations = torch.gather(self.observations, 1, obs_indices).reshape(self.n_env * batch_size, self.n_obs)
        actions = torch.gather(self.actions, 1, act_indices).reshape(self.n_env * batch_size, self.n_act)
        critic_observations = torch.gather(self.critic_observations, 1, critic_obs_indices).reshape(
            self.n_env * batch_size, self.n_critic_obs
        )

        seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
        all_indices = (indices.unsqueeze(-1) + seq_offsets) % self.buffer_size

        all_rewards = torch.gather(self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
        all_dones = torch.gather(self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
        all_truncations = torch.gather(
            self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
        )

        all_dones_shifted = torch.cat([torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2)
        done_masks = torch.cumprod(1.0 - all_dones_shifted, dim=2)
        effective_n_steps = done_masks.sum(2)

        discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device))
        masked_rewards = all_rewards * done_masks
        discounted_rewards = masked_rewards * discounts.view(1, 1, -1)
        n_step_rewards = discounted_rewards.sum(dim=2)

        first_done = torch.argmax((all_dones > 0).float(), dim=2)
        first_trunc = torch.argmax((all_truncations > 0).float(), dim=2)
        no_dones = all_dones.sum(dim=2) == 0
        no_truncs = all_truncations.sum(dim=2) == 0
        first_done = torch.where(no_dones, self.n_steps - 1, first_done)
        first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)
        final_indices = torch.minimum(first_done, first_trunc)

        final_next_obs_indices = torch.gather(all_indices, 2, final_indices.unsqueeze(-1)).squeeze(-1)
        final_next_observations = self.next_observations.gather(
            1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
        )
        final_dones = self.dones.gather(1, final_next_obs_indices)
        final_truncations = self.truncations.gather(1, final_next_obs_indices)

        final_next_critic_obs = self.next_critic_observations.gather(
            1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
        )
        next_critic_observations = final_next_critic_obs.reshape(self.n_env * batch_size, self.n_critic_obs)

        rewards = n_step_rewards.reshape(self.n_env * batch_size)
        dones = final_dones.reshape(self.n_env * batch_size)
        truncations = final_truncations.reshape(self.n_env * batch_size)
        effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
        next_observations = final_next_observations.reshape(self.n_env * batch_size, self.n_obs)

        if self.ptr >= self.buffer_size:
            self.truncations[:, current_pos - 1] = curr_truncations

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        out["critic_observations"] = critic_observations
        out["next"]["critic_observations"] = next_critic_observations
        return out
