"""RGMT attention-based policy network.

Paper: "Robust and Generalized Humanoid Motion Tracking" (RGMT)

Architecture:
  - History Encoder: causal self-attention over K+1 proprioceptive frames → dynamics emb h_t
  - Command Encoder: cross-attention (dynamics emb as query over 2L+1 command frames) → cmd emb u_t
  - Actor head: MLP over [current_obs, h_t, u_t] → action
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.modules.mlp import MLP
from rsl_rl.utils import resolve_callable, unpad_trajectories


def _sinusoidal_position_encoding(max_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Return sinusoidal position encoding [max_len, d_model]."""
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _MLPBlock(nn.Module):
    """MLP block with LayerNorm and residual connection."""

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.fc2 = nn.Linear(dim * hidden_mult, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc2(self.dropout(self.act(self.fc1(x))))
        return x + residual


class _CausalSelfAttention(nn.Module):
    """Causal (unidirectional) self-attention over time."""

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, embed_dim]."""
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask: can only attend to current and past
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class _CrossAttention(nn.Module):
    """Cross-attention: query attends over key/value memory."""

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """query: [B, 1, D], mem: [B, M, D]."""
        B, M, D = mem.shape
        q = self.q_proj(query).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(mem).reshape(B, M, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, heads, M, head_dim]
        k, v = kv[0], kv[1]

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, heads, 1, head_dim]
        out = out.transpose(1, 2).reshape(B, 1, D)
        return self.out_proj(out)


class HistoryEncoder(nn.Module):
    """Causal self-attention encoder over proprioceptive history.

    Input: [batch_size, K+1, prop_dim] — K past frames + current frame
    Output: dynamics embedding h_t [batch_size, embed_dim]
    """

    def __init__(self, prop_dim: int, embed_dim: int = 128, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(prop_dim, embed_dim)
        self.self_attn = _CausalSelfAttention(embed_dim, num_heads)
        self.mlp_block = _MLPBlock(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, prop_dim]."""
        B, T, _ = x.shape
        # Project + position encoding
        x = self.proj(x)
        pe = _sinusoidal_position_encoding(T, self.embed_dim, x.device)
        x = x + pe.unsqueeze(0)

        # Causal self-attention + MLP block
        x = self.self_attn(x) + x
        x = self.mlp_block(x)
        x = self.norm(x)

        # Max pooling over time → [B, embed_dim]
        x, _ = x.max(dim=1)
        return x


class CommandEncoder(nn.Module):
    """Cross-attention encoder over command window.

    Input: dynamics emb h_t [B, embed_dim], command window [B, M, cmd_dim]
    Output: command embedding u_t [B, embed_dim]
    """

    def __init__(self, cmd_dim: int, embed_dim: int = 128, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.cmd_proj = nn.Linear(cmd_dim, embed_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cross_attn = _CrossAttention(embed_dim, num_heads)
        self.mlp_block = _MLPBlock(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, dynamics_emb: torch.Tensor, cmd_window: torch.Tensor) -> torch.Tensor:
        """dynamics_emb: [B, D], cmd_window: [B, M, cmd_dim]."""
        B, M, _ = cmd_window.shape
        # Project commands + position encoding
        cmd_tokens = self.cmd_proj(cmd_window)  # [B, M, D]
        pe = _sinusoidal_position_encoding(M, self.embed_dim, cmd_tokens.device)
        cmd_tokens = cmd_tokens + pe.unsqueeze(0)

        # Dynamics emb → query
        q = self.query_mlp(dynamics_emb).unsqueeze(1)  # [B, 1, D]

        # Cross-attention + MLP block
        u = self.cross_attn(q, cmd_tokens).squeeze(1)  # [B, D]
        u = self.mlp_block(u) + u
        u = self.norm(u)
        return u


class RGMTModel(nn.Module):
    """RGMT attention-based actor model.

    Follows RSL-RL's MLPModel interface for compatibility with the PPO algorithm.
    Splits the flat policy observation into current_obs, prop_history, and command_window,
    then processes them through the RGMT attention architecture.
    """

    is_recurrent: bool = False
    """This model is stateless — all context is embedded in the observation."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        # RGMT-specific parameters
        prop_history_len: int = 10,
        command_window_half_size: int = 10,
        prop_dim: int = 90,
        cmd_dim: int = 50,
        embed_dim: int = 128,
        num_heads: int = 1,
    ) -> None:
        super().__init__()

        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.prop_history_len = prop_history_len
        self.cmd_window_len = 2 * command_window_half_size + 1
        self.prop_dim = prop_dim
        self.cmd_dim = cmd_dim
        self.embed_dim = embed_dim

        # Dynamic dimension computation: [current_obs | prop_history (K × 90) | cmd_window ((2L+1) × 50)]
        self.history_dim = prop_history_len * prop_dim
        self.cmd_window_dim = self.cmd_window_len * cmd_dim
        self.current_obs_dim = self.obs_dim - self.history_dim - self.cmd_window_dim

        if self.current_obs_dim < 0:
            raise ValueError(
                f"Observation dimension {self.obs_dim} is too small for history ({self.history_dim}) "
                f"+ command window ({self.cmd_window_dim}). "
                f"Ensure prop_history and command_window observation terms are added to the policy group."
            )

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # RGMT encoders
        self.history_encoder = HistoryEncoder(prop_dim, embed_dim, num_heads)
        self.command_encoder = CommandEncoder(cmd_dim, embed_dim, num_heads)

        # Actor MLP head: [current_obs | dynamics_emb | cmd_emb]
        self.head_input_dim = self.current_obs_dim + embed_dim + embed_dim
        self.mlp = MLP(self.head_input_dim, mlp_output_dim, hidden_dims, activation)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Extracts current_obs, prop_history, and command_window from the flat
        observation, runs the RGMT encoders, and outputs an action (or value).
        """
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent: [current_obs, dynamics_emb, cmd_emb]."""
        # Concatenate and normalize observations
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        flat_obs = torch.cat(obs_list, dim=-1)
        flat_obs = self.obs_normalizer(flat_obs)

        # Split: [current_obs | prop_history | command_window]
        cur_start = 0
        cur_end = self.current_obs_dim
        hist_end = cur_end + self.history_dim
        cmd_end = hist_end + self.cmd_window_dim

        current_obs = flat_obs[:, cur_start:cur_end]
        prop_history = flat_obs[:, cur_end:hist_end].reshape(
            -1, self.prop_history_len, self.prop_dim
        )
        cmd_window = flat_obs[:, hist_end:cmd_end].reshape(
            -1, self.cmd_window_len, self.cmd_dim
        )

        # History encoder: K past frames (last frame = current proprio from buffer)
        dynamics_emb = self.history_encoder(prop_history)

        # Command encoder: dynamics_emb as query, cross-attend over command window
        cmd_emb = self.command_encoder(dynamics_emb, cmd_window)

        # Concatenate for MLP head
        return torch.cat([current_obs, dynamics_emb, cmd_emb], dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset internal state (no-op for stateless model)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return hidden state (None for stateless model)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden state (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return JIT-exportable version."""
        return _TorchRGMTModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return ONNX-exportable version."""
        return _OnnxRGMTModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization statistics."""
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            flat_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(flat_obs)

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"RGMTModel only supports 1D observations, "
                    f"got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchRGMTModel(nn.Module):
    """Exportable RGMT model for JIT."""

    def __init__(self, model: RGMTModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.command_encoder = copy.deepcopy(model.command_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        # Store split config
        self.current_obs_dim = model.current_obs_dim
        self.history_dim = model.history_dim
        self.cmd_window_dim = model.cmd_window_dim
        self.prop_history_len = model.prop_history_len
        self.cmd_window_len = model.cmd_window_len
        self.prop_dim = model.prop_dim
        self.cmd_dim = model.cmd_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        cur_end = self.current_obs_dim
        hist_end = cur_end + self.history_dim
        current_obs = x[:, :cur_end]
        prop_history = x[:, cur_end:hist_end].reshape(-1, self.prop_history_len, self.prop_dim)
        cmd_window = x[:, hist_end:hist_end + self.cmd_window_dim].reshape(-1, self.cmd_window_len, self.cmd_dim)

        dynamics_emb = self.history_encoder(prop_history)
        cmd_emb = self.command_encoder(dynamics_emb, cmd_window)
        latent = torch.cat([current_obs, dynamics_emb, cmd_emb], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxRGMTModel(nn.Module):
    """Exportable RGMT model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: RGMTModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.command_encoder = copy.deepcopy(model.command_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

        # Store split config
        self.current_obs_dim = model.current_obs_dim
        self.history_dim = model.history_dim
        self.cmd_window_dim = model.cmd_window_dim
        self.prop_history_len = model.prop_history_len
        self.cmd_window_len = model.cmd_window_len
        self.prop_dim = model.prop_dim
        self.cmd_dim = model.cmd_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        cur_end = self.current_obs_dim
        hist_end = cur_end + self.history_dim
        current_obs = x[:, :cur_end]
        prop_history = x[:, cur_end:hist_end].reshape(-1, self.prop_history_len, self.prop_dim)
        cmd_window = x[:, hist_end:hist_end + self.cmd_window_dim].reshape(-1, self.cmd_window_len, self.cmd_dim)

        dynamics_emb = self.history_encoder(prop_history)
        cmd_emb = self.command_encoder(dynamics_emb, cmd_window)
        latent = torch.cat([current_obs, dynamics_emb, cmd_emb], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
