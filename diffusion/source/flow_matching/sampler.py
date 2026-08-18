"""Euler and Heun ODE integration for a noise-to-data rectified flow."""

from __future__ import annotations

import torch


@torch.inference_mode()
def sample_flow(
  model: torch.nn.Module,
  x_noise: torch.Tensor,
  terrain: torch.Tensor | None,
  proprio: torch.Tensor | None,
  sampling_steps: int = 10,
  method: str = "euler",
  time_embedding_scale: float = 1000.0,
) -> torch.Tensor:
  """Integrate dx/dt=v(x,t,c) from noise at t=1 to data at t=0."""
  if method not in ("euler", "heun"):
    raise ValueError(f"Unknown flow sampler {method!r}")
  if sampling_steps <= 0:
    raise ValueError("sampling_steps must be positive")

  x = x_noise
  batch = x.shape[0]
  times = torch.linspace(1.0, 0.0, sampling_steps + 1, device=x.device)
  for index in range(sampling_steps):
    t = times[index]
    t_next = times[index + 1]
    dt = t_next - t
    t_batch = torch.full(
      (batch,), t * time_embedding_scale, dtype=x.dtype, device=x.device
    )
    velocity = model(x, t_batch, terrain=terrain, proprio=proprio)
    if method == "euler":
      x = x + dt * velocity
      continue

    prediction = x + dt * velocity
    t_next_batch = torch.full(
      (batch,), t_next * time_embedding_scale, dtype=x.dtype, device=x.device
    )
    velocity_next = model(
      prediction, t_next_batch, terrain=terrain, proprio=proprio
    )
    x = x + 0.5 * dt * (velocity + velocity_next)
  return x
