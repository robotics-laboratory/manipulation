"""Flow-SDE utilities for PPO finetuning of SmolVLA.

Background
----------
SmolVLA's inference is a deterministic ODE: x_{k+1} = x_k + dt * v_θ(x_k, t_k),
where dt = -1/K and t_k = 1 - k/K.  This determinism prevents computing a
tractable log π(action | obs), which is needed for PPO.

The piRL paper (arXiv:2510.25889) solves this by converting the ODE into an
equivalent SDE that preserves marginal action distributions.  Each denoising
step becomes a Gaussian transition p(x_{k+1} | x_k) ~ N(mu, Σ), giving a
tractable log-prob.

Hybrid ODE-SDE sampling (from piRL)
------------------------------------
Instead of making every step stochastic (expensive), we randomly pick ONE
step index k_sde ∈ [1, K-2] to be the SDE step; all others are deterministic
ODE steps.  This reduces the effective MDP horizon to 1 stochastic transition
per chunk, keeping training complexity manageable.

SDE math (in SmolVLA t-space where t = 1 - k/K)
-------------------------------------------------
The SDE equivalent of the ODE dx = v_θ dt is (in rectified-flow / tau-space):

    tau = 1 - t  (tau increases 0→1 as denoising progresses)
    v_tau = -v_t   (sign flip from SmolVLA's convention)

    sigma²(tau) = a² * tau / (1 - tau)  [a = noise_level hyperparameter]
    delta = 1/K

    drift correction:
        drift_corr = sigma²(tau) / (2*tau) * (x_k + (1 - tau) * v_tau)

    mu   = x_k + v_tau * delta + drift_corr * delta
    var  = sigma²(tau) * delta

    x_{k+1} ~ N(mu, var * I)
    log_prob = -0.5 * [D * log(2π * var) + ||x_{k+1} - mu||² / var]

This file does NOT modify the installed lerobot package.  It calls into
SmolVLA internals through the public VLAFlowMatching.denoise_step interface.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching


# ---------------------------------------------------------------------------
# Core SDE step
# ---------------------------------------------------------------------------


def flow_sde_step(
    x_k: torch.Tensor,
    v_t: torch.Tensor,
    k: int,
    K: int,
    noise_level: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one SDE denoising step and return the next state + log-probability.

    Args:
        x_k:         Current noisy action state, shape (B, chunk_size, action_dim).
        v_t:         Velocity predicted by the action head at t_k = 1 - k/K,
                     same shape as x_k.  This is SmolVLA's v_t (points from
                     noise → action, i.e. v_tau = -v_t in tau-space).
        k:           Current denoising step index in [1, K-2].
        K:           Total number of denoising steps.
        noise_level: Noise magnitude hyperparameter `a` (default 0.5 per piRL).

    Returns:
        x_next:   Sampled next state (B, chunk_size, action_dim).
        log_prob: Scalar log-probability summed over all action dimensions and
                  chunk steps, shape (B,).

    Raises:
        ValueError: If k is a boundary step (k == 0 or k == K-1) where the SDE
                    formula is singular.
    """
    if k == 0 or k == K - 1:
        raise ValueError(
            f"SDE step cannot be applied at boundary steps k=0 or k=K-1 (k={k}, K={K}). "
            "Use ODE step at boundaries."
        )

    tau = k / K              # tau ∈ (0, 1) for k ∈ [1, K-2]
    delta = 1.0 / K
    device = x_k.device
    dtype = torch.float32

    x_k = x_k.to(dtype)
    v_t = v_t.to(dtype)

    # In piRL tau-space: v_tau = -v_t (SmolVLA uses negative time direction).
    v_tau = -v_t

    # SDE variance schedule: sigma²(tau) = a² * tau / (1 - tau)
    sigma2_tau = (noise_level ** 2) * tau / (1.0 - tau)
    var = sigma2_tau * delta  # scalar

    # Drift correction term (from ODE→SDE conversion)
    drift_corr = (sigma2_tau / (2.0 * tau)) * (x_k + (1.0 - tau) * v_tau)

    # SDE mean: x_k + v_tau * delta + drift_corr * delta
    mu = x_k + v_tau * delta + drift_corr * delta

    # Sample x_{k+1} ~ N(mu, var * I)
    eps = torch.randn_like(mu)
    x_next = mu + math.sqrt(var) * eps

    # Log-prob: sum over all dims per sample
    # log N(x_next; mu, var*I) = -0.5 * [D*log(2*pi*var) + ||x_next - mu||² / var]
    D = float(mu[0].numel())  # total number of action elements (chunk_size * action_dim)
    sq_err = ((x_next - mu) ** 2).sum(dim=(-1, -2))  # (B,)
    log_prob = -0.5 * (D * math.log(2.0 * math.pi * var) + sq_err / var)  # (B,)

    return x_next, log_prob


def flow_sde_logprob(
    x_k: torch.Tensor,
    x_next: torch.Tensor,
    v_t: torch.Tensor,
    k: int,
    K: int,
    noise_level: float = 0.5,
) -> torch.Tensor:
    """Recompute log-probability of a stored (x_k → x_next) SDE transition.

    Used in the PPO update step where we need to recompute log π_new(a|s)
    with current network weights but stored state pairs from rollout.

    Args:
        x_k:         State before the SDE step (B, chunk_size, action_dim).
        x_next:      State after the SDE step (B, chunk_size, action_dim).
        v_t:         New velocity at t_k from the current policy weights.
        k, K:        Step index and total steps.
        noise_level: Noise magnitude (must match rollout value).

    Returns:
        log_prob: Shape (B,).
    """
    tau = k / K
    delta = 1.0 / K
    dtype = torch.float32

    x_k = x_k.to(dtype)
    x_next = x_next.to(dtype)
    v_t = v_t.to(dtype)

    v_tau = -v_t
    sigma2_tau = (noise_level ** 2) * tau / (1.0 - tau)
    var = sigma2_tau * delta

    drift_corr = (sigma2_tau / (2.0 * tau)) * (x_k + (1.0 - tau) * v_tau)
    mu = x_k + v_tau * delta + drift_corr * delta

    D = float(mu[0].numel())
    sq_err = ((x_next - mu) ** 2).sum(dim=(-1, -2))  # (B,)
    log_prob = -0.5 * (D * math.log(2.0 * math.pi * var) + sq_err / var)

    return log_prob


# ---------------------------------------------------------------------------
# Hybrid ODE-SDE sampling (full chunk)
# ---------------------------------------------------------------------------


def hybrid_ode_sde_sample(
    flow_model: "VLAFlowMatching",
    prefix_pad_masks: torch.Tensor,
    past_key_values: object,
    noise: torch.Tensor,
    K: int,
    noise_level: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Run K-step hybrid ODE-SDE denoising with ONE stochastic step.

    Picks a random SDE step index k_sde ∈ [1, K-2].  All other steps use
    the deterministic ODE update x_{k+1} = x_k + dt * v_t.

    Args:
        flow_model:      VLAFlowMatching instance (policy.model).
        prefix_pad_masks: Prefix padding masks from embed_prefix, shape (B, prefix_len).
        past_key_values: KV-cache from the frozen VLM prefix forward pass.
        noise:           Initial noise (B, chunk_size, action_dim).
        K:               Number of denoising steps.
        noise_level:     SDE noise magnitude `a`.

    Returns:
        action_chunk:  Final denoised action (B, chunk_size, action_dim).
        log_prob:      Log-probability of the SDE transition, shape (B,).
        k_sde:         The stochastic step index used (int).
        x_k_sde:       State x_k at the SDE step (before sampling), detached CPU tensor.
        x_next_sde:    State x_{k+1} from the SDE step (after sampling), detached CPU tensor.
    """
    device = noise.device
    bsize = noise.shape[0]
    x_t = noise.float()
    dt = -1.0 / K

    # Pick a random interior step for the SDE injection.
    k_sde = torch.randint(1, K - 1, (1,)).item()

    x_k_sde = None
    x_next_sde = None
    log_prob = torch.zeros(bsize, device=device)

    with torch.no_grad():
        for step in range(K):
            t_val = 1.0 + step * dt  # descends from 1.0
            time_tensor = torch.full((bsize,), t_val, dtype=torch.float32, device=device)

            v_t = flow_model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )

            if step == k_sde:
                # Stochastic step — save state for PPO recomputation.
                x_k_sde = x_t.detach().cpu()
                x_next, lp = flow_sde_step(x_t, v_t, k=step, K=K, noise_level=noise_level)
                x_next_sde = x_next.detach().cpu()
                log_prob = lp.detach()  # (B,)
                x_t = x_next
            else:
                # Deterministic ODE step.
                x_t = x_t + dt * v_t

    return x_t, log_prob, int(k_sde), x_k_sde, x_next_sde


# ---------------------------------------------------------------------------
# Log-prob recomputation for PPO update
# ---------------------------------------------------------------------------


def recompute_log_prob(
    flow_model: "VLAFlowMatching",
    prefix_pad_masks: torch.Tensor,
    past_key_values: object,
    x_k_sde: torch.Tensor,
    x_next_sde: torch.Tensor,
    k_sde: int,
    K: int,
    noise_level: float = 0.5,
) -> torch.Tensor:
    """Recompute log π_new(a|s) using current policy weights.

    Only runs a SINGLE denoise_step call (at the saved SDE step index),
    making PPO updates efficient.  The prefix KV cache must be precomputed
    from the stored observation.

    Args:
        flow_model:       VLAFlowMatching instance with current weights.
        prefix_pad_masks: (B, prefix_len) from embed_prefix of stored obs.
        past_key_values:  KV cache from the stored observation prefix forward.
        x_k_sde:          Stored x_k at the SDE step (B, chunk_size, action_dim).
        x_next_sde:       Stored x_{k+1} at the SDE step (B, chunk_size, action_dim).
        k_sde:            The SDE step index that was stochastic during rollout.
        K:                Total number of denoising steps.
        noise_level:      Noise magnitude (must match rollout value).

    Returns:
        log_prob: Shape (B,) with gradient w.r.t. action head parameters.
    """
    device = x_k_sde.device
    bsize = x_k_sde.shape[0]
    t_k = 1.0 + k_sde * (-1.0 / K)  # SmolVLA time at step k_sde
    time_tensor = torch.full((bsize,), t_k, dtype=torch.float32, device=device)

    # Forward pass with gradient (action head is trainable).
    v_t = flow_model.denoise_step(
        x_t=x_k_sde.to(device),
        prefix_pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,
        timestep=time_tensor,
    )

    log_prob = flow_sde_logprob(
        x_k=x_k_sde.to(device),
        x_next=x_next_sde.to(device),
        v_t=v_t,
        k=k_sde,
        K=K,
        noise_level=noise_level,
    )
    return log_prob
