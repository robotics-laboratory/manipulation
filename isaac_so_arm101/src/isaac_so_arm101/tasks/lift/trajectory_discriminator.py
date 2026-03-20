from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TrajectoryDiscriminatorCfg:
    """Discriminator config for a (p, delta_p, g) -> logit model."""

    p_dim: int = 3
    delta_dim: int = 3
    g_dim: int = 3
    hidden_dims: tuple[int, ...] = (256, 128)


class TrajectoryDiscriminator(nn.Module):
    """Small MLP discriminator: D(p_t, delta_p_t, g) -> probability.

    This mirrors the paper's discriminator-guided exploration idea, but adapted
    to your EE-based trajectory representation.
    """

    def __init__(self, cfg: TrajectoryDiscriminatorCfg):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.p_dim + cfg.delta_dim + cfg.g_dim
        dims = (input_dim,) + tuple(cfg.hidden_dims) + (1,)

        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))  # logits
        self.net = nn.Sequential(*layers)

        # Normalization stats (set by `load()`).
        self.register_buffer("p_mean", torch.zeros(cfg.p_dim), persistent=False)
        self.register_buffer("p_std", torch.ones(cfg.p_dim), persistent=False)
        self.register_buffer("delta_mean", torch.zeros(cfg.delta_dim), persistent=False)
        self.register_buffer("delta_std", torch.ones(cfg.delta_dim), persistent=False)
        self.register_buffer("g_mean", torch.zeros(cfg.g_dim), persistent=False)
        self.register_buffer("g_std", torch.ones(cfg.g_dim), persistent=False)

    def _standardize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        std = torch.clamp(std, min=1.0e-6)
        return (x - mean) / std

    def forward(self, p: torch.Tensor, delta_p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Returns discriminator logits."""
        p_n = self._standardize(p, self.p_mean, self.p_std)
        delta_n = self._standardize(delta_p, self.delta_mean, self.delta_std)
        g_n = self._standardize(g, self.g_mean, self.g_std)
        x = torch.cat([p_n, delta_n, g_n], dim=1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def log_d_scores(
        self,
        p: torch.Tensor,
        delta_p: torch.Tensor,
        g: torch.Tensor,
        logit_temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_d_raw, log_d_eps_floor, logits)`` for RL and diagnostics.

        - ``log_d_raw``: :math:`\\log \\sigma(\\text{logits}/T)` with **no** epsilon floor.
        - ``log_d_eps_floor``: same with lower clamp at ``log(eps)`` (matches legacy reward path).
        - ``logits``: **pre-temperature** discriminator output (same as :meth:`forward`).
        """
        logits = self(p, delta_p, g)
        t = max(float(logit_temperature), 1e-6)
        z = logits / t
        log_d = -F.softplus(-z)
        raw = log_d
        eps_t = torch.log(torch.tensor(float(eps), device=log_d.device, dtype=log_d.dtype))
        floored = torch.clamp(log_d, min=eps_t)
        return raw, floored, logits

    @torch.no_grad()
    def log_prob_positive(
        self,
        p: torch.Tensor,
        delta_p: torch.Tensor,
        g: torch.Tensor,
        eps: float = 1e-6,
        logit_temperature: float = 1.0,
    ) -> torch.Tensor:
        """Computes log(sigmoid(logits)) in a numerically stable way.

        ``logit_temperature`` divides logits before ``softplus`` (inference-only). Keep **near 1**
        unless debugging saturation; large values collapse all envs to ~\\log 0.5.
        """
        _, floored, _ = self.log_d_scores(p, delta_p, g, logit_temperature=logit_temperature, eps=eps)
        return floored

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "TrajectoryDiscriminator":
        raw: dict[str, Any] = torch.load(path, map_location="cpu")
        cfg_raw = raw["model_cfg"]
        cfg = TrajectoryDiscriminatorCfg(
            p_dim=int(cfg_raw["p_dim"]),
            delta_dim=int(cfg_raw["delta_dim"]),
            g_dim=int(cfg_raw["g_dim"]),
            hidden_dims=tuple(int(x) for x in cfg_raw["hidden_dims"]),
        )
        model = cls(cfg=cfg)
        model.load_state_dict(raw["model_state_dict"], strict=True)
        model.p_mean.copy_(raw["p_mean"])
        model.p_std.copy_(raw["p_std"])
        model.delta_mean.copy_(raw["delta_mean"])
        model.delta_std.copy_(raw["delta_std"])
        model.g_mean.copy_(raw["g_mean"])
        model.g_std.copy_(raw["g_std"])
        model.to(device=device)
        model.eval()
        return model

    def to_dict(self) -> dict[str, Any]:
        """Serialize in the same structure expected by `load()`."""
        return {
            "model_cfg": {
                "p_dim": self.cfg.p_dim,
                "delta_dim": self.cfg.delta_dim,
                "g_dim": self.cfg.g_dim,
                "hidden_dims": list(self.cfg.hidden_dims),
            },
            "model_state_dict": self.state_dict(),
            "p_mean": self.p_mean.detach().cpu(),
            "p_std": self.p_std.detach().cpu(),
            "delta_mean": self.delta_mean.detach().cpu(),
            "delta_std": self.delta_std.detach().cpu(),
            "g_mean": self.g_mean.detach().cpu(),
            "g_std": self.g_std.detach().cpu(),
        }

