from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(
    in_dim: int,
    hidden_dims: tuple[int, ...],
    out_dim: int,
    *,
    use_layer_norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev, hidden))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

class ResidualActor(nn.Module):
    """Residual policy head: a_res = pi_res(obs, a_base)."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.net = build_mlp(
            in_dim=obs_dim + act_dim,
            hidden_dims=hidden_dims,
            out_dim=act_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, obs: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, base_action], dim=-1)
        return self.net(x)


class Critic(nn.Module):
    """Single Q-function over full executed action Q(s, a_exec)."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.net = build_mlp(
            in_dim=obs_dim + act_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class TwinCritic(nn.Module):
    """Twin-critic module for clipped double-Q style updates."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.q1 = Critic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        )
        self.q2 = Critic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.minimum(q1, q2)
