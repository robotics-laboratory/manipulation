from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import ResidualRLConfig
from .networks import ResidualActor, TwinCritic
from .replay import ReplayBatch


@dataclass
class ResidualUpdateStats:
    critic_loss: float
    actor_loss: float | None
    q1_mean: float
    q2_mean: float
    target_q_mean: float
    grad_norm_actor: float | None
    grad_norm_critic: float


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.mul_(1.0 - tau).add_(tau * source_param)


def _global_grad_norm(parameters) -> float:
    sq_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        sq_norm += float(torch.sum(p.grad.detach() ** 2).item())
    return float(sq_norm ** 0.5)


class ResidualLearner:
    """Residual off-policy learner (TD3-style update over residual actions).

    This class is intentionally policy-agnostic:
    - It assumes `base_action` is provided by a frozen external policy.
    - It optimizes only the residual actor and critic.
    """

    def __init__(self, cfg: ResidualRLConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = ResidualActor(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dims=cfg.actor_hidden,
            use_layer_norm=cfg.layer_norm_actor,
        ).to(self.device)
        self.actor_target = ResidualActor(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dims=cfg.actor_hidden,
            use_layer_norm=cfg.layer_norm_actor,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TwinCritic(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dims=cfg.critic_hidden,
            use_layer_norm=cfg.layer_norm_critic,
        ).to(self.device)
        self.critic_target = TwinCritic(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dims=cfg.critic_hidden,
            use_layer_norm=cfg.layer_norm_critic,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.AdamW(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            weight_decay=cfg.weight_decay,
        )
        self.critic_optim = torch.optim.AdamW(
            self.critic.parameters(),
            lr=cfg.critic_lr,
            weight_decay=cfg.weight_decay,
        )

        self._update_step = 0

    def clamp_residual(self, residual_action: torch.Tensor) -> torch.Tensor:
        """Apply per-dimension residual limits (SO-101 default convention)."""
        if residual_action.shape[-1] != self.cfg.act_dim:
            raise ValueError(f"Expected residual action dim {self.cfg.act_dim}, got {residual_action.shape[-1]}")
        clamped = residual_action.clone()
        if self.cfg.act_dim >= 6:
            clamped[..., :5] = torch.clamp(clamped[..., :5], -self.cfg.residual_arm_limit_rad, self.cfg.residual_arm_limit_rad)
            clamped[..., 5:6] = torch.clamp(
                clamped[..., 5:6],
                -self.cfg.residual_gripper_limit,
                self.cfg.residual_gripper_limit,
            )
        else:
            clamped = torch.clamp(clamped, -self.cfg.residual_arm_limit_rad, self.cfg.residual_arm_limit_rad)
        return clamped

    def compose_action(
        self,
        *,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Compose full executed action."""
        return base_action + float(alpha) * self.clamp_residual(residual_action)

    @torch.no_grad()
    def infer_residual(self, obs: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        self.actor.eval()
        out = self.actor(obs, base_action)
        self.actor.train()
        return out

    def update(self, batch: ReplayBatch) -> ResidualUpdateStats:
        """Run one learner update.

        TODO(user):
            - Add n-step target support if you store n-step transitions.
            - Mix offline/demo and online replay here if desired.
        """
        cfg = self.cfg
        self._update_step += 1

        with torch.no_grad():
            next_residual = self.actor_target(batch.next_obs, batch.next_base_action)
            next_residual = self.clamp_residual(next_residual)

            noise = torch.randn_like(next_residual) * cfg.target_noise_std
            noise = torch.clamp(noise, -cfg.target_noise_clip, cfg.target_noise_clip)
            next_residual = self.clamp_residual(next_residual + noise)

            next_exec = self.compose_action(
                base_action=batch.next_base_action,
                residual_action=next_residual,
                alpha=1.0,  # replay stores already-executed actions; alpha is absorbed in learning distribution
            )
            target_q = self.critic_target.min_q(batch.next_obs, next_exec)
            target = batch.reward + (1.0 - batch.done) * cfg.gamma * target_q

        q1, q2 = self.critic(batch.obs, batch.exec_action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = _global_grad_norm(self.critic.parameters())
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=cfg.grad_clip_norm)
        self.critic_optim.step()

        actor_loss_value: float | None = None
        actor_grad_norm: float | None = None
        if self._update_step % cfg.actor_update_interval == 0:
            residual = self.actor(batch.obs, batch.base_action)
            exec_action = self.compose_action(
                base_action=batch.base_action,
                residual_action=residual,
                alpha=1.0,
            )
            actor_loss = -self.critic.min_q(batch.obs, exec_action).mean()

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = _global_grad_norm(self.actor.parameters())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=cfg.grad_clip_norm)
            self.actor_optim.step()

            actor_loss_value = float(actor_loss.detach().item())

            _soft_update(self.actor_target, self.actor, cfg.tau)
            _soft_update(self.critic_target, self.critic, cfg.tau)
        else:
            _soft_update(self.critic_target, self.critic, cfg.tau)

        return ResidualUpdateStats(
            critic_loss=float(critic_loss.detach().item()),
            actor_loss=actor_loss_value,
            q1_mean=float(q1.detach().mean().item()),
            q2_mean=float(q2.detach().mean().item()),
            target_q_mean=float(target.detach().mean().item()),
            grad_norm_actor=actor_grad_norm,
            grad_norm_critic=critic_grad_norm,
        )

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "cfg": dataclasses.asdict(self.cfg),
            "update_step": self._update_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        self._update_step = int(state.get("update_step", 0))
