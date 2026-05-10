from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TransitionBatch:
    """One-step transition data pushed into replay.

    Stored schema follows residual RL notation:
      (s_t, a_base_t, a_exec_t, r_t, s_{t+1}, a_base_{t+1}, done_t)
    """

    obs: np.ndarray
    base_action: np.ndarray
    exec_action: np.ndarray
    reward: float
    next_obs: np.ndarray
    next_base_action: np.ndarray
    done: bool


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    base_action: torch.Tensor
    exec_action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    next_base_action: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """Simple numpy replay buffer with torch sampling."""

    def __init__(self, *, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.base_action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.exec_action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_base_action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: TransitionBatch) -> None:
        idx = self._ptr
        self.obs[idx] = transition.obs
        self.base_action[idx] = transition.base_action
        self.exec_action[idx] = transition.exec_action
        self.reward[idx, 0] = float(transition.reward)
        self.next_obs[idx] = transition.next_obs
        self.next_base_action[idx] = transition.next_base_action
        self.done[idx, 0] = float(transition.done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, *, batch_size: int, device: torch.device) -> ReplayBatch:
        if self._size < batch_size:
            raise ValueError(f"Replay size {self._size} smaller than batch_size={batch_size}.")

        indices = np.random.randint(0, self._size, size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[indices], device=device, dtype=torch.float32),
            base_action=torch.as_tensor(self.base_action[indices], device=device, dtype=torch.float32),
            exec_action=torch.as_tensor(self.exec_action[indices], device=device, dtype=torch.float32),
            reward=torch.as_tensor(self.reward[indices], device=device, dtype=torch.float32),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device, dtype=torch.float32),
            next_base_action=torch.as_tensor(self.next_base_action[indices], device=device, dtype=torch.float32),
            done=torch.as_tensor(self.done[indices], device=device, dtype=torch.float32),
        )
