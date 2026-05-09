# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead


@dataclass
class SmolVLAConfig:
    model_path: str = "/path/to/model/smolvla"
    model_type: str = "smolvla"
    action_dim: int = 6
    num_action_chunks: int = 1
    add_value_head: bool = True
    state_dim: int = 6
    hidden_dim: int = 256
    residual_scale: float = 0.2
    action_low: float = -3.2
    action_high: float = 3.2
    log_std_init: float = -2.0
    noise_method: str = "gaussian_fixed"
    noise_level: float = 0.1
    noise_anneal: bool = False
    noise_params: tuple[float, float, int] = (0.2, 0.05, 20000)
    policy_language_instruction: str = "Lift the red cube up."


class SmolVLAActionModel(nn.Module, BasePolicy):
    """SmolVLA adapter for RLinf embodied PPO on SO101."""

    _STATE_NAMES = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )

    _JOINT_LOWS = np.array([-110.0, -100.0, -100.0, -95.0, -160.0, -10.0])
    _JOINT_HIGHS = np.array([110.0, 100.0, 90.0, 95.0, 160.0, 100.0])
    _MOTOR_LOWS = np.array([-100.0, -100.0, -100.0, -100.0, -100.0, 0.0])
    _MOTOR_HIGHS = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

    def __init__(self, model_cfg: SmolVLAConfig):
        super().__init__()
        self.cfg = model_cfg
        self.global_step = 0
        self._init_smolvla_backend()

        self.residual_head = nn.Sequential(
            nn.Linear(self.cfg.state_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.cfg.hidden_dim, self.cfg.num_action_chunks * self.cfg.action_dim
            ),
        )
        self.log_std = nn.Parameter(
            torch.full(
                (1, self.cfg.num_action_chunks, self.cfg.action_dim),
                self.cfg.log_std_init,
            )
        )

        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=self.cfg.state_dim,
                hidden_sizes=(256, 256, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

    def _init_smolvla_backend(self) -> None:
        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.utils import build_inference_frame
        except ImportError as exc:
            raise RuntimeError(
                "SmolVLA model_type requires lerobot smolvla dependencies in the runtime "
                "environment. Install lerobot with SmolVLA support in the container."
            ) from exc

        self._build_inference_frame = build_inference_frame
        self._policy = SmolVLAPolicy.from_pretrained(self.cfg.model_path).eval()
        if hasattr(self._policy, "config") and hasattr(self._policy.config, "n_action_steps"):
            self._policy.config.n_action_steps = self.cfg.num_action_chunks

        preprocess, postprocess = make_pre_post_processors(
            self._policy.config,
            self.cfg.model_path,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self._preprocess = preprocess
        self._postprocess = postprocess
        self._ds_features = {
            "observation.state": {
                "type": "STATE",
                "dtype": "float32",
                "shape": [self.cfg.state_dim],
                "names": list(self._STATE_NAMES),
            },
            "observation.images.front": {
                "type": "VISUAL",
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "type": "VISUAL",
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
        }

    @staticmethod
    def _lerobot_to_leisaac(actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, SmolVLAActionModel._MOTOR_LOWS, SmolVLAActionModel._MOTOR_HIGHS)
        motor_fraction = (actions - SmolVLAActionModel._MOTOR_LOWS) / (
            SmolVLAActionModel._MOTOR_HIGHS - SmolVLAActionModel._MOTOR_LOWS
        )
        joint_degrees = (
            motor_fraction
            * (SmolVLAActionModel._JOINT_HIGHS - SmolVLAActionModel._JOINT_LOWS)
            + SmolVLAActionModel._JOINT_LOWS
        )
        return joint_degrees / 180.0 * np.pi

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def _noise_scale(self, mode: Literal["train", "eval"]) -> float:
        if mode == "eval":
            return 0.0
        if self.cfg.noise_anneal:
            start, end, steps = self.cfg.noise_params
            frac = min(float(self.global_step) / max(float(steps), 1.0), 1.0)
            return float(start + (end - start) * frac)
        return float(self.cfg.noise_level)

    def _get_state_tensor(self, env_obs: dict[str, Any]) -> torch.Tensor:
        if "states" in env_obs:
            return env_obs["states"].to(dtype=torch.float32)
        if "state" in env_obs:
            return env_obs["state"].to(dtype=torch.float32)
        raise KeyError("SmolVLA adapter expects `states` in env observation.")

    def _base_action_one(self, front_img: torch.Tensor, wrist_img: torch.Tensor, state: np.ndarray, prompt: str) -> np.ndarray:
        front = front_img.detach().cpu().numpy().astype(np.uint8)
        wrist = wrist_img.detach().cpu().numpy().astype(np.uint8)
        policy_obs = {"front": front, "wrist": wrist}
        for i, name in enumerate(self._STATE_NAMES):
            policy_obs[name] = float(state[i])
        obs_frame = self._build_inference_frame(
            observation=policy_obs,
            ds_features=self._ds_features,
            device=next(self._policy.parameters()).device,
            task=str(prompt),
            robot_type="",
        )
        model_input = self._preprocess(obs_frame)
        with torch.no_grad():
            action = self._policy.select_action(model_input)
        action = self._postprocess(action)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if action.ndim == 1:
            action = action[None, :]
        if action.shape[0] > self.cfg.num_action_chunks:
            action = action[: self.cfg.num_action_chunks]
        return self._lerobot_to_leisaac(action[:, : self.cfg.action_dim])

    def _compute_base_actions(self, env_obs: dict[str, Any]) -> torch.Tensor:
        states = self._get_state_tensor(env_obs)
        batch_size = states.shape[0]
        fronts = env_obs["main_images"]
        wrists = env_obs.get("wrist_images")
        prompts = env_obs.get("task_descriptions")
        if prompts is None:
            prompts = [self.cfg.policy_language_instruction] * batch_size
        base_actions = []
        for i in range(batch_size):
            wrist_img = fronts[i] if wrists is None else wrists[i]
            action_np = self._base_action_one(
                front_img=fronts[i],
                wrist_img=wrist_img,
                state=states[i].detach().cpu().numpy(),
                prompt=prompts[i],
            )
            base_actions.append(torch.from_numpy(action_np).to(dtype=torch.float32))
        return torch.stack(base_actions, dim=0)

    def _residual_actions(self, states: torch.Tensor) -> torch.Tensor:
        residual = self.residual_head(states)
        residual = residual.view(-1, self.cfg.num_action_chunks, self.cfg.action_dim)
        return torch.tanh(residual) * self.cfg.residual_scale

    @staticmethod
    def _normal_logprob(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        var = std.square().clamp_min(1e-8)
        return -0.5 * (((action - mean).square() / var) + torch.log(2 * torch.pi * var))

    def _compose_action(
        self, base_action: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        action = base_action.to(residual.device) + residual
        return torch.clamp(action, self.cfg.action_low, self.cfg.action_high)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        states = forward_inputs["states"].to(dtype=torch.float32)
        base_action = forward_inputs["base_action"].to(dtype=torch.float32)
        base_action = base_action.view(-1, self.cfg.num_action_chunks, self.cfg.action_dim)
        action = forward_inputs["action"].to(dtype=torch.float32)
        action = action.view(-1, self.cfg.num_action_chunks, self.cfg.action_dim)

        residual = self._residual_actions(states)
        action_mean = self._compose_action(base_action, residual)
        action_std = torch.exp(self.log_std).expand_as(action_mean).clamp_min(1e-4)

        output_dict = {}
        if compute_logprobs:
            output_dict["logprobs"] = self._normal_logprob(action, action_mean, action_std)
        if compute_entropy:
            entropy = 0.5 + 0.5 * torch.log(2 * torch.pi * action_std.square())
            output_dict["entropy"] = entropy
        if compute_values:
            if hasattr(self, "value_head"):
                output_dict["values"] = self.value_head(states)
            else:
                raise NotImplementedError
        return output_dict

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        calculate_values=True,
        return_obs=True,
        **kwargs,
    ):
        device = next(self.residual_head.parameters()).device
        states = self._get_state_tensor(env_obs).to(device=device, dtype=torch.float32)
        base_actions = self._compute_base_actions(env_obs).to(device=device, dtype=torch.float32)
        residual = self._residual_actions(states)
        action_mean = self._compose_action(base_actions, residual)

        noise_scale = self._noise_scale(mode)
        if noise_scale > 0:
            actions = torch.clamp(
                action_mean + torch.randn_like(action_mean) * noise_scale,
                self.cfg.action_low,
                self.cfg.action_high,
            )
        else:
            actions = action_mean

        action_std = torch.exp(self.log_std).expand_as(action_mean).clamp_min(1e-4)
        chunk_logprobs = self._normal_logprob(actions, action_mean, action_std)

        if hasattr(self, "value_head") and calculate_values:
            chunk_values = self.value_head(states)
        else:
            chunk_values = torch.zeros((states.shape[0], 1), device=device, dtype=torch.float32)

        forward_inputs = {
            "action": actions.reshape(actions.shape[0], -1).contiguous(),
            "model_action": action_mean.reshape(action_mean.shape[0], -1).contiguous(),
            "base_action": base_actions.reshape(base_actions.shape[0], -1).contiguous(),
            "states": states.contiguous(),
        }

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return actions, result

