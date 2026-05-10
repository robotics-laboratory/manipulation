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

from omegaconf import DictConfig

from rlinf.models.embodiment.smolvla.smolvla_action_model import (
    SmolVLAActionModel,
    SmolVLAConfig,
)


def get_model(cfg: DictConfig, torch_dtype=None):
    model_cfg = SmolVLAConfig()

    for key, value in cfg.items():
        if key == "smolvla":
            continue
        if hasattr(model_cfg, key):
            setattr(model_cfg, key, value)

    smolvla_cfg = getattr(cfg, "smolvla", None)
    if smolvla_cfg is not None:
        for key, value in smolvla_cfg.items():
            if hasattr(model_cfg, key):
                setattr(model_cfg, key, value)

    model = SmolVLAActionModel(model_cfg=model_cfg)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    return model

