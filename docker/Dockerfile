# Isaac SO-ARM101 + LeRobot/SmolVLA Bridge
# Based on the official Isaac Lab container (Isaac Lab 2.3.x + Isaac Sim 5.1.0)
#
# Build:
#   cd docker && docker compose build
# Run (headless):
#   cd docker && docker compose run --rm isaac-bridge
# Run (with X11 GUI):
#   xhost +local:docker
#   cd docker && docker compose run --rm isaac-bridge-x11

ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

LABEL description="Isaac Lab SO-ARM101 + LeRobot/SmolVLA bridge environment"

# ---- System deps ----
USER root
# ffmpeg: shared libs (libavutil, libavcodec, …) required by torchcodec when LeRobot
# decodes dataset videos during lerobot-train (see lerobot/datasets/video_utils.py).
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Project directory ----
ENV PROJECT_DIR=/workspace/isaac-bridge
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_EXTRA_INDEX_URL=
ENV PIP_CONFIG_FILE=/dev/null
# Wheel links on mirrors often point at files.pythonhosted.org; trust both without using pypi.org as an index.
ENV PIP_TRUSTED_HOST="mirrors.aliyun.com files.pythonhosted.org download.pytorch.org"
# Pip-installed CLIs (huggingface-cli, hf, …) live next to Isaac Sim's Python.
ENV PATH="/isaac-sim/kit/python/bin:${PATH}"
# Force RLinf's isaac_sim/python.sh wrapper to use Isaac Sim's Python runtime.
ENV PYTHONEXE=/isaac-sim/kit/python/bin/python3
WORKDIR ${PROJECT_DIR}

# ---- Copy project files ----
COPY camera_poses.json ${PROJECT_DIR}/camera_poses.json
COPY camera_poses.example.json ${PROJECT_DIR}/camera_poses.example.json
COPY isaac_so_arm101/ ${PROJECT_DIR}/isaac_so_arm101/
COPY src/ ${PROJECT_DIR}/src/
COPY scripts/ ${PROJECT_DIR}/scripts/

# ---- Install Python deps ----
#
# lerobot has two deps that break Isaac Sim's Python environment:
#   - rerun-sdk>=0.24 requires numpy>=2 (Omniverse needs numpy 1.x)
#   - packaging>=24.2 corrupts pip._vendor.packaging (pip 24.3.1)
#
# Fix: install lerobot with --no-deps, then install its actual deps
# manually — skipping rerun-sdk (visualization only, not used in our
# Isaac Sim pipeline). This keeps numpy at 1.x and pip healthy.
# See docs/numpy-fix-notes.md and docs/torch-fix-notes.md for details.
#
RUN --mount=type=cache,target=/root/.cache/pip \
    /isaac-sim/python.sh -m pip uninstall -y torch 2>/dev/null ; \
    rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch && \
    /isaac-sim/python.sh -m pip install \
        "torch==2.7.0" --extra-index-url https://download.pytorch.org/whl/cu128 && \
    TORCH_DIR=$(/isaac-sim/python.sh -c "import torch, os; print(os.path.dirname(torch.__file__))") && \
    ln -s "$TORCH_DIR" /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch && \
    /isaac-sim/python.sh -m pip install --no-deps \
    "lerobot[smolvla,async] @ git+https://github.com/huggingface/lerobot.git@v0.4.3" && \
    /isaac-sim/python.sh -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "num2words>=0.5.14,<0.6.0" && \
    /isaac-sim/python.sh -m pip install \
        "typing_extensions>=4.12.0" \
        "requests>=2.31.0" \
        "datasets>=4.0.0,<4.2.0" \
        "diffusers>=0.27.2,<0.36.0" \
        "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0" \
        "accelerate>=1.10.0,<2.0.0" \
        "cmake>=3.29.0.1,<4.2.0" \
        "av>=15.0.0,<16.0.0" \
        "jsonlines>=4.0.0,<5.0.0" \
        "pynput>=1.7.7,<1.9.0" \
        "pyserial>=3.5,<4.0" \
        "wandb>=0.24.0,<0.25.0" \
        "torchcodec>=0.2.1,<0.6.0" \
        "draccus==0.10.0" \
        "gymnasium>=1.1.1,<2.0.0" \
        "deepdiff>=7.0.1,<9.0.0" \
        "imageio[ffmpeg]>=2.34.0,<3.0.0" \
        "transformers>=4.57.1,<5.0.0" && \
    # /isaac-sim/python.sh -m pip install "lerobot[async] @ git+https://github.com/huggingface/lerobot.git@v0.4.3" --no-deps && \
    # /isaac-sim/python.sh -m pip install 'leisaac[isaaclab,lerobot-async] @ git+https://github.com/LightwheelAI/leisaac.git#subdirectory=source/leisaac' --extra-index-url https://pypi.nvidia.com && \
    cd ${PROJECT_DIR}/scripts/leisaac/source/leisaac && \
    /isaac-sim/python.sh -m pip install -e . && \
    cd ${PROJECT_DIR} && \
    /isaac-sim/python.sh -c "import torch; print(f'torch {torch.__version__} OK')" && \
    /isaac-sim/python.sh -c "import numpy; print(f'numpy {numpy.__version__} OK')" && \
    /isaac-sim/python.sh -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('SmolVLA OK')" && \
    mkdir -p /isaac-sim/kit/python/lib/assets && \
    cp -r ${PROJECT_DIR}/scripts/leisaac/assets/* /isaac-sim/kit/python/lib/assets/

# ---- RLinf runtime deps ----
# RLinf source is mounted at runtime to /workspace/RLinf via docker-compose.
# Install the core Python dependencies here so RLinf configs/train entrypoints run
# without manual pip steps inside the container.
RUN --mount=type=cache,target=/root/.cache/pip \
    /isaac-sim/python.sh -m pip install \
      "omegaconf>=2.3.0,<2.5.0" \
      "hydra-core>=1.3.0,<1.5.0" \
      "ray[default]>=2.47.0" \
      "tensorboard>=2.15.0" \
      "scipy>=1.10.0" \
      "einops>=0.7.0" \
      "nvitop>=1.3.0" \
      "timm>=1.0.0" \
      "gymnasium>=0.29.0" \
      "gym==0.23.1"

# ---- Install the isaac_so_arm101 extension ----
# Registers Gym environments (Isaac-SO-ARM101-Lift-Cube-v0, etc.)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd ${PROJECT_DIR}/isaac_so_arm101 && \
    /isaac-sim/python.sh -m pip install --no-deps --no-build-isolation -e . 2>/dev/null || \
    echo "Extension installed via PYTHONPATH (pip editable install not critical)"

# ---- Ensure extension is on PYTHONPATH ----
ENV PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/isaac_so_arm101/src:/workspace/RLinf:${PYTHONPATH}"

# ---- Convenience aliases ----
RUN echo "export PROJECT_DIR=${PROJECT_DIR}" >> /root/.bashrc && \
    echo "alias python='/isaac-sim/python.sh'" >> /root/.bashrc && \
    echo "alias isaaclab='/workspace/isaaclab/isaaclab.sh'" >> /root/.bashrc && \
    echo "alias run-smolvla='/workspace/isaaclab/isaaclab.sh -p ${PROJECT_DIR}/scripts/run_smolvla_isaac.py'" >> /root/.bashrc && \
    echo "alias save-cameras='/workspace/isaaclab/isaaclab.sh -p ${PROJECT_DIR}/scripts/save_env_cameras.py'" >> /root/.bashrc && \
    echo "cd ${PROJECT_DIR}" >> /root/.bashrc

# ---- Default entrypoint ----
ENTRYPOINT ["bash"]
