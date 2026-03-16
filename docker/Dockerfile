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
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Project directory ----
ENV PROJECT_DIR=/workspace/isaac-bridge
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
    /isaac-sim/python.sh -m pip install --no-deps "lerobot[smolvla]==0.4.3" && \
    /isaac-sim/python.sh -m pip install \
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
        "num2words>=0.5.14,<0.6.0" \
        "transformers>=4.57.1,<5.0.0" && \
    /isaac-sim/python.sh -c "import torch; print(f'torch {torch.__version__} OK')" && \
    /isaac-sim/python.sh -c "import numpy; print(f'numpy {numpy.__version__} OK')" && \
    /isaac-sim/python.sh -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('SmolVLA OK')"

# ---- Install the isaac_so_arm101 extension ----
# Registers Gym environments (Isaac-SO-ARM101-Lift-Cube-v0, etc.)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd ${PROJECT_DIR}/isaac_so_arm101 && \
    /isaac-sim/python.sh -m pip install --no-deps --no-build-isolation -e . 2>/dev/null || \
    echo "Extension installed via PYTHONPATH (pip editable install not critical)"

# ---- Ensure extension is on PYTHONPATH ----
ENV PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/isaac_so_arm101/src:${PYTHONPATH}"

# ---- Convenience aliases ----
RUN echo "export PROJECT_DIR=${PROJECT_DIR}" >> /root/.bashrc && \
    echo "alias python='/isaac-sim/python.sh'" >> /root/.bashrc && \
    echo "alias isaaclab='/workspace/isaaclab/isaaclab.sh'" >> /root/.bashrc && \
    echo "alias run-smolvla='/workspace/isaaclab/isaaclab.sh -p ${PROJECT_DIR}/scripts/run_smolvla_isaac.py'" >> /root/.bashrc && \
    echo "alias save-cameras='/workspace/isaaclab/isaaclab.sh -p ${PROJECT_DIR}/scripts/save_env_cameras.py'" >> /root/.bashrc && \
    echo "cd ${PROJECT_DIR}" >> /root/.bashrc

# ---- Default entrypoint ----
ENTRYPOINT ["bash"]
