#!/usr/bin/env bash
set -euo pipefail

# Restore the OpenPI + LeIsaac pi0-FAST PPO/logprob work on a fresh instance.
#
# Usage:
#   OPENPI_DIR=/workspace/openpi \
#   MANIPULATION_DIR=/workspace/manipulation \
#   bash scripts/leisaac/scripts/training/openpi_ppo_restore/restore_openpi_ppo.sh
#
# Set SKIP_INSTALL=1 to only apply patches and skip Python/IsaacLab install steps.

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="${OPENPI_DIR:-/workspace/openpi}"
MANIPULATION_DIR="${MANIPULATION_DIR:-/workspace/manipulation}"
UV_BIN="${UV_BIN:-/root/.local/bin/uv}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${MANIPULATION_DIR}/scripts/leisaac/dependencies/IsaacLab}"
ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-/isaac-sim}"

apply_patch_once() {
  local repo_dir="$1"
  local patch_file="$2"
  local label="$3"

  if git -C "${repo_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
    echo "[restore] Applying ${label}: ${patch_file}"
    git -C "${repo_dir}" apply "${patch_file}"
    return
  fi

  if git -C "${repo_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "[restore] ${label} already appears to be applied; skipping."
    return
  fi

  echo "[restore] ERROR: ${label} patch does not apply cleanly."
  echo "          Repo:  ${repo_dir}"
  echo "          Patch: ${patch_file}"
  echo "          Inspect with: git -C ${repo_dir} apply --check ${patch_file}"
  exit 1
}

if [[ ! -d "${OPENPI_DIR}/.git" ]]; then
  echo "[restore] ERROR: OPENPI_DIR is not a git checkout: ${OPENPI_DIR}"
  exit 1
fi

if [[ ! -d "${MANIPULATION_DIR}/.git" ]]; then
  echo "[restore] ERROR: MANIPULATION_DIR is not a git checkout: ${MANIPULATION_DIR}"
  exit 1
fi

apply_patch_once "${OPENPI_DIR}" "${BUNDLE_DIR}/openpi_ppo_logprob.patch" "OpenPI PPO/logprob"
apply_patch_once "${MANIPULATION_DIR}" "${BUNDLE_DIR}/manipulation_leisaac_openpi_rollout.patch" "LeIsaac rollout"

if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
  echo "[restore] SKIP_INSTALL=1, done after applying patches."
  exit 0
fi

if [[ ! -x "${UV_BIN}" ]]; then
  echo "[restore] ERROR: uv not found at ${UV_BIN}."
  echo "          Install uv first or set UV_BIN to the correct path."
  exit 1
fi

echo "[restore] Installing OpenPI editable package with uv."
(
  cd "${OPENPI_DIR}"
  "${UV_BIN}" sync
  "${UV_BIN}" pip install -e .
)

if [[ -d "${ISAACLAB_DIR}" ]]; then
  echo "[restore] Ensuring /workspace/isaaclab points to IsaacLab."
  ln -sfn "${ISAACLAB_DIR}" /workspace/isaaclab

  if [[ -d "${ISAAC_SIM_DIR}" ]]; then
    echo "[restore] Ensuring IsaacLab _isaac_sim points to ${ISAAC_SIM_DIR}."
    ln -sfn "${ISAAC_SIM_DIR}" "${ISAACLAB_DIR}/_isaac_sim"
  else
    echo "[restore] WARNING: Isaac Sim directory not found at ${ISAAC_SIM_DIR}; skipping _isaac_sim link."
  fi

  echo "[restore] Installing IsaacLab/LeIsaac extensions into Isaac Sim Python."
  TERM=xterm "${ISAACLAB_DIR}/isaaclab.sh" -i all
else
  echo "[restore] WARNING: IsaacLab directory not found at ${ISAACLAB_DIR}; skipping IsaacLab install."
fi

echo "[restore] Done. See RESTORE_OPENPI_PPO.md for smoke commands."
