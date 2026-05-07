#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/RLinf" >&2
  exit 2
fi

RLINF_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_DIR="${SCRIPT_DIR}/overlays/rlinf"

if [[ ! -d "${RLINF_DIR}" ]]; then
  echo "RLinf directory does not exist: ${RLINF_DIR}" >&2
  exit 1
fi

if [[ ! -f "${RLINF_DIR}/pyproject.toml" || ! -d "${RLINF_DIR}/rlinf" ]]; then
  echo "Target does not look like an RLinf checkout: ${RLINF_DIR}" >&2
  exit 1
fi

cp -a "${OVERLAY_DIR}/." "${RLINF_DIR}/"
echo "Applied SO-101 RLinf overlay to ${RLINF_DIR}"
