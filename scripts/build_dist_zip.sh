#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="${1:-financial-flow-dist.zip}"

if command -v python3 >/dev/null 2>&1; then
    python3 "${SCRIPT_DIR}/build_dist_zip.py" "${TARGET}"
elif command -v python >/dev/null 2>&1; then
    python "${SCRIPT_DIR}/build_dist_zip.py" "${TARGET}"
else
    echo "Error: Python is required to build the distribution zip." >&2
    exit 1
fi
