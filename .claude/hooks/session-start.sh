#!/bin/bash
# Installs the Python dependencies so tests / linters work in Claude Code cloud sessions.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

if ! python3 -c "import torch" 2>/dev/null; then
  # CPU wheels are much smaller; fall back to PyPI if download.pytorch.org is not reachable
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu || pip install -q torch
fi
pip install -q -r requirements-dev.txt
