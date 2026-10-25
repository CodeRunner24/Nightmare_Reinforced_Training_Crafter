#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [[ ! -x .tools/uv ]]; then
  mkdir -p .tools
  curl -L --fail --progress-bar https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-unknown-linux-gnu.tar.gz -o .tools/uv.tar.gz
  tar -xzf .tools/uv.tar.gz -C .tools --strip-components=1
fi
if [[ ! -x .venv-wsl/bin/python ]]; then
  .tools/uv venv --python 3.11 --seed .venv-wsl
fi
if [[ -s setup/requirements-crafter-rtx50.lock.txt ]]; then
  .tools/uv pip sync --python .venv-wsl/bin/python --link-mode copy setup/requirements-crafter-rtx50.lock.txt
else
  .tools/uv pip install --python .venv-wsl/bin/python --link-mode copy -r setup/requirements-crafter-rtx50.txt
fi
.tools/uv pip check --python .venv-wsl/bin/python
# Installation only. No training or game execution.
