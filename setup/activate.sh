#!/usr/bin/env bash
# Source this file from WSL Ubuntu: source setup/activate.sh
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$TASK_ROOT/.venv-wsl/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
cd "$TASK_ROOT"
