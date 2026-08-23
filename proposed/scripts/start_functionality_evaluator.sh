#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

FUNC_PYTHON="${FUNC_PYTHON:-$PROJECT_ROOT/env/adv_RL_env/func_venv/bin/python}"
ADV_REPO_ROOT="$PROJECT_ROOT/env/adv_RL_env"
ADV_SYSTEM_ROOT="$ADV_REPO_ROOT/malware_rl_system"

if [ ! -x "$FUNC_PYTHON" ]; then
  echo "[!] Functionality venv Python not found: $FUNC_PYTHON"
  echo "    Create it with:"
  echo "    python3 -m venv env/adv_RL_env/func_venv"
  echo "    env/adv_RL_env/func_venv/bin/python -m pip install -r env/adv_RL_env/requirements_functionality.txt"
  echo "    env/adv_RL_env/func_venv/bin/python -m pip install ./env/adv_RL_env/examples/comparing_apis"
  exit 1
fi

export PYTHONPATH="$ADV_REPO_ROOT:$ADV_SYSTEM_ROOT:${PYTHONPATH:-}"

echo "[*] Starting FunctionalityEvaluator with: $FUNC_PYTHON"
echo "[*] PYTHONPATH=$PYTHONPATH"

exec env PYTHONUNBUFFERED=1 "$FUNC_PYTHON" -c "from evaluation.functionality_evaluator import FunctionalityEvaluator; FunctionalityEvaluator().run()"
