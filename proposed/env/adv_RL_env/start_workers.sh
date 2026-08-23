#!/bin/bash

echo "[*] Killing old worker processes..."
pkill -f "worker_api"
pkill -f "worker_syscall"
pkill -f "worker_static"
pkill -f "worker_angr"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export CAPE_NODES_CONFIG="$PROJECT_ROOT/assets/cape_nodes.json"
ADV_ENV_PYTHON="${ADV_ENV_PYTHON:-$PROJECT_ROOT/venv/bin/python}"

echo "[*] Using CAPE config: $CAPE_NODES_CONFIG"
echo "[*] Using worker Python: $ADV_ENV_PYTHON"

echo "[*] Starting CAPE API Worker..."
env PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/malware_rl_system" nohup "$ADV_ENV_PYTHON" -m malware_rl_system.workers.worker_api > worker_api.log 2>&1 &

echo "[*] Starting Syscall/Frida Worker..."
env PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/malware_rl_system" nohup "$ADV_ENV_PYTHON" -m malware_rl_system.workers.worker_syscall > worker_syscall.log 2>&1 &

echo "[*] Starting Static Worker..."
env PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/malware_rl_system" nohup "$ADV_ENV_PYTHON" -m malware_rl_system.workers.worker_static > worker_static.log 2>&1 &

echo "[*] Starting Angr Worker..."
env PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/malware_rl_system" nohup "$ADV_ENV_PYTHON" -m malware_rl_system.workers.worker_angr > worker_angr.log 2>&1 &

echo "[+] All 4 workers successfully started in the background!"
echo "    You can check their status using: tail -f worker_api.log"
