#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/start_eval_workers.sh \
    --name shard_1 \
    --redis-db 1 \
    --cape-config assets/cape_nodes_eval_shard_1.json \
    --work-dir /tmp/veritas_eval_shard_1 \
    --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_1
EOF
}

NAME=""
REDIS_DB=""
CAPE_CONFIG=""
WORK_DIR_ARG=""
LOG_DIR=""
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="$2"
      shift 2
      ;;
    --redis-db)
      REDIS_DB="$2"
      shift 2
      ;;
    --cape-config)
      CAPE_CONFIG="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR_ARG="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$NAME" || -z "$REDIS_DB" || -z "$CAPE_CONFIG" || -z "$WORK_DIR_ARG" || -z "$LOG_DIR" ]]; then
  echo "[ERR] Missing required argument." >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ADV_ENV_ROOT="$PROJECT_ROOT/env/adv_RL_env"
ADV_ENV_PYTHON="${ADV_ENV_PYTHON:-$PROJECT_ROOT/venv/bin/python}"

if [[ "$CAPE_CONFIG" != /* ]]; then
  CAPE_CONFIG="$PROJECT_ROOT/$CAPE_CONFIG"
fi
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="$PROJECT_ROOT/$LOG_DIR"
fi

if [[ ! -f "$CAPE_CONFIG" ]]; then
  echo "[ERR] CAPE config not found: $CAPE_CONFIG" >&2
  exit 1
fi
if [[ ! -x "$ADV_ENV_PYTHON" ]]; then
  echo "[ERR] Worker Python not executable: $ADV_ENV_PYTHON" >&2
  exit 1
fi

mkdir -p "$WORK_DIR_ARG" "$LOG_DIR/pids"

export PYTHONPATH="$ADV_ENV_ROOT:$ADV_ENV_ROOT/malware_rl_system"
export REDIS_HOST="$REDIS_HOST"
export REDIS_PORT="$REDIS_PORT"
export REDIS_DB="$REDIS_DB"
export CAPE_NODES_CONFIG="$CAPE_CONFIG"
export WORK_DIR="$WORK_DIR_ARG"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CAPE_SUBMIT_JITTER_MIN="${CAPE_SUBMIT_JITTER_MIN:-3}"
export CAPE_SUBMIT_JITTER_MAX="${CAPE_SUBMIT_JITTER_MAX:-8}"
export CAPE_MAX_RETRIES="${CAPE_MAX_RETRIES:-5}"
export CAPE_BACKOFF_BASE_SEC="${CAPE_BACKOFF_BASE_SEC:-60}"
export CAPE_BACKOFF_MAX_SEC="${CAPE_BACKOFF_MAX_SEC:-300}"
export ANGR_TIMEOUT_SEC="${ANGR_TIMEOUT_SEC:-300}"
export ANGR_CFG_TIMEOUT_SEC="${ANGR_CFG_TIMEOUT_SEC:-300}"

start_worker() {
  local module="$1"
  local short_name="$2"
  local pid_file="$LOG_DIR/pids/${short_name}.pid"
  local log_file="$LOG_DIR/${short_name}.log"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[ERR] $short_name already running for $NAME with PID $old_pid ($pid_file)." >&2
      exit 1
    fi
  fi

  nohup "$ADV_ENV_PYTHON" -m "$module" > "$log_file" 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "[+] $NAME started $short_name pid=$pid log=$log_file"
}

echo "[*] Starting eval workers for $NAME"
echo "    redis=${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}"
echo "    work_dir=$WORK_DIR_ARG"
echo "    cape_config=$CAPE_CONFIG"
echo "    log_dir=$LOG_DIR"

start_worker "malware_rl_system.workers.worker_api" "worker_api"
start_worker "malware_rl_system.workers.worker_syscall" "worker_syscall"
start_worker "malware_rl_system.workers.worker_static" "worker_static"
start_worker "malware_rl_system.workers.worker_angr" "worker_angr"

echo "[+] Worker pool $NAME started."
