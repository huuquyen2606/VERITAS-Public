#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/stop_eval_workers.sh \
    --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_1
EOF
}

LOG_DIR=""
WAIT_SEC=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --wait-sec)
      WAIT_SEC="$2"
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

if [[ -z "$LOG_DIR" ]]; then
  echo "[ERR] Missing --log-dir." >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="$PROJECT_ROOT/$LOG_DIR"
fi

PID_DIR="$LOG_DIR/pids"
if [[ ! -d "$PID_DIR" ]]; then
  echo "[WARN] PID directory not found: $PID_DIR"
  exit 0
fi

for pid_file in "$PID_DIR"/*.pid; do
  [[ -e "$pid_file" ]] || continue
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  name="$(basename "$pid_file" .pid)"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    continue
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "[*] Stopping $name pid=$pid"
    kill "$pid" 2>/dev/null || true
  else
    echo "[*] $name pid=$pid already stopped"
  fi
done

deadline=$((SECONDS + WAIT_SEC))
while [[ $SECONDS -lt $deadline ]]; do
  alive=0
  for pid_file in "$PID_DIR"/*.pid; do
    [[ -e "$pid_file" ]] || continue
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 1
done

for pid_file in "$PID_DIR"/*.pid; do
  [[ -e "$pid_file" ]] || continue
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[WARN] Force stopping pid=$pid from $pid_file"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
done

echo "[+] Stopped workers tracked by $PID_DIR"
