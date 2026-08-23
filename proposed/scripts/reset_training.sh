#!/bin/bash
# reset_training.sh — remove artifacts from the previous VERITAS training run.
#
# This script only resets train outputs. It does NOT touch:
#   - data/dataset/
#   - data/benign_content/
#   - data/precomputed/
#   - env/adv_RL_env/
#   - model/detector weights
#
# Usage:
#   bash scripts/reset_training.sh            # dry-run, prints targets only
#   bash scripts/reset_training.sh --yes      # delete and recreate dirs
#   bash scripts/reset_training.sh --config configs/train.yaml --yes
#   bash scripts/reset_training.sh --yes --keep-tee-logs --keep-debug-json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="$ROOT/configs/train.yaml"
ASSUME_YES=0
KEEP_TEE_LOGS=0
KEEP_DEBUG_JSON=0

usage() {
  sed -n '1,18p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)
      ASSUME_YES=1
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --keep-tee-logs)
      KEEP_TEE_LOGS=1
      shift
      ;;
    --keep-debug-json)
      KEEP_DEBUG_JSON=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[!] Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$ROOT/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[!] Config not found: $CONFIG"
  exit 1
fi

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

mapfile -t TARGET_LINES < <(
  ROOT="$ROOT" CONFIG="$CONFIG" "$PYTHON" - <<'PY'
import os
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Run from the project venv or install PyYAML.") from exc

root = Path(os.environ["ROOT"]).resolve()
config = Path(os.environ["CONFIG"]).resolve()
cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}

def norm(value):
    if value in (None, "", "null"):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()

train = cfg.get("train", {}) or {}
sandbox = cfg.get("sandbox", {}) or {}

targets = [
    ("checkpoint_dir", norm(train.get("checkpoint_dir", "models/checkpoints/run_001"))),
    ("log_dir", norm(train.get("log_dir", "logs/runs/run_001"))),
    ("evasive_folder", norm(train.get("evasive_folder", "outputs/evasive/run_001"))),
    ("debug_json_dir", norm(sandbox.get("save_json_folder", "outputs/cuckoo_json/"))),
]

for label, path in targets:
    if path is not None:
        print(f"{label}\t{path}")
PY
)

is_under_root() {
  local p="$1"
  [[ "$p" == "$ROOT/"* ]]
}

print_path_state() {
  local label="$1"
  local p="$2"
  if [[ -e "$p" ]]; then
    local size
    size="$(du -sh "$p" 2>/dev/null | awk '{print $1}')"
    printf "  %-16s %-8s %s\n" "$label" "$size" "$p"
  else
    printf "  %-16s %-8s %s\n" "$label" "missing" "$p"
  fi
}

echo "Project root: $ROOT"
echo "Config      : $CONFIG"
echo ""
echo "Training reset targets:"

for line in "${TARGET_LINES[@]}"; do
  label="${line%%$'\t'*}"
  path="${line#*$'\t'}"
  if [[ "$KEEP_DEBUG_JSON" -eq 1 && "$label" == "debug_json_dir" ]]; then
    continue
  fi
  if ! is_under_root "$path"; then
    echo "[!] Refusing target outside project root: $path"
    exit 1
  fi
  print_path_state "$label" "$path"
done

if [[ "$KEEP_TEE_LOGS" -eq 0 ]]; then
  echo ""
  echo "Top-level train tee logs:"
  if compgen -G "$ROOT/logs/train*.log" >/dev/null; then
    for f in "$ROOT"/logs/train*.log; do
      print_path_state "tee_log" "$f"
    done
  else
    echo "  none"
  fi
fi

echo ""
if [[ "$ASSUME_YES" -ne 1 ]]; then
  echo "Dry run only. Re-run with --yes to delete these training artifacts."
  exit 0
fi

echo "[*] Deleting training artifacts..."

for line in "${TARGET_LINES[@]}"; do
  label="${line%%$'\t'*}"
  path="${line#*$'\t'}"
  if [[ "$KEEP_DEBUG_JSON" -eq 1 && "$label" == "debug_json_dir" ]]; then
    continue
  fi
  if ! is_under_root "$path"; then
    echo "[!] Refusing target outside project root: $path"
    exit 1
  fi
  rm -rf "$path"
  mkdir -p "$path"
done

if [[ "$KEEP_TEE_LOGS" -eq 0 ]]; then
  mkdir -p "$ROOT/logs"
  find "$ROOT/logs" -maxdepth 1 -type f -name 'train*.log' -delete
fi

echo "[+] Reset done. You can start a fresh training run now."
