#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[*] Resetting evaluation artifacts only..."

shopt -s nullglob

# Evaluate reports
eval_reports=(
  artifacts/runs/eval_history_*.json
  artifacts/runs/eval_summary_*.json
)

if ((${#eval_reports[@]} > 0)); then
  rm -f "${eval_reports[@]}"
  echo "[*] Removed ${#eval_reports[@]} eval report file(s)."
else
  echo "[*] No eval report files found."
fi

# Optional eval-only cache/output folders (safe to remove if present)
if [[ -d artifacts/cache/eval ]]; then
  rm -rf artifacts/cache/eval
  echo "[*] Removed artifacts/cache/eval"
fi

if [[ -d artifacts/evaded/eval ]]; then
  find artifacts/evaded/eval -type f -delete
  echo "[*] Cleared artifacts/evaded/eval"
fi

echo "[*] Done. Training artifacts were not touched."
