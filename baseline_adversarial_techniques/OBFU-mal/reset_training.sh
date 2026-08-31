#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[*] Removing run artifacts and caches..."
rm -rf artifacts/runs artifacts/evaded artifacts/models artifacts/reports
rm -rf .pytest_cache
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo "[*] Done."
