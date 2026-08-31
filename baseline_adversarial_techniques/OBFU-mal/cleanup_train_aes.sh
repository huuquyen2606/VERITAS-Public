#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[*] Removing generated Adversarial Examples (AEs) from training..."
if [ -d "artifacts/evaded" ]; then
    rm -rf artifacts/evaded
    echo " -> Removed artifacts/evaded/ (All generated AEs deleted)"
else
    echo " -> No artifacts/evaded/ directory found. Nothing to delete."
fi

echo "[*] Cleanup complete! The trained agent in artifacts/models/ is still kept."
