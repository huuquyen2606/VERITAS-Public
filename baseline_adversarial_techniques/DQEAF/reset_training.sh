#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"

usage() {
    cat <<'EOF'
Usage:
  bash reset_training.sh
  bash reset_training.sh <family_name>

Examples:
  bash reset_training.sh
  bash reset_training.sh Zeroaccess

Behavior:
  - Always removes DQEAF __pycache__ folders
  - Always removes temporary /tmp/*.exe files created during PE mutation
  - With no family_name: removes outputs/dqeaf and outputs/*_dqeaf
  - With family_name: removes outputs/<family_name>_dqeaf only
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

TARGET_FAMILY="${1:-}"

echo "[*] Resetting DQEAF training artifacts..."
echo "[i] Project root: $ROOT_DIR"

find "$ROOT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
echo "[+] Removed __pycache__ directories."

rm -f /tmp/*.exe
echo "[+] Removed temporary /tmp/*.exe files."

mkdir -p "$OUTPUT_DIR"

if [[ -n "$TARGET_FAMILY" ]]; then
    TARGET_PATH="$OUTPUT_DIR/${TARGET_FAMILY}_dqeaf"
    if [[ -d "$TARGET_PATH" ]]; then
        rm -rf "$TARGET_PATH"
        echo "[+] Removed family training output: $TARGET_PATH"
    else
        echo "[-] No family training output found at: $TARGET_PATH"
    fi
else
    shopt -s nullglob
    targets=("$OUTPUT_DIR/dqeaf" "$OUTPUT_DIR"/*_dqeaf)
    shopt -u nullglob

    if [[ ${#targets[@]} -eq 0 ]]; then
        echo "[-] No training output directories found in: $OUTPUT_DIR"
    else
        rm -rf "${targets[@]}"
        echo "[+] Removed training output directories:"
        printf '    - %s\n' "${targets[@]}"
    fi
fi

echo "[*] Training reset complete."
