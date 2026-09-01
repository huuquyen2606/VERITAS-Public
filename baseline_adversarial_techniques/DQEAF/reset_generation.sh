#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash reset_generation.sh
  bash reset_generation.sh <family_name>

Examples:
  bash reset_generation.sh
  bash reset_generation.sh Zeroaccess

Behavior:
  - With no family_name: removes evaded_malware, evaded_malware_debug,
    evaded_malware_debug2, and Evaded_Malware, then recreates evaded_malware
  - With family_name: removes evaded_malware/<family_name> only
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

TARGET_FAMILY="${1:-}"

echo "[*] Resetting DQEAF generation artifacts..."
echo "[i] Project root: $ROOT_DIR"

if [[ -n "$TARGET_FAMILY" ]]; then
    TARGET_PATH="$ROOT_DIR/evaded_malware/$TARGET_FAMILY"
    if [[ -d "$TARGET_PATH" ]]; then
        rm -rf "$TARGET_PATH"
        echo "[+] Removed family AE directory: $TARGET_PATH"
    else
        echo "[-] No family AE directory found at: $TARGET_PATH"
    fi
else
    shopt -s nullglob
    targets=(
        "$ROOT_DIR/evaded_malware"
        "$ROOT_DIR/evaded_malware_debug"
        "$ROOT_DIR/evaded_malware_debug2"
        "$ROOT_DIR/Evaded_Malware"
    )
    shopt -u nullglob

    existing_targets=()
    for path in "${targets[@]}"; do
        if [[ -e "$path" ]]; then
            existing_targets+=("$path")
        fi
    done

    if [[ ${#existing_targets[@]} -eq 0 ]]; then
        echo "[-] No generation output directories found."
    else
        rm -rf "${existing_targets[@]}"
        echo "[+] Removed generation output directories:"
        printf '    - %s\n' "${existing_targets[@]}"
    fi

    mkdir -p "$ROOT_DIR/evaded_malware"
    echo "[+] Recreated empty directory: $ROOT_DIR/evaded_malware"
fi

echo "[*] Generation reset complete."
