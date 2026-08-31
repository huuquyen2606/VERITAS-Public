#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/samples/rl/adversarial_examples/train"
DRY_RUN=0
ASSUME_YES=0

usage() {
  cat <<'USAGE'
Usage: clean_aimedrl_train_aes.sh [--dry-run] [--yes]

Deletes only adversarial examples generated during AIMED-RL training.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    --yes)
      ASSUME_YES=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "No train AE directory found at: $TARGET_DIR"
  exit 0
fi

mapfile -t ENTRIES < <(find "$TARGET_DIR" -mindepth 1 | sort)
if [[ ${#ENTRIES[@]} -eq 0 ]]; then
  echo "Train AE directory is already clean: $TARGET_DIR"
  exit 0
fi

echo "The following train AE artifacts will be removed:"
printf '  %s\n' "${ENTRIES[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Nothing was deleted."
  exit 0
fi

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Delete these AIMED-RL training AEs? [y/N] " REPLY
  if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

find "$TARGET_DIR" -mindepth 1 -depth -exec rm -rf {} +
mkdir -p "$TARGET_DIR"

echo "Removed AIMED-RL training AEs from: $TARGET_DIR"
