#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
ASSUME_YES=0

DIR_TARGETS=(
  "$ROOT_DIR/samples/rl/agent"
  "$ROOT_DIR/samples/rl/adversarial_examples"
  "$ROOT_DIR/db/rl/training_reports"
  "$ROOT_DIR/db/rl/evaluating_reports"
)

RECREATE_DIRS=(
  "$ROOT_DIR/samples/rl"
  "$ROOT_DIR/samples/rl/evaluation_set"
  "$ROOT_DIR/samples/rl/agent"
  "$ROOT_DIR/samples/rl/adversarial_examples"
  "$ROOT_DIR/db/rl"
  "$ROOT_DIR/samples/mod"
)

usage() {
  cat <<'USAGE'
Usage: reset_aimedrl_artifacts.sh [--dry-run] [--yes]

Deletes AIMED-RL train/eval artifacts while preserving datasets and repo code.
Targets:
  - samples/rl/agent
  - samples/rl/adversarial_examples
  - db/rl/training_reports
  - db/rl/evaluating_reports
  - samples/mod/*_m.exe
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

TO_DELETE=()
for path in "${DIR_TARGETS[@]}"; do
  if [[ -e "$path" ]]; then
    TO_DELETE+=("$path")
  fi
done

if [[ -d "$ROOT_DIR/samples/mod" ]]; then
  while IFS= read -r file; do
    TO_DELETE+=("$file")
  done < <(find "$ROOT_DIR/samples/mod" -maxdepth 1 -type f -name '*_m.exe' | sort)
fi

if [[ ${#TO_DELETE[@]} -eq 0 ]]; then
  echo "No AIMED-RL artifacts found to reset."
  exit 0
fi

echo "The following AIMED-RL artifacts will be removed:"
printf '  %s\n' "${TO_DELETE[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Nothing was deleted."
  exit 0
fi

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Reset all AIMED-RL train/eval artifacts listed above? [y/N] " REPLY
  if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

for path in "${TO_DELETE[@]}"; do
  rm -rf "$path"
done

for path in "${RECREATE_DIRS[@]}"; do
  mkdir -p "$path"
done

echo "AIMED-RL artifacts were reset successfully."
