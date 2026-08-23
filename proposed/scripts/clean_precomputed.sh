#!/bin/bash
# clean_precomputed.sh — Wipe ALL precompute outputs (CT + CR, train + test)
# before a clean re-run. Keeps the directory skeleton so the xargs commands in
# README run without re-creating subdirs.
#
# Does NOT touch:
#   - data/dataset/                 (input PE files)
#   - data/benign_content/          (benign_bank.pkl)
#   - data/precomputed/_legacy/     (archived legacy outputs)
#   - tools/obfuscator_llvm_14/     (OLLVM build)
#   - venv/, scripts/, env/, agent/, actions/, ...
#
# Usage:
#   bash scripts/clean_precomputed.sh           # interactive (prompts y/N)
#   bash scripts/clean_precomputed.sh --yes     # non-interactive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE="$ROOT/data/precomputed"
LOGS="$ROOT/logs"

ASSUME_YES=0
[[ "${1:-}" == "--yes" ]] && ASSUME_YES=1

print_state() {
    echo "===== $1 ====="
    for phase in train test; do
        for action in code-translation code-randomize; do
            for kind in exe json; do
                d="$BASE/$phase/$action/$kind"
                if [ -d "$d" ]; then
                    cnt=$(find "$d" -mindepth 1 -maxdepth 1 -type f 2>/dev/null | wc -l)
                else
                    cnt="(dir missing)"
                fi
                printf "  %-50s %s\n" "$d" "$cnt files"
            done
        done
    done
    # Tmpfs work dirs (CT only — CR runs in-memory).
    for d in /dev/shm/ct_work_train /dev/shm/ct_work_test; do
        if [ -d "$d" ]; then
            cnt=$(find "$d" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
            printf "  %-50s %s\n" "$d" "$cnt subdirs"
        fi
    done
    if [ -d "$LOGS" ]; then
        for f in ct_train.log ct_test.log cr_train.log cr_test.log; do
            p="$LOGS/$f"
            if [ -f "$p" ]; then
                size=$(du -h "$p" | cut -f1)
                printf "  %-50s %s\n" "$p" "$size"
            fi
        done
    fi
}

print_state "Pre-clean state"
echo ""
echo "Will DELETE all files in:"
echo "  $BASE/{train,test}/{code-translation,code-randomize}/{exe,json}/"
echo "  /dev/shm/ct_work_train  /dev/shm/ct_work_test"
echo "  $LOGS/{ct,cr}_{train,test}.log"
echo ""

if [ "$ASSUME_YES" -eq 0 ]; then
    read -rp "Proceed? [y/N] " ans
    case "${ans,,}" in
        y|yes) ;;
        *) echo "aborted, no files deleted"; exit 1 ;;
    esac
fi

# 1. Wipe precompute outputs (keep dir skeleton).
for phase in train test; do
    for action in code-translation code-randomize; do
        for kind in exe json; do
            d="$BASE/$phase/$action/$kind"
            mkdir -p "$d"
            # find/-delete is safer than rm -rf "$d/*" against weird filenames.
            find "$d" -mindepth 1 -delete 2>/dev/null || true
        done
    done
done

# 2. Wipe tmpfs work dirs (CT pipeline writes per-PID subdirs there).
rm -rf /dev/shm/ct_work_train /dev/shm/ct_work_test 2>/dev/null || true

# 3. Wipe logs.
for f in ct_train.log ct_test.log cr_train.log cr_test.log; do
    rm -f "$LOGS/$f"
done

echo ""
print_state "Post-clean state"
echo ""
echo "Clean done. Re-run with the four tmux sessions in README:"
echo "  Phase 1 (CT, ~1.5h): tmux new -s ct-train  /  tmux new -s ct-test"
echo "  Phase 2 (CR,  ~30m): tmux new -s cr-train  /  tmux new -s cr-test"
