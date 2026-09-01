#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[*] Resetting GAPGAN training artifacts under: $ROOT_DIR..."

# Remove Checkpoints and learned weights
rm -rf "$ROOT_DIR/checkpoints/generator_"*.pth
rm -rf "$ROOT_DIR/checkpoints/discriminator_"*.pth

# Remove Exported MalConv weights and run folders
rm -rf "$ROOT_DIR/output_malconv/malconv.pth"
rm -rf "$ROOT_DIR/predic_models/malconv_training"

# Remove Attack outputs
rm -rf "$ROOT_DIR/adversarial_samples"
rm -rf "$ROOT_DIR/adversarial_samples_output"

# Remove all logs in relevant directories
rm -rf "$ROOT_DIR"/*.log "$ROOT_DIR/checkpoints/"*.log "$ROOT_DIR/output_malconv/"*.log

echo "[*] Reset complete!"
echo "[*] Datasets such as Adv_agent, Adv_detector, Test were left untouched."