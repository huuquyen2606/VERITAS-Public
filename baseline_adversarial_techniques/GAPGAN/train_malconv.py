import os
import argparse
import shutil

import torch

from binary_malconv import BinaryMalConv


TRAIN_PATHS = ["Adv_detector"]
VAL_PATHS = ["Test"]
EXPERIMENT_NAME = "malconv_training"
BEST_MODEL_PATH = f"predic_models/{EXPERIMENT_NAME}/best_model.pth"
EXPORT_MODEL_PATH = "output_malconv/malconv.pth"


def assert_required_paths(paths):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing required dataset paths: {', '.join(missing)}"
        )


def summarize_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint format mismatch: expected wrapped checkpoint with 'model_state_dict'."
        )

    print(f"[*] Checkpoint verified: {checkpoint_path}")
    print(f"[*] Checkpoint keys: {sorted(checkpoint.keys())}")

    model_cfg = checkpoint.get("model_cfg", {})
    train_cfg = checkpoint.get("train_cfg", {})
    if model_cfg:
        print(
            "[*] model_cfg:",
            {
                "max_len": model_cfg.get("max_len"),
                "channels": model_cfg.get("channels"),
                "window_size": model_cfg.get("window_size"),
                "embedding_dim": model_cfg.get("embedding_dim"),
            },
        )
    if train_cfg:
        print(
            "[*] train_cfg:",
            {
                "batch_size": train_cfg.get("batch_size"),
                "accum_steps": train_cfg.get("accum_steps"),
                "learning_rate": train_cfg.get("learning_rate"),
                "padding_byte": train_cfg.get(
                    "padding_byte", train_cfg.get("padding_char")
                ),
            },
        )


def main():
    parser = argparse.ArgumentParser(description="Train Binary MalConv Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--accum_steps", type=int, default=64, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    model = BinaryMalConv()
    assert_required_paths(TRAIN_PATHS + VAL_PATHS)

    # Overrides based on user arguments
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "learning_rate": args.lr,
        "padding_byte": 0,
    }

    # Datasets
    train_paths = TRAIN_PATHS
    val_paths = VAL_PATHS

    # Label mapping isn't strictly needed as binary_malconv auto-labels based on 'benign' folder/file substring
    label_map = {"Benign": 0, "Malware": 1}

    print("[*] Starting MalConv training...")
    print(f"[*] Train paths: {train_paths}")
    print(f"[*] Val paths: {val_paths}")
    print("[*] Paper-truth mode: zero-padding raw bytes and wrapped checkpoint export enabled.")

    model.train(
        train_paths=train_paths,
        val_paths=val_paths,
        label_mapping=label_map,
        experiment_name=EXPERIMENT_NAME,
        overrides=overrides
    )

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Training finished but checkpoint not found at {BEST_MODEL_PATH}"
        )

    summarize_checkpoint(BEST_MODEL_PATH)

    print(f"[*] Training completed! Weights saved to {BEST_MODEL_PATH}")

    # Move to requested output_malconv dir
    os.makedirs("output_malconv", exist_ok=True)
    try:
        shutil.copy2(BEST_MODEL_PATH, EXPORT_MODEL_PATH)
        print(f"[*] Model successfully copied to {EXPORT_MODEL_PATH}")
    except Exception as e:
        print(f"[!] Warning: Could not copy model to output_malconv: {e}")

if __name__ == "__main__":
    main()
