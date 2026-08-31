import os
import json
import argparse

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from malgpt import MalGPTModel


def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Automated MalGPT Workflow Runner")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()

    # 2. Load JSON Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(args.config, "r") as f:
        config = json.load(f)

    command = config.get("command", "").lower()
    models_to_run = config.get("models", [])
    run_name = config.get("name", "Default_Run")

    print("=" * 60)
    print(f"[*] Starting Automation Workflow: {run_name}")
    print(f"[*] Command: {command.upper()}")
    print("=" * 60)

    # We only process if "malgpt" is in the models list
    if "malgpt" not in [m.lower() for m in models_to_run]:
        print("[-] MalGPT not found in 'models' list. Exiting.")
        return

    # Initialize the MalGPT Model
    malgpt_model = MalGPTModel()

    # 3. Route to the appropriate function
    if command == "train":
        benign_dataset_path = config.get("dataset")
        # UPGRADE: Grab the save_path from JSON
        save_path = config.get("save_path")

        if not benign_dataset_path or not os.path.exists(benign_dataset_path):
            raise ValueError(
                "Valid 'dataset' path (benign .npz) is required for training."
            )

        print(f"[*] Initiating Training Pipeline...")
        malgpt_model.train(
            benign_npz_path=benign_dataset_path,
            experiment_name=run_name,
            save_path=save_path,  # Pass it here!
        )
        print("[+] Training completed successfully.")

    elif command == "generate":
        target_dataset_path = config.get("dataset")
        output_dir = config.get("output_dir", "./Functional_Adv_Samples")
        # UPGRADE: Grab the load_path from JSON
        load_path = config.get("load_path")
        vt_api_key = config.get("vt_api_key") or os.environ.get("VT_API_KEY")

        if not target_dataset_path or not os.path.exists(target_dataset_path):
            raise ValueError(
                "Valid 'dataset' path (malware .npz) is required for generation."
            )

        print(f"[*] Loading trained weights from {load_path or 'default directory'}...")
        malgpt_model.load_weights(weights_path=load_path)  # Pass it here!

        print(f"[*] Initiating Generation Pipeline...")
        malgpt_model.generate_functional_samples(
            target_npz_path=target_dataset_path,
            output_dir=output_dir,
            vt_api_key=vt_api_key,
        )

    else:
        print(
            f"[-] Unknown command: {command}. Supported commands are 'train' and 'generate'."
        )


if __name__ == "__main__":
    main()
