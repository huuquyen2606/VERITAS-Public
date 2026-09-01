import argparse
import json
import os
import sys
import resource


# ==========================================
# 0. GLOBAL RESOURCE CONFIGURATION
# ==========================================
import psutil
import multiprocessing

try:
    total_ram_gb = max(16, int(psutil.virtual_memory().total / (1024**3)) - 2)
except:
    total_ram_gb = 30
try:
    total_cores = max(4, multiprocessing.cpu_count() - 1)
except:
    total_cores = 4

RESOURCE_CONFIG = {
    "max_cpu_ram_gb": total_ram_gb,
    "max_cpu_cores": total_cores,
    "force_cpu_mode": False,
    "gpu_vram_limit": 1.0,
}


def apply_resource_limits():
    """Applies the safety settings defined in RESOURCE_CONFIG."""
    is_main = multiprocessing.current_process().name == 'MainProcess'
    
    if is_main:
        print("\n" + "=" * 40)
        print("      RESOURCE SAFETY CONTROLS")
        print("=" * 40)

    cores = str(RESOURCE_CONFIG["max_cpu_cores"])
    os.environ["OMP_NUM_THREADS"] = cores
    os.environ["MKL_NUM_THREADS"] = cores
    os.environ["TORCH_NUM_THREADS"] = cores
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    if is_main:
        print(f"[*] CPU Core Limit: {cores} Cores (Active)")

    # Removed resource.setrlimit(resource.RLIMIT_AS) because PyTorch CUDA allocator 
    # requires massive virtual memory space (VSZ) for memory mapping, which causes 
    # artificial 'OS error 12' and 'CUDA out of memory' errors even when physical RAM/VRAM is free.

    if RESOURCE_CONFIG["force_cpu_mode"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if is_main:
            print("[*] GPU Mode: DISABLED (Forced CPU-only)")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                fraction = RESOURCE_CONFIG["gpu_vram_limit"]
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                torch.backends.cudnn.enabled = False
                if is_main:
                    print(f"[*] GPU Mode: ENABLED (cuDNN Bypass Active)")
                    print(f"[*] GPU VRAM Limit: {fraction * 100}%")
            else:
                if is_main:
                    print("[!] GPU Mode: Requested, but no GPU found.")
        except Exception:
            pass

    if is_main:
        print("=" * 40 + "\n")

# APPLY LIMITS IMMEDIATELY
apply_resource_limits()


from baseline_detectors import MODEL_REGISTRY
from baseline_detectors.ensemble import EnsembleManager
from baseline_detectors.utils import print_evasion_summary


def load_json(path):
    if not os.path.exists(path):
        print(f"[!] Error: File not found at {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def run_from_config(config):
    """
    Main logic that decides between SINGLE mode and ENSEMBLE mode.
    """
    mode = "single"
    manager = None
    model_instance = None
    model_name = None  # Needed for single model reporting

    # ==========================================
    # 1. DETECT MODE (Single vs Ensemble)
    # ==========================================

    # CHECK A: Is it a list of models? -> ENSEMBLE MODE
    if "models" in config and isinstance(config["models"], list):
        mode = "ensemble"
        model_names = config["models"]
        weights_map = config.get("weights", {})
        try:
            manager = EnsembleManager(model_names, weights_map)
        except Exception as e:
            print(f"[!] Error initializing ensemble: {e}")
            return

    # CHECK B: Is it a single model string? -> SINGLE MODE
    elif "model" in config:
        mode = "single"
        model_name = config["model"]
        if model_name not in MODEL_REGISTRY:
            print(f"[!] Error: Model '{model_name}' not found.")
            return

        ModelClass = MODEL_REGISTRY[model_name]
        model_instance = ModelClass()
        print(f"[*] Initialized Single Model: {model_name}")

    else:
        print("[!] Error: Config must specify 'model' (string) or 'models' (list).")
        return

    command = config.get("command")

    if manager is None and model_instance is None:
        print("[!] Error: No model or ensemble could be initialized.")
        return

    # ==========================================
    # 2. EXECUTE COMMAND
    # ==========================================

    if command == "train":
        # A. LOAD MAPPING (With Global Int Fix)
        map_path = config.get("map")
        if not map_path:
            print("[!] Error: Training requires '--map'.")
            return

        raw_map = load_json(map_path)

        # GLOBAL FIEXD Force convert all values to Integer
        label_map = {}
        try:
            for k, v in raw_map.items():
                label_map[k] = int(v)
        except ValueError:
            print(
                f"[!] Critical Error: Label mapping values must be convertible to integers."
            )
            print(f"    Found invalid value in map: {raw_map}")
            return

        print(f"[*] Loaded Label Mapping: {label_map}")

        # B. LOAD DATASET CONFIG (Fixed: This was missing in your code!)
        train_datasets = config.get("dataset")
        if not train_datasets:
            print(
                "[!] Error: Training requires 'dataset' path (e.g., path/to/Train_full.npz)."
            )
            return

        if isinstance(train_datasets, str):
            train_datasets = [train_datasets]

        val_datasets = config.get("val_dataset")
        if not val_datasets:
            val_datasets = train_datasets
        elif isinstance(val_datasets, str):
            val_datasets = [val_datasets]

        # C. EXPERIMENT NAME
        experiment_name = config.get("name")

        # D. EXECUTION
        if mode == "ensemble":
            manager.train(
                train_datasets, val_datasets, label_map, experiment_name=experiment_name
            )
        else:
            model_instance.train(
                train_datasets, val_datasets, label_map, experiment_name=experiment_name
            )

    elif command == "predict":
        inp = config.get("input")
        if not inp:
            print("[!] Error: Predict requires 'input'.")
            return

        # 1. RUN PREDICTION
        from baseline_detectors.utils import predict_raw_or_npz
        
        if mode == "ensemble":
            results = predict_raw_or_npz(manager.predict, inp)
        else:
            weights = config.get("weights")
            if not weights:
                print("[!] Error: Single model requires 'weights'.")
                return
            model_instance.load_weights(weights)
            results = predict_raw_or_npz(model_instance.predict, inp)

        # 2. PRINT RESULTS
        print("\n" + "=" * 80)
        print(f"{'FULL PREDICTION BREAKDOWN':^80}")
        print("=" * 80)

        for fpath, data in results.items():
            fname = os.path.basename(fpath)
            print(f"\nFILE: {fname}")
            print("-" * 60)

            # HELPER FUNCTION TO PRINT ONE MODEL'S RESULT
            def print_model_result(model_name, res_dict):
                # 1. Identify the winner
                winner = res_dict.get("final_label", "Unknown")

                print(f"[{model_name.upper()}] -> Winner: {winner}")

                # 2. Print the detailed scores
                # We sort them by score (highest first) for better readability
                try:
                    # Filter out non-class keys
                    scores = {
                        k: v
                        for k, v in res_dict.items()
                        if k not in ["final_label", "label", "confidence", "details"]
                    }

                    # Sort: largest score first
                    sorted_scores = sorted(
                        scores.items(), key=lambda item: item[1], reverse=True
                    )

                    for label, score in sorted_scores:
                        # Highlight the winner with a star, others are just listed
                        marker = "*" if label == winner else " "
                        print(f"   {marker} {label:<15}: {score:.4f}")

                except Exception as e:
                    print(f"   [!] Error parsing details: {e}")
                print("-" * 30)

            # LOGIC FOR ENSEMBLE VS SINGLE
            if mode == "ensemble":
                for m_name, res in data.items():
                    print_model_result(m_name, res)
            else:
                # In single mode, 'data' is the result dict directly
                # We use the model name from config or generic "Model"
                m_name = config.get("model", "SingleModel")
                print_model_result(m_name, data)

        print("=" * 80)

        # 3. SAVE DETAILED PER-FILE LOGS
        out_file = "per_file_prediction_results.json"
        try:
            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"\n[*] Detailed per-file logs saved to '{out_file}'")
        except Exception as e:
            print(f"\n[!] Error saving detailed per-file logs: {e}")

    elif command == "evaluate":
        test_dir = config.get("test_dir")
        if not test_dir:
            print("[!] Error: Evaluate requires 'test_dir'.")
            return

        out_dir = config.get("output_dir", "results")

        if mode == "ensemble":
            manager.evaluate(test_dir, output_dir=out_dir)
        else:
            weights = config.get("weights")
            if not weights:
                print("[!] Error: weights path is required for evaluation.")
                return
            model_instance.load_weights(weights)

            metrics = model_instance.evaluate(test_dir, output_dir=out_dir)

            if metrics:
                print("\n" + "=" * 40)
                print(f"      FINAL EVALUATION METRICS")
                print("=" * 40)
                print(f"[*] Accuracy  : {metrics['accuracy']:.4f}")
                print(f"[*] Precision : {metrics['precision_macro']:.4f}")
                print(f"[*] Recall    : {metrics['recall_macro']:.4f}")
                print(f"[*] F1 Score  : {metrics['f1_macro']:.4f}")
                print("=" * 40 + "\n")

    elif command == "evade":
        # 1. VALIDATE INPUTS
        adv_dirs = config.get("adv_dirs")
        if not adv_dirs or not isinstance(adv_dirs, list):
            print("[!] Error: Evasion test requires 'adv_dirs' (list of paths).")
            return

        map_path = config.get("map")
        if not map_path:
            print("[!] Error: Evasion test requires '--map'.")
            return

        raw_map = load_json(map_path)
        # Apply Global Int Fix to Evade Map as well for consistency
        try:
            label_map = {k: int(v) for k, v in raw_map.items()}
        except ValueError:
            print(f"[!] Error: Label map values must be integers.")
            return

        # 2. EXECUTION
        if mode == "ensemble":
            manager.evade(adv_dirs, label_map)
        else:
            # SINGLE MODEL LOGIC
            weights = config.get("weights")
            if not weights:
                print("[!] Error: Single model requires 'weights'.")
                return

            model_instance.load_weights(weights)

            # --- MONKEY PATCH TO INTERCEPT PER-FILE PREDICTIONS ---
            file_level_results = {}
            original_predict = model_instance.predict
            
            def patched_predict(input_path, *args, **kwargs):
                preds = original_predict(input_path, *args, **kwargs)
                tech_name = (
                    os.path.basename(input_path.rstrip("/\\"))
                    .replace(".npz", "")
                    .replace("_adv_samples", "")
                )
                
                for fpath_key, res in preds.items():
                    if fpath_key not in file_level_results:
                        file_level_results[fpath_key] = {
                            "dataset": tech_name,
                            "true_label_class": "Unknown",
                            "true_label_binary": "Unknown",
                            "results": {}
                        }
                    file_level_results[fpath_key]["results"][model_name] = res
                return preds

            model_instance.predict = patched_predict

            # Delegate to model
            raw_results = model_instance.evade(adv_dirs, label_map)

            # --- EXTRACT TRUE LABELS AND SAVE PER-FILE REPORT ---
            print("\n[*] Extracting True Labels for per-file report...")
            import numpy as np
            from baseline_detectors.utils import is_benign_label
            
            for tech_path in adv_dirs:
                if tech_path.endswith(".npz"):
                    try:
                        data = np.load(tech_path, allow_pickle=True)
                        labels = data.get("label", [])
                        names = data.get("name", [])
                        if len(names) == 0 and len(labels) > 0:
                            names = [f"sample_{i}" for i in range(len(labels))]
                            
                        for i in range(len(names)):
                            fkey = f"{tech_path}::{names[i]}"
                            t_class = str(labels[i]).strip()
                            t_binary = "Benign" if is_benign_label(t_class, label_map) else "Malware"
                            
                            if fkey in file_level_results:
                                file_level_results[fkey]["true_label_class"] = t_class
                                file_level_results[fkey]["true_label_binary"] = t_binary
                            elif names[i] in file_level_results:
                                file_level_results[names[i]]["true_label_class"] = t_class
                                file_level_results[names[i]]["true_label_binary"] = t_binary
                    except Exception as e:
                        print(f"[!] Warning: Could not extract true labels from {tech_path}: {e}")
                else:
                    for root, _, files in os.walk(tech_path):
                        for f in files:
                            fpath = os.path.join(root, f)
                            if fpath in file_level_results:
                                t_binary = "Benign" if is_benign_label(fpath, label_map) or "benign" in fpath.lower() else "Malware"
                                file_level_results[fpath]["true_label_class"] = "Unknown (Raw Dir)"
                                file_level_results[fpath]["true_label_binary"] = t_binary

            # --- REFORMAT JSON STRUCTURE TO { dataset: { filename: { ... } } } ---
            formatted_results = {}
            for fkey, fdata in file_level_results.items():
                dataset_name = fdata.pop("dataset", "UnknownDataset")
                
                if "::" in fkey:
                    single_file_name = fkey.split("::")[-1]
                else:
                    single_file_name = os.path.basename(fkey)
                    
                if dataset_name not in formatted_results:
                    formatted_results[dataset_name] = {}
                    
                formatted_results[dataset_name][single_file_name] = fdata

            out_file = "per_file_evasion_results.json"
            try:
                import json
                with open(out_file, "w") as f:
                    json.dump(formatted_results, f, indent=4)
                print(f"[*] Per-file evasion results saved to '{out_file}'")
            except Exception as e:
                print(f"[!] Error saving per-file results: {e}")


            # Format for shared printer: {tech: {model: stats}}
            final_report = {}
            for tech_name, stats in raw_results.items():
                final_report[tech_name] = {model_name: stats}

            print_evasion_summary(final_report)
    else:
        print(f"[!] Error: Unknown or missing command '{command}'.")
        print("    Valid commands are: 'train', 'predict', 'evaluate', 'evade'.")


def main():
    parser = argparse.ArgumentParser(description="AI Malware Research Suite")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config file"
    )

    parser.add_argument(
        "--name", type=str, help="Override experiment name for this run"
    )

    args = parser.parse_args()

    config_dict = load_json(args.config)

    if args.name:
        config_dict["name"] = args.name
        print(f"[*] Command Line Override: Experiment Name set to '{args.name}'")

    run_from_config(config_dict)


if __name__ == "__main__":
    main()
