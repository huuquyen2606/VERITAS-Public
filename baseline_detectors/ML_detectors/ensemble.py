import os
import collections
import datetime
import gc

try:
    import torch
except ImportError:
    pass

from baseline_detectors import MODEL_REGISTRY
from baseline_detectors.utils import plot_model_comparison, Logger, print_evasion_summary


class EnsembleManager:
    """
    The central controller (The 'General') that commands multiple models simultaneously.
    VRAM-SAFE VERSION: Loads and purges models one-by-one to prevent GPU OOM crashes.
    """

    def __init__(self, model_names, weights_map=None):
        self.model_names = [n for n in model_names if n in MODEL_REGISTRY]
        self.weights_map = weights_map or {}

        print(f"\n[*] Initializing VRAM-Safe Ensemble Council with: {self.model_names}")

        if not self.model_names:
            raise RuntimeError(
                "Ensemble initialization failed: No valid models found in registry."
            )

    def _purge_vram(self):
        """Forces the GPU to empty its cache and run Python garbage collection."""
        if "torch" in globals():
            torch.cuda.empty_cache()
        gc.collect()

    def train(self, train_paths, val_paths, label_mapping, experiment_name=None):
        print(f"[*] Starting Ensemble Training for: {self.model_names}")
        experiment_name = experiment_name or "Ensemble_Run"

        for name in self.model_names:
            print(f"\n" + "=" * 40)
            print(f"TRAINING: {name.upper()}")
            print("=" * 40)

            try:
                model = MODEL_REGISTRY[name]()
                model.train(
                    train_paths=train_paths,
                    val_paths=val_paths,
                    label_mapping=label_mapping,
                    experiment_name=experiment_name,
                )
                print(f"    -> {name} Training Cycle Complete.")

                # PURGE VRAM
                del model
                self._purge_vram()

            except Exception as e:
                print(f"[!] Error training {name}: {e}")

        print(f"\n[*] All models trained.")

    def predict(self, input_path):
        """Perform inference using all models sequentially."""
        file_results = collections.defaultdict(dict)
        print(f"[*] Ensemble Scanning: {input_path}")

        for name in self.model_names:
            try:
                model = MODEL_REGISTRY[name]()

                # --- JSON STRING SAFETY NET ---
                w_path = None
                if isinstance(self.weights_map, dict):
                    w_path = self.weights_map.get(name)
                elif isinstance(self.weights_map, str) and len(self.model_names) == 1:
                    w_path = self.weights_map

                if w_path:
                    model.load_weights(w_path)

                preds = model.predict(input_path)

                for fpath, result in preds.items():
                    file_results[fpath][name] = result

                # PURGE VRAM
                del model
                self._purge_vram()

            except Exception as e:
                print(f"[!] Error getting predictions from {name}: {e}")
                key = input_path if os.path.isfile(input_path) else "unknown_file"
                if key not in file_results:
                    file_results[key] = {}
                file_results[key][name] = {
                    "label": "ERROR",
                    "Virus": 1.0,
                    "details": str(e),
                }

        return dict(file_results)

    def evaluate(self, test_dir, output_dir="results"):
        print(f"[*] Starting Ensemble Evaluation on: {test_dir}")
        os.makedirs(output_dir, exist_ok=True)

        logger = Logger(output_dir)
        logger.log(f"BATTLE REPORT - {datetime.datetime.now()}")
        logger.log(f"Test Directory: {test_dir}")
        logger.log("=" * 40)

        battle_stats = {}

        for name in self.model_names:
            print(f"\nEvaluating: {name.upper()}")

            try:
                model = MODEL_REGISTRY[name]()

                # --- JSON STRING SAFETY NET ---
                w_path = None
                if isinstance(self.weights_map, dict):
                    w_path = self.weights_map.get(name)
                elif isinstance(self.weights_map, str) and len(self.model_names) == 1:
                    w_path = self.weights_map

                if w_path:
                    model.load_weights(w_path)

                metrics = model.evaluate(test_dir, output_dir=None)
                if metrics:
                    battle_stats[name] = metrics
                    logger.log(f"\nMODEL: {name.upper()}")
                    logger.log(f"[*] Accuracy  : {metrics['accuracy']:.4f}")
                    logger.log(f"[*] Precision : {metrics['precision_macro']:.4f}")
                    logger.log(f"[*] Recall    : {metrics['recall_macro']:.4f}")
                    logger.log(f"[*] F1 Score  : {metrics['f1_macro']:.4f}")

                del model
                self._purge_vram()

            except Exception as e:
                logger.log(f"[!] Error evaluating {name}: {e}")

        if battle_stats:
            print(f"\n[*] Generating Battle Comparison Graph...")
            plot_model_comparison(battle_stats, save_dir=output_dir)
            logger.log("\n" + "=" * 40)
            logger.log("Battle Complete. Comparison graph generated.")

        logger.close()

    def evade(self, technique_dirs, label_mapping):
        print(
            f"[*] Commencing Ensemble Evasion Testing on {len(technique_dirs)} techniques..."
        )
        print("[*] Evasion Evaluation: Filtering out benign samples from evaluation progress and metrics calculations.")
        master_report = {}
        file_level_results = {}

        # Initialize the master report structure
        for tech_path in technique_dirs:
            tech_name = (
                os.path.basename(tech_path.rstrip("/\\"))
                .replace(".npz", "")
                .replace("_adv_samples", "")
            )
            master_report[tech_name] = {}

        for name in self.model_names:
            print(f"\n" + "=" * 40)
            print(f"[*] Ensemble Evasion Test: {name.upper()}")
            print("=" * 40)

            try:
                model = MODEL_REGISTRY[name]()

                # --- JSON STRING SAFETY NET ---
                w_path = None
                if isinstance(self.weights_map, dict):
                    w_path = self.weights_map.get(name)
                elif isinstance(self.weights_map, str) and len(self.model_names) == 1:
                    w_path = self.weights_map

                if w_path:
                    model.load_weights(w_path)
                else:
                    print(f"[!] Warning: No weights provided for {name}. Skipping.")
                    continue

                # --- MONKEY PATCH TO INTERCEPT PER-FILE PREDICTIONS ---
                original_predict = model.predict
                
                # Default arg capture
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
                        file_level_results[fpath_key]["results"][name] = res
                    return preds

                model.predict = patched_predict

                # --- NATIVE DELEGATION FIX! ---
                # We stop using os.walk() and pass the .npz directly to the model's custom logic
                raw_report = model.evade(technique_dirs, label_mapping)

                # Map the results back to the master report (Safely)
                for returned_tech_name, stats in raw_report.items():
                    # Clean the name so it matches the master report perfectly
                    clean_name = returned_tech_name.replace(".npz", "").replace(
                        "_adv_samples", ""
                    )

                    if clean_name not in master_report:
                        master_report[clean_name] = {}

                    master_report[clean_name][name] = stats
                # PURGE VRAM
                del model
                self._purge_vram()

            except Exception as e:
                print(f"[!] Error during evasion testing for {name}: {e}")

        # --- EXTRACT TRUE LABELS AND SAVE PER-FILE REPORT ---
        print("\n[*] Extracting True Labels for per-file report...")
        import numpy as np
        from baseline_detectors.utils import is_benign_label
        
        for tech_path in technique_dirs:
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
                        t_binary = "Benign" if is_benign_label(t_class, label_mapping) else "Malware"
                        
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
                            t_binary = "Benign" if is_benign_label(fpath, label_mapping) or "benign" in fpath.lower() else "Malware"
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

        print_evasion_summary(master_report)
        return master_report
