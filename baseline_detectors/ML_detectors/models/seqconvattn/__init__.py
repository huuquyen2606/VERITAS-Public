import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import copy

# Internal imports
from .arch import SeqConvAttn
from ...core import MalwareModelBase
from ...utils import ChunkRandomSampler, calculate_optimal_chunk_size

# Lab Tools
from ...utils import (
    generate_confusion_matrix,
    calculate_metrics,
    save_training_plot,
    plot_confidence_distribution,
    Logger,
    ExperimentManager,
    get_ignored_samples,
    is_benign_label,
)

# ==========================================
# 1. CONFIG: EXACT FROM PAPER
# ==========================================

CONFIG = {
    "max_len": 400000,
    "batch_size": 25,
    "epochs": 25,
    "learning_rate": 1e-4,
    "betas": (0.9, 0.999),
    "num_workers": 2,
    "vocab_size": 257,
    "emb_dim": 8,
    "n_filters": 128,
    "kernel_size": 500,
    "n_transformer_blocks": 3,
    "n_heads": 8,
    "ff_expansion": 4,
}

# ==========================================
# 2. DATASET CLASS
# ==========================================


class PESequenceDataset(Dataset):
    def __init__(self, paths, label_mapping, max_len=400000):
        self.max_len = max_len
        self.samples = []
        self.npz_metadata = []

        # Dictionary populated ONLY inside background workers to prevent pickling crashes!
        self.worker_open_files = {}

        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}

        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            if path.endswith(".npz"):
                # Use 'with' to peek inside and instantly close the file!
                with np.load(path, allow_pickle=True) as data:
                    labels = data["label"]
                    for i in range(len(labels)):
                        l_str = str(labels[i]).strip().lower()
                        l_int = self.label_mapping.get(l_str, -1)
                        if l_int != -1:
                            # Only store routing metadata, NOT the open file
                            self.npz_metadata.append((path, i, l_int))
            else:
                path_obj = Path(path)
                for dirpath, _, filenames in os.walk(path_obj):
                    folder_name = os.path.basename(dirpath).lower()

                    if folder_name in self.label_mapping:
                        label_id = self.label_mapping[folder_name]
                        for f in filenames:
                            filepath = os.path.join(dirpath, f)
                            if os.path.isfile(filepath):
                                self.samples.append((filepath, label_id))

        if len(self.samples) == 0 and len(self.npz_metadata) == 0:
            print(f"[!] Warning: No samples found in {paths}")

    def __len__(self):
        return len(self.samples) + len(self.npz_metadata)

    def __getitem__(self, idx):
        if idx < len(self.samples):
            filepath, label = self.samples[idx]
            try:
                with open(filepath, "rb") as f:
                    data = f.read(self.max_len)
                arr = np.frombuffer(data, dtype=np.uint8).copy()
            except Exception as e:
                print(f"[!] Error loading {filepath}: {e}")
                arr = np.full(self.max_len, 256, dtype=np.uint8)
        else:
            npz_idx = idx - len(self.samples)
            npz_path, item_idx, label = self.npz_metadata[npz_idx]

            # FAST LAZY LOADING: Cache the extracted arrays inside the spawned worker
            if npz_path not in self.worker_open_files:
                npz = np.load(npz_path, allow_pickle=True)
                self.worker_open_files[npz_path] = {
                    "raw_byte": npz.get("raw_byte", None)
                }

            raw_array = self.worker_open_files[npz_path]["raw_byte"]
            raw_val = raw_array[item_idx] if raw_array is not None else []
            arr = np.array(raw_val, dtype=np.uint8)

        if arr.size < self.max_len:
            arr = np.pad(arr, (0, self.max_len - arr.size), constant_values=256)
        else:
            arr = arr[: self.max_len]

        return torch.from_numpy(arr).long(), torch.tensor(label, dtype=torch.long)


# ==========================================
# 3. MAIN MODEL CONTROLLER
# ==========================================


class SeqConvAttnModel(MalwareModelBase):
    def __init__(self, model_dir="trained_models/seqconvattn"):
        super().__init__()
        self.model = None
        self.idx_to_label = {}
        self.label_mapping = {}

        # Standardized model directory logic
        self.model_dir = model_dir
        self.local_weights_dir = os.path.join(self.model_dir, "weights")
        self.local_results_dir = os.path.join(self.model_dir, "results")

    # FIXED: Default weight path allows main.py to call this empty
    def load_weights(self, weights_path=None, num_classes=None):
        final_path = (
            weights_path
            if weights_path
            else os.path.join(self.model_dir, "seqconvattn.pth")
        )

        if not os.path.exists(final_path):
            print(f"[!] Warning: Could not find weights file at {final_path}")
            return

        print(f"[*] Loading weights from: {final_path}")
        checkpoint = torch.load(
            final_path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.label_mapping = checkpoint.get("label_mapping", {})
            if self.label_mapping:
                self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
                num_classes = len(self.label_mapping)

            hp = checkpoint.get("hyperparameters", CONFIG)
            state_dict = checkpoint["model_state"]
        else:
            print("    [!] Warning: Loading raw state_dict without metadata.")
            state_dict = checkpoint
            hp = CONFIG
            if num_classes is None:
                raise ValueError(
                    "num_classes must be provided explicitly when loading raw weights."
                )

        self.model = SeqConvAttn(
            num_classes=num_classes,
            vocab_size=hp.get("vocab_size", 257),
            emb_dim=hp.get("emb_dim", 8),
            n_filters=hp.get("n_filters", 128),
            kernel_size=hp.get("kernel_size", 500),
            n_transformer_blocks=hp.get("n_transformer_blocks", 3),
            n_heads=hp.get("n_heads", 8),
            ff_expansion=hp.get("ff_expansion", 4),
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("    ✓ Model loaded successfully.")

    def train(self, train_paths, val_paths, label_mapping, experiment_name=None):
        if experiment_name:
            self.model_dir = os.path.join("trained_models", experiment_name)
            os.makedirs(self.model_dir, exist_ok=True)

        logger = Logger(os.path.join(self.model_dir, "training.log"))
        logger.log(f"[*] Starting SeqConvAttn Training Session")

        self.label_mapping = label_mapping
        num_classes = len(label_mapping)
        self.idx_to_label = {v: k for k, v in label_mapping.items()}

        logger.log("[*] Loading Datasets...")

        # Handle NPZ extraction
        train_npz = train_paths[0] if isinstance(train_paths, list) else train_paths
        val_npz = val_paths[0] if isinstance(val_paths, list) else val_paths

        train_ds = PESequenceDataset(
            train_npz, label_mapping, max_len=CONFIG["max_len"]
        )
        val_ds = PESequenceDataset(val_npz, label_mapping, max_len=CONFIG["max_len"])

        chunk_sz = calculate_optimal_chunk_size(element_size_bytes=200000)
        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            sampler=ChunkRandomSampler(train_ds, chunk_sz),
            num_workers=CONFIG["num_workers"],
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
        )

        self.model = SeqConvAttn(
            num_classes=num_classes,
            **{
                k: v
                for k, v in CONFIG.items()
                if k
                not in [
                    "max_len",
                    "batch_size",
                    "epochs",
                    "learning_rate",
                    "betas",
                    "num_workers",
                ]
            },
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(), lr=CONFIG["learning_rate"], betas=CONFIG["betas"]
        )
        criterion = nn.CrossEntropyLoss()

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_weighted": [],
        }
        best_acc, best_epoch, best_model_state = 0.0, 0, None

        for epoch in range(CONFIG["epochs"]):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}", leave=True
            )

            for sequences, labels in pbar:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}"})

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total

            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = (
                        sequences.to(self.device),
                        labels.to(self.device),
                    )
                    outputs = self.model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            if val_total > 0:
                val_epoch_loss = val_loss / len(val_loader)
                val_epoch_acc = val_correct / val_total
                val_f1_weighted = f1_score(
                    all_labels, all_preds, average="weighted", zero_division=0
                )
            else:
                val_epoch_loss, val_epoch_acc, val_f1_weighted = 0.0, 0.0, 0.0

            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)
            history["val_loss"].append(val_epoch_loss)
            history["val_acc"].append(val_epoch_acc)
            history["val_f1_weighted"].append(val_f1_weighted)

            logger.log(
                f"Epoch [{epoch + 1}/{CONFIG['epochs']}] Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f} F1-Weighted: {val_f1_weighted:.4f}"
            )

            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(self.model.state_dict())
                logger.log(f"  ✓ Best model updated (Accuracy: {val_epoch_acc:.4f})")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            save_path = os.path.join(self.model_dir, "seqconvattn.pth")
            torch.save(
                {
                    "model_state": best_model_state,
                    "label_mapping": label_mapping,
                    "hyperparameters": CONFIG,
                    "history": history,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "accuracy": best_acc,
                    "f1_weighted": history["val_f1_weighted"][best_epoch - 1],
                    "best_epoch": best_epoch,
                    "num_epochs": CONFIG["epochs"],
                },
                save_path,
            )
            logger.log(f"[*] Training Complete. Best Weights saved to: {save_path}")

        save_training_plot(history, save_dir=self.model_dir)
        logger.close()

    def predict(self, input_path, ignore_list=None):
        # FIXED: Auto-loads weights instead of crashing
        if self.model is None:
            self.load_weights()

        results = {}
        self.model.eval()

        if str(input_path).endswith(".npz"):
            data = np.load(input_path, allow_pickle=True)
            names = data["name"]
            raw_bytes = data["raw_byte"]

            with torch.no_grad():
                for i in tqdm(range(len(names)), desc="Predicting NPZ"):
                    malware_hash = names[i]
                    if ignore_list and malware_hash.split(".")[0] in ignore_list:
                        results[f"{input_path}::{malware_hash}"] = {"final_label": "IGNORED"}
                        continue
                        
                    arr = np.array(raw_bytes[i], dtype=np.uint8)
                    if arr.size < CONFIG["max_len"]:
                        arr = np.pad(
                            arr, (0, CONFIG["max_len"] - arr.size), constant_values=256
                        )
                    else:
                        arr = arr[: CONFIG["max_len"]]

                    tensor = torch.from_numpy(arr).long().unsqueeze(0).to(self.device)
                    outputs = self.model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()

                    file_result = {}
                    for idx, class_name in self.idx_to_label.items():
                        file_result[class_name] = probs[0][idx].item()
                    file_result["final_label"] = self.idx_to_label.get(
                        pred_idx, str(pred_idx)
                    )
                    results[f"{input_path}::{names[i]}"] = file_result
        else:
            files = [input_path] if os.path.isfile(input_path) else []
            if os.path.isdir(input_path):
                for r, _, fs in os.walk(input_path):
                    files.extend([os.path.join(r, f) for f in fs])

            with torch.no_grad():
                for fpath in tqdm(files, desc="Predicting Files"):
                    base_name = os.path.basename(fpath)
                    if ignore_list and base_name.split(".")[0] in ignore_list:
                        results[fpath] = {"final_label": "IGNORED"}
                        continue
                        
                    try:
                        with open(fpath, "rb") as f:
                            data = f.read(CONFIG["max_len"])
                        arr = np.frombuffer(data, dtype=np.uint8).copy()
                        if arr.size < CONFIG["max_len"]:
                            arr = np.pad(
                                arr,
                                (0, CONFIG["max_len"] - arr.size),
                                constant_values=256,
                            )
                        else:
                            arr = arr[: CONFIG["max_len"]]

                        tensor = (
                            torch.from_numpy(arr).long().unsqueeze(0).to(self.device)
                        )
                        outputs = self.model(tensor)
                        probs = torch.softmax(outputs, dim=1)
                        pred_idx = torch.argmax(probs, dim=1).item()

                        file_result = {}
                        for idx, class_name in self.idx_to_label.items():
                            file_result[class_name] = probs[0][idx].item()
                        file_result["final_label"] = self.idx_to_label.get(
                            pred_idx, str(pred_idx)
                        )
                        results[fpath] = file_result
                    except Exception as e:
                        results[fpath] = {"final_label": "ERROR", "details": str(e)}

        return results

    def evaluate(self, test_path, output_dir=None) -> dict:
        if self.model is None:
            self.load_weights()

        # Route output to the framework's requested folder
        out_dir = (
            output_dir if output_dir else os.path.join(self.model_dir, "evaluation")
        )
        os.makedirs(out_dir, exist_ok=True)

        # Pass test_path directly instead of extracting from a dict
        ds = PESequenceDataset(test_path, self.label_mapping, max_len=CONFIG["max_len"])
        loader = DataLoader(
            ds,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
        )

        all_preds, all_labels, all_confs, correct_mask = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for sequences, lbls in tqdm(loader, desc="Evaluating SeqConvAttn"):
                sequences, lbls = sequences.to(self.device), lbls.to(self.device)
                outputs = self.model(sequences)
                probs = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())
                all_confs.extend(confidences.cpu().numpy())
                correct_mask.extend((predicted.cpu() == lbls.cpu()).numpy())

        class_names = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]
        metrics = calculate_metrics(all_labels, all_preds, class_names)
        generate_confusion_matrix(all_labels, all_preds, class_names, save_dir=out_dir)
        plot_confidence_distribution(all_confs, correct_mask, save_dir=out_dir)
        return metrics

    def evade(self, technique_dirs, label_mapping) -> dict:
        if self.model is None:
            self.load_weights()

        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}

        results_report = {}
        for tech_path in technique_dirs:
            tech_name = os.path.basename(tech_path.rstrip("/\\")).replace(
                "_adv_samples", ""
            )
            passed, total = 0, 0
            
            ignore_list = get_ignored_samples(tech_name, tech_path=tech_path, label_mapping=self.label_mapping)

            if tech_path.endswith(".npz"):
                preds = self.predict(tech_path, ignore_list=ignore_list)
                data = np.load(tech_path, allow_pickle=True)
                names, labels = data["name"], data["label"]

                for i in range(len(names)):
                    true_label = str(labels[i]).strip().lower()
                    if is_benign_label(true_label, self.label_mapping):
                        continue
                    res = preds.get(f"{tech_path}::{names[i]}", {})
                    if res.get("final_label") == "ERROR" or not res:
                        continue
                    total += 1
                    
                    if res.get("final_label") == "IGNORED":
                        continue
                        
                    if str(res.get("final_label")).lower() != true_label:
                        passed += 1
            else:
                for label_str in self.label_mapping.keys():
                    if is_benign_label(label_str, self.label_mapping):
                        continue
                    class_dir = os.path.join(tech_path, label_str)
                    if not os.path.isdir(class_dir):
                        continue

                    preds = self.predict(class_dir, ignore_list=ignore_list)
                    for fpath, res in preds.items():
                        if res.get("final_label", "ERROR") == "ERROR":
                            continue
                        total += 1
                        
                        if res.get("final_label") == "IGNORED":
                            continue
                            
                        if str(res.get("final_label")).lower() != label_str.lower():
                            passed += 1

            results_report[tech_name] = {
                "passed": passed,
                "total": total,
                "rate": (passed / total) if total > 0 else 0.0,
            }
            print(f"[*] Evasion {tech_name}: {passed}/{total} passed")

        return results_report
