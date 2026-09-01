import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Internal imports
from .arch import MalConv
from ...core import MalwareModelBase
from ...utils import ChunkRandomSampler, calculate_optimal_chunk_size
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
# 1. CONFIG
# ==========================================
CONFIG = {
    "max_len": 2**21,  # 2MB
    "batch_size": 4,  # Small batch size (Paper limits)
    "epochs": 50,
    "learning_rate": 0.001,
    "num_workers": 2,
    "vocab_size": 257,
    "emb_dim": 8,
    "n_filters": 128,
    "kernel_size": 500,
    "stride": 500,
    "decov_strength": 0.1,  # DeCov weight parameter
}


# ==========================================
# HELPER: DECOV LOSS
# ==========================================
def decov_loss(features):
    """
    Calculates DeCov loss to penalize correlation between activations.
    """
    if features.size(0) <= 1:
        return torch.tensor(0.0, device=features.device)

    features_mean = features - torch.mean(features, dim=0, keepdim=True)
    cov_matrix = (features_mean.t() @ features_mean) / (features.size(0) - 1)

    frobenius_norm_sq = torch.sum(cov_matrix**2)
    diag_norm_sq = torch.sum(torch.diag(cov_matrix) ** 2)

    loss = 0.5 * (frobenius_norm_sq - diag_norm_sq)
    return loss


# ==========================================
# 2. DATASET CLASS (MEMORY-SAFE LAZY LOADER)
# ==========================================
class PESequenceDataset(Dataset):
    """
    PyTorch Dataset for loading Windows PE files as raw byte sequences.
    VRAM/RAM Safe: Uses lazy-loading for .npz files.
    """

    def __init__(self, paths, label_mapping, max_len=2**21):
        self.max_len = max_len
        self.samples = []
        self.npz_data = []

        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}

        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            if path.endswith(".npz"):
                # ONLY load the labels during init to get the length and map IDs.
                # DO NOT touch the raw_byte array here to prevent RAM crashes.
                data = np.load(path, allow_pickle=True)
                labels = data["label"]

                for i in range(len(labels)):
                    l_str = str(labels[i]).strip().lower()
                    l_int = self.label_mapping.get(l_str, -1)
                    if l_int != -1:
                        # Store the path, the index, and the label.
                        self.npz_data.append((path, i, l_int))

                del data  # Free memory immediately
                gc.collect()
            else:
                path_obj = Path(path)
                for dirpath, _, filenames in os.walk(path_obj):
                    folder_name = os.path.basename(dirpath).lower()
                    found_label = self.label_mapping.get(folder_name, None)

                    if found_label is not None:
                        for f in filenames:
                            filepath = os.path.join(dirpath, f)
                            if os.path.isfile(filepath):
                                self.samples.append((filepath, found_label))

        if len(self.samples) == 0 and len(self.npz_data) == 0:
            print(f"[!] Warning: No samples found in {paths}")

    def __len__(self):
        return len(self.samples) + len(self.npz_data)

    def __getitem__(self, idx):
        # 1. Handle Raw PE Directory
        if idx < len(self.samples):
            filepath, label = self.samples[idx]
            try:
                with open(filepath, "rb") as f:
                    data = f.read(self.max_len)
                arr = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
            except Exception as e:
                print(f"[!] Error loading {filepath}: {e}")
                arr = np.full(self.max_len, 256, dtype=np.int16)

        # 2. Handle NPZ Lazy Loading
        else:
            npz_idx = idx - len(self.samples)
            npz_path, item_index, label = self.npz_data[npz_idx]

            # Lazy load the dictionary into the specific worker thread
            if not hasattr(self, "worker_open_files"):
                self.worker_open_files = {}

            if npz_path not in self.worker_open_files:
                npz = np.load(npz_path, allow_pickle=True)
                self.worker_open_files[npz_path] = {
                    "raw_byte": npz.get("raw_byte", None)
                }

            # Extract only the single byte array needed for this batch
            raw_array = self.worker_open_files[npz_path]["raw_byte"]
            raw_val = raw_array[item_index] if raw_array is not None else []
            arr = np.array(raw_val, dtype=np.int16)

        # Padding / Truncating
        if arr.size < self.max_len:
            arr = np.pad(arr, (0, self.max_len - arr.size), constant_values=256)
        else:
            arr = arr[: self.max_len]

        return torch.from_numpy(arr).long(), torch.tensor(int(label), dtype=torch.long)


# ==========================================
# 3. MAIN MODEL CONTROLLER
# ==========================================
class MalConvModel(MalwareModelBase):
    def __init__(self):
        super().__init__()
        self.model = None
        self.idx_to_label = {}
        self.label_mapping = {}

        base_dir = os.path.dirname(__file__)
        self.local_weights_dir = os.path.join(base_dir, "weights")
        self.local_results_dir = os.path.join(base_dir, "results")

    def load_weights(self, weights_path):
        if not os.path.exists(weights_path):
            weights_path = os.path.join(self.local_weights_dir, weights_path)
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights not found: {weights_path}")

        print(f"[*] Loading weights: {weights_path}")
        checkpoint = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.label_mapping = checkpoint.get("label_mapping", {})
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        num_classes = len(self.label_mapping) if self.label_mapping else 6

        self.model = MalConv(
            vocab_size=CONFIG["vocab_size"],
            emb_dim=CONFIG["emb_dim"],
            n_filters=CONFIG["n_filters"],
            kernel_size=CONFIG["kernel_size"],
            stride=CONFIG["stride"],
            num_classes=num_classes,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def train(self, train_paths, val_paths, label_mapping, experiment_name=None):
        manager = ExperimentManager(os.path.dirname(__file__), experiment_name)
        logger = Logger(manager.run_dir)

        logger.log(f"[*] Starting MalConv Training (Multi-class + DeCov)")
        manager.save_config(CONFIG)

        self.label_mapping = label_mapping
        self.idx_to_label = {v: k for k, v in label_mapping.items()}

        train_ds = PESequenceDataset(train_paths, label_mapping, CONFIG["max_len"])
        val_ds = PESequenceDataset(val_paths, label_mapping, CONFIG["max_len"])

        logger.log(f"[*] Loaded Datasets: {len(train_ds)} Train | {len(val_ds)} Val")

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

        self.model = MalConv(num_classes=len(label_mapping)).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_acc = 0.0

        for epoch in range(CONFIG["epochs"]):
            self.model.train()
            total_loss, correct, total = 0, 0, 0
            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}", leave=False
            )

            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                logits, features = self.model(x)

                cls_loss = criterion(logits, y)
                reg_loss = decov_loss(features)
                loss = cls_loss + (CONFIG["decov_strength"] * reg_loss)

                loss.backward()
                if self.device.type == "xla":
                    import torch_xla.core.xla_model as xm

                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                total_loss += loss.item()
                _, pred = torch.max(logits, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total

            # Validation
            self.model.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits, _ = self.model(x)

                    v_loss += criterion(logits, y).item()
                    _, pred = torch.max(logits, 1)
                    v_total += y.size(0)
                    v_correct += (pred == y).sum().item()

            val_loss = v_loss / len(val_loader) if v_total > 0 else 0
            val_acc = v_correct / v_total if v_total > 0 else 0

            logger.log(
                f"Epoch {epoch + 1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "label_mapping": label_mapping,
                        "accuracy": best_acc,
                        "config": CONFIG,
                    },
                    manager.get_path("malconv.pth"),
                )

        # Memory Cleanup after training
        del train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_training_plot(history, manager.run_dir)
        logger.close()

    def predict(self, input_path, ignore_list=None):
        if not self.model:
            raise RuntimeError("Model not loaded")
        self.model.eval()
        results = {}

        if input_path.endswith(".npz"):
            data = np.load(input_path, allow_pickle=True)
            names = data.get("name", [])
            raw_bytes_cache = data.get("raw_byte", None)

            with torch.no_grad():
                for i in tqdm(range(len(names)), desc="Predicting NPZ"):
                    malware_hash = names[i]
                    if ignore_list and malware_hash.split(".")[0] in ignore_list:
                        results[f"{input_path}::{malware_hash}"] = {"final_label": "IGNORED"}
                        continue
                        
                    # Extract array one by one to avoid massive memory spikes
                    raw_val = raw_bytes_cache[i] if raw_bytes_cache is not None else []
                    arr = np.array(raw_val, dtype=np.int16)
                    arr = np.pad(
                        arr,
                        (0, max(0, CONFIG["max_len"] - arr.size)),
                        constant_values=256,
                    )[: CONFIG["max_len"]]

                    x = torch.from_numpy(arr).long().unsqueeze(0).to(self.device)
                    logits, _ = self.model(x)
                    probs = torch.softmax(logits, dim=1)
                    _, pred = torch.max(probs, 1)

                    res = {
                        self.idx_to_label.get(idx, str(idx)): probs[0][idx].item()
                        for idx in range(probs.size(1))
                    }
                    res["final_label"] = self.idx_to_label.get(
                        pred.item(), str(pred.item())
                    )
                    results[f"{input_path}::{names[i]}"] = res

            del data
            gc.collect()
        else:
            files = (
                [input_path]
                if os.path.isfile(input_path)
                else [
                    os.path.join(r, f) for r, _, fs in os.walk(input_path) for f in fs
                ]
            )
            with torch.no_grad():
                for fpath in tqdm(files, desc="Predicting Files"):
                    base_name = os.path.basename(fpath)
                    if ignore_list and base_name.split(".")[0] in ignore_list:
                        results[fpath] = {"final_label": "IGNORED"}
                        continue
                        
                    try:
                        with open(fpath, "rb") as f:
                            data = f.read(CONFIG["max_len"])
                        arr = np.pad(
                            np.frombuffer(data, dtype=np.uint8).astype(np.int16),
                            (0, max(0, CONFIG["max_len"] - len(data))),
                            constant_values=256,
                        )[: CONFIG["max_len"]]

                        x = torch.from_numpy(arr).long().unsqueeze(0).to(self.device)
                        logits, _ = self.model(x)
                        probs = torch.softmax(logits, dim=1)
                        _, pred = torch.max(probs, 1)

                        res = {
                            self.idx_to_label.get(idx, str(idx)): probs[0][idx].item()
                            for idx in range(probs.size(1))
                        }
                        res["final_label"] = self.idx_to_label.get(
                            pred.item(), str(pred.item())
                        )
                        results[fpath] = res
                    except Exception as e:
                        results[fpath] = {"final_label": "ERROR", "details": str(e)}

        return results

    def evaluate(self, test_path, output_dir=None):
        if not self.model:
            raise RuntimeError("Model not loaded")
        output_dir = output_dir or self.local_results_dir
        os.makedirs(output_dir, exist_ok=True)

        ds = PESequenceDataset(test_path, self.label_mapping, CONFIG["max_len"])
        loader = DataLoader(
            ds, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"]
        )

        y_true, y_pred, confs, correct_mask = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Evaluating MalConv"):
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, 1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                confs.extend(conf.cpu().numpy())
                correct_mask.extend((pred == y).cpu().numpy())

        class_names = [
            self.idx_to_label.get(i, str(i)) for i in range(len(self.idx_to_label))
        ]
        metrics = calculate_metrics(y_true, y_pred, class_names)
        generate_confusion_matrix(y_true, y_pred, class_names, output_dir)
        plot_confidence_distribution(confs, correct_mask, output_dir)
        return metrics

    def evade(self, technique_paths, label_mapping) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        self.label_mapping = label_mapping
        self.idx_to_label = {v: k for k, v in label_mapping.items()}
        report = {}
        self.model.eval()

        for tech_path in tqdm(technique_paths, desc="Evasion Testing"):
            tech_name = os.path.basename(tech_path.rstrip("/\\")).replace(
                "_adv_samples", ""
            )
            passed, total = 0, 0
            
            ignore_list = get_ignored_samples(tech_name, tech_path=tech_path, label_mapping=self.label_mapping)

            if tech_path.endswith(".npz"):
                preds = self.predict(tech_path, ignore_list=ignore_list)
                data = np.load(tech_path, allow_pickle=True)
                for i in range(len(data["name"])):
                    true_label = str(data["label"][i]).strip().lower()
                    if is_benign_label(true_label, self.label_mapping):
                        continue
                    res = preds.get(f"{tech_path}::{data['name'][i]}", {})
                    if res.get("final_label") == "ERROR" or not res:
                        continue
                    total += 1
                    
                    if res.get("final_label") == "IGNORED":
                        continue
                        
                    if str(res.get("final_label")).lower() != true_label:
                        passed += 1
            else:
                for label_name, label_id in label_mapping.items():
                    if is_benign_label(label_name, self.label_mapping):
                        continue
                    target_dir = next(
                        (
                            os.path.join(r, d)
                            for r, ds, _ in os.walk(tech_path)
                            for d in ds
                            if d.lower() == label_name.lower()
                        ),
                        None,
                    )
                    if target_dir:
                        preds = self.predict(target_dir, ignore_list=ignore_list)
                        for res in preds.values():
                            if res.get("final_label", "ERROR") == "ERROR":
                                continue
                            total += 1
                            
                            if res.get("final_label") == "IGNORED":
                                continue
                                
                            if (
                                str(res.get("final_label")).lower()
                                != label_name.lower()
                            ):
                                passed += 1

            report[tech_name] = {
                "passed": passed,
                "total": total,
                "rate": (passed / total) if total > 0 else 0,
            }
            print(f"[*] Evasion Analysis: {tech_name} -> {passed}/{total} passed")

        return report
