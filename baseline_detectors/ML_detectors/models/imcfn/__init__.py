import os
import cv2
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import transforms

# Internal imports
from .arch import IMCFNArch
from ...core import MalwareModelBase
from ...utils import ChunkRandomSampler, calculate_optimal_chunk_size
from ...utils import (
    generate_confusion_matrix,
    calculate_metrics,
    plot_confidence_distribution,
    save_training_plot,
    Logger,
    ExperimentManager,
    get_ignored_samples,
    is_benign_label,
)

# ==========================================
# 1. CONFIGURATION (From Paper & Kaggle)
# ==========================================
CONFIG = {
    "image_size": (224, 224),
    "batch_size": 32,
    "learning_rate": 5e-6,  # Exact from paper
    "weight_decay": 1.0,  # Lambda for Eq. 2 in paper
    "momentum": 0.9,  # use the default standard
    "epochs": 10,
    "num_workers": 2,
    "fc1_units": 2048,
    "fc2_units": 2048,
    "dropout_rate": 0.5,
}


# ==========================================
# 2. IMAGE CONVERSION LOGIC
# ==========================================
def get_image_width(file_size_bytes):
    """
    Determines image width based on file size (standard malware image technique).
    """
    kb = file_size_bytes / 1024
    if kb < 10:
        return 32
    elif kb < 30:
        return 64
    elif kb < 60:
        return 128
    elif kb < 100:
        return 256
    elif kb < 200:
        return 384
    elif kb < 500:
        return 512
    elif kb < 1000:
        return 768
    else:
        return 1024


def binary_to_image(byte_array):
    """
    Converts a 1D byte array into a 224x224 RGB image with JET colormap.
    """
    if len(byte_array) == 0:
        return torch.zeros((3, 224, 224), dtype=torch.float32)

    width = get_image_width(len(byte_array))
    height = math.ceil(len(byte_array) / width)

    # Pad array to form a perfect rectangle
    padded_size = width * height
    if len(byte_array) < padded_size:
        padded_array = np.pad(
            byte_array, (0, padded_size - len(byte_array)), "constant"
        )
    else:
        padded_array = byte_array

    # Reshape to 2D
    img_2d = padded_array.reshape((height, width)).astype(np.uint8)

    # Apply Jet Colormap (as specified in IMCFN paper)
    img_color = cv2.applyColorMap(img_2d, cv2.COLORMAP_JET)

    # Resize to 224x224 (VGG16 input size)
    img_resized = cv2.resize(
        img_color, CONFIG["image_size"], interpolation=cv2.INTER_CUBIC
    )

    # Convert to PyTorch Tensor format (Channels, Height, Width) and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

    # Standard ImageNet Normalization (required for VGG16)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return normalize(tensor)


# ==========================================
# 3. DATASET CLASS (WITH OVERSAMPLING)
# ==========================================
class IMCFNDataset(Dataset):
    """
    Handles both physical files and .npz extracted datasets.
    Implements Data Balancing (Oversampling) to handle malware family imbalance.
    Safe for Multiprocessing (Lazy NPZ Loading).
    """

    def __init__(self, paths, label_mapping, balance_data=False):
        self.samples = []
        self.npz_metadata = []

        # Dictionary populated ONLY inside background workers to prevent pickling crashes!
        self.worker_open_files = {}

        # Force lowercase matching
        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}

        # Load data pointers
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            if path.endswith(".npz"):
                # Use 'with' to peek inside and instantly close the file!
                with np.load(path, allow_pickle=True) as data:
                    labels = data["label"]
                    for i in range(len(labels)):
                        l_str = str(labels[i]).strip().lower()
                        if l_str in self.label_mapping:
                            # Only store routing metadata (Path, Index, Label ID)
                            self.npz_metadata.append(
                                (path, i, self.label_mapping[l_str])
                            )
            else:
                for root, dirs, _ in os.walk(path):
                    folder_name = os.path.basename(root).lower()
                    if folder_name in self.label_mapping:
                        label_id = self.label_mapping[folder_name]
                        for f in os.listdir(root):
                            fpath = os.path.join(root, f)
                            if os.path.isfile(fpath):
                                self.samples.append((fpath, label_id))

        self.all_data = [{"type": "file", "data": s} for s in self.samples] + [
            {"type": "npz", "data": n} for n in self.npz_metadata
        ]

        if len(self.all_data) == 0:
            print(f"[!] Warning: No samples found in {paths}")

        # Explicit Data Balancing (Oversampling)
        if balance_data and len(self.all_data) > 0:
            self._balance_dataset()

    def _balance_dataset(self):
        class_counts = {}
        for item in self.all_data:
            # item["data"] is either (fpath, label) or (npz_path, idx, label)
            lbl = item["data"][-1]
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        max_count = max(class_counts.values())
        print(f"[*] Balancing Dataset. Target samples per class: {max_count}")

        balanced_data = []
        for class_idx in class_counts.keys():
            class_samples = [
                item for item in self.all_data if item["data"][-1] == class_idx
            ]
            multiplier = math.ceil(max_count / len(class_samples))
            oversampled = (class_samples * multiplier)[:max_count]
            balanced_data.extend(oversampled)

        self.all_data = balanced_data
        print(f"[*] Total samples after balancing: {len(self.all_data)}")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        item = self.all_data[idx]
        if item["type"] == "file":
            fpath, label = item["data"]
            try:
                with open(fpath, "rb") as f:
                    byte_array = np.frombuffer(f.read(), dtype=np.uint8)
            except:
                byte_array = np.array([], dtype=np.uint8)
        else:
            npz_path, item_idx, label = item["data"]

            # FAST LAZY LOADING: Cache the extracted arrays inside the spawned worker
            if npz_path not in self.worker_open_files:
                npz = np.load(npz_path, allow_pickle=True)
                self.worker_open_files[npz_path] = {
                    "raw_byte": npz.get("raw_byte", None)
                }

            raw_array = self.worker_open_files[npz_path]["raw_byte"]
            raw_val = raw_array[item_idx] if raw_array is not None else []
            byte_array = np.array(raw_val, dtype=np.uint8)

        img_tensor = binary_to_image(byte_array)
        return img_tensor, torch.tensor(label, dtype=torch.long)


# ==========================================
# 4. MAIN MODEL CONTROLLER
# ==========================================
class IMCFNModel(MalwareModelBase):
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

        print(f"[*] Loading IMCFN weights: {weights_path}")
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

        self.model = IMCFNArch(num_classes=num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def train(self, train_paths, val_paths, label_mapping, experiment_name=None):
        manager = ExperimentManager(
            os.path.dirname(__file__), experiment_name or "IMCFN_Run"
        )
        logger = Logger(manager.run_dir)
        manager.save_config(CONFIG)

        # FIXED: Match casing
        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}

        logger.log("[*] Initializing Datasets (Applying Oversampling to Training Set)")
        train_ds = IMCFNDataset(train_paths, self.label_mapping, balance_data=True)
        val_ds = IMCFNDataset(val_paths, self.label_mapping, balance_data=False)

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

        self.model = IMCFNArch(num_classes=len(self.label_mapping)).to(self.device)

        # Optimizer with L2 Regularization (Weight Decay) to satisfy paper's Eq 2
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=CONFIG["learning_rate"],
            momentum=CONFIG["momentum"],
            weight_decay=CONFIG["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

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
                logits = self.model(x)
                loss = criterion(logits, y)
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
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_acc = correct / total if total > 0 else 0

            # Validation
            self.model.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    v_loss += criterion(logits, y).item()
                    _, pred = torch.max(logits, 1)
                    v_total += y.size(0)
                    v_correct += (pred == y).sum().item()

            val_loss = v_loss / len(val_loader) if v_total > 0 else 0
            val_acc = v_correct / v_total if v_total > 0 else 0

            scheduler.step(val_acc)

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
                        "label_mapping": self.label_mapping,
                        "accuracy": best_acc,
                        "config": CONFIG,
                    },
                    manager.get_path("imcfn.pth"),
                )

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
                for i in tqdm(range(len(names)), desc="Predicting NPZ", leave=False):
                    malware_hash = names[i]
                    if ignore_list and malware_hash.split(".")[0] in ignore_list:
                        results[f"{input_path}::{malware_hash}"] = {"final_label": "IGNORED"}
                        continue

                    # FIX: Extract directly from the cached array
                    raw_val = raw_bytes_cache[i] if raw_bytes_cache is not None else []
                    byte_arr = np.array(raw_val, dtype=np.uint8)
                    tensor = binary_to_image(byte_arr).unsqueeze(0).to(self.device)

                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    winner_idx = torch.argmax(probs).item()

                    res = {
                        self.idx_to_label.get(idx, str(idx)): p.item()
                        for idx, p in enumerate(probs)
                    }
                    res["final_label"] = self.idx_to_label.get(
                        winner_idx, str(winner_idx)
                    )
                    results[f"{input_path}::{malware_hash}"] = res

            # FIX: Manually clean up memory after prediction is complete
            del data
            gc.collect()

        else:
            files = [input_path] if os.path.isfile(input_path) else []
            if os.path.isdir(input_path):
                for r, _, fs in os.walk(input_path):
                    files.extend([os.path.join(r, f) for f in fs])

            with torch.no_grad():
                for fpath in tqdm(files, desc="Predicting Files", leave=False):
                    base_name = os.path.basename(fpath)
                    if ignore_list and base_name.split(".")[0] in ignore_list:
                        results[fpath] = {"final_label": "IGNORED"}
                        continue
                        
                    try:
                        with open(fpath, "rb") as f:
                            byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
                        tensor = binary_to_image(byte_arr).unsqueeze(0).to(self.device)

                        logits = self.model(tensor)
                        probs = torch.softmax(logits, dim=1)[0]
                        winner_idx = torch.argmax(probs).item()

                        res = {
                            self.idx_to_label.get(idx, str(idx)): p.item()
                            for idx, p in enumerate(probs)
                        }
                        res["final_label"] = self.idx_to_label.get(
                            winner_idx, str(winner_idx)
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

        ds = IMCFNDataset(test_path, self.label_mapping, balance_data=False)
        loader = DataLoader(
            ds, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"]
        )

        y_true, y_pred, confs, correct_mask = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Evaluating", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
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

        # FIXED: Lowercase mapping
        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}

        report = {}
        for tech_path in tqdm(technique_paths, desc="Evasion Testing"):
            tech_name = os.path.basename(tech_path.rstrip("/\\")).replace(
                "_adv_samples", ""
            )
            passed, total = 0, 0
            
            ignore_list = get_ignored_samples(tech_name, tech_path=tech_path, label_mapping=self.label_mapping)
            preds = self.predict(tech_path, ignore_list=ignore_list)

            if tech_path.endswith(".npz"):
                data = np.load(tech_path, allow_pickle=True)
                for i, true_label in enumerate(data["label"]):
                    true_label_lower = str(true_label).strip().lower()
                    if is_benign_label(true_label_lower, self.label_mapping):
                        continue
                    res = preds.get(f"{tech_path}::{data['name'][i]}", {})
                    if res.get("final_label") == "ERROR" or not res:
                        continue
                    
                    total += 1
                    if res.get("final_label") == "IGNORED":
                        continue
                        
                    if str(res.get("final_label")).lower() != true_label_lower:
                        passed += 1
            else:
                for label_name in self.label_mapping.keys():
                    if is_benign_label(label_name, self.label_mapping):
                        continue
                    class_dir = os.path.join(tech_path, label_name)
                    if not os.path.isdir(class_dir):
                        continue

                    for fpath, res in preds.items():
                        if class_dir in fpath:
                            if res.get("final_label", "ERROR") == "ERROR":
                                continue
                            
                            total += 1
                            if res.get("final_label") == "IGNORED":
                                continue
                                
                            if str(res.get("final_label")).lower() != label_name:
                                passed += 1

            report[tech_name] = {
                "passed": passed,
                "total": total,
                "rate": (passed / total) if total > 0 else 0,
            }
            print(f"[*] Evasion Analysis: {tech_name} -> {passed}/{total} passed")

        return report
