import os
import copy
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import cv2
import zlib

# External dependencies
try:
    import pefile
except ImportError:
    pass

# Internal imports
from .arch import MAttnHealthArch
from ...core import MalwareModelBase
from ...utils import ChunkRandomSampler, calculate_optimal_chunk_size
from ...utils import (
    generate_confusion_matrix,
    calculate_metrics,
    plot_confidence_distribution,
    save_training_plot,
    Logger,
)

# ==========================================
# 1. CONFIGURATION (KAGGLE OPTIMIZED)
# ==========================================
CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 30,
    "num_workers": 0,  # CRITICAL: Prevents Kaggle shared memory deadlocks
    "max_raw_bytes": 1024,
    "max_sections": 4,
    "max_imports": 1000,
    "max_apis": 100,
}


# ==========================================
# 2. BULLETPROOF MULTI-CLASS DATASET LOADER
# ==========================================
class MAttnDataset(Dataset):
    def __init__(
        self, npz_path, api_vocab=None, label_map=None, max_apis=100, is_training=True
    ):
        self.npz_path = npz_path
        self.max_apis = max_apis
        self.label_map = label_map if label_map else {}
        self.worker_open_files = {}

        with np.load(npz_path, allow_pickle=True) as data:
            self.raw_labels = data["label"]

            self.api_vocab = api_vocab
            if is_training and self.api_vocab is None:
                apis = data["api_cuckoo"]
                self.api_vocab = self._build_vocab(apis)
                del apis

    def _build_vocab(self, apis_list):
        vocab = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for seq in apis_list:
            if seq is None:
                continue
            for api in seq:
                api_str = (
                    str(api) if not isinstance(api, dict) else str(list(api.keys())[0])
                )
                if api_str not in vocab:
                    vocab[api_str] = idx
                    idx += 1
        return vocab

    def _extract_numerics(self, obj):
        numerics = []
        if isinstance(obj, dict):
            for v in obj.values():
                numerics.extend(self._extract_numerics(v))
        elif isinstance(obj, (list, tuple, np.ndarray)):
            for v in obj:
                numerics.extend(self._extract_numerics(v))
        elif isinstance(obj, (int, float, np.integer, np.floating)):
            numerics.append(float(obj))
        elif isinstance(obj, str):
            try:
                numerics.append(float(obj))
            except:
                pass
        return numerics

    def _extract_strings(self, obj):
        strings = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                strings.append(str(k))
                strings.extend(self._extract_strings(v))
        elif isinstance(obj, (list, tuple, np.ndarray)):
            for v in obj:
                strings.extend(self._extract_strings(v))
        elif isinstance(obj, str):
            strings.append(obj)
        return strings

    def __len__(self):
        return len(self.raw_labels)

    def __getitem__(self, idx):
        # FAST LAZY LOADING: Cache the extracted arrays inside the spawned worker
        if self.npz_path not in self.worker_open_files:
            npz = np.load(self.npz_path, allow_pickle=True)
            self.worker_open_files[self.npz_path] = {
                "api_cuckoo": npz.get("api_cuckoo", None),
                "raw_byte": npz.get("raw_byte", None),
                "pe_sections": npz.get("pe_sections", None),
                "pe_imports": npz.get("pe_imports", None),
            }

        data = self.worker_open_files[self.npz_path]

        api_array = data["api_cuckoo"]
        api_raw = api_array[idx] if (api_array is not None and len(api_array) > idx and api_array[idx] is not None) else []
        api_tokens = []
        for api in api_raw:
            api_str = (
                str(api) if not isinstance(api, dict) else str(list(api.keys())[0])
            )
            api_tokens.append(self.api_vocab.get(api_str, 1) if self.api_vocab else 1)

        api_tokens = api_tokens[: self.max_apis]
        api_tokens += [0] * (self.max_apis - len(api_tokens))

        raw_bytes = np.array(data["raw_byte"][idx], dtype=np.float32)
        length = len(raw_bytes)
        width = int(np.ceil(np.sqrt(length))) if length > 0 else 32
        padded_bytes = np.pad(raw_bytes, (0, width * width - length))
        square_img = padded_bytes.reshape((width, width))

        img_2d_base = cv2.resize(square_img, (32, 32), interpolation=cv2.INTER_NEAREST)
        img_2d_base = img_2d_base / 255.0

        img_1d = img_2d_base.reshape(1, 1024)
        img_2d = np.stack([img_2d_base, img_2d_base, img_2d_base], axis=0)

        header_vals = self._extract_numerics(data["pe_sections"][idx])
        header_feat = np.array(header_vals, dtype=np.float32)
        if len(header_feat) > 4:
            header_feat = header_feat[:4]
        elif len(header_feat) < 4:
            header_feat = np.pad(header_feat, (0, 4 - len(header_feat)))

        import_strings = self._extract_strings(data["pe_imports"][idx])
        import_feat = np.zeros(1000, dtype=np.float32)
        for s in import_strings:
            idx_hash = zlib.crc32(s.encode("utf-8")) % 1000
            import_feat[idx_hash] = 1.0

        lbl_str = str(self.raw_labels[idx]).strip().lower()
        label_idx = self.label_map.get(lbl_str, 0)
        label = torch.tensor(label_idx, dtype=torch.long)

        return (
            torch.tensor(header_feat, dtype=torch.float32),
            torch.tensor(import_feat, dtype=torch.float32),
            torch.tensor(img_1d, dtype=torch.float32),
            torch.tensor(img_2d, dtype=torch.float32),
            torch.tensor(api_tokens, dtype=torch.float32).unsqueeze(0),
        ), label


# ==========================================
# 3. MULTI-CLASS MODEL WRAPPER
# ==========================================
class MAttnHealthModel(MalwareModelBase):
    def __init__(self, model_dir="trained_models/m_attn_health_multi"):
        super().__init__()
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.label_mapping = {}
        self.idx_to_label = {}
        self.api_vocab = {}

    def load_weights(self, weights_path=None):
        final_path = (
            weights_path
            if weights_path
            else os.path.join(self.model_dir, "m_attn_health_multi.pth")
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
            self.api_vocab = checkpoint.get("api_vocab", {})

            if self.label_mapping:
                self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
                num_classes = len(self.label_mapping)
            else:
                num_classes = 6

            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
            num_classes = 6

        self.model = MAttnHealthArch(num_classes=num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("    ✓ Model loaded successfully.")

    # RESTORED: Your original complete training logic
    def train(self, train_paths, val_paths, label_mapping, experiment_name=None):
        if experiment_name:
            self.model_dir = os.path.join("trained_models", experiment_name)
            os.makedirs(self.model_dir, exist_ok=True)

        logger = Logger(os.path.join(self.model_dir, "training.log"))

        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}

        num_classes = len(self.label_mapping)
        self.model = MAttnHealthArch(num_classes=num_classes)
        self.model.to(self.device)
        logger.log(
            f"[*] Starting M-Attn-Health MULTI-CLASS Training (Classes: {num_classes})"
        )

        train_npz = train_paths[0] if isinstance(train_paths, list) else train_paths
        val_npz = val_paths[0] if isinstance(val_paths, list) else val_paths

        train_dataset = MAttnDataset(
            train_npz, label_map=self.label_mapping, is_training=True
        )
        val_dataset = MAttnDataset(
            val_npz,
            api_vocab=train_dataset.api_vocab,
            label_map=self.label_mapping,
            is_training=False,
        )

        self.api_vocab = train_dataset.api_vocab
        class_names = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]

        chunk_sz = calculate_optimal_chunk_size(element_size_bytes=200000)
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False, sampler=ChunkRandomSampler(train_dataset, chunk_sz),
            num_workers=CONFIG["num_workers"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["learning_rate"])

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        best_f1 = 0.0
        best_epoch = 0
        best_model_state = None
        epochs = CONFIG["epochs"]

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for inputs, targets in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs}"
            ):
                inputs = [x.to(self.device) for x in inputs]
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(*inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)

            epoch_loss = train_loss / len(train_loader)
            epoch_acc = train_correct / train_total

            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_targets = [], []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = [x.to(self.device) for x in inputs]
                    targets = targets.to(self.device)

                    outputs = self.model(*inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            val_epoch_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            val_epoch_acc = val_correct / val_total if val_total > 0 else 0.0
            val_f1 = calculate_metrics(all_targets, all_preds, class_names)["f1_macro"]

            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)
            history["val_loss"].append(val_epoch_loss)
            history["val_acc"].append(val_epoch_acc)
            history["val_f1"].append(val_f1)

            logger.log(
                f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f} | "
                f"Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f} | Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(self.model.state_dict())
                logger.log(f"  ✓ Best model updated (F1: {best_f1:.4f})")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            save_path = os.path.join(self.model_dir, "m_attn_health_multi.pth")

            torch.save(
                {
                    "model_state": best_model_state,
                    "label_mapping": self.label_mapping,
                    "api_vocab": self.api_vocab,
                    "hyperparameters": CONFIG,
                    "history": history,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "f1_score": best_f1,
                    "best_epoch": best_epoch,
                    "num_epochs": CONFIG["epochs"],
                },
                save_path,
            )

            logger.log(f"[*] Training Complete. Best Weights saved to: {save_path}")

        save_training_plot(history, save_dir=self.model_dir)
        logger.close()

    def predict(self, target_path) -> dict:
        if self.model is None:
            self.load_weights()

        self.model.eval()
        results = {}

        if str(target_path).endswith(".npz"):
            data = np.load(target_path, allow_pickle=True)
            names = (
                data["name"]
                if "name" in data.files
                else [f"sample_{i}" for i in range(len(data["label"]))]
            )

            dataset = MAttnDataset(
                target_path,
                api_vocab=self.api_vocab,
                label_map=self.label_mapping,
                is_training=False,
            )
            loader = DataLoader(
                dataset,
                batch_size=CONFIG["batch_size"],
                shuffle=False,
                num_workers=CONFIG["num_workers"],
            )

            idx = 0
            with torch.no_grad():
                for inputs, _ in tqdm(loader, desc="Predicting M-Attn"):
                    inputs = [x.to(self.device) for x in inputs]
                    outputs = self.model(*inputs)

                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()

                    for i in range(len(probs)):
                        pred_idx = preds[i]
                        final_label = self.idx_to_label.get(
                            pred_idx, f"Unknown_{pred_idx}"
                        )

                        sample_res = {
                            self.idx_to_label.get(c_idx, f"Class_{c_idx}"): float(p)
                            for c_idx, p in enumerate(probs[i])
                        }
                        sample_res["final_label"] = final_label
                        results[f"{target_path}::{names[idx]}"] = sample_res
                        idx += 1
        return results

    def evaluate(self, test_path, output_dir=None) -> dict:
        if self.model is None:
            self.load_weights()

        preds = self.predict(test_path)

        y_true, y_pred, confs, correct_mask = [], [], [], []
        data = np.load(test_path, allow_pickle=True)
        names = (
            data["name"]
            if "name" in data.files
            else [f"sample_{i}" for i in range(len(data["label"]))]
        )
        labels = data["label"]

        for i in range(len(names)):
            # THE FIX: Force strings into Master Integer Map for utils.py compatibility
            true_lbl_str = str(labels[i]).strip().lower()
            true_idx = self.label_mapping.get(true_lbl_str, 0)

            res = preds.get(f"{test_path}::{names[i]}")
            if not res:
                continue

            final_label_str = res["final_label"].lower()
            pred_idx = self.label_mapping.get(final_label_str, 0)

            # Pass the integers directly!
            y_true.append(true_idx)
            y_pred.append(pred_idx)

            confs.append(res.get(res["final_label"], 0.0))
            correct_mask.append(final_label_str == true_lbl_str)

        out_dir = (
            output_dir if output_dir else os.path.join(self.model_dir, "evaluation")
        )
        os.makedirs(out_dir, exist_ok=True)

        class_names = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]
        metrics = calculate_metrics(y_true, y_pred, class_names)
        generate_confusion_matrix(y_true, y_pred, class_names, out_dir)
        plot_confidence_distribution(confs, correct_mask, out_dir)
        return metrics

    def evade(self, technique_paths, label_mapping) -> dict:
        if self.model is None:
            self.load_weights()

        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}

        report = {}
        for tech_path in tqdm(technique_paths, desc="Evasion Testing"):
            tech_name = os.path.basename(tech_path.rstrip("/\\")).replace(
                "_adv_samples", ""
            )
            passed, total = 0, 0
            preds = self.predict(tech_path)

            if tech_path.endswith(".npz"):
                data = np.load(tech_path, allow_pickle=True)
                names = data.get(
                    "name", [f"sample_{i}" for i in range(len(data["label"]))]
                )
                labels = data["label"]

                for i in range(len(names)):
                    true_label = str(labels[i]).strip().lower()
                    res = preds.get(f"{tech_path}::{names[i]}", {})
                    if not res:
                        continue
                    total += 1

                    if str(res.get("final_label")).lower() != true_label:
                        passed += 1
            else:
                for label_str in self.label_mapping.keys():
                    class_dir = os.path.join(tech_path, label_str)
                    if not os.path.isdir(class_dir):
                        continue

                    preds = self.predict(class_dir)
                    for fpath, res in preds.items():
                        if res.get("final_label", "ERROR") == "ERROR":
                            continue
                        total += 1

                        if str(res.get("final_label")).lower() != label_str.lower():
                            passed += 1

            rate = (passed / total) if total > 0 else 0.0
            report[tech_name] = {"passed": passed, "total": total, "rate": rate}
            print(
                f"[*] M-Attn-Health (Multi-Class) Evasion Analysis: {tech_name} -> {passed}/{total} passed"
            )

        return report
