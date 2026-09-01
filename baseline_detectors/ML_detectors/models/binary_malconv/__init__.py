import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
import json

from baseline_detectors.core import MalwareModelBase
from baseline_detectors.utils import (
    ExperimentManager,
    Logger,
    generate_confusion_matrix,
    calculate_metrics,
    plot_confidence_distribution,
    save_training_plot,
    get_ignored_samples,
    is_benign_label,
)
from .arch import MalConvArch, MALCONV_CFG

torch.set_num_threads(12)


# ==========================================
# HELPER: DeCov Loss
# ==========================================
def decov_loss(features):
    """
    Calculates DeCov loss to reduce overfitting.
    """
    features_mean = torch.mean(features, dim=0, keepdim=True)
    features_centered = features - features_mean
    n = features.size(0)

    if n <= 1:
        return torch.tensor(0.0, device=features.device)

    corr = (1 / (n - 1)) * torch.matmul(features_centered.t(), features_centered)
    loss = 0.5 * (
        torch.norm(corr, p="fro") ** 2 - torch.norm(torch.diag(corr), p=2) ** 2
    )
    return loss


# ==========================================
# DATASET (LAZY WORKER LOADING FOR INFINITE SCALE)
# ==========================================
class BytesDataset(Dataset):
    def __init__(self, paths, max_len, padding_val):
        self.max_len = max_len
        self.padding_val = padding_val
        self.samples = []  # (filepath, binary_label)
        self.npz_metadata = []  # (npz_path, item_idx, binary_label)

        # This dictionary will be populated INSIDE the background workers
        # ensuring PyTorch never tries to pickle an open file handle!
        self.worker_open_files = {}

        if isinstance(paths, str):
            paths = [paths]

        for p in paths:
            if p.endswith(".npz"):
                # 1. Peek inside temporarily just to map the routing index
                with np.load(p, allow_pickle=True) as data:
                    labels = data["label"]
                    for i in range(len(labels)):
                        binary_label = 0 if "benign" in str(labels[i]).lower() else 1
                        # Save the routing instructions, NOT the data
                        self.npz_metadata.append((p, i, binary_label))
            else:
                for root, dirs, _ in os.walk(p):
                    for d in dirs:
                        binary_label = 0 if "benign" in d.lower() else 1
                        target_dir = os.path.join(root, d)
                        for r, _, files in os.walk(target_dir):
                            for f in files:
                                if not f.startswith("."):
                                    self.samples.append(
                                        (os.path.join(r, f), binary_label)
                                    )

    def __len__(self):
        return len(self.samples) + len(self.npz_metadata)

    def __getitem__(self, idx):
        if idx < len(self.samples):
            path, label = self.samples[idx]
            try:
                with open(path, "rb") as f:
                    content = f.read(self.max_len)
                byte_arr = np.frombuffer(content, dtype=np.uint8)
            except:
                byte_arr = np.array([], dtype=np.uint8)
        else:
            npz_idx = idx - len(self.samples)
            npz_path, item_idx, label = self.npz_metadata[npz_idx]

            # 2. FAST LAZY LOADING: Cache the extracted arrays inside the spawned worker
            if npz_path not in self.worker_open_files:
                npz = np.load(npz_path, allow_pickle=True)
                self.worker_open_files[npz_path] = {
                    "raw_byte": npz.get("raw_byte", None)
                }

            # Stream the specific raw bytes directly from the cached array
            raw_array = self.worker_open_files[npz_path]["raw_byte"]
            raw_bytes = raw_array[item_idx] if raw_array is not None else []
            byte_arr = np.array(raw_bytes, dtype=np.uint8)

        padded = np.ones(self.max_len, dtype=np.int64) * self.padding_val
        ln = min(len(byte_arr), self.max_len)
        padded[:ln] = byte_arr[:ln]

        return torch.from_numpy(padded), torch.tensor(label, dtype=torch.long)


# ==========================================
# MAIN CONTROLLER
# ==========================================
class BinaryMalConv(MalwareModelBase):
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_cfg = MALCONV_CFG.copy()

        self.model_cfg["num_classes"] = 1
        self.model_cfg["max_len"] = 2000000
        self.class_names = ["Benign", "Malware"]

        self.train_cfg = {
            "batch_size": 16,
            "accum_steps": 16,
            "epochs": 10,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "num_workers": 2,
            "padding_char": 256,
            "decov_strength": 0.1,
        }

    def load_weights(self, weights_path):
        print(f"[*] Loading weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model_cfg = ckpt.get("cfg", self.model_cfg)
        self.model_cfg["num_classes"] = 1
        self.model_cfg["max_len"] = 2000000

        self.model = MalConvArch(self.model_cfg).to(self.device)
        self.model.load_state_dict(
            ckpt["model_state"] if "model_state" in ckpt else ckpt
        )
        self.model.eval()

    def train(
        self,
        train_paths,
        val_paths,
        label_mapping,
        experiment_name=None,
        overrides=None,
    ):
        if overrides:
            self.train_cfg.update(overrides)

        manager = ExperimentManager(
            "predic_models", experiment_name=experiment_name or "Binary_MalConv_Run"
        )
        logger = Logger(manager.run_dir)

        logger.log("[*] Starting Binary Auto-Label Training")

        train_loader = DataLoader(
            BytesDataset(
                train_paths, self.model_cfg["max_len"], self.train_cfg["padding_char"]
            ),
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            num_workers=self.train_cfg["num_workers"],
        )
        val_loader = DataLoader(
            BytesDataset(
                val_paths, self.model_cfg["max_len"], self.train_cfg["padding_char"]
            ),
            batch_size=self.train_cfg["batch_size"],
            shuffle=False,
            num_workers=self.train_cfg["num_workers"],
        )

        self.model = MalConvArch(self.model_cfg).to(self.device)
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            momentum=self.train_cfg["momentum"],
            nesterov=True,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_acc = 0.0
        accum = self.train_cfg["accum_steps"]
        decov_lambda = self.train_cfg.get("decov_strength", 0.1)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.train_cfg["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            logger.log(f"\n--- Epoch {epoch + 1} ---")

            loop = tqdm(train_loader, desc="Train")
            total_loss, correct, total = 0, 0, 0

            for i, (x, y) in enumerate(loop):
                x, y = x.to(self.device), y.to(self.device)
                out, hidden = self.model(x)

                cls_loss = criterion(out.view(-1), y.float())
                reg_loss = decov_loss(hidden)
                loss = cls_loss + (decov_lambda * reg_loss)

                (loss / accum).backward()

                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    if self.device.type == "xla":
                        import torch_xla.core.xla_model as xm

                        xm.optimizer_step(optimizer, barrier=True)
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                total_loss += loss.item()
                preds = (torch.sigmoid(out) > 0.5).long().view(-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                loop.set_postfix(loss=f"{loss.item():.4f}")

            train_acc = correct / total if total > 0 else 0
            val_acc = self._evaluate_loop(val_loader, criterion)

            history["train_loss"].append(
                total_loss / len(train_loader) if len(train_loader) > 0 else 0
            )
            history["train_acc"].append(train_acc)
            history["val_loss"].append(0)
            history["val_acc"].append(val_acc)

            logger.log(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "cfg": self.model_cfg,
                        "classes": self.class_names,
                    },
                    manager.get_path("best_model.pth"),
                )
                logger.log("  ✓ Best model saved")

        save_training_plot(history, manager.run_dir)
        logger.close()

    def _evaluate_loop(self, loader, criterion):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.model(x)
                preds = (torch.sigmoid(out) > 0.5).long().view(-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0

    def predict(self, input_path, ignore_list=None) -> dict:
        if not self.model:
            raise RuntimeError("Load weights first")
        self.model.eval()
        results = {}
        maxlen = self.model_cfg["max_len"]
        pad = self.train_cfg["padding_char"]

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
                        
                    # FIX: Extract directly from the dictionary block one at a time
                    byte_arr = np.array(
                        raw_bytes_cache[i] if raw_bytes_cache is not None else [],
                        dtype=np.uint8,
                    )
                    padded = np.ones(maxlen, dtype=np.int64) * pad
                    ln = min(len(byte_arr), maxlen)
                    padded[:ln] = byte_arr[:ln]

                    inp = torch.from_numpy(padded).unsqueeze(0).to(self.device)
                    out, _ = self.model(inp)

                    virus_prob = torch.sigmoid(out).item()
                    results[f"{input_path}::{names[i]}"] = {
                        "Benign": 1.0 - virus_prob,
                        "Malware": virus_prob,
                        "final_label": "Malware" if virus_prob > 0.5 else "Benign",
                    }

            # FIX: Manually clean up memory after prediction is complete
            del data
            gc.collect()

        else:
            files = [input_path] if os.path.isfile(input_path) else []
            if os.path.isdir(input_path):
                for r, _, fl in os.walk(input_path):
                    files.extend([os.path.join(r, f) for f in fl])

            with torch.no_grad():
                for fpath in tqdm(files, desc="Predicting Files"):
                    base_name = os.path.basename(fpath)
                    if ignore_list and base_name.split(".")[0] in ignore_list:
                        results[fpath] = {'final_label': 'IGNORED'}
                        continue
                        
                    try:
                        with open(fpath, "rb") as f:
                            content = f.read(maxlen)
                        byte_arr = np.frombuffer(content, dtype=np.uint8)
                        padded = np.ones(maxlen, dtype=np.int64) * pad
                        ln = min(len(byte_arr), maxlen)
                        padded[:ln] = byte_arr[:ln]

                        inp = torch.from_numpy(padded).unsqueeze(0).to(self.device)
                        out, _ = self.model(inp)

                        virus_prob = torch.sigmoid(out).item()
                        results[fpath] = {
                            "Benign": 1.0 - virus_prob,
                            "Malware": virus_prob,
                            "final_label": "Malware" if virus_prob > 0.5 else "Benign",
                        }
                    except Exception as e:
                        results[fpath] = {"final_label": "ERROR", "details": str(e)}
        return results

    def evaluate(self, test_path, output_dir=None) -> dict:
        if not self.model:
            raise RuntimeError("Model not loaded")
        output_dir = output_dir or "results"
        os.makedirs(output_dir, exist_ok=True)

        ds = BytesDataset(
            test_path, self.model_cfg["max_len"], self.train_cfg["padding_char"]
        )
        loader = DataLoader(
            ds,
            batch_size=self.train_cfg["batch_size"],
            shuffle=False,
            num_workers=self.train_cfg["num_workers"],
        )

        y_true, y_pred, confs, correct_mask = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Evaluating"):
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.model(x)

                probs = torch.sigmoid(out).view(-1)
                preds = (probs > 0.5).long()

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                batch_confs = torch.max(probs, 1.0 - probs)
                confs.extend(batch_confs.cpu().numpy())
                correct_mask.extend((preds == y).cpu().numpy())

        metrics = calculate_metrics(y_true, y_pred, self.class_names)
        generate_confusion_matrix(y_true, y_pred, self.class_names, output_dir)
        plot_confidence_distribution(confs, correct_mask, output_dir)
        return metrics

    def evade(self, technique_paths, label_mapping) -> dict:
        report = {}
        for tech_path in tqdm(technique_paths, desc="Evasion Testing"):
            tech_name = os.path.basename(tech_path.rstrip("/\\")).replace(
                "_adv_samples", ""
            )
            passed, total = 0, 0
            
            ignore_list = get_ignored_samples(tech_name, tech_path=tech_path, label_mapping=label_mapping)
            preds = self.predict(tech_path, ignore_list=ignore_list)

            if tech_path.endswith(".npz"):
                data = np.load(tech_path, allow_pickle=True)
                labels = data.get("label", [])
                names = data.get("name", [f"sample_{i}" for i in range(len(labels))])
                for i in range(len(names)):
                    true_label = str(labels[i]).strip().lower()
                    if is_benign_label(true_label, label_mapping):
                        continue
                    res = preds.get(f"{tech_path}::{names[i]}", preds.get(names[i], {}))
                    if res.get("final_label") == "ERROR" or not res:
                        continue
                    total += 1
                    if res.get("final_label") == "IGNORED":
                        continue
                    # For binary models, all non-benign samples have ground truth label 'Malware'.
                    # Evasion passes if the model fails to predict 'Malware' (e.g., predicts 'Benign').
                    if str(res.get("final_label")).lower() != "malware":
                        passed += 1
            else:
                for path_key, res in preds.items():
                    if res.get("final_label") == "ERROR" or not res:
                        continue

                    true_label = "Benign" if is_benign_label(path_key, label_mapping) or "benign" in path_key.lower() else "Malware"
                    if true_label == "Benign":
                        continue

                    total += 1

                    if res.get("final_label") == "IGNORED":
                        continue

                    if res.get("final_label") != true_label:
                        passed += 1

            report[tech_name] = {
                "passed": passed,
                "total": total,
                "rate": (passed / total) if total > 0 else 0,
            }
            print(f"[*] Evasion {tech_name}: {passed}/{total} passed")

        return report
