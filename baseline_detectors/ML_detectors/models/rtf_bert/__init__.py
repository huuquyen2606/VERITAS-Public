import os
import glob
import json
import torch
import torch.nn as nn
import numpy as np
import warnings
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import transformers

transformers.logging.set_verbosity_error()
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
import pandas as pd
import gc

# Internal imports
from ...core import MalwareModelBase
from ...utils import ChunkRandomSampler, calculate_optimal_chunk_size
from ...utils import (
    generate_confusion_matrix,
    calculate_metrics,
    plot_confidence_distribution,
    Logger,
    ExperimentManager,
    get_ignored_samples,
    is_benign_label,
)
from .arch import BertClassifier

# Environment settings
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# External dependencies
try:
    import pefile
except ImportError:
    print("⚠️ pefile not installed. Install with: pip install pefile")
    raise

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================

CONFIG = {
    "seed": 42,
    "n_estimators": 6,
    "max_length": 512,
    "hidden_dim": 768,
    "batch_size": 4,  # 8
    "hidden_dim": 768,
    "learning_rate": 2e-5,
    "num_epochs": 4,
    "validation_split": 0.2,
    "model_name": "bert-base-uncased",
    "early_stopping_patience": 2,
}

# ==========================================
# 2. API EXTRACTION & PREPROCESSING
# ==========================================


def extract_api_calls(pe_file_path):
    """Static extracts API calls from a PE file."""
    try:
        pe = pefile.PE(pe_file_path)
        api_calls = [
            imp.name.decode("utf-8", errors="ignore")
            for entry in pe.DIRECTORY_ENTRY_IMPORT
            for imp in entry.imports
            if imp.name
        ]
        res = " ".join(api_calls)
        return res if res else "CreateFile"
    except:
        return "CreateFile"


def _remove_consecutive_repeats(sequence, n):
    """Helper function to remove consecutive repetitive n-grams."""
    if len(sequence) < 2 * n:
        return sequence

    new_seq = []
    i = 0
    while i < len(sequence):
        if i + 2 * n <= len(sequence):
            chunk1 = tuple(sequence[i : i + n])
            chunk2 = tuple(sequence[i + n : i + 2 * n])

            if chunk1 == chunk2:
                new_seq.extend(sequence[i : i + n])
                i += n
                while i + n <= len(sequence) and tuple(sequence[i : i + n]) == chunk1:
                    i += n
                continue

        if n == 1:
            if not new_seq or sequence[i] != new_seq[-1]:
                new_seq.append(sequence[i])
            i += 1
        else:
            new_seq.append(sequence[i])
            i += 1

    return new_seq


def preprocess_api_sequence(api_string):
    """Applies the 3-step pre-processing described in the Paper."""
    if not api_string:
        return ""

    apis = api_string.strip().split()
    apis = _remove_consecutive_repeats(apis, 1)
    apis = _remove_consecutive_repeats(apis, 2)
    apis = _remove_consecutive_repeats(apis, 3)
    return " ".join(apis)


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_bootstrap_split(X, y, seed, val_split=0.2):
    """Generates a bootstrapped training and validation split."""
    np.random.seed(seed)
    boot_indices = np.random.choice(len(X), size=len(X), replace=True)
    X_boot = X[boot_indices]
    y_boot = y[boot_indices]
    return train_test_split(
        X_boot, y_boot, test_size=val_split, random_state=seed, stratify=y_boot
    )


# ==========================================
# 3. DATASET CLASS
# ==========================================


class MalwareDataset(Dataset):
    """PyTorch Dataset for handling API call sequences with BERT tokenization."""

    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = self.labels[idx]

        # FIXED: Call tokenizer directly!
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ==========================================
# 4. MAIN MODEL CONTROLLER
# ==========================================


class RTFBertModel(MalwareModelBase):
    """Random Tree Forest + BERT (RTF-BERT) Ensemble Model."""

    def __init__(self):
        super().__init__()
        self.models = []
        self.tokenizer = None
        self.idx_to_label = {}
        self.label_mapping = {}

        base_dir = os.path.dirname(__file__)
        self.local_weights_dir = os.path.join(base_dir, "weights")
        self.local_results_dir = os.path.join(base_dir, "results")

    def _resolve_weights_path(self, path):
        if os.path.exists(path) and os.path.isdir(path):
            return path
        local_path = os.path.join(self.local_weights_dir, path)
        if os.path.exists(local_path) and os.path.isdir(local_path):
            return local_path
        raise FileNotFoundError(f"Could not find weights directory at {path}")

    def load_weights(self, weights_dir, num_classes=None):
        final_path = self._resolve_weights_path(weights_dir)
        print(f"[*] Loading RTF-BERT ensemble from: {final_path}")

        self.tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])

        json_map_path = os.path.join(final_path, "label_map.json")
        if os.path.exists(json_map_path):
            with open(json_map_path, "r") as f:
                self.label_mapping = json.load(f)
            self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
            num_classes = len(self.label_mapping)
        elif num_classes is not None:
            self.label_mapping = {str(i): i for i in range(num_classes)}
            self.idx_to_label = {i: str(i) for i in range(num_classes)}

        self.models = []

        for i in range(CONFIG["n_estimators"]):
            model_path = os.path.join(final_path, f"bert_model_{i}.pth")

            if i == 0 and not os.path.exists(model_path):
                best_path = os.path.join(final_path, "rtf_best.pth")
                if os.path.exists(best_path):
                    model_path = best_path

            if not os.path.exists(model_path):
                continue

            try:
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )

                if not self.label_mapping and "label_mapping" in checkpoint:
                    self.label_mapping = checkpoint["label_mapping"]
                    self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
                    num_classes = len(self.label_mapping)

                if num_classes is None:
                    num_classes = 6

                model = BertClassifier(
                    model_name=CONFIG["model_name"],
                    num_classes=num_classes,
                    hidden_dim=CONFIG["hidden_dim"],
                )

                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                else:
                    state_dict = checkpoint

                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"    - Loaded: {os.path.basename(model_path)}")

            except Exception as e:
                print(f"[!] Error loading {os.path.basename(model_path)}: {e}")

        if not self.models:
            raise RuntimeError("No RTF-BERT models loaded!")

    def _predict_sequence(self, api_seq):
        """Helper to run inference on a single API string."""
        if not api_seq or not isinstance(api_seq, str):
            api_seq = "CreateFile"

        # FIXED: Call tokenizer directly!
        encoding = self.tokenizer(
            api_seq,
            add_special_tokens=True,
            max_length=CONFIG["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        mask = encoding["attention_mask"].to(self.device)

        all_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(input_ids, mask)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        pred_idx = torch.argmax(avg_probs, dim=1).item()

        file_result = {}
        for idx, class_name in self.idx_to_label.items():
            file_result[class_name] = avg_probs[0, idx].item()

        file_result["final_label"] = self.idx_to_label.get(pred_idx, str(pred_idx))
        return file_result

    def predict(self, input_path, ignore_list=None):
        if not self.models:
            raise RuntimeError("Models not loaded.")

        results = {}

        if input_path.endswith(".npz"):
            print(
                f"[*] Predicting on extracted dataset (Cuckoo Sandbox APIs): {input_path}"
            )
            data = np.load(input_path, allow_pickle=True)
            names = data["name"]
            cuckoo_apis = data["api_cuckoo"]

            for i in tqdm(range(len(names)), desc="Predicting NPZ"):
                malware_hash = names[i]
                if ignore_list and malware_hash.split(".")[0] in ignore_list:
                    results[f"{input_path}::{malware_hash}"] = {"final_label": "IGNORED"}
                    continue
                    
                raw_api = cuckoo_apis[i]

                if isinstance(raw_api, list) or isinstance(raw_api, np.ndarray):
                    raw_api = " ".join([str(x) for x in raw_api])

                clean_api_seq = preprocess_api_sequence(str(raw_api))

                try:
                    results[f"{input_path}::{malware_hash}"] = self._predict_sequence(
                        clean_api_seq
                    )
                except Exception as e:
                    results[f"{input_path}::{malware_hash}"] = {
                        "final_label": "ERROR",
                        "details": str(e),
                    }
        else:
            files = [input_path] if os.path.isfile(input_path) else []
            if os.path.isdir(input_path):
                for r, _, fs in os.walk(input_path):
                    for f in fs:
                        files.append(os.path.join(r, f))

            for fpath in tqdm(files, desc="Predicting Files"):
                base_name = os.path.basename(fpath)
                if ignore_list and base_name.split(".")[0] in ignore_list:
                    results[fpath] = {"final_label": "IGNORED"}
                    continue
                    
                try:
                    raw_api_seq = extract_api_calls(fpath)
                    clean_api_seq = preprocess_api_sequence(raw_api_seq)
                    results[fpath] = self._predict_sequence(clean_api_seq)
                except Exception as e:
                    results[fpath] = {"final_label": "ERROR", "details": str(e)}

        return results

    # FIXED: Signature now uses train_paths / val_paths
    def train(self, train_paths, val_paths, label_mapping, experiment_name=None):
        manager = ExperimentManager(
            base_dir=os.path.dirname(__file__), experiment_name=experiment_name
        )
        logger = Logger(manager.run_dir)
        logger.log(f"[*] Starting RTF-BERT Training Session")
        manager.save_config(CONFIG)

        with open(os.path.join(manager.run_dir, "label_map.json"), "w") as f:
            json.dump(label_mapping, f, indent=4)

        set_seed(CONFIG["seed"])

        # FIXED: Enforce lowercase mappings
        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])

        train_samples = []
        train_paths_list = (
            train_paths if isinstance(train_paths, list) else [train_paths]
        )

        logger.log("[*] Extracting API sequences for training...")
        for path in train_paths_list:
            if path.endswith(".npz"):
                data = np.load(path, allow_pickle=True)
                labels = data["label"]
                cuckoo_apis = data["api_cuckoo"]
                for i in range(len(labels)):
                    # FIXED: Strip and lowercase NPZ labels
                    l_str = str(labels[i]).strip().lower()
                    if l_str in self.label_mapping:
                        raw_api = cuckoo_apis[i]
                        if isinstance(raw_api, list) or isinstance(raw_api, np.ndarray):
                            raw_api = " ".join([str(x) for x in raw_api])
                        clean_api = preprocess_api_sequence(str(raw_api))
                        if clean_api and len(clean_api.split()) > 5:
                            train_samples.append(
                                {
                                    "api_seq": clean_api,
                                    "label": self.label_mapping[l_str],
                                }
                            )
            else:
                for dirpath, _, filenames in os.walk(path):
                    folder_name = os.path.basename(dirpath).lower()
                    if folder_name in self.label_mapping:
                        label_id = self.label_mapping[folder_name]
                        for f in filenames:
                            fpath = os.path.join(dirpath, f)
                            if os.path.isfile(fpath):
                                raw_seq = extract_api_calls(fpath)
                                clean_seq = preprocess_api_sequence(raw_seq)
                                if clean_seq and len(clean_seq.split()) > 5:
                                    train_samples.append(
                                        {"api_seq": clean_seq, "label": label_id}
                                    )

        df = pd.DataFrame(train_samples)

        # FIXED: Catch empty dataset immediately to prevent KeyError
        if df.empty:
            raise ValueError(
                "No training samples matched the label mapping! Dataset is empty."
            )

        X_all = df["api_seq"].values
        y_all = df["label"].values

        logger.log(f"[*] Loaded {len(X_all)} valid samples")
        os.makedirs(os.path.join(manager.run_dir, "models"), exist_ok=True)

        for i in range(CONFIG["n_estimators"]):
            logger.log(f"\nTraining Estimator {i + 1}/{CONFIG['n_estimators']}")

            X_train, X_val, y_train, y_val = generate_bootstrap_split(
                X_all,
                y_all,
                seed=CONFIG["seed"] + i,
                val_split=CONFIG["validation_split"],
            )

            train_ds = MalwareDataset(
                X_train, y_train, self.tokenizer, CONFIG["max_length"]
            )
            val_ds = MalwareDataset(X_val, y_val, self.tokenizer, CONFIG["max_length"])

            chunk_sz = calculate_optimal_chunk_size(element_size_bytes=200000)
            train_dl = DataLoader(
                train_ds,
                batch_size=CONFIG["batch_size"],
                shuffle=False,
                sampler=ChunkRandomSampler(train_ds, chunk_sz),
                num_workers=0,
            )
            val_dl = DataLoader(
                val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
            )

            model = BertClassifier(
                model_name=CONFIG["model_name"],
                num_classes=len(label_mapping),
                hidden_dim=CONFIG["hidden_dim"],
            ).to(self.device)

            optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
            total_steps = len(train_dl) * CONFIG["num_epochs"]
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps
            )
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            for epoch in range(CONFIG["num_epochs"]):
                model.train()
                train_loss = 0
                for batch in tqdm(train_dl, desc=f"Ep{epoch + 1}", leave=False):
                    ids = batch["input_ids"].to(self.device)
                    mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    logits = model(ids, mask)
                    loss = criterion(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    if self.device.type == "xla":
                        import torch_xla.core.xla_model as xm

                        xm.optimizer_step(optimizer, barrier=True)
                    else:
                        optimizer.step()
                    scheduler.step()
                    train_loss += loss.item()

                model.eval()
                val_loss = 0
                preds, true_lbs = [], []

                with torch.no_grad():
                    for batch in val_dl:
                        ids = batch["input_ids"].to(self.device)
                        mask = batch["attention_mask"].to(self.device)
                        labels = batch["label"].to(self.device)

                        logits = model(ids, mask)
                        loss = criterion(logits, labels)
                        val_loss += loss.item()

                        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        true_lbs.extend(labels.cpu().numpy())

                val_loss /= len(val_dl)
                f1 = f1_score(true_lbs, preds, average="macro", zero_division=0)

                logger.log(
                    f"  Ep {epoch + 1}: Val Loss={val_loss:.4f}, Val F1={f1:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_f1_macro": f1,
                        "label_mapping": self.label_mapping,
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= CONFIG["early_stopping_patience"]:
                        logger.log("  Early stopping")
                        break

            if best_state:
                save_path = os.path.join(manager.run_dir, f"bert_model_{i}.pth")
                torch.save(best_state, save_path)
                logger.log(
                    f"Estimator {i + 1} Saved. Best F1: {best_state['val_f1_macro']:.4f}"
                )

            # ==========================================
            # RAM FIX: Force clear memory before next loop
            # ==========================================
            del model
            del optimizer
            del scheduler
            del train_dl
            del val_dl
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.close()

    def evaluate(self, test_path, output_dir=None):
        if output_dir is None:
            output_dir = self.local_results_dir
        os.makedirs(output_dir, exist_ok=True)

        preds_list, labels_list, conf_list = [], [], []

        if test_path.endswith(".npz"):
            preds = self.predict(test_path)
            data = np.load(test_path, allow_pickle=True)
            names, labels = data["name"], data["label"]

            for i in range(len(names)):
                # FIXED: Force lowercase
                true_label_str = str(labels[i]).strip().lower()
                if true_label_str not in self.label_mapping:
                    continue

                res = preds.get(f"{test_path}::{names[i]}", {})
                if res.get("final_label", "ERROR") == "ERROR":
                    continue

                real_label_idx = self.label_mapping[true_label_str]
                pred_label_idx = self.label_mapping.get(res["final_label"], -1)

                preds_list.append(pred_label_idx)
                labels_list.append(real_label_idx)

        else:
            for root, _, files in os.walk(test_path):
                folder_name = os.path.basename(root).lower()
                if folder_name not in self.label_mapping:
                    continue

                real_label_idx = self.label_mapping[folder_name]

                for f in files:
                    fpath = os.path.join(root, f)
                    res = self.predict(fpath).get(fpath, {})
                    if res.get("final_label", "ERROR") != "ERROR":
                        pred_label_idx = self.label_mapping.get(res["final_label"], -1)
                        preds_list.append(pred_label_idx)
                        labels_list.append(real_label_idx)

        class_names = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]
        metrics = calculate_metrics(labels_list, preds_list, class_names)
        generate_confusion_matrix(
            labels_list, preds_list, class_names, save_dir=output_dir
        )

        return metrics

    def evade(self, technique_paths, label_mapping) -> dict:
        if not self.models:
            raise RuntimeError("RTF-BERT models must be loaded before evasion testing.")

        # FIXED: Lowercase mapping
        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        results_report = {}

        for tech_path in tqdm(technique_paths, desc="Evasion Testing"):
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
                    # FIXED: Lowercase true label
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
                            
                        if str(res.get("final_label")).lower() != str(label_str).lower():
                            passed += 1

            rate = (passed / total) if total > 0 else 0.0
            results_report[tech_name] = {"passed": passed, "total": total, "rate": rate}
            print(f"[*] RTF-BERT Evasion analysis complete for: {tech_name}")

        return results_report
