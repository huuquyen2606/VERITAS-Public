import os
import gc
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import pandas as pd
import json
import warnings
from tqdm.auto import tqdm

# --- ENVIRONMENT CONFIG ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# External dependencies
try:
    import pefile
except ImportError:
    print("pefile not installed. Install with: pip install pefile")
    raise

from transformers import (
    CanineTokenizer,
    CanineForSequenceClassification,
)
import transformers

transformers.logging.set_verbosity_error()

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score

# Internal imports
from ...core import MalwareModelBase
from ...utils import (
    generate_confusion_matrix,
    calculate_metrics,
    plot_confidence_distribution,
    Logger,
    ExperimentManager,
    get_ignored_samples,
    is_benign_label,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================

CONFIG = {
    "seed": 42,
    "n_estimators": 10,  # closest to the highest f1 score in the paper
    "max_length": 2048,  # full size
    "batch_size": 2,  # we got to lower this to not overflood the kaggle GPU !, original is 8
    "learning_rate": 3e-5,
    "weight_decay": 1e-3,
    "num_epochs": 5,
    "early_stopping_patience": 2,
    "warmup_ratio": 0.1,
    "validation_split": 0.2,
    "model_name": "google/canine-s",
}

# ==========================================
# 2. API EXTRACTION
# ==========================================


def extract_api_calls(pe_file_path):
    """
    Static extracts the sequence of imported API calls from a PE file.
    """
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


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================


def stratified_bootstrap(df, random_state=None):
    """
    Performs stratified resampling (bootstrapping) on a DataFrame.
    """
    return (
        df.groupby("label", group_keys=False)
        .apply(
            lambda x: resample(
                x, n_samples=len(x), replace=True, random_state=random_state
            )
        )
        .reset_index(drop=True)
    )


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 4. MAIN MODEL CONTROLLER
# ==========================================


class RTFCannieModel(MalwareModelBase):
    """
    Random Tree Forest + CANINE (RTF-CANNIE) Ensemble Model.
    """

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
        print(f"[*] Loading RTF-CANNIE ensemble from: {final_path}")

        self.tokenizer = CanineTokenizer.from_pretrained(CONFIG["model_name"])

        # Load Mapping
        json_map_path = os.path.join(final_path, "label_map.json")
        if os.path.exists(json_map_path):
            print(f"[*] Found label_map.json at {json_map_path}")
            with open(json_map_path, "r") as f:
                self.label_mapping = json.load(f)
            self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
            num_classes = len(self.label_mapping)
        elif num_classes is not None:
            print(f"[!] Warning: using numeric labels 0-{num_classes - 1}")
            self.label_mapping = {str(i): i for i in range(num_classes)}
            self.idx_to_label = {i: str(i) for i in range(num_classes)}

        self.models = []

        for i in range(1, CONFIG["n_estimators"] + 1):
            model_path = os.path.join(final_path, f"RTF_CANINE_model_{i}.pth")

            if not os.path.exists(model_path):
                model_path = os.path.join(final_path, f"RTF_CANNIE_model_{i}.pth")

            if not os.path.exists(model_path):
                print(f"[!] Warning: Model file {model_path} not found. Skipping.")
                continue

            try:
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )

                if not self.label_mapping and "label_mapping" in checkpoint:
                    self.label_mapping = checkpoint["label_mapping"]
                    self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
                    num_classes = len(self.label_mapping)

                model = CanineForSequenceClassification.from_pretrained(
                    CONFIG["model_name"],
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
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
                print(f"[*] Loaded Estimator {i}/{CONFIG['n_estimators']}")

                # Clear up CPU RAM used during downloading/loading state_dicts
                del checkpoint
                del state_dict
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[!] Error loading {os.path.basename(model_path)}: {e}")

        if not self.models:
            raise RuntimeError("No models loaded! Check path and filenames.")

        print(f"[*] RTF-CANNIE Forest ready with {len(self.models)} estimators")

    def _predict_sequence(self, api_seq):
        """Helper to run inference on a single API string."""
        if not api_seq or not isinstance(api_seq, str):
            api_seq = "CreateFile"

        encoding = self.tokenizer(
            api_seq,
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
                outputs = model(input_ids=input_ids, attention_mask=mask)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.append(probs)

        stacked_probs = torch.stack(all_probs)
        avg_probs = torch.mean(stacked_probs, dim=0)

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
            print(f"[*] Predicting on extracted dataset: {input_path}")
            data = np.load(input_path, allow_pickle=True)
            names = data["name"]
            apis = data["api_pefile"]

            for i in tqdm(range(len(names)), desc="Predicting NPZ", leave=False):
                malware_hash = names[i]
                if ignore_list and malware_hash.split(".")[0] in ignore_list:
                    results[f"{input_path}::{malware_hash}"] = {"final_label": "IGNORED"}
                    continue
                    
                api_seq = apis[i]
                try:
                    results[f"{input_path}::{malware_hash}"] = self._predict_sequence(
                        api_seq
                    )
                except Exception as e:
                    results[f"{input_path}::{malware_hash}"] = {
                        "final_label": "ERROR",
                        "details": str(e),
                    }
        else:
            files = (
                [input_path]
                if os.path.isfile(input_path)
                else [
                    os.path.join(r, f) for r, _, fs in os.walk(input_path) for f in fs
                ]
            )
            print(f"[*] Running inference on {len(files)} files...")

            for fpath in tqdm(files, desc="Predicting Files", leave=False):
                base_name = os.path.basename(fpath)
                if ignore_list and base_name.split(".")[0] in ignore_list:
                    results[fpath] = {"final_label": "IGNORED"}
                    continue
                    
                try:
                    api_seq = extract_api_calls(fpath)
                    results[fpath] = self._predict_sequence(api_seq)
                except Exception as e:
                    print(f"[!] Error predicting {fpath}: {e}")
                    results[fpath] = {"final_label": "ERROR", "details": str(e)}

        return results

    def train(self, train_paths, val_paths, label_mapping, experiment_name=None):
        manager = ExperimentManager(
            base_dir=os.path.dirname(__file__), experiment_name=experiment_name
        )
        logger = Logger(manager.run_dir)
        logger.log(f"[*] Starting Training")
        manager.save_config(CONFIG)

        with open(os.path.join(manager.run_dir, "label_map.json"), "w") as f:
            json.dump(label_mapping, f, indent=4)

        set_seed(CONFIG["seed"])

        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        self.tokenizer = CanineTokenizer.from_pretrained(CONFIG["model_name"])

        train_samples = []
        train_paths_list = (
            train_paths if isinstance(train_paths, list) else [train_paths]
        )

        logger.log("[*] Extracting API sequences for training...")
        for path in train_paths_list:
            if path.endswith(".npz"):
                data = np.load(path, allow_pickle=True)
                labels = data["label"]
                apis = data["api_pefile"]
                for i in range(len(labels)):
                    l_str = str(labels[i]).strip().lower()
                    if l_str in self.label_mapping:
                        train_samples.append(
                            {
                                "api_seq": apis[i] if apis[i] else "CreateFile",
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
                                train_samples.append(
                                    {
                                        "api_seq": extract_api_calls(fpath),
                                        "label": label_id,
                                    }
                                )

        train_df = pd.DataFrame(train_samples)

        if train_df.empty:
            raise ValueError(
                "No training samples matched the label mapping! Dataset is empty."
            )

        train_data, val_data = train_test_split(
            train_df,
            test_size=CONFIG["validation_split"],
            stratify=train_df["label"],
            random_state=CONFIG["seed"],
        )

        for i in range(CONFIG["n_estimators"]):
            logger.log(f"Training Model {i + 1}...")

            bootstrap_train = stratified_bootstrap(
                train_data, random_state=CONFIG["seed"] + i
            )

            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, df, tokenizer):
                    self.data = df.reset_index(drop=True)
                    self.tokenizer = tokenizer

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    txt = self.data.iloc[idx]["api_seq"]
                    label = self.data.iloc[idx]["label"]
                    enc = self.tokenizer(
                        txt,
                        max_length=CONFIG["max_length"],
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    return (
                        enc["input_ids"].squeeze(0),
                        enc["attention_mask"].squeeze(0),
                        torch.tensor(label),
                    )

            train_ds = SimpleDataset(bootstrap_train, self.tokenizer)
            val_ds = SimpleDataset(val_data, self.tokenizer)

            train_dl = torch.utils.data.DataLoader(
                train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0
            )
            val_dl = torch.utils.data.DataLoader(
                val_ds, batch_size=CONFIG["batch_size"], num_workers=0
            )

            model = CanineForSequenceClassification.from_pretrained(
                CONFIG["model_name"],
                num_labels=len(self.label_mapping),
                ignore_mismatched_sizes=True,
            ).to(self.device)
            optim = AdamW(model.parameters(), lr=CONFIG["learning_rate"])

            best_f1 = 0.0
            for epoch in range(CONFIG["num_epochs"]):
                model.train()
                for ids, mask, lbls in tqdm(
                    train_dl, desc=f"M{i + 1} Epoch {epoch + 1}", leave=False
                ):
                    ids, mask, lbls = (
                        ids.to(self.device),
                        mask.to(self.device),
                        lbls.to(self.device),
                    )
                    optim.zero_grad()
                    loss = model(ids, mask, labels=lbls).loss
                    loss.backward()
                    optim.step()

                model.eval()
                preds, true_lbs = [], []
                with torch.no_grad():
                    for ids, mask, lbls in val_dl:
                        ids, mask = ids.to(self.device), mask.to(self.device)
                        out = model(ids, mask)
                        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                        true_lbs.extend(lbls.numpy())

                f1 = f1_score(true_lbs, preds, average="macro")
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "val_f1_macro": f1,
                            "epoch": epoch + 1,
                            "label_mapping": self.label_mapping,
                        },
                        os.path.join(manager.run_dir, f"RTF_CANINE_model_{i + 1}.pth"),
                    )

            logger.log(f"Model {i + 1} Finished. Best F1: {best_f1:.4f}")

            # ==========================================
            # RAM FIX: Force clear memory before next loop
            # ==========================================
            del model
            del optim
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
            raise RuntimeError(
                "RTF-CANNIE models must be loaded before evasion testing."
            )

        self.label_mapping = {str(k).lower(): int(v) for k, v in label_mapping.items()}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}

        if self.tokenizer is None:
            self.tokenizer = CanineTokenizer.from_pretrained(CONFIG["model_name"])

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

            rate = (passed / total) if total > 0 else 0.0
            results_report[tech_name] = {"passed": passed, "total": total, "rate": rate}

            print(f"[*] RTF-CANNIE Evasion analysis complete for: {tech_name}")

        return results_report
