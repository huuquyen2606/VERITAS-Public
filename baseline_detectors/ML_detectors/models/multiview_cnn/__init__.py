import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# External dependencies
try:
    import pefile
except ImportError:
    print("[!] pefile not installed. Install with: pip install pefile")
    raise

# Internal imports
from .arch import MultiViewCNNArch
from ...core import MalwareModelBase
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
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 200,
    "num_workers": 2,
    "max_raw_bytes": 1024,
    "max_sections": 4,
    "max_imports": 1000,
    "max_apis": 100,
    "fused_length": 2128,  # 4 + 1000 + 100 + 1024
}


# ==========================================
# 2. RAW PE EXTRACTION (FALLBACK LOGIC)
# ==========================================
def extract_pe_features(filepath):
    """
    Extracts the 4 views from a raw PE file.
    Maps static PE APIs to the dynamic API slot as requested.
    """
    # 1. Raw Bytes (First 1024)
    try:
        with open(filepath, "rb") as f:
            raw = f.read(CONFIG["max_raw_bytes"])
        raw_bytes = np.frombuffer(raw, dtype=np.uint8)
        if len(raw_bytes) < CONFIG["max_raw_bytes"]:
            raw_bytes = np.pad(raw_bytes, (0, CONFIG["max_raw_bytes"] - len(raw_bytes)))
        else:
            raw_bytes = raw_bytes[: CONFIG["max_raw_bytes"]]
    except:
        raw_bytes = np.zeros(CONFIG["max_raw_bytes"], dtype=np.uint8)

    # 2. PE Parsing (Sections, Imports, APIs)
    sec_str, imp_str, api_str = "", "", ""
    try:
        pe = pefile.PE(filepath)
        sec_str = " ".join(
            [s.Name.decode("utf-8", "ignore").strip("\\x00") for s in pe.sections]
        )

        imports, apis = [], []
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                if entry.dll:
                    imports.append(entry.dll.decode("utf-8", "ignore"))
                for imp in entry.imports:
                    if imp.name:
                        apis.append(imp.name.decode("utf-8", "ignore"))

        imp_str = " ".join(imports)
        api_str = " ".join(apis)
    except:
        pass

    return raw_bytes, sec_str, imp_str, api_str


# ==========================================
# 3. DATASET CLASS
# ==========================================
class MultiViewDataset(Dataset):
    """
    PyTorch Dataset that serves the pre-fused and scaled 2128-length vectors.
    """

    def __init__(self, X_fused, y_labels):
        self.X = torch.tensor(X_fused, dtype=torch.float32).unsqueeze(
            1
        )  # Shape: (N, 1, 2128)
        self.y = torch.tensor(
            y_labels, dtype=torch.float32
        )  # Float32 required for BCEWithLogitsLoss

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 4. MAIN MODEL CONTROLLER
# ==========================================
class MultiViewCNNModel(MalwareModelBase):
    def __init__(self):
        super().__init__()
        self.model = None

        # Enforce Binary Classes
        self.class_names = ["Benign", "Malware"]

        # Artifacts (Crucial for Vectorization and Scaling)
        self.artifacts = {
            "vec_sec": None,
            "vec_imp": None,
            "vec_api": None,
            "scaler": None,
        }

        base_dir = os.path.dirname(__file__)
        self.local_weights_dir = os.path.join(base_dir, "weights")
        self.local_results_dir = os.path.join(base_dir, "results")

    def load_weights(self, weights_path):
        if not os.path.exists(weights_path):
            weights_path = os.path.join(self.local_weights_dir, weights_path)
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights not found: {weights_path}")

        print(
            f"[*] Loading Binary Multi-View CNN weights from: {os.path.dirname(weights_path)}"
        )

        # Load Model State
        checkpoint = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )

        self.model = MultiViewCNNArch()
        self.model.load_state_dict(checkpoint.get("model_state", checkpoint))
        self.model.to(self.device)
        self.model.eval()

        # Load Artifacts (Vectorizers and Scaler)
        artifact_path = os.path.join(os.path.dirname(weights_path), "artifacts.pkl")
        if os.path.exists(artifact_path):
            with open(artifact_path, "rb") as f:
                self.artifacts = pickle.load(f)
            print("    ✓ Vectorizer Artifacts loaded successfully.")
        else:
            raise FileNotFoundError(
                f"Missing artifacts.pkl at {artifact_path}. Required for prediction."
            )

    def _process_and_fuse(
        self, raw_bytes_list, sec_list, imp_list, api_list, is_training=False
    ):
        """
        Applies NLP Vectorization and scales the features into the final 2128-length array.
        """
        if is_training:
            self.artifacts["vec_sec"] = CountVectorizer(
                max_features=CONFIG["max_sections"]
            )
            self.artifacts["vec_imp"] = CountVectorizer(
                max_features=CONFIG["max_imports"]
            )
            self.artifacts["vec_api"] = CountVectorizer(max_features=CONFIG["max_apis"])
            self.artifacts["scaler"] = StandardScaler()

            X_sec = self.artifacts["vec_sec"].fit_transform(sec_list).toarray()
            X_imp = self.artifacts["vec_imp"].fit_transform(imp_list).toarray()
            X_api = self.artifacts["vec_api"].fit_transform(api_list).toarray()
        else:
            X_sec = self.artifacts["vec_sec"].transform(sec_list).toarray()
            X_imp = self.artifacts["vec_imp"].transform(imp_list).toarray()
            X_api = self.artifacts["vec_api"].transform(api_list).toarray()

        # Pad dynamically in case vocabularies couldn't reach max_features
        if X_sec.shape[1] < CONFIG["max_sections"]:
            X_sec = np.pad(
                X_sec, ((0, 0), (0, CONFIG["max_sections"] - X_sec.shape[1]))
            )
        if X_imp.shape[1] < CONFIG["max_imports"]:
            X_imp = np.pad(X_imp, ((0, 0), (0, CONFIG["max_imports"] - X_imp.shape[1])))
        if X_api.shape[1] < CONFIG["max_apis"]:
            X_api = np.pad(X_api, ((0, 0), (0, CONFIG["max_apis"] - X_api.shape[1])))

        X_raw = np.array(raw_bytes_list)
        X_fused = np.hstack([X_sec, X_imp, X_api, X_raw])

        if is_training:
            return self.artifacts["scaler"].fit_transform(X_fused)
        return self.artifacts["scaler"].transform(X_fused)

    def train(self, train_paths, val_paths, label_mapping, experiment_name=None):
        manager = ExperimentManager(
            os.path.dirname(__file__), experiment_name or "MultiView_Binary_Run"
        )
        logger = Logger(manager.run_dir)
        manager.save_config(CONFIG)

        logger.log(
            "[*] Collecting Multi-View Data into Memory (Binary Auto-Labeling Enabled)..."
        )

        def load_data(paths):
            raw_lst, sec_lst, imp_lst, api_lst, lbl_lst = [], [], [], [], []
            for path in paths if isinstance(paths, list) else [paths]:
                if path.endswith(".npz"):
                    data = np.load(path, allow_pickle=True)
                    labels = data["label"]
                    raw_bytes_cache = data.get("raw_byte", None)
                    pe_sections_cache = data.get("pe_sections", None)
                    pe_imports_cache = data.get("pe_imports", None)
                    api_cuckoo_cache = data.get("api_cuckoo", None)

                    for i in range(len(labels)):
                        # BINARY AUTO-LABEL LOGIC
                        lbl_lst.append(0 if "benign" in str(labels[i]).lower() else 1)

                        # Pre-slice to avoid allocating massive 4MB+ numpy arrays for no reason
                        rb_val = (
                            raw_bytes_cache[i] if raw_bytes_cache is not None else []
                        )
                        sliced_val = rb_val[: CONFIG["max_raw_bytes"]]
                        rb = np.array(sliced_val, dtype=np.uint8)
                        if len(rb) < CONFIG["max_raw_bytes"]:
                            rb = np.pad(rb, (0, CONFIG["max_raw_bytes"] - len(rb)))
                        raw_lst.append(rb)

                        # Progressively free massive strings/byte arrays to prevent OOM
                        if raw_bytes_cache is not None:
                            raw_bytes_cache[i] = None

                        sec_val = (
                            pe_sections_cache[i]
                            if pe_sections_cache is not None
                            else ""
                        )
                        sec_lst.append(str(sec_val))

                        imp_val = (
                            pe_imports_cache[i] if pe_imports_cache is not None else ""
                        )
                        imp_lst.append(str(imp_val))

                        raw_api = (
                            api_cuckoo_cache[i] if api_cuckoo_cache is not None else ""
                        )
                        if isinstance(raw_api, list) or isinstance(raw_api, np.ndarray):
                            raw_api = " ".join([str(x) for x in raw_api])
                        api_lst.append(str(raw_api))
                else:
                    for root, _, files in os.walk(path):
                        folder_name = os.path.basename(root).lower()
                        # BINARY AUTO-LABEL LOGIC
                        binary_label = 0 if "benign" in folder_name else 1
                        for f in files:
                            fpath = os.path.join(root, f)
                            rb, sec, imp, api = extract_pe_features(fpath)
                            raw_lst.append(rb)
                            sec_lst.append(sec)
                            imp_lst.append(imp)
                            api_lst.append(api)
                            lbl_lst.append(binary_label)
            return raw_lst, sec_lst, imp_lst, api_lst, lbl_lst

        # Load and Fuse Training Data
        tr_raw, tr_sec, tr_imp, tr_api, tr_lbl = load_data(train_paths)
        X_train_fused = self._process_and_fuse(
            tr_raw, tr_sec, tr_imp, tr_api, is_training=True
        )
        train_loader = DataLoader(
            MultiViewDataset(X_train_fused, tr_lbl),
            batch_size=CONFIG["batch_size"],
            shuffle=True,
        )

        # Load and Fuse Validation Data
        val_raw, val_sec, val_imp, val_api, val_lbl = load_data(val_paths)
        X_val_fused = self._process_and_fuse(
            val_raw, val_sec, val_imp, val_api, is_training=False
        )
        val_loader = DataLoader(
            MultiViewDataset(X_val_fused, val_lbl),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
        )

        # Save Artifacts
        with open(manager.get_path("artifacts.pkl"), "wb") as f:
            pickle.dump(self.artifacts, f)

        # Training Setup
        self.model = MultiViewCNNArch().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["learning_rate"])
        criterion = nn.BCEWithLogitsLoss()  # BINARY LOSS FUNCTION

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_acc = 0.0

        for epoch in range(CONFIG["epochs"]):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for x, y in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}", leave=False
            ):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x).view(-1)
                loss = criterion(logits, y)
                loss.backward()
                if self.device.type == "xla":
                    import torch_xla.core.xla_model as xm

                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                total_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                total += y.size(0)
                correct += (preds == y).sum().item()

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total

            # Validation
            self.model.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x).view(-1)
                    v_loss += criterion(logits, y).item()
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    v_total += y.size(0)
                    v_correct += (preds == y).sum().item()

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
                        "accuracy": best_acc,
                        "config": CONFIG,
                    },
                    manager.get_path("multiview_binary.pth"),
                )

        save_training_plot(history, manager.run_dir)
        logger.close()

    def predict(self, input_path, ignore_list=None):
        if not self.model or not self.artifacts["scaler"]:
            raise RuntimeError("Model or artifacts not loaded")

        self.model.eval()
        results = {}

        hashes, r_list, s_list, i_list, a_list = [], [], [], [], []

        if input_path.endswith(".npz"):
            data = np.load(input_path, allow_pickle=True)
            names = data["name"]
            raw_bytes_cache = data.get("raw_byte", None)
            pe_sections_cache = data.get("pe_sections", None)
            pe_imports_cache = data.get("pe_imports", None)
            api_cuckoo_cache = data.get("api_cuckoo", None)

            for i in range(len(names)):
                malware_hash = names[i]
                if ignore_list and malware_hash.split(".")[0] in ignore_list:
                    results[f"{input_path}::{malware_hash}"] = {"final_label": "IGNORED"}
                    continue

                hashes.append(f"{input_path}::{malware_hash}")
                rb_val = raw_bytes_cache[i] if raw_bytes_cache is not None else []
                # Pre-slice to avoid OOM when converting large lists to numpy arrays
                sliced_val = rb_val[: CONFIG["max_raw_bytes"]]
                rb = np.array(sliced_val, dtype=np.uint8)
                if len(rb) < CONFIG["max_raw_bytes"]:
                    rb = np.pad(rb, (0, CONFIG["max_raw_bytes"] - len(rb)))
                r_list.append(rb)

                # Progressively free massive objects from RAM
                if raw_bytes_cache is not None:
                    raw_bytes_cache[i] = None

                sec_val = pe_sections_cache[i] if pe_sections_cache is not None else ""
                s_list.append(str(sec_val))

                imp_val = pe_imports_cache[i] if pe_imports_cache is not None else ""
                i_list.append(str(imp_val))

                raw_api = api_cuckoo_cache[i] if api_cuckoo_cache is not None else ""
                if isinstance(raw_api, list) or isinstance(raw_api, np.ndarray):
                    raw_api = " ".join([str(x) for x in raw_api])
                a_list.append(str(raw_api))
        else:
            files = [input_path] if os.path.isfile(input_path) else []
            if os.path.isdir(input_path):
                for r, _, fs in os.walk(input_path):
                    files.extend([os.path.join(r, f) for f in fs])
            for fpath in files:
                base_name = os.path.basename(fpath)
                if ignore_list and base_name.split(".")[0] in ignore_list:
                    results[fpath] = {"final_label": "IGNORED"}
                    continue
                    
                hashes.append(fpath)
                rb, sec, imp, api = extract_pe_features(fpath)
                r_list.append(rb)
                s_list.append(sec)
                i_list.append(imp)
                a_list.append(api)

        if not hashes:
            return results

        X_fused = self._process_and_fuse(
            r_list, s_list, i_list, a_list, is_training=False
        )
        # DON'T push to device here yet! Keep it on CPU.
        X_tensor = torch.tensor(X_fused, dtype=torch.float32).unsqueeze(1)

        virus_probs = []
        batch_size = CONFIG.get("batch_size", 32)

        # Process in batches to prevent GPU OOM!
        with torch.no_grad():
            for i in tqdm(range(0, len(X_tensor), batch_size), desc="Predicting Batch"):
                batch_x = X_tensor[i : i + batch_size].to(self.device)
                logits = self.model(batch_x).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                virus_probs.extend(probs)

        for i, h in enumerate(hashes):
            vp = float(virus_probs[i])
            results[h] = {
                "Benign": 1.0 - vp,
                "Malware": vp,
                "final_label": "Malware" if vp > 0.5 else "Benign",
            }

        return results

    def evaluate(self, test_path, output_dir=None):
        if not self.model:
            raise RuntimeError("Model not loaded")
        output_dir = output_dir or self.local_results_dir
        os.makedirs(output_dir, exist_ok=True)

        preds = self.predict(test_path)
        y_true, y_pred, confs, correct_mask = [], [], [], []

        if test_path.endswith(".npz"):
            data = np.load(test_path, allow_pickle=True)
            for i, name in enumerate(data["name"]):
                true_lbl_str = data["label"][i]
                res = preds.get(f"{test_path}::{name}", {})
                if not res or res.get("final_label") == "ERROR":
                    continue

                true_idx = 0 if "benign" in str(true_lbl_str).lower() else 1
                pred_idx = 1 if res["final_label"] == "Malware" else 0

                y_true.append(true_idx)
                y_pred.append(pred_idx)
                confs.append(max(res["Benign"], res["Malware"]))
                correct_mask.append(pred_idx == true_idx)
        else:
            for fpath, res in preds.items():
                if res.get("final_label", "ERROR") == "ERROR":
                    continue
                folder_name = os.path.basename(os.path.dirname(fpath)).lower()

                true_idx = 0 if "benign" in folder_name else 1
                pred_idx = 1 if res["final_label"] == "Malware" else 0

                y_true.append(true_idx)
                y_pred.append(pred_idx)
                confs.append(max(res["Benign"], res["Malware"]))
                correct_mask.append(pred_idx == true_idx)

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
