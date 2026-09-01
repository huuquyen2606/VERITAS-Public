import os
import sys
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

import torch
import psutil
from torch.utils.data import Sampler
import random


import multiprocessing


# find the correct hardware
def get_optimal_device():
    """
    Unified smart hardware detector.
    Priority: GPU (CUDA) -> TPU (XLA) -> Apple Silicon (MPS) -> CPU
    """
    is_main = multiprocessing.current_process().name == "MainProcess"

    # 1. Check for NVIDIA GPU
    if torch.cuda.is_available():
        if is_main:
            print("[*] Hardware Accelerator: GPU (CUDA) detected.")
        return torch.device("cuda")

    # 2. Check for Google TPU (Requires torch_xla on Kaggle/Colab)
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        if is_main:
            print("[*] Hardware Accelerator: TPU (XLA) detected.")
        return device
    except ImportError:
        pass  # torch_xla not installed or not on a TPU machine

    # 3. Check for Apple Silicon GPU (Mac M1/M2/M3)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if is_main:
            print("[*] Hardware Accelerator: Apple Silicon (MPS) detected.")
        return torch.device("mps")

    # 4. Fallback to CPU
    if is_main:
        print("[!] Hardware Accelerator: CPU fallback. Training will be slow.")
    return torch.device("cpu")


def calculate_optimal_chunk_size(
    element_size_bytes, safety_margin=0.4, min_chunk=200, max_chunk=10000
):
    """
    Calculates how many items to load per trunk/chunk based on free system RAM & VRAM.
    Args:
        element_size_bytes: Estimated size of a single data element in bytes.
    """
    available_cpu_ram = psutil.virtual_memory().available

    if torch.cuda.is_available():
        free_vram, _ = torch.cuda.mem_get_info()
        # Bound by whichever is smaller, the CPU RAM or GPU VRAM, ensuring we never OOM.
        available_mem = min(available_cpu_ram, free_vram)
    else:
        available_mem = available_cpu_ram

    usable_mem = available_mem * safety_margin
    chunk_size = int(usable_mem / element_size_bytes)

    # Return within strict sanity bounds
    return max(min_chunk, min(chunk_size, max_chunk))


class ChunkRandomSampler(Sampler):
    """
    Samples indices pseudo-randomly but strictly within contiguous 'trunks' (chunks).
    This prevents disk thrashing when reading from raw files or NPZ archives by ensuring
    that when a chunk is loaded to RAM, all its items are batched and processed before moving on.
    """

    def __init__(self, data_source, chunk_size):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_samples = len(data_source)

    def __iter__(self):
        chunks = []
        for i in range(0, self.num_samples, self.chunk_size):
            chunks.append(list(range(i, min(i + self.chunk_size, self.num_samples))))

        # Shuffle the order of the trunks
        random.shuffle(chunks)

        # Shuffle internally within the trunk
        for chunk in chunks:
            random.shuffle(chunk)
            yield from chunk

    def __len__(self):
        return self.num_samples


# ==========================================
# 1. CORE INFRASTRUCTURE
# ==========================================


class Logger:
    """
    The 'Live Monitor' for experiments.

    This class handles logging messages simultaneously to the standard console output
    and to a persistent text file ("experiment_log.txt") within the run directory.
    It ensures that even if a long training session crashes, the logs up to that
    point are saved.

    Attributes:
        log_path (str): Full path to the log file.
        file (file object): The open file handle for writing logs.
    """

    def __init__(self, log_dir):
        """
        Initialize the Logger.

        Args:
            log_dir (str): Directory where the log file should be created.
        """

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "experiment_log.txt")
        self.file = open(self.log_path, "a", encoding="utf-8")

        # Write header
        self.log("-" * 40)
        self.log(
            f"SESSION STARTED: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.log("-" * 40)

    def log(self, message):
        """
        Write a message to both the console and the log file.

        Args:
            message (str): The text content to log.
        """

        # 1. Print to Console
        print(message)

        # 2. Write to File
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S] ")
        self.file.write(timestamp + str(message) + "\n")
        self.file.flush()  # Ensure it saves immediately if crash happens

    def close(self):
        """
        Finalize the log session and close the file handle.
        """

        if self.file:
            self.log("-" * 40)
            self.log("SESSION ENDED")
            self.file.close()


class ExperimentManager:
    """
    The 'Lab Notebook' for managing experiment directories.

    This class automatically generates structured, timestamped folders for every
    training run. It prevents overwriting previous results and provides a standardized
    way to save configuration snapshots and model weights.

    Attributes:
        run_dir (str): The full path to the directory created for this specific run.
    """

    def __init__(self, base_dir, experiment_name=None):
        """
        Initialize a new experiment directory.

        Creates a folder structure:
            base_dir/
              └── experiments/
                   └── YYYY-MM-DD_HH-MM_{experiment_name}/

        Args:
            base_dir (str): The root directory of the model module (e.g., .../models/malconv/).
            experiment_name (str, optional): A descriptive suffix for the folder.
                                             Defaults to "Run" if not provided.
        """

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M")

        if experiment_name:
            folder_name = f"{timestamp}_{experiment_name}"
        else:
            folder_name = f"{timestamp}_Run"

        self.run_dir = os.path.join(base_dir, "experiments", folder_name)
        os.makedirs(self.run_dir, exist_ok=True)

        print(f"[*] New Experiment Created: {self.run_dir}")

    def get_path(self, filename):
        """
        Get the full absolute path for saving a file within this experiment.

        Args:
            filename (str): The name of the file (e.g., 'best_model.pth').

        Returns:
            str: Full path (e.g., '.../experiments/2026.../best_model.pth').
        """

        return os.path.join(self.run_dir, filename)

    def save_config(self, config_dict):
        """
        Save the experiment configuration as a JSON snapshot.

        This ensures reproducibility by recording exactly which parameters were used.

        Args:
            config_dict (dict): The configuration dictionary (e.g., CONFIG).
        """

        path = self.get_path("config_snapshot.json")
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)


# ==========================================
# 2. VISUALIZATION TOOLS
# ==========================================


def save_training_plot(history, save_dir):
    """
    Generate and save a plot of Training vs. Validation Loss and Accuracy.

    Args:
        history (dict): Dictionary containing lists for 'train_loss', 'val_loss',
                        'train_acc', and 'val_acc'.
        save_dir (str): Directory where 'training_plot.png' will be saved.
    """

    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    plt.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_plot.png")
    plt.savefig(path)
    plt.close()


def generate_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """
    Generate and save a Confusion Matrix heatmap.

    Args:
        y_true (list/array): Ground truth integer labels.
        y_pred (list/array): Predicted integer labels.
        class_names (list): List of class strings corresponding to label indices.
        save_dir (str): Directory where 'confusion_matrix.png' will be saved.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Generate labels indices to match class_names length
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()


def plot_confidence_distribution(confidences, correct_mask, save_dir):
    """
    Plot histograms showing the distribution of model confidence scores.

    This visualization separates Correct predictions (Green) from Incorrect ones (Red),
    helping to identify if the model is 'confidently wrong' or uncertain.

    Args:
        confidences (list/array): Float confidence scores (0.0 to 1.0) for each prediction.
        correct_mask (list/array): Boolean mask where True indicates a correct prediction.
        save_dir (str): Directory where 'confidence_distribution.png' will be saved.
    """

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Correct Predictions (Green)
    correct_confs = [
        c for c, is_correct in zip(confidences, correct_mask) if is_correct
    ]
    if correct_confs:
        sns.histplot(
            correct_confs, color="green", label="Correct", kde=True, bins=20, alpha=0.5
        )

    # Incorrect Predictions (Red)
    wrong_confs = [
        c for c, is_correct in zip(confidences, correct_mask) if not is_correct
    ]
    if wrong_confs:
        sns.histplot(
            wrong_confs, color="red", label="Wrong", kde=True, bins=20, alpha=0.5
        )

    plt.title("Model Confidence Distribution")
    plt.xlabel("Confidence Score (0.0 - 1.0)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "confidence_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()


def calculate_metrics(y_true, y_pred, class_names):
    labels_indices = list(range(len(class_names)))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 30)
    print("       CLASSIFICATION REPORT")
    print("=" * 30)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels_indices,
            target_names=class_names,
            zero_division=0,
        )
    )

    # EXACT 4 METRICS - MACRO AVERAGED
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "full_report": report_dict,
    }
    return results


def plot_model_comparison(battle_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    data = []
    for model_name, metrics in battle_results.items():
        data.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", 0),
                "Precision": metrics.get("precision_macro", 0),
                "Recall": metrics.get("recall_macro", 0),
                "F1-Score": metrics.get("f1_macro", 0),
            }
        )

    if not data:
        return

    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    chart = sns.barplot(
        x="Model", y="Score", hue="Metric", data=df_melted, palette="viridis"
    )

    for container in chart.containers:
        chart.bar_label(
            container, fmt="%.4f", padding=3
        )  # <--- EXACTLY 4 DECIMALS ON GRAPH

    plt.title("Model Comparison: 4 Standard Research Metrics")
    plt.ylim(0, 1.1)
    plt.legend(loc="lower right")

    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[+] Comparison graph saved to {path}")


def is_benign_label(label, label_mapping=None):
    """
    Checks if a given label string or integer index corresponds to a benign class.
    Handles label indices by checking against label_mapping.
    """
    if label is None:
        return False
    lbl_str = str(label).strip().lower()
    if "benign" in lbl_str:
        return True
    
    if label_mapping:
        for k, v in label_mapping.items():
            if "benign" in str(k).strip().lower():
                if lbl_str == str(v).strip().lower() or lbl_str == str(k).strip().lower():
                    return True
    elif lbl_str == "0" or lbl_str == "0.0":
        return True
    return False


def print_evasion_summary(report):
    if not report:
        return

    techniques = list(report.keys())
    first_tech = techniques[0]
    models = list(report[first_tech].keys())

    def print_sep(width=110):
        print("-" * width)

    col_width = 24
    tech_width = 25
    header = f"{'Technique':<{tech_width}} | " + " | ".join(
        [f"{m:^{col_width}}" for m in models]
    )

    print("\n" + "=" * len(header))
    print(f"{'EVASION RATE (Passed/Total Samples)':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print_sep(len(header))

    for tech in techniques:
        row_cells = []
        for m in models:
            stats = report[tech][m]
            passed = stats.get("passed", 0)
            total = stats.get("total", 0)
            rate = stats.get("rate", 0.0)
            cell_str = f"{passed}/{total} ({rate:.4f})"
            row_cells.append(f"{cell_str:^{col_width}}")
        print(f"{tech:<{tech_width}} | " + " | ".join(row_cells))
    print("=" * len(header) + "\n")


def predict_raw_or_npz(predictor_func, input_path):
    """
    Wraps the predictor_func to support dynamically querying a folder of PE files.
    """
    if str(input_path).endswith(".npz") or os.path.isfile(input_path):
        return predictor_func(input_path)

    # It's a directory of PE files
    from baseline_detectors.dynamic_extractor import process_raw_pe_to_npz

    results = {}
    print("\n" + "=" * 80)
    print(f"{'DYNAMIC RAW PE SCANNING & PREDICTION':^80}")
    print("=" * 80)

    for root, _, files in os.walk(input_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"\n[*] Target: {file_path}")
            temp_npz = "/tmp/temp_dynamic_scan.npz"

            success = process_raw_pe_to_npz(file_path, temp_npz)
            if not success:
                print(f" [!] Skipping {file}")
                results[file_path] = {"final_label": "ERROR (Extraction Failed)"}
                continue

            # Sub-predict on the 1-sample NPZ created
            res = predictor_func(temp_npz)

            # Merge dictionary mappings seamlessly to mimic raw parsing
            for k, v in res.items():
                results[file_path] = v
                
    return results

# Cache to avoid reloading the JSON report multiple times during an ensemble run
_EVAL_REPORT_CACHE = None

def get_ignored_samples(tech_name, report_path=None, tech_path=None, label_mapping=None):
    """
    Returns a set of base filenames (extensions stripped) that failed functionality or integrity checks,
    or correspond to benign samples (to exclude them from evaluation progress and metric calculations).
    Matches against `comprehensive_detailed_reports.json` and checks dataset ground truth.
    """
    global _EVAL_REPORT_CACHE

    ignored_set = set()

    # 1. Filter out benign samples at the ensemble/evaluator step before prediction
    if tech_path and os.path.exists(tech_path) and str(tech_path).endswith(".npz"):
        try:
            data = np.load(tech_path, allow_pickle=True)
            names = data.get("name", [])
            labels = data.get("label", [])
            for i in range(min(len(names), len(labels))):
                if is_benign_label(labels[i], label_mapping):
                    clean_key = str(names[i]).split(".")[0]
                    ignored_set.add(clean_key)
        except Exception as e:
            print(f"[!] Warning: Could not scan {tech_path} for benign samples: {e}")

    # 2. Filter out non-functional / dead samples
    if report_path is None:
        # Load from the baseline_detectors directory itself as requested
        base_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(base_dir, "comprehensive_detailed_reports.json")

    if not os.path.exists(report_path):
        if len(ignored_set) > 0:
            print(f"[*] Identified {len(ignored_set)} benign samples for {tech_name}. These will be skipped from evaluation progress.")
        return ignored_set

    if _EVAL_REPORT_CACHE is None:
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                _EVAL_REPORT_CACHE = json.load(f)
        except Exception as e:
            print(f"[!] Error loading {report_path}: {e}")
            return ignored_set

    # Smart substring match to find the actual technique name in the report
    actual_tech_key = None
    tech_name_lower = tech_name.lower()
    for key in _EVAL_REPORT_CACHE.keys():
        if key.lower() in tech_name_lower:
            actual_tech_key = key
            break

    # Check if the specific evasion technique exists in the report
    if actual_tech_key is not None:
        tech_report = _EVAL_REPORT_CACHE[actual_tech_key]
        for key, status in tech_report.items():
            is_functional = status.get("is_functional", False)
            is_alive = status.get("is_alive", False)
            
            if not is_functional or not is_alive:
                clean_key = key.split(".")[0]
                ignored_set.add(clean_key)

    if len(ignored_set) > 0:
        print(f"[*] Identified {len(ignored_set)} ignored samples (benign or non-functional/dead) for {tech_name}. These will be skipped.")

    return ignored_set
