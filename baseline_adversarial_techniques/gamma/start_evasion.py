import os
import sys
import json
import requests
import numpy as np
import uuid
import argparse
import gc  # <-- Added for memory management

# SecML Imports
from secml.ml.classifiers import CClassifier
from secml.array import CArray
from secml_malware.attack.blackbox.c_gamma_sections_evasion import (
    CGammaSectionsEvasionProblem,
)

# ==============================================================================
# 0. SETUP PATHS
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
root_dir = os.path.abspath(os.path.join(parent_dir, ".."))

from gamma import create_adversarial_malware, CFG

# ==============================================================================
# 1. DEFAULT CONFIGURATION (Fallback if no JSON is provided)
# ==============================================================================
DEFAULT_CONFIG = {
    "dataset_dir": os.path.join(root_dir, "test_dataset", "Test", "Virus"),
    "benign_dir": os.path.join(root_dir, "..", "test_dataset", "Test", "Benign"),
    "output_dir": os.path.join(
        root_dir, "tmp", "adversarial_malware", "test_gamma_network"
    ),
    "server_url": "http://0.0.0.0:8000",
    "model_config": {
        "model": "binary_malconv",
        "single_weights": "/absolute/path/to/weights.pth",
    },
}

# ==============================================================================
# 2. MODEL ADAPTER (Client)
# ==============================================================================
class RemoteModelAdapter(CClassifier):
    def __init__(self, server_url, auto_load_config=None):
        super(RemoteModelAdapter, self).__init__()

        self._classes = CArray([0, 1])
        self._n_features = 2**20
        
        # --- FIX: Strip trailing slash to prevent routing errors ---
        self.base_url = server_url.rstrip("/")
        print(f"[*] Connecting to Model Server at {self.base_url}...")

        if not os.path.exists("/dev/shm"):
            print("[!] WARNING: /dev/shm not found. RAM-disk optimization might fail.")

        try:
            requests.get(f"{self.base_url}/status", timeout=2)
            print("[*] Server is ONLINE.")
        except:
            print("[!] CRITICAL: Server appears offline.")
            sys.exit(1)

        if auto_load_config:
            self._load_model_on_server(auto_load_config)

    def _load_model_on_server(self, config):
        print(f"[*] Instructing Server to load model...")
        load_url = f"{self.base_url}/load"
        try:
            response = requests.post(load_url, json=config)
            if response.status_code == 200:
                print(f"    [SUCCESS] Server replied: {response.json().get('message')}")
            else:
                print(
                    f"    [!] Failed to load model! Server sent {response.status_code}: {response.text}"
                )
                sys.exit(1)
        except Exception as e:
            print(f"    [!] Connection Error during loading: {e}")
            sys.exit(1)

    def predict_proba(self, byte_data, original_label=None):
        return self._query_server(byte_data)

    def _forward(self, x):
        x_numpy = x.tondarray().astype(np.uint8).flatten()
        byte_data = x_numpy.tobytes()
        prob_virus = self._query_server(byte_data)
        prob_benign = 1.0 - prob_virus
        return CArray([[prob_benign, prob_virus]])

    def _query_server(self, byte_data):
        unique_name = f"scan_{uuid.uuid4()}.exe"
        temporary_malware_sample = os.path.join("/dev/shm", unique_name)

        try:
            with open(temporary_malware_sample, "wb") as f:
                f.write(byte_data)

            payload = {"input": temporary_malware_sample}
            predict_url = f"{self.base_url}/predict"
            response = requests.post(
                predict_url, json=payload, headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                resp_json = response.json()
                results = resp_json.get("results", {})
                file_result = results.get(temporary_malware_sample, {})

                if "Ensemble" in file_result:
                    score = file_result["Ensemble"].get("Virus", 0.0)
                elif "SingleModel" in file_result:
                    score = file_result["SingleModel"].get("Virus", 0.0)
                else:
                    first_key = next(iter(file_result))
                    score = file_result[first_key].get("Virus", 0.0)

                return float(score)
            else:
                return 1.0
        except Exception as e:
            return 1.0
        finally:
            if os.path.exists(temporary_malware_sample):
                try:
                    os.remove(temporary_malware_sample)
                except OSError:
                    pass

    def _fit(self, dataset):
        return self

    def _backward(self, w):
        return CArray.zeros(self.n_features)

# ==============================================================================
# 3. EXECUTION LOGIC
# ==============================================================================
def main():
    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Run GAMMA Evasion Batch")
    parser.add_argument(
        "--config", type=str, help="Path to a custom config.json file", default=None
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.isfile(args.config):
        print(f"[*] Loading custom configuration from: {args.config}")
        with open(args.config, "r") as f:
            user_config = json.load(f)
            config.update(user_config)
    else:
        print("[*] Using internal default configuration.")

    dataset_dir = config.get("dataset_dir")
    benign_dir = config["benign_dir"]
    output_dir = config["output_dir"]

    # --- Validation ---
    if not os.path.isdir(dataset_dir):
        print(f"[!] Target dataset directory not found: {dataset_dir}")
        sys.exit(1)
    if not os.path.exists(benign_dir):
        print(f"[!] Benign directory missing: {benign_dir}")
        sys.exit(1)

    # --- FIX: Strict Extension Filter ---
    tasks = []
    valid_extensions = {".exe", ".dll"}
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, dataset_dir)
                tasks.append((full_path, rel_dir))
            else:
                print(f"[-] Skipping non-executable file: {file}")

    print(f"[*] Found {len(tasks)} malware samples to process across subdirectories.")

    # --- Initialize Adapter & Load Model ---
    adapter = RemoteModelAdapter(
        server_url=config["server_url"], auto_load_config=config["model_config"]
    )

    # --- Pre-load Benign Sections (Optimized) ---
    print(f"\n[+] Extracting {CFG['N_SECTIONS']} benign sections into RAM once...")
    section_population, _ = (
        CGammaSectionsEvasionProblem.create_section_population_from_folder(
            folder=benign_dir,
            how_many=CFG["N_SECTIONS"],
            sections_to_extract=CFG["SECTION_TYPES"],
            to_ignore=[],
        )
    )
    print("[+] Extraction complete. Starting generation loop...\n")

    # --- Run Attack ---
    for i, (malware_path, rel_dir) in enumerate(tasks):
        # 1. Create matching output subdirectory
        out_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)

        # 2. Identify the dynamic label based on the top-level folder name
        label_name = rel_dir.split(os.sep)[0] if rel_dir != "." else "Unknown"

        print(
            f"[*] {i + 1}/{len(tasks)}: [{label_name}] {os.path.basename(malware_path)}"
        )

        # 3. Generate Adversarial Malware
        create_adversarial_malware(
            malware_path=malware_path,
            section_population=section_population,
            output_dir=out_subdir,  # Saves exactly to output_dir/Label/
            custom_model=adapter,
            original_label=label_name,  # Passes the dynamic folder name as the label
            config=CFG,
        )
        
        # --- FIX: Memory Cleanup to prevent OOM ---
        gc.collect()

    print("\n[+] Batch Process Finished.")

if __name__ == "__main__":
    main()