import os
import glob
import json
import numpy as np
import re

DATASETS_DIR = "datasets/APIs"
ORIGINAL_DATASET = "Test_full.npz"


def ensure_str(val):
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    elif hasattr(val, "item"):
        item_val = val.item()
        if isinstance(item_val, bytes):
            return item_val.decode("utf-8", errors="ignore")
        return str(item_val)
    return str(val)


def get_base_name(filename):
    filename = ensure_str(filename)

    # Extract the MD5/SHA1/SHA256 hash part from the filename
    match = re.search(r"([a-fA-F0-9]{32,64})", filename)
    if match:
        return match.group(1)
    return filename.split(".exe")[0] + ".exe"


def get_features(data, key_choices):
    for key in key_choices:
        if key in data.files:
            val = data[key]
            return val.tolist() if hasattr(val, "tolist") else list(val)
    # Return empty list if no keys matched
    return [[] for _ in range(len(data["name"]))]


def main():
    original_path = os.path.join(DATASETS_DIR, ORIGINAL_DATASET)
    if not os.path.exists(original_path):
        print(f"[!] Original dataset not found: {original_path}")
        return

    print(f"[*] Loading original dataset: {ORIGINAL_DATASET}")
    orig_data = np.load(original_path, allow_pickle=True)

    orig_names = (
        orig_data["name"].tolist()
        if hasattr(orig_data["name"], "tolist")
        else list(orig_data["name"])
    )

    # Map base name so we know which files actually belong to the baseline
    original_signatures = {}
    for name in orig_names:
        base = get_base_name(name)
        original_signatures[base] = True

    print(f"    Loaded {len(original_signatures)} original base samples.")

    # Find all adversarial datasets
    npz_files = glob.glob(os.path.join(DATASETS_DIR, "*.npz"))

    detailed_reports = {}
    summaries = {}

    # For DQEAF, we check APIs
    feature_keys = ["api_cuckoo", "api"]

    for npz_file in npz_files:
        dataset_name = os.path.basename(npz_file)
        if dataset_name == ORIGINAL_DATASET or dataset_name == "Test_full.npz":
            continue

        print(f"\n[*] Processing adversarial dataset: {dataset_name}")
        adv_data = np.load(npz_file, allow_pickle=True)
        adv_names = (
            adv_data["name"].tolist()
            if hasattr(adv_data["name"], "tolist")
            else list(adv_data["name"])
        )
        
        # Get the APIs for the adversarial dataset
        adv_apis = get_features(adv_data, feature_keys)

        dataset_report = {}
        integrity_count = 0
        total_samples = 0

        for adv_name, apis in zip(adv_names, adv_apis):
            adv_name_str = ensure_str(adv_name)
            
            if apis is None:
                apis = []
            elif hasattr(apis, "tolist"):
                apis = apis.tolist()

            base_name = get_base_name(adv_name_str)

            if base_name in original_signatures:
                total_samples += 1
                
                # DQEAF Mechanism: If having APIs call means Integrity + 1
                api_call_count = len(apis)
                is_alive = api_call_count > 0
                
                if is_alive:
                    integrity_count += 1

                dataset_report[adv_name_str] = {
                    "base_name": base_name,
                    "is_alive": is_alive,
                    "api_call_count": api_call_count
                }
            else:
                # Base name not found in original dataset, skip or mark as dead
                dataset_report[adv_name_str] = {
                    "base_name": base_name,
                    "is_alive": False,
                    "error": "Original sample not found",
                }

        detailed_reports[dataset_name] = dataset_report
        percentage = (
            (integrity_count / total_samples * 100) if total_samples > 0 else 0.0
        )

        summaries[dataset_name] = {
            "total_samples": total_samples,
            "integrity_count": integrity_count,
            "percentage": percentage,
        }

        print(f"    Total Samples: {total_samples}")
        print(f"    Integrity Count: {integrity_count}")
        print(f"    Percentage: {percentage:.2f}%")

    # Save details
    print("\n[*] Saving detailed reports to dqeaf_detailed_reports.json")
    with open("dqeaf_detailed_reports.json", "w") as f:
        json.dump(detailed_reports, f, indent=4)

    print("[*] Saving summaries to dqeaf_summary.json")
    with open("dqeaf_summary.json", "w") as f:
        json.dump(summaries, f, indent=4)

    print("\n[+] Done!")


if __name__ == "__main__":
    main()
