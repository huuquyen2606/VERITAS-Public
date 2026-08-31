import os
import glob
import json
import numpy as np
import re

DATASETS_DIR = "datasets/APIs"
ORIGINAL_DATASET = "Test_full.npz"
THRESHOLD = 0.8


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


def calculate_jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / union if union > 0 else 0.0


def get_features(data, key_choices):
    for key in key_choices:
        if key in data.files:
            val = data[key]
            return val.tolist() if hasattr(val, "tolist") else list(val)
    # Return empty list if no keys matched
    return [[] for _ in range(len(data["name"]))]


def extract_feature_names(raw_sigs):
    if not raw_sigs:
        return set()
    
    # If the entire row is a single JSON string, parse it first
    if isinstance(raw_sigs, str):
        try:
            raw_sigs = json.loads(raw_sigs)
        except:
            pass
            
    # If it's still a string, just return it as a single feature
    if isinstance(raw_sigs, str):
        return {raw_sigs}
        
    features = set()
    for item in raw_sigs:
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    name = parsed.get("name", parsed.get("api", ""))
                    if name: features.add(name)
                else:
                    features.add(item)
            except:
                features.add(item)
        elif isinstance(item, dict):
            name = item.get("name", item.get("api", ""))
            if name: features.add(name)
    return features


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

    # Explicitly use only signatures for MAB Technique
    feature_keys = ["signatures"]
    orig_sigs = get_features(orig_data, feature_keys)

    # Map base name to a set of signatures
    original_signatures = {}
    for name, sigs in zip(orig_names, orig_sigs):
        if sigs is None:
            sigs = []
        elif hasattr(sigs, "tolist"):
            sigs = sigs.tolist()
        base = get_base_name(name)
        original_signatures[base] = extract_feature_names(sigs)

    print(f"    Loaded {len(original_signatures)} original samples.")

    # Find all adversarial datasets
    npz_files = glob.glob(os.path.join(DATASETS_DIR, "*.npz"))

    detailed_reports = {}
    summaries = {}

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
        adv_sigs = get_features(adv_data, feature_keys)

        dataset_report = {}
        integrity_count = 0
        total_samples = 0

        for adv_name, sigs in zip(adv_names, adv_sigs):
            adv_name_str = ensure_str(adv_name)
            if sigs is None:
                sigs = []
            elif hasattr(sigs, "tolist"):
                sigs = sigs.tolist()

            adv_sig_set = extract_feature_names(sigs)
            base_name = get_base_name(adv_name_str)

            if base_name in original_signatures:
                total_samples += 1
                orig_sig_set = original_signatures[base_name]
                similarity = calculate_jaccard_similarity(orig_sig_set, adv_sig_set)

                is_functional = similarity > THRESHOLD
                if is_functional:
                    integrity_count += 1

                dataset_report[adv_name_str] = {
                    "base_name": base_name,
                    "similarity": similarity,
                    "is_functional": is_functional,
                    "orig_sig_count": len(orig_sig_set),
                    "adv_sig_count": len(adv_sig_set),
                }
            else:
                # Base name not found in original dataset, skip or mark as 0
                dataset_report[adv_name_str] = {
                    "base_name": base_name,
                    "similarity": 0.0,
                    "is_functional": False,
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
    print("\n[*] Saving detailed reports to mab_detailed_reports.json")
    with open("mab_detailed_reports.json", "w") as f:
        json.dump(detailed_reports, f, indent=4)

    print("[*] Saving summaries to mab_summary.json")
    with open("mab_summary.json", "w") as f:
        json.dump(summaries, f, indent=4)

    print("\n[+] Done!")


if __name__ == "__main__":
    main()
