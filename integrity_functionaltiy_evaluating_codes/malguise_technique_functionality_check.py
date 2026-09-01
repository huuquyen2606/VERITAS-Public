import os
import json
import re
import sys
import time
import numpy as np
from typing import List, Dict, Any

import editdistance  # The C-compiled library for lightning-fast Levenshtein

# Import the dynamic progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    print(" [!] Missing 'tqdm' module. Proceeding without the progress bar. (Run: pip install tqdm)")
    HAS_TQDM = False

# Re-import your sequence deduplicator
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from examples.sequence_deduplicator import deduplicate_sequence
    HAS_DEDUP = True
except ImportError:
    print(" [!] Missing sequence_deduplicator module. Proceeding without deduplication.")
    HAS_DEDUP = False

# ==========================================
# Configuration & Paths (STRICTLY APIs)
# ==========================================
DATASETS_API_DIR = "datasets/APIs"
ORIG_API_FILENAME = "Test_full.npz"
MALGUISE_API_FILENAME = "malguise_full.npz"

# Default distance threshold (dist_norm < 0.20 corresponds to > 80% sequence similarity)
DEFAULT_DIST_DELTA = 0.20 

# ==========================================
# Helper Functions for Data Processing
# ==========================================
def get_base_name(filename: Any) -> str:
    """Extracts standard hash or base name from sample identifiers."""
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='ignore')
    elif hasattr(filename, 'item'):
        val = filename.item()
        filename = val.decode('utf-8', errors='ignore') if isinstance(val, bytes) else str(val)
    else:
        filename = str(filename)
        
    match = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if match:
        return match.group(1)
    return filename.split('.exe')[0] + '.exe'


def get_features(data: np.lib.npyio.NpzFile, key_choices: List[str], count: int) -> list:
    """Safely extracts feature arrays from .npz files."""
    for key in key_choices:
        if key in data.files:
            val = data[key]
            return val.tolist() if hasattr(val, 'tolist') else list(val)
    return [[] for _ in range(count)]


def extract_json_if_needed(data: Any) -> list:
    """Normalizes API traces and JSON strings into Python lists."""
    if hasattr(data, 'tolist'):
        data = data.tolist()
    if isinstance(data, str):
        try:
            return json.loads(data)
        except Exception:
            return []
    return data if data is not None else []


def extract_api_names(api_chain: list) -> List[str]:
    """Extracts a flat list of API names from dictionary or string representations."""
    names = []
    for item in api_chain:
        if isinstance(item, dict):
            name = item.get("name", item.get("api", ""))
            if name:
                names.append(str(name))
        elif isinstance(item, str) and item.strip():
            names.append(item.strip())
    return names


# ==========================================
# Paper Metric: Normalized Edit Distance
# ==========================================
def compute_normalized_edit_distance(seq_orig: List[str], seq_adv: List[str]) -> float:
    """
    Computes Eq. (7) from the MalGuise paper using a lightning-fast C-extension.
    dist_norm(z, z_adv) = Distance(API_z, API_zadv) / max(len(API_z), len(API_zadv))
    """
    len_orig = len(seq_orig)
    len_adv = len(seq_adv)
    max_len = max(len_orig, len_adv)

    if max_len == 0:
        return 0.0

    # This runs in C, taking milliseconds instead of minutes
    edit_dist = editdistance.eval(seq_orig, seq_adv)
    
    return edit_dist / max_len


# ==========================================
# Main Evaluation Engine (MalGuise ONLY)
# ==========================================
def main():
    print(f"[*] Initializing Optimized MalGuise API Functionality Evaluator")

    orig_api_path = os.path.join(DATASETS_API_DIR, ORIG_API_FILENAME)
    malguise_api_path = os.path.join(DATASETS_API_DIR, MALGUISE_API_FILENAME)

    if not os.path.exists(orig_api_path):
        print(f"[!] Error: Baseline dataset not found at {orig_api_path}")
        return
    if not os.path.exists(malguise_api_path):
        print(f"[!] Error: MalGuise dataset not found at {malguise_api_path}")
        return

    # 1. Load Baseline Dataset
    print(f"[*] Loading Baseline API Dataset ({ORIG_API_FILENAME})...")
    orig_api_data = np.load(orig_api_path, allow_pickle=True)
    orig_api_names = get_features(orig_api_data, ['name'], 0)
    orig_api_chains = get_features(orig_api_data, ['api_cuckoo', 'api'], len(orig_api_names))
    orig_api_labels = get_features(orig_api_data, ['label'], len(orig_api_names))

    baseline: Dict[str, Dict[str, Any]] = {}
    for name, api, label in zip(orig_api_names, orig_api_chains, orig_api_labels):
        base = get_base_name(name)
        lbl_str = label.decode('utf-8') if isinstance(label, bytes) else (str(label) if label != [] else "")
        
        parsed_api_raw = extract_json_if_needed(api)
        flat_api_list = extract_api_names(parsed_api_raw)

        baseline[base] = {
            "name": name,
            "api_list": flat_api_list,
            "label": lbl_str,
            "is_alive": len(flat_api_list) > 0
        }
    print(f"[+] Loaded {len(baseline)} original baseline samples.")

    # 2. Load MalGuise Dataset
    print(f"[*] Loading MalGuise API Dataset ({MALGUISE_API_FILENAME})...")
    adv_api_data = np.load(malguise_api_path, allow_pickle=True)
    adv_api_names = get_features(adv_api_data, ['name'], 0)
    adv_api_chains = get_features(adv_api_data, ['api_cuckoo', 'api'], len(adv_api_names))

    malguise_data_map = {}
    for name, api in zip(adv_api_names, adv_api_chains):
        base = get_base_name(name)
        parsed_api = extract_json_if_needed(api)
        flat_api = extract_api_names(parsed_api)
        malguise_data_map[base] = {
            "name": name,
            "api_list": flat_api,
            "is_alive": len(flat_api) > 0
        }
    print(f"[+] Loaded {len(malguise_data_map)} MalGuise samples.")

    # 3. Process & Compare
    print(f"\n==================================================")
    print(f"[*] Evaluating MalGuise Semantics Preservation Rate")
    print(f"==================================================")

    start_time = time.time()
    total_malware_evaluated = 0
    live_count = 0
    preserved_semantics_count = 0
    
    dist_scores = []
    malguise_detailed = {}
    malguise_lived_samples = {}

    # Setup the Progress Bar
    if HAS_TQDM:
        pbar = tqdm(baseline.items(), desc="Crunching Samples", unit="file", dynamic_ncols=True, 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        iterable = pbar
    else:
        pbar = None
        iterable = baseline.items()

    for base, orig_info in iterable:
        # Filter out benign samples
        if orig_info.get("label", "").lower() == "benign":
            continue

        total_malware_evaluated += 1
        adv_info = malguise_data_map.get(base, {"name": base, "api_list": [], "is_alive": False})

        # Apply Deduplication if available to strip out API spam
        orig_api_raw = orig_info["api_list"]
        adv_api_raw = adv_info["api_list"]

        if HAS_DEDUP:
            orig_api_clean = deduplicate_sequence(orig_api_raw, lm=5, k=2)
            adv_api_clean = deduplicate_sequence(adv_api_raw, lm=5, k=2)
        else:
            orig_api_clean = orig_api_raw
            adv_api_clean = adv_api_raw

        # Re-check dynamic execution integrity (Live definition: non-empty API trace after dedup)
        orig_is_alive = len(orig_api_clean) > 0
        adv_is_alive = len(adv_api_clean) > 0

        if not orig_is_alive or not adv_is_alive:
            malguise_detailed[str(adv_info["name"])] = {
                "is_alive": adv_is_alive,
                "orig_alive": orig_is_alive,
                "dist_norm": 1.0,
                "similarity_score": 0.0,
                "is_semantics_preserved": False,
                "status": "DEAD_OR_NO_API_TRACE"
            }
        else:
            # Sample is verified ALIVE
            live_count += 1

            # Compute Normalized Edit Distance (Eq. 7) using fast C-extension
            dist_norm = compute_normalized_edit_distance(orig_api_clean, adv_api_clean)
            similarity_score = (1.0 - dist_norm) * 100.0
            dist_scores.append(dist_norm)

            # Check Semantics Equivalence (Eq. 8)
            is_preserved = dist_norm < DEFAULT_DIST_DELTA
            if is_preserved:
                preserved_semantics_count += 1
                malguise_lived_samples[base] = {
                    "sample_name": str(adv_info["name"]),
                    "orig_api_len": len(orig_api_clean),
                    "adv_api_len": len(adv_api_clean),
                    "dist_norm": dist_norm,
                    "similarity_score": similarity_score
                }

            malguise_detailed[str(adv_info["name"])] = {
                "is_alive": True,
                "orig_alive": True,
                "dist_norm": dist_norm,
                "similarity_score": round(similarity_score, 2),
                "is_semantics_preserved": is_preserved,
                "status": "PRESERVED" if is_preserved else "SEMANTICS_DIVERGED"
            }

        # Dynamically update the visual statistics on the progress bar
        if pbar:
            current_spr = (preserved_semantics_count / live_count * 100) if live_count > 0 else 0.0
            pbar.set_postfix({
                "Eval": total_malware_evaluated,
                "Live": live_count,
                "SPR": f"{current_spr:.1f}%"
            })

    # Close the progress bar cleanly when done
    if pbar:
        pbar.close()

    # Calculate final summary rates
    execution_time = time.time() - start_time
    live_rate = (live_count / total_malware_evaluated * 100) if total_malware_evaluated > 0 else 0.0
    spr = (preserved_semantics_count / live_count * 100) if live_count > 0 else 0.0
    overall_spr = (preserved_semantics_count / total_malware_evaluated * 100) if total_malware_evaluated > 0 else 0.0
    avg_dist = float(np.mean(dist_scores)) if dist_scores else 1.0

    print(f"\n==================================================")
    print(f" -> Total Malware Evaluated : {total_malware_evaluated}")
    print(f" -> Lived Samples (Alive)   : {live_count} ({live_rate:.2f}%)")
    print(f" -> Semantics Preserved     : {preserved_semantics_count}")
    print(f" -> SPR (Semantics Rate/Live): {spr:.2f}%")
    print(f" -> Overall Retention Rate  : {overall_spr:.2f}%")
    print(f" -> Avg Normalized Distance : {avg_dist:.4f}")
    print(f" -> Execution Time          : {execution_time:.2f} seconds")
    print(f"==================================================")

    # 4. Export Reports
    out_live_file = "malguise_filtered_live_samples.json"
    out_detailed_file = "malguise_detailed_api_report.json"
    out_summary_file = "malguise_functionality_summary.json"

    print(f"\n[*] Writing filtered lived samples to: {out_live_file}")
    with open(out_live_file, "w") as f:
        json.dump({"malguise": malguise_lived_samples}, f, indent=4)

    print(f"[*] Writing detailed API metrics to: {out_detailed_file}")
    with open(out_detailed_file, "w") as f:
        json.dump({"malguise": malguise_detailed}, f, indent=4)

    print(f"[*] Writing summary statistics to: {out_summary_file}")
    with open(out_summary_file, "w") as f:
        json.dump({
            "malguise": {
                "total_samples": total_malware_evaluated,
                "live_count": live_count,
                "live_percentage": round(live_rate, 2),
                "preserved_semantics_count": preserved_semantics_count,
                "spr_percentage": round(spr, 2),
                "overall_retention_percentage": round(overall_spr, 2),
                "average_dist_norm": round(avg_dist, 4),
                "threshold_dist_delta": DEFAULT_DIST_DELTA
            }
        }, f, indent=4)

    print("\n[+] MalGuise evaluation successfully completed!")

if __name__ == "__main__":
    main()