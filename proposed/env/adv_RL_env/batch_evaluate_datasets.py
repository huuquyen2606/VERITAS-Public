"""
Batch Dataset Evaluator
=======================
Automated test script to orchestrate the semantic evaluation of 3 dataset types:
- Ground Truth (Exact clones)
- False Dataset (Different files)
- Obfuscation Dataset (Packed files)

This script submits all files to the CAPE/Redis workers in parallel, waits for
extraction to finish, and computes the API similarity scores using the Smith-Waterman
evaluator. It then prints a comprehensive statistical report.
"""

import os
import sys
import json
import time
import shutil
import redis
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'malware_rl_system'))

from core.redis_orchestrator import RedisOrchestrator
from evaluation.functionality_evaluator import FunctionalityEvaluator

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
WORK_DIR = "/tmp"
POLL_INTERVAL = 5
MAX_WAIT = 3600  

def extract_pairs(dataset_path):
    """Scan the dataset folder for pair_* directories and return the file paths."""
    pairs = []
    p = Path(dataset_path)
    if not p.exists():
        return pairs

    for pair_dir in p.iterdir():
        if pair_dir.is_dir() and pair_dir.name.startswith("pair_"):
            files = list(pair_dir.glob("*"))
            if len(files) == 2:
                file_a = None
                file_b = None
                for f in files:
                    lower_name = f.name.lower()
                    if 'original' in lower_name or '_a.' in lower_name:
                        file_a = f
                    else:
                        file_b = f
                
                if file_a is None or file_b is None:
                    file_a, file_b = files[0], files[1]

                pairs.append((file_a, file_b))
    return pairs

def main():
    base_dir = Path(__file__).parent
    datasets = {
        "Ground Truth": base_dir / "tmp/ground_truth_dataset",
        "False Dataset": base_dir / "tmp/false_dataset",
        "Obfuscation Dataset": base_dir / "tmp/obfuscation_dataset"
    }

    print("\n" + "="*70)
    print(" BATCH DATASET EVALUATION PIPELINE")
    print("="*70)

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        r.ping()
        print("[✓] Redis connection OK.")
    except redis.ConnectionError:
        print("[!] Cannot connect to Redis. Make sure it is running!")
        sys.exit(1)

    orchestrator = RedisOrchestrator()
    evaluator = FunctionalityEvaluator()

    all_tasks = []
    dataset_records = {}

    for ds_name, ds_path in datasets.items():
        if not ds_path.exists():
            print(f"[!] Warning: Dataset path not found: {ds_path}")
            continue

        pairs = extract_pairs(ds_path)
        print(f"[*] Found {len(pairs)} pairs in '{ds_name}'.")
        
        dataset_records[ds_name] = []
        
        for file_a, file_b in pairs:
            ori_name = file_a.name
            adv_name = file_b.name
            
            shutil.copy2(file_a, Path(WORK_DIR) / ori_name)
            shutil.copy2(file_b, Path(WORK_DIR) / adv_name)

            record = {
                "ori_name": ori_name,
                "adv_name": adv_name,
                "ori_barrier": f"baseline:{ori_name}:tasks_remaining",
                "adv_barrier": f"episode:{adv_name}:tasks_remaining",
                "score": None,
                "method": None
            }
            dataset_records[ds_name].append(record)
            all_tasks.append(record)

            r.hset(f"baseline:{ori_name}:data", "true_label", ds_name)
            orchestrator.dispatch_task(ori_name, parent_file_name=ori_name, is_original=True, episode_id="batch_eval")
            
            orchestrator.dispatch_task(adv_name, parent_file_name=ori_name, is_original=False, episode_id="batch_eval")

    if not all_tasks:
        print("[!] No files found to evaluate. Please generate datasets first.")
        sys.exit(1)

    print(f"\n[*] Successfully dispatched {len(all_tasks) * 2} files to Redis queues.")
    print(f"[*] Waiting for all workers to complete extraction (MAX {MAX_WAIT}s)...")

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > MAX_WAIT:
            print(f"\n[!] Timeout waiting for workers after {MAX_WAIT}s.")
            break

        all_cleared = True
        remaining_tasks = 0

        for task in all_tasks:
            ori_val = r.get(task["ori_barrier"])
            adv_val = r.get(task["adv_barrier"])
            
            ori_rem = int(ori_val) if ori_val else 0
            adv_rem = int(adv_val) if adv_val else 0
            
            rem = ori_rem + adv_rem
            remaining_tasks += rem
            if rem > 0:
                all_cleared = False

        print(f"    ... remaining subtasks across all files: {remaining_tasks}      ", end="\r")

        if all_cleared:
            print(f"\n\n[✓] All files have been successfully extracted by CAPE & Angr!")
            break

        time.sleep(POLL_INTERVAL)

    print("\n[*] Running Semantic Functionality Evaluator (Smith-Waterman/Z3) on all pairs...")
    
    for ds_name, records in dataset_records.items():
        for rec in records:
            adv_key = f"episode:{rec['adv_name']}:data"
            
            try:
                evaluator.evaluate(rec["adv_name"], rec["ori_name"])
                
                final_data = r.hgetall(adv_key)
                func_score = float(final_data.get("functionality_score", 0.0))
                method = final_data.get("evaluation_method", "UNKNOWN")
                
                rec["score"] = func_score
                rec["method"] = method
            except Exception as e:
                print(f" [!] Error evaluating {rec['adv_name']}: {e}")
                rec["score"] = 0.0
                rec["method"] = "ERROR"

    print("\n" + "="*70)
    print(" 📊 COMPREHENSIVE DATASET EVALUATION REPORT")
    print("="*70)

    report_json = {}

    for ds_name, records in dataset_records.items():
        if not records:
            continue
            
        valid_scores = [r["score"] for r in records if r["score"] is not None]
        
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            max_score = max(valid_scores)
            min_score = min(valid_scores)
        else:
            avg_score = max_score = min_score = 0.0

        print(f"\n📁 Dataset: {ds_name} ({len(records)} pairs)")
        print(f"   ├─ Average Similarity : {avg_score:>6.2f} %")
        print(f"   ├─ Max Similarity     : {max_score:>6.2f} %")
        print(f"   └─ Min Similarity     : {min_score:>6.2f} %")
        
        report_json[ds_name] = {
            "average": avg_score,
            "max": max_score,
            "min": min_score,
            "pairs": records
        }

    out_path = "dataset_evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report_json, f, indent=4)
        
    print("\n" + "="*70)
    print(f"[✓] Detailed pair-by-pair results saved to: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
