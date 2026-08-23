import json
import redis
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'malware_rl_system'))
from evaluation.functionality_evaluator import FunctionalityEvaluator

r = redis.Redis(decode_responses=True)
evaluator = FunctionalityEvaluator()

def analyze():
    base_dir = Path(__file__).parent
    datasets = ["ground_truth_dataset", "false_dataset", "obfuscation_dataset"]
    
    for ds_name in datasets:
        ds_path = base_dir / "tmp" / ds_name
        if not ds_path.exists(): 
            print(f"Path does not exist: {ds_path}")
            continue
        
        print(f"\n--- Analyzing {ds_name} ---")
        pairs = list(ds_path.glob("pair_*"))
        
        for pair_dir in pairs:
            files = list(pair_dir.glob("*"))
            if len(files) != 2: continue
            
            file_a, file_b = files[0].name, files[1].name
            if 'original' in file_a.lower() or '_a.' in file_a.lower():
                ori_name, adv_name = file_a, file_b
            else:
                ori_name, adv_name = file_b, file_a
                
            base_key = f"baseline:{ori_name}:data"
            adv_key = f"episode:{adv_name}:data"
            
            ori_data = r.hgetall(base_key)
            adv_data = r.hgetall(adv_key)
            
            if not ori_data or not adv_data:
                print(f"Missing data for {pair_dir.name}")
                continue
                
            is_alive, ori_sys, adv_sys = evaluator.check_integrity(ori_data, adv_data)
            print(f"[{pair_dir.name}] is_alive={is_alive} ori_sys={ori_sys} adv_sys={adv_sys}")
            
            api1 = len(json.loads(ori_data.get("api_chain", "[]")))
            api2 = len(json.loads(adv_data.get("api_chain", "[]")))
            print(f"   api_chain lengths: ori={api1}, adv={api2}")

analyze()
