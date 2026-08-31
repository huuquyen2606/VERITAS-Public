#!/usr/bin/env python3
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Merge Functionality and Integrity reports into a FIR report.")
    parser.add_argument("--func-report", required=True, help="Path to the functionality JSON report.")
    parser.add_argument("--integ-report", required=True, help="Path to the integrity JSON report.")
    parser.add_argument("--technique", required=True, help="Name of the technique (e.g., dqeaf, malguise). Used to locate keys in some reports.")
    parser.add_argument("--output", required=True, help="Output FIR JSON report path.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.func_report):
        print(f"[!] Functionality report not found: {args.func_report}")
        return
        
    if not os.path.exists(args.integ_report):
        print(f"[!] Integrity report not found: {args.integ_report}")
        return

    print(f"[*] Loading functionality report: {args.func_report}")
    with open(args.func_report, 'r') as f:
        func_data = json.load(f)
        
    print(f"[*] Loading integrity report: {args.integ_report}")
    with open(args.integ_report, 'r') as f:
        integ_data = json.load(f)

    # Resolve functionality data
    # Some reports have a flat structure, some are nested like 'results_by_directory' -> <technique> -> 'details'
    func_details = []
    if "results_by_directory" in func_data:
        dir_data = func_data["results_by_directory"].get(args.technique, {})
        func_details = dir_data.get("details", [])
    else:
        # Check if the technique name is the top level key
        if args.technique in func_data:
            func_details = func_data[args.technique]
        else:
            # Assume flat dictionary of samples
            # Try to guess if it's flat by looking at a value
            first_val = next(iter(func_data.values()), None) if isinstance(func_data, dict) else None
            if isinstance(first_val, dict) and "is_functional" in first_val:
                func_details = func_data
            else:
                print(f"[!] Warning: Could not easily parse functionality report format. Falling back to raw dictionary.")
                func_details = func_data

    # Convert func_details to a standardized dict mapping sample_name -> is_functional
    func_map = {}
    if isinstance(func_details, list):
        for item in func_details:
            name = item.get("adversarial_file", item.get("sample_name", ""))
            if not name:
                continue
            is_func = item.get("status") == "preserved" or item.get("is_functional") or item.get("is_semantics_preserved")
            func_map[name] = bool(is_func)
    elif isinstance(func_details, dict):
        for name, item in func_details.items():
            if isinstance(item, dict):
                is_func = item.get("is_functional") or item.get("is_semantics_preserved") or item.get("status") == "preserved"
                func_map[name] = bool(is_func)
            else:
                func_map[name] = bool(item)

    # Resolve integrity data
    # Some are flat, some might have '<technique>_full.npz' as key
    integ_map = {}
    integ_dataset_key = None
    for k in integ_data.keys():
        if args.technique.lower() in k.lower():
            integ_dataset_key = k
            break
            
    if integ_dataset_key:
        integ_samples = integ_data[integ_dataset_key]
    else:
        # Check if it is a flat dictionary of samples
        first_val = next(iter(integ_data.values()), None) if isinstance(integ_data, dict) else None
        if isinstance(first_val, dict) and "is_alive" in first_val:
            integ_samples = integ_data
        else:
            print(f"[!] Warning: Could not cleanly find '{args.technique}' in integrity report. Using entire root.")
            integ_samples = integ_data
            
    for name, item in integ_samples.items():
        if isinstance(item, dict):
            integ_map[name] = bool(item.get("is_alive", False))
        else:
            integ_map[name] = bool(item)
            
    # Combine the reports
    all_samples = set(func_map.keys()).union(set(integ_map.keys()))
    
    fir_results = {}
    counts = {"BOTH": 0, "FUNCTIONAL_ONLY": 0, "INTEGRITY_ONLY": 0, "NEITHER": 0}
    
    for sample in all_samples:
        is_func = func_map.get(sample, False)
        is_integ = integ_map.get(sample, False)
        
        status = "NEITHER"
        if is_func and is_integ:
            status = "BOTH"
        elif is_func:
            status = "FUNCTIONAL_ONLY"
        elif is_integ:
            status = "INTEGRITY_ONLY"
            
        counts[status] += 1
        
        fir_results[sample] = {
            "dataset": args.technique,
            "is_functional": is_func,
            "is_integrity_preserved": is_integ,
            "combined_status": status
        }
        
    final_report = {
        args.technique: fir_results,
        "summary": counts
    }
    
    print(f"[*] Saving combined FIR report to {args.output}")
    print(f"    Summary: {counts}")
    
    with open(args.output, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print("[+] Done!")

if __name__ == '__main__':
    main()
