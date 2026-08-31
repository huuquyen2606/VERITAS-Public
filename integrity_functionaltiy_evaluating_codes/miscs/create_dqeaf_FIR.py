import json
import os

def main():
    base_dir = '/home/gnomeright/programming/nghien_cuu/new_results/dqeaf_handles/dqeaf'
    
    with open(os.path.join(base_dir, 'dqeaf_integrity_functionality_reports.json'), 'r') as f:
        base_report = json.load(f)
        
    with open(os.path.join(base_dir, 'dqeaf_with_fiam_included_functionlaity_check.json'), 'r') as f:
        func_report = json.load(f)
        
    with open(os.path.join(base_dir, 'dqeaf_with_fiam_included_integrity_check.json'), 'r') as f:
        integ_report = json.load(f)
        
    if 'proposal' in base_report:
        del base_report['proposal']
        
    fiam_data = {}
    
    # Integ lookup
    fiam_integ = integ_report.get('fiam_full.npz', {})
    
    # Func lookup
    fiam_func_details = func_report.get('results_by_directory', {}).get('fiam_1500', {}).get('details', [])
    
    # Some samples might be in integ but not in func, or vice-versa.
    # We should iterate over both or just one? The details in func seems to list adversarial files.
    # Let's iterate over fiam_func_details first
    
    processed_samples = set()
    
    for detail in fiam_func_details:
        sample = detail.get('adversarial_file')
        if not sample:
            continue
            
        processed_samples.add(sample)
        
        status = detail.get('status')
        is_functional = (status == 'preserved')
        cfg_status = status
        
        integ_info = fiam_integ.get(sample, {})
        is_integrity_preserved = integ_info.get('is_alive', False)
        
        combined = "NEITHER"
        if is_functional and is_integrity_preserved:
            combined = "BOTH"
        elif is_functional:
            combined = "FUNCTIONAL_ONLY"
        elif is_integrity_preserved:
            combined = "INTEGRITY_ONLY"
            
        fiam_data[sample] = {
            "dataset": "fiam",
            "is_functional": is_functional,
            "is_integrity_preserved": is_integrity_preserved,
            "cfg_status": cfg_status,
            "combined_status": combined
        }
        
    # Also process any samples in integ_report that were missed (if any)
    for sample, integ_info in fiam_integ.items():
        if sample not in processed_samples:
            is_functional = False
            is_integrity_preserved = integ_info.get('is_alive', False)
            
            combined = "NEITHER"
            if is_functional and is_integrity_preserved:
                combined = "BOTH"
            elif is_functional:
                combined = "FUNCTIONAL_ONLY"
            elif is_integrity_preserved:
                combined = "INTEGRITY_ONLY"
                
            fiam_data[sample] = {
                "dataset": "fiam",
                "is_functional": is_functional,
                "is_integrity_preserved": is_integrity_preserved,
                "cfg_status": "broken", # Default if missing from func check
                "combined_status": combined
            }

    base_report['fiam'] = fiam_data
    
    out_path = os.path.join(base_dir, 'dqeaf_FIR.json')
    with open(out_path, 'w') as f:
        json.dump(base_report, f, indent=4)
        
    print(f"Created {out_path} with {len(fiam_data)} FIAM samples.")

if __name__ == '__main__':
    main()
