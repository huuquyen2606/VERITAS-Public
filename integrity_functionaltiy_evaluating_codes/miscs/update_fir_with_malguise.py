import json
import os

def main():
    # 1. Update DQEAF FIR
    with open('dqeaf_FIR.json', 'r') as f:
        dqeaf_fir = json.load(f)
        
    with open('malguise/dqeaf_functionality_check.json', 'r') as f:
        func_report = json.load(f)
        
    with open('malguise/dqeaf_integrity_check.json', 'r') as f:
        integ_report = json.load(f)

    # In integ_report, the key is 'malguise_full.npz' or 'malguise_api.npz'?
    # Earlier I saw 'malguise_full.npz' in the keys of integ_report.
    malguise_integ = integ_report.get('malguise_full.npz', {})
    
    # In func_report, the key is 'malguise' inside 'results_by_directory'
    malguise_func_details = func_report.get('results_by_directory', {}).get('malguise', {}).get('details', [])
    
    processed_samples = set()
    malguise_dqeaf_data = {}
    
    for detail in malguise_func_details:
        sample = detail.get('adversarial_file')
        if not sample:
            continue
            
        processed_samples.add(sample)
        
        status = detail.get('status')
        is_functional = (status == 'preserved')
        cfg_status = status
        
        integ_info = malguise_integ.get(sample, {})
        is_integrity_preserved = integ_info.get('is_alive', False)
        
        combined = "NEITHER"
        if is_functional and is_integrity_preserved:
            combined = "BOTH"
        elif is_functional:
            combined = "FUNCTIONAL_ONLY"
        elif is_integrity_preserved:
            combined = "INTEGRITY_ONLY"
            
        malguise_dqeaf_data[sample] = {
            "dataset": "malguise",
            "is_functional": is_functional,
            "is_integrity_preserved": is_integrity_preserved,
            "cfg_status": cfg_status,
            "combined_status": combined
        }
        
    # Also process any samples in integ_report that were missed
    for sample, integ_info in malguise_integ.items():
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
                
            malguise_dqeaf_data[sample] = {
                "dataset": "malguise",
                "is_functional": is_functional,
                "is_integrity_preserved": is_integrity_preserved,
                "cfg_status": "broken",
                "combined_status": combined
            }
            
    dqeaf_fir['malguise'] = malguise_dqeaf_data
    
    with open('dqeaf_FIR.json', 'w') as f:
        json.dump(dqeaf_fir, f, indent=4)
        
    print(f"Updated dqeaf_FIR.json with {len(malguise_dqeaf_data)} malguise samples.")
    
    # 2. Update FIAM FIR
    with open('fiam_FIR.json', 'r') as f:
        fiam_fir = json.load(f)
        
    with open('malguise/fiam_full.json', 'r') as f:
        fiam_full = json.load(f)
        
    if 'malguise' in fiam_full:
        fiam_fir['malguise'] = fiam_full['malguise']
        with open('fiam_FIR.json', 'w') as f:
            json.dump(fiam_fir, f, indent=4)
        print(f"Updated fiam_FIR.json with {len(fiam_full['malguise'])} malguise samples.")

if __name__ == '__main__':
    main()
