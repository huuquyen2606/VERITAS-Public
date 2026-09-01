import json

def main():
    print("Loading data...")
    with open('malware_detector_results_raw.json', 'r') as f:
        raw_data = json.load(f)
        
    with open('dqeaf_handles/dqeaf/dqeaf_integrity_functionality_reports.json', 'r') as f:
        dqeaf_report = json.load(f)
        
    with open('dqeaf_handles/mab/mab_detailed_reports.json', 'r') as f:
        mab_report = json.load(f)
        
    with open('dqeaf_handles/fiam_FIR.json', 'r') as f:
        fiam_report = json.load(f)

    print("Filtering dqeaf...")
    # dqeaf
    dqeaf_alive = set()
    for sample, data in dqeaf_report.get('dqef', {}).items():
        if data.get('is_functional') and data.get('is_integrity_preserved'):
            dqeaf_alive.add(sample)
    
    if 'dqef_full_lived' in raw_data:
        filtered_dqeaf = {k: v for k, v in raw_data['dqef_full_lived'].items() if k in dqeaf_alive}
        raw_data['dqef_full_lived'] = filtered_dqeaf
        print(f"Dqeaf: filtered to {len(filtered_dqeaf)}")
    
    print("Filtering mab...")
    # mab
    mab_alive = set()
    for sample, data in mab_report.get('mab_full.npz', {}).items():
        if data.get('is_functional'):
            mab_alive.add(sample)
            
    if 'MAB_full_lived' in raw_data:
        filtered_mab = {k: v for k, v in raw_data['MAB_full_lived'].items() if k in mab_alive}
        raw_data['MAB_full_lived'] = filtered_mab
        print(f"MAB: filtered to {len(filtered_mab)}")

    print("Filtering fiam...")
    # fiam
    fiam_alive = set()
    for ds in fiam_report.keys():
        for sample, data in fiam_report.get(ds, {}).items():
            if data.get('is_functional') and data.get('is_alive'):
                 fiam_alive.add(sample)
                 
    if 'fiam_1500_filtered' in raw_data:
        filtered_fiam = {k: v for k, v in raw_data['fiam_1500_filtered'].items() if k in fiam_alive}
        print(f"FIAM: filtered to {len(filtered_fiam)}")
        raw_data['fiam_1500_filtered'] = filtered_fiam
        
    print("Saving...")
    with open('malware_detector_results_filtered.json', 'w') as f:
        json.dump(raw_data, f, indent=4)
    print("Done!")

if __name__ == '__main__':
    main()
