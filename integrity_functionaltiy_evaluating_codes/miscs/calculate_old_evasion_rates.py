import json
import re
import csv
import argparse

def extract_hash(filename):
    match = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if match:
        return match.group(1).lower()
    return None

def is_bypassed(final_label, true_class, true_binary, strict_bypass=False):
    if not final_label:
        return True # if no label, consider bypassed

    fl = str(final_label).lower()
    tc = str(true_class).lower()
    
    if fl == "benign":
        return True
        
    if strict_bypass:
        return False
        
    if fl == "malware":
        return False
        
    return fl != tc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='malware_detector_results_filtered.json')
    parser.add_argument('--dqeaf_fir_file', type=str, default='dqeaf_FIR.json')
    parser.add_argument('--fiam_fir_file', type=str, default='fiam_FIR.json')
    parser.add_argument('--output_prefix', type=str, default='old_evasion_rates')
    args = parser.parse_args()
    
    print("Loading data...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
        
    with open(args.dqeaf_fir_file, 'r') as f:
        dqeaf_fir_data = json.load(f)
        
    with open(args.fiam_fir_file, 'r') as f:
        fiam_fir_data = json.load(f)
        
    tech_map = {
        'MAB_full_lived': 'mab',
        'Malgpt_full': 'malgpt',
        'OBFU-mal_full': 'obfu',
        'aimed_full': 'aimed',
        'dqef_full_lived': 'dqef',
        'fiam_1500_filtered': 'fiam',
        'gamma_adv_full': 'gamma',
        'gapgan_full': 'gapgan'
    }
    
    # Pre-process FIR data by hash for O(1) lookups
    dqeaf_fir_hashes = {}
    for fir_tech, samples in dqeaf_fir_data.items():
        dqeaf_fir_hashes[fir_tech] = {}
        for filename, attrs in samples.items():
            h = extract_hash(filename)
            if h:
                dqeaf_fir_hashes[fir_tech][h] = (attrs.get('combined_status') == 'BOTH')
                
    fiam_fir_hashes = {}
    for fir_tech, samples in fiam_fir_data.items():
        fiam_fir_hashes[fir_tech] = {}
        for filename, attrs in samples.items():
            h = extract_hash(filename)
            if h:
                fiam_fir_hashes[fir_tech][h] = (attrs.get('is_functional') == True and attrs.get('is_alive') == True)
        
    test_full = data.get('Test_full', {})
    if not test_full:
        print("Error: 'Test_full' dataset not found in the JSON file.")
        return
        
    # Extract ground truth hashes from Test_full and count original malwares
    original_samples = {}
    total_original_malware = 0
    for filename, sample_data in test_full.items():
        h = extract_hash(filename)
        if h:
            original_samples[h] = sample_data
            if sample_data.get('true_label_binary') == 'Malware':
                total_original_malware += 1
            
    print(f"Loaded {len(original_samples)} valid original samples from Test_full")
    print(f"Total Original Malware Count (Denominator): {total_original_malware}")
    
    # Detectors
    detectors = set()
    for sample_data in original_samples.values():
        for d in sample_data.get('results', {}).keys():
            detectors.add(d)
    detectors = sorted(list(detectors))

    techniques = [k for k in data.keys() if k != 'Test_full']
    
    def process_rates(strict):
        csv_rows = []
        for tech in techniques:
            tech_data = data[tech]
            fir_tech_key = tech_map.get(tech)
            
            # Extract hashes for the technique
            adv_samples = {}
            for filename, sample_data in tech_data.items():
                h = extract_hash(filename)
                if h:
                    adv_samples[h] = sample_data
                    
            tech_stats = {d: {
                'adv_bypassed_files': 0,
                'dqeaf_fir_bypassed_files': 0,
                'fiam_fir_bypassed_files': 0
            } for d in detectors}
            
            for h, adv_data in adv_samples.items():
                orig_data = original_samples.get(h)
                if not orig_data:
                    continue # Need true label
                    
                # We only consider files that are actually malware originally as part of the evasion rate denominator
                if orig_data.get('true_label_binary') != 'Malware':
                    continue
                    
                is_valid_dqeaf = False
                is_valid_fiam = False
                if fir_tech_key:
                    is_valid_dqeaf = dqeaf_fir_hashes.get(fir_tech_key, {}).get(h, False)
                    is_valid_fiam = fiam_fir_hashes.get(fir_tech_key, {}).get(h, False)
                
                adv_results = adv_data.get('results', {})
                for detector in detectors:
                    adv_res = adv_results.get(detector, {})
                    adv_fl = adv_res.get('final_label')
                    
                    tc = orig_data.get('true_label_class')
                    tb = orig_data.get('true_label_binary')
                    
                    adv_bypassed = is_bypassed(adv_fl, tc, tb, strict_bypass=strict)
                    
                    if adv_bypassed:
                        tech_stats[detector]['adv_bypassed_files'] += 1
                        if is_valid_dqeaf:
                            tech_stats[detector]['dqeaf_fir_bypassed_files'] += 1
                        if is_valid_fiam:
                            tech_stats[detector]['fiam_fir_bypassed_files'] += 1
                            
            for detector in detectors:
                stats = tech_stats[detector]
                evasion_rate = stats['adv_bypassed_files'] / total_original_malware
                dqeaf_fir_evasion_rate = stats['dqeaf_fir_bypassed_files'] / total_original_malware
                fiam_fir_evasion_rate = stats['fiam_fir_bypassed_files'] / total_original_malware
                
                csv_rows.append({
                    'Technique': tech,
                    'Detector': detector,
                    'Evasion_Files': stats['adv_bypassed_files'],
                    'Valid evasion files (dqeaf)': stats['dqeaf_fir_bypassed_files'],
                    'Valid evasion files (fiam)': stats['fiam_fir_bypassed_files'],
                    'Total_Original_Malware': total_original_malware,
                    'Evasion_Rate': f"{evasion_rate:.4f}",
                    'Valid Evasion rate (dqeaf)': f"{dqeaf_fir_evasion_rate:.4f}",
                    'Valid evasion rate (fiam)': f"{fiam_fir_evasion_rate:.4f}"
                })
        return csv_rows

    # Generate Standard
    print("Calculating standard evasion rates...")
    std_rows = process_rates(strict=False)
    with open(f'{args.output_prefix}.csv', 'w', newline='') as f:
        fieldnames = [
            'Technique', 'Detector', 'Evasion_Files', 'Valid evasion files (dqeaf)', 'Valid evasion files (fiam)',
            'Total_Original_Malware', 'Evasion_Rate', 'Valid Evasion rate (dqeaf)', 'Valid evasion rate (fiam)'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(std_rows)
        
    # Generate Strict
    print("Calculating strict evasion rates...")
    strict_rows = process_rates(strict=True)
    with open(f'{args.output_prefix}_strict.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(strict_rows)
        
    print(f"Done! Generated {args.output_prefix}.csv and {args.output_prefix}_strict.csv")

if __name__ == '__main__':
    main()
