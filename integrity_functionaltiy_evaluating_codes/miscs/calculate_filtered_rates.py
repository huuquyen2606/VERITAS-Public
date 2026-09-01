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
    parser.add_argument('--strict_bypass', action='store_true', help="If set, changing families is NOT a bypass.")
    parser.add_argument('--input_file', type=str, default='malware_detector_results_filtered.json')
    parser.add_argument('--dqeaf_fir_file', type=str, default='dqeaf_FIR.json')
    parser.add_argument('--fiam_fir_file', type=str, default='fiam_FIR.json')
    parser.add_argument('--csv_out', type=str, default='')
    parser.add_argument('--json_out', type=str, default='')
    args = parser.parse_args()
    
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
        
    # Extract ground truth hashes from Test_full
    original_samples = {} 
    for filename, sample_data in test_full.items():
        h = extract_hash(filename)
        if h:
            original_samples[h] = sample_data
            
    print(f"Loaded {len(original_samples)} valid original samples from Test_full")
    
    # Calculate Original Detected and Undetected per detector
    original_stats = {}
    
    for h, sample_data in original_samples.items():
        results = sample_data.get('results', {})
        for detector, res in results.items():
            if detector not in original_stats:
                original_stats[detector] = {'detected': 0, 'undetected': 0}
                
            fl = res.get('final_label')
            tc = sample_data.get('true_label_class')
            tb = sample_data.get('true_label_binary')
            
            bypassed = is_bypassed(fl, tc, tb, strict_bypass=args.strict_bypass)
            if bypassed:
                original_stats[detector]['undetected'] += 1
            else:
                original_stats[detector]['detected'] += 1

    techniques = [k for k in data.keys() if k != 'Test_full']
    
    csv_rows = []
    json_output = {
        'original_stats': original_stats,
        'techniques': {}
    }
    
    for tech in techniques:
        json_output['techniques'][tech] = {}
        tech_data = data[tech]
        
        fir_tech_key = tech_map.get(tech)
        
        # Extract hashes for the technique
        adv_samples = {}
        for filename, sample_data in tech_data.items():
            h = extract_hash(filename)
            if h:
                adv_samples[h] = sample_data
                
        detectors = list(original_stats.keys())
        tech_stats = {d: {
            'success_orig_det': 0, 
            'success_orig_undet': 0,
            'adv_det': 0,
            'adv_undet': 0,
            'evasion_files': 0, 
            'spoiled_files': 0, 
            'valid_evasion_files_dqeaf': 0,
            'valid_evasion_files_fiam': 0
        } for d in detectors}
        
        # Compare adversarial to original
        for h, orig_data in original_samples.items():
            if h not in adv_samples:
                continue
                
            adv_data = adv_samples[h]
            orig_results = orig_data.get('results', {})
            adv_results = adv_data.get('results', {})
            
            is_valid_dqeaf = False
            is_valid_fiam = False
            if fir_tech_key:
                is_valid_dqeaf = dqeaf_fir_hashes.get(fir_tech_key, {}).get(h, False)
                is_valid_fiam = fiam_fir_hashes.get(fir_tech_key, {}).get(h, False)
            
            for detector in detectors:
                orig_res = orig_results.get(detector, {})
                adv_res = adv_results.get(detector, {})
                
                orig_fl = orig_res.get('final_label')
                adv_fl = adv_res.get('final_label')
                
                tc = orig_data.get('true_label_class')
                tb = orig_data.get('true_label_binary')
                
                orig_bypassed = is_bypassed(orig_fl, tc, tb, strict_bypass=args.strict_bypass)
                adv_bypassed = is_bypassed(adv_fl, tc, tb, strict_bypass=args.strict_bypass)
                
                if orig_bypassed:
                    tech_stats[detector]['success_orig_undet'] += 1
                else:
                    tech_stats[detector]['success_orig_det'] += 1
                    
                if adv_bypassed:
                    tech_stats[detector]['adv_undet'] += 1
                else:
                    tech_stats[detector]['adv_det'] += 1
                
                # Evasion: Originally detected (not bypassed), but now bypassed in adversarial
                if not orig_bypassed and adv_bypassed:
                    tech_stats[detector]['evasion_files'] += 1
                    
                    if is_valid_dqeaf:
                        tech_stats[detector]['valid_evasion_files_dqeaf'] += 1
                    if is_valid_fiam:
                        tech_stats[detector]['valid_evasion_files_fiam'] += 1
                    
                # Spoiled: Originally undetected (bypassed), but now detected in adversarial
                if orig_bypassed and not adv_bypassed:
                    tech_stats[detector]['spoiled_files'] += 1
                    
        # Calculate rates
        for detector in detectors:
            stats = tech_stats[detector]
            orig_det = original_stats[detector]['detected']
            orig_undet = original_stats[detector]['undetected']
            
            evasion_rate = (stats['evasion_files'] / orig_det) if orig_det > 0 else 0
            valid_evasion_rate_dqeaf = (stats['valid_evasion_files_dqeaf'] / orig_det) if orig_det > 0 else 0
            valid_evasion_rate_fiam = (stats['valid_evasion_files_fiam'] / orig_det) if orig_det > 0 else 0
            spoiled_rate = (stats['spoiled_files'] / orig_undet) if orig_undet > 0 else 0
            
            stats['evasion_rate'] = evasion_rate
            stats['valid_evasion_rate_dqeaf'] = valid_evasion_rate_dqeaf
            stats['valid_evasion_rate_fiam'] = valid_evasion_rate_fiam
            stats['spoiled_rate'] = spoiled_rate
            
            # Save to JSON
            json_output['techniques'][tech][detector] = {
                'success_orig_det': stats['success_orig_det'],
                'success_orig_undet': stats['success_orig_undet'],
                'adv_detected': stats['adv_det'],
                'adv_undetected': stats['adv_undet'],
                'evasion_files': stats['evasion_files'],
                'valid_evasion_files_dqeaf': stats['valid_evasion_files_dqeaf'],
                'valid_evasion_files_fiam': stats['valid_evasion_files_fiam'],
                'spoiled_files': stats['spoiled_files'],
                'evasion_rate': evasion_rate,
                'valid_evasion_rate_dqeaf': valid_evasion_rate_dqeaf,
                'valid_evasion_rate_fiam': valid_evasion_rate_fiam,
                'spoiled_rate': spoiled_rate
            }
            
            # Add to CSV
            csv_rows.append({
                'Technique': tech,
                'Detector': detector,
                'Original_Detected': orig_det,
                'Original_Undetected': orig_undet,
                'Success_Gen_Orig_Det': stats['success_orig_det'],
                'Success_Gen_Orig_Undet': stats['success_orig_undet'],
                'Adv_Detected': stats['adv_det'],
                'Adv_Undetected': stats['adv_undet'],
                'Evasion_Files': stats['evasion_files'],
                'Valid evasion files (dqeaf)': stats['valid_evasion_files_dqeaf'],
                'Valid evasion files (fiam)': stats['valid_evasion_files_fiam'],
                'Spoiled_Files': stats['spoiled_files'],
                'Evasion_Rate': f"{evasion_rate:.4f}",
                'Valid Evasion rate (dqeaf)': f"{valid_evasion_rate_dqeaf:.4f}",
                'Valid evasion rate (fiam)': f"{valid_evasion_rate_fiam:.4f}",
                'Spoiled_Rate': f"{spoiled_rate:.4f}"
            })
            
    csv_filename = args.csv_out if args.csv_out else ('filtered_analysis_results_strict.csv' if args.strict_bypass else 'filtered_analysis_results.csv')
    json_filename = args.json_out if args.json_out else ('filtered_analysis_results_strict.json' if args.strict_bypass else 'filtered_analysis_results.json')

    # Write CSV
    with open(csv_filename, 'w', newline='') as f:
        fieldnames = [
            'Technique', 'Detector', 'Original_Detected', 'Original_Undetected', 
            'Success_Gen_Orig_Det', 'Success_Gen_Orig_Undet', 
            'Adv_Detected', 'Adv_Undetected',
            'Evasion_Files', 'Valid evasion files (dqeaf)', 'Valid evasion files (fiam)', 'Spoiled_Files', 
            'Evasion_Rate', 'Valid Evasion rate (dqeaf)', 'Valid evasion rate (fiam)', 'Spoiled_Rate'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    # Write JSON
    with open(json_filename, 'w') as f:
        json.dump(json_output, f, indent=4)
        
    print(f"Exported {csv_filename} and {json_filename}")

if __name__ == '__main__':
    main()
