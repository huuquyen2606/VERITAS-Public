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
        # Only benign or ignored is considered a bypass
        return False
        
    if fl == "malware":
        # Binary classifier correctly identified it as malware
        return False
        
    # Family classifier: correct only if it matches the true family class
    return fl != tc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict_bypass', action='store_true', help="If set, changing families is NOT a bypass.")
    parser.add_argument('--input_file', type=str, default='malware_detector_results.json')
    parser.add_argument('--fir_file', type=str, default='fiam_FIR.json')
    args = parser.parse_args()
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)
        
    with open(args.fir_file, 'r') as f:
        fir_data = json.load(f)
        
    # Map technique names in detector results to the keys used in FIR JSON
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
    fir_hashes = {}
    for fir_tech, samples in fir_data.items():
        fir_hashes[fir_tech] = {}
        for filename, attrs in samples.items():
            h = extract_hash(filename)
            if h:
                fir_hashes[fir_tech][h] = attrs.get('is_functional', False)
        
    test_full = data.get('Test_full', {})
    if not test_full:
        print("Error: 'Test_full' dataset not found in the JSON file.")
        return
        
    # Extract ground truth hashes from Test_full
    original_samples = {} # hash -> data
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

    # Now evaluate each technique
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
                
        # Initialize detector stats for this technique
        detectors = list(original_stats.keys())
        tech_stats = {d: {'evasion_files': 0, 'spoiled_files': 0, 'valid_evasion_files': 0} for d in detectors}
        
        # Compare adversarial to original
        for h, orig_data in original_samples.items():
            if h not in adv_samples:
                # File failed to generate, does not count towards evasion or spoiled.
                continue
                
            adv_data = adv_samples[h]
            orig_results = orig_data.get('results', {})
            adv_results = adv_data.get('results', {})
            
            # Check if it has integrity and functionality
            is_valid_fir = False
            if fir_tech_key and fir_tech_key in fir_hashes:
                is_valid_fir = fir_hashes[fir_tech_key].get(h, False)
            
            for detector in detectors:
                orig_res = orig_results.get(detector, {})
                adv_res = adv_results.get(detector, {})
                
                orig_fl = orig_res.get('final_label')
                adv_fl = adv_res.get('final_label')
                
                tc = orig_data.get('true_label_class')
                tb = orig_data.get('true_label_binary')
                
                orig_bypassed = is_bypassed(orig_fl, tc, tb, strict_bypass=args.strict_bypass)
                adv_bypassed = is_bypassed(adv_fl, tc, tb, strict_bypass=args.strict_bypass)
                
                # Evasion: Originally detected (not bypassed), but now bypassed in adversarial
                if not orig_bypassed and adv_bypassed:
                    tech_stats[detector]['evasion_files'] += 1
                    
                    if is_valid_fir:
                        tech_stats[detector]['valid_evasion_files'] += 1
                    
                # Spoiled: Originally undetected (bypassed), but now detected in adversarial
                if orig_bypassed and not adv_bypassed:
                    tech_stats[detector]['spoiled_files'] += 1
                    
        # Calculate rates
        for detector in detectors:
            evasion_files = tech_stats[detector]['evasion_files']
            valid_evasion_files = tech_stats[detector]['valid_evasion_files']
            spoiled_files = tech_stats[detector]['spoiled_files']
            
            orig_det = original_stats[detector]['detected']
            orig_undet = original_stats[detector]['undetected']
            
            evasion_rate = (evasion_files / orig_det) if orig_det > 0 else 0
            valid_evasion_rate = (valid_evasion_files / orig_det) if orig_det > 0 else 0
            spoiled_rate = (spoiled_files / orig_undet) if orig_undet > 0 else 0
            
            tech_stats[detector]['evasion_rate'] = evasion_rate
            tech_stats[detector]['valid_evasion_rate'] = valid_evasion_rate
            tech_stats[detector]['spoiled_rate'] = spoiled_rate
            
            # Save to JSON
            json_output['techniques'][tech][detector] = {
                'evasion_files': evasion_files,
                'valid_evasion_files': valid_evasion_files,
                'spoiled_files': spoiled_files,
                'evasion_rate': evasion_rate,
                'valid_evasion_rate': valid_evasion_rate,
                'spoiled_rate': spoiled_rate
            }
            
            # Add to CSV
            csv_rows.append({
                'Technique': tech,
                'Detector': detector,
                'Original_Detected': orig_det,
                'Original_Undetected': orig_undet,
                'Evasion_Files': evasion_files,
                'Valid_Evasion_Files': valid_evasion_files,
                'Spoiled_Files': spoiled_files,
                'Evasion_Rate': f"{evasion_rate:.4f}",
                'Valid_Evasion_Rate': f"{valid_evasion_rate:.4f}",
                'Spoiled_Rate': f"{spoiled_rate:.4f}"
            })
            
    # Determine output filenames based on strict_bypass flag
    csv_filename = 'analysis_results_strict.csv' if args.strict_bypass else 'analysis_results.csv'
    json_filename = 'analysis_results_strict.json' if args.strict_bypass else 'analysis_results.json'

    # Write CSV
    with open(csv_filename, 'w', newline='') as f:
        fieldnames = [
            'Technique', 'Detector', 'Original_Detected', 'Original_Undetected', 
            'Evasion_Files', 'Valid_Evasion_Files', 'Spoiled_Files', 
            'Evasion_Rate', 'Valid_Evasion_Rate', 'Spoiled_Rate'
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
