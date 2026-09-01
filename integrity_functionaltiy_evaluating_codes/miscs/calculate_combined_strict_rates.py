import json
import re
import csv
import argparse
import os

def extract_hash(filename):
    match = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if match:
        return match.group(1).lower()
    return None

def is_bypassed(final_label, true_class, true_binary, strict_bypass=True):
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
    parser.add_argument('--dl_file', type=str, default='malware_detector_results_filtered.json')
    parser.add_argument('--heuristic_file', type=str, default='heuristic_rulebase_results_filtered.json')
    parser.add_argument('--dqeaf_fir_file', type=str, default='dqeaf_FIR.json')
    parser.add_argument('--fiam_fir_file', type=str, default='fiam_FIR.json')
    parser.add_argument('--csv_out', type=str, default='final_result/fixed_csvs/combined_strict_rates.csv')
    args = parser.parse_args()
    
    # Load FIR data
    with open(args.dqeaf_fir_file, 'r') as f:
        dqeaf_fir_data = json.load(f)
        
    with open(args.fiam_fir_file, 'r') as f:
        fiam_fir_data = json.load(f)
        
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

    csv_rows = []

    def process_file(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        test_full = data.get('Test_full', {})
        if not test_full:
            return
            
        original_samples = {}
        total_original_malware = 0
        for filename, sample_data in test_full.items():
            h = extract_hash(filename)
            if h:
                original_samples[h] = sample_data
                if sample_data.get('true_label_binary') == 'Malware':
                    total_original_malware += 1
                    
        detectors = set()
        for sample_data in original_samples.values():
            for d in sample_data.get('results', {}).keys():
                detectors.add(d)
        detectors = sorted(list(detectors))

        techniques = [k for k in data.keys() if k != 'Test_full']
        
        for tech in techniques:
            tech_data = data[tech]
            fir_tech_key = tech_map.get(tech)
            
            adv_samples = {}
            for filename, sample_data in tech_data.items():
                h = extract_hash(filename)
                if h:
                    adv_samples[h] = sample_data
                    
            for detector in detectors:
                orig_det = 0
                orig_undet = 0
                
                success_gen_orig_det = 0
                success_gen_orig_undet = 0
                
                orig_det_adv_detected = 0
                orig_det_adv_undetected = 0
                orig_det_dqeaf_evasion = 0
                orig_det_fiam_evasion = 0
                
                orig_undet_adv_detected = 0
                orig_undet_adv_undetected = 0
                orig_undet_dqeaf_evasion = 0
                orig_undet_fiam_evasion = 0

                # Determine Original Stats
                for h, orig_data in original_samples.items():
                    if orig_data.get('true_label_binary') != 'Malware':
                        continue
                        
                    tc = orig_data.get('true_label_class')
                    tb = orig_data.get('true_label_binary')
                    orig_res = orig_data.get('results', {}).get(detector, {})
                    orig_fl = orig_res.get('final_label')
                    
                    orig_bypassed = is_bypassed(orig_fl, tc, tb, strict_bypass=True)
                    if orig_bypassed:
                        orig_undet += 1
                        was_detected = False
                    else:
                        orig_det += 1
                        was_detected = True
                        
                    # Check adversarial
                    if h in adv_samples:
                        adv_data = adv_samples[h]
                        adv_res = adv_data.get('results', {}).get(detector, {})
                        adv_fl = adv_res.get('final_label')
                        adv_bypassed = is_bypassed(adv_fl, tc, tb, strict_bypass=True)
                        
                        is_valid_dqeaf = False
                        is_valid_fiam = False
                        if fir_tech_key:
                            is_valid_dqeaf = dqeaf_fir_hashes.get(fir_tech_key, {}).get(h, False)
                            is_valid_fiam = fiam_fir_hashes.get(fir_tech_key, {}).get(h, False)
                            
                        if was_detected:
                            success_gen_orig_det += 1
                            if adv_bypassed:
                                orig_det_adv_undetected += 1
                                if is_valid_dqeaf: orig_det_dqeaf_evasion += 1
                                if is_valid_fiam: orig_det_fiam_evasion += 1
                            else:
                                orig_det_adv_detected += 1
                        else:
                            success_gen_orig_undet += 1
                            if adv_bypassed:
                                orig_undet_adv_undetected += 1
                                if is_valid_dqeaf: orig_undet_dqeaf_evasion += 1
                                if is_valid_fiam: orig_undet_fiam_evasion += 1
                            else:
                                orig_undet_adv_detected += 1
                
                # Calculate Rates
                # Group: Orig_Det
                orig_det_evasion_rate = orig_det_adv_undetected / orig_det if orig_det > 0 else 0
                orig_det_dqeaf_rate = orig_det_dqeaf_evasion / orig_det if orig_det > 0 else 0
                orig_det_fiam_rate = orig_det_fiam_evasion / orig_det if orig_det > 0 else 0
                
                # Group: Orig_UnDet
                orig_undet_evasion_rate = orig_undet_adv_undetected / orig_undet if orig_undet > 0 else 0
                orig_undet_dqeaf_rate = orig_undet_dqeaf_evasion / orig_undet if orig_undet > 0 else 0
                orig_undet_fiam_rate = orig_undet_fiam_evasion / orig_undet if orig_undet > 0 else 0

                csv_rows.append([
                    tech, detector, total_original_malware, orig_det, orig_undet, success_gen_orig_det, success_gen_orig_undet,
                    # Orig_Det block
                    orig_det_adv_detected, orig_det_adv_undetected, orig_det_dqeaf_evasion, orig_det_fiam_evasion,
                    f"{orig_det_evasion_rate * 100:.2f}%", f"{orig_det_dqeaf_rate * 100:.2f}%", f"{orig_det_fiam_rate * 100:.2f}%",
                    # Orig_UnDet block
                    orig_undet_adv_detected, orig_undet_adv_undetected, orig_undet_dqeaf_evasion, orig_undet_fiam_evasion,
                    f"{orig_undet_evasion_rate * 100:.2f}%", f"{orig_undet_dqeaf_rate * 100:.2f}%", f"{orig_undet_fiam_rate * 100:.2f}%"
                ])

    process_file(args.dl_file)
    process_file(args.heuristic_file)

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    
    with open(args.csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write Multi-Index Headers
        header_row_1 = [
            'General Info', '', '', '', '', '', '',
            'Original_Detected Group', '', '', '', '', '', '',
            'Original_Undetected Group', '', '', '', '', '', ''
        ]
        
        header_row_2 = [
            'Technique', 'Detector', 'Test', 'Original_Detected', 'Original_Undetected', 'Success_Gen_Orig_Det', 'Success_Gen_Orig_Undet',
            'Adv_Detected', 'Adv_Undetected', 'Valid evasion files (dqeaf)', 'Valid evasion files (fiam)', 'Evasion_Rate', 'Valid Evasion rate (dqeaf)', 'Valid evasion rate (fiam)',
            'Adv_Detected', 'Adv_Undetected', 'Valid evasion files (dqeaf)', 'Valid evasion files (fiam)', 'Evasion_Rate', 'Valid Evasion rate (dqeaf)', 'Valid evasion rate (fiam)'
        ]
        
        writer.writerow(header_row_1)
        writer.writerow(header_row_2)
        writer.writerows(csv_rows)

    print(f"Generated combined strict table at: {args.csv_out}")

if __name__ == '__main__':
    main()
