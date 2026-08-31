import json
import re
import csv
import argparse
import os
import glob
from pathlib import Path

def extract_hash(filename):
    match = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if match:
        return match.group(1).lower()
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_csv', type=str, default='final_result/fixed_csvs/combined_strict_rates.csv')
    parser.add_argument('--dqeaf_fir_file', type=str, default='dqeaf_FIR.json')
    parser.add_argument('--fiam_fir_file', type=str, default='fiam_FIR.json')
    parser.add_argument('--vt_dir', type=str, default='virus_total_result/VirusTotal-CLI-Tool/results')
    parser.add_argument('--csv_out', type=str, default='final_result/fixed_csvs/combined_strict_rates_with_vt.csv')
    args = parser.parse_args()

    # 1. Load existing CSV
    existing_rows = []
    header_row_1 = []
    header_row_2 = []
    if os.path.exists(args.base_csv):
        with open(args.base_csv, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) >= 2:
                header_row_1 = rows[0]
                header_row_2 = rows[1]
                existing_rows = rows[2:]

    # 2. Load FIR Data
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

    vt_map = {
        'AEs_AIMEDRL': 'aimed_full',
        'AEs_DQEAF': 'dqef_full_lived',
        'AEs_FIAM': 'fiam_1500_filtered',
        'AEs_GAMMA': 'gamma_adv_full',
        'AEs_GAPGAN': 'gapgan_full',
        'AEs_MAB-malware': 'MAB_full_lived',
        'AEs_MalGPT': 'Malgpt_full',
        'AEs_OBFU-mal': 'OBFU-mal_full'
    }

    tech_to_fir = {
        'MAB_full_lived': 'mab',
        'Malgpt_full': 'malgpt',
        'OBFU-mal_full': 'obfu',
        'aimed_full': 'aimed',
        'dqef_full_lived': 'dqef',
        'fiam_1500_filtered': 'fiam',
        'gamma_adv_full': 'gamma',
        'gapgan_full': 'gapgan'
    }

    # 3. Load Original VT Data
    original_samples = {}
    test_dir = os.path.join(args.vt_dir, 'test')
    total_original_malware = 0
    
    test_jsons = glob.glob(os.path.join(test_dir, '**', '*.json'), recursive=True)
    for f in test_jsons:
        if os.path.basename(f).startswith('failed_'):
            continue
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
            stats = data.get('data', {}).get('attributes', {}).get('stats', {})
            # fallback to last_analysis_stats if stats is not available
            if not stats:
                stats = data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
            
            malicious = stats.get('malicious', 0)
            total_scanned = sum(stats.values())
            
            h = extract_hash(os.path.basename(f))
            if h and total_scanned > 0:
                original_samples[h] = {
                    'malicious': malicious,
                    'total': total_scanned,
                    'det_ratio': (malicious / total_scanned) * 100
                }
                total_original_malware += 1
        except Exception:
            pass

    print(f"Loaded {total_original_malware} original VT samples.")

    # 4. Load Adversarial VT Data
    adv_datasets = {}
    for vt_folder, standard_tech in vt_map.items():
        folder_path = os.path.join(args.vt_dir, vt_folder)
        adv_datasets[standard_tech] = {}
        
        if not os.path.exists(folder_path):
            continue
            
        adv_jsons = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
        for f in adv_jsons:
            if os.path.basename(f).startswith('failed_'):
                continue
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                stats = data.get('data', {}).get('attributes', {}).get('stats', {})
                if not stats:
                    stats = data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
                    
                malicious = stats.get('malicious', 0)
                total_scanned = sum(stats.values())
                
                h = extract_hash(os.path.basename(f))
                if h and total_scanned > 0:
                    adv_datasets[standard_tech][h] = {
                        'malicious': malicious,
                        'total': total_scanned,
                        'det_ratio': (malicious / total_scanned) * 100
                    }
            except Exception:
                pass

    # 5. Compute Metrics for each TER
    vt_rows = []
    ters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for standard_tech in vt_map.values():
        adv_samples = adv_datasets.get(standard_tech, {})
        fir_tech_key = tech_to_fir.get(standard_tech)
        
        for ter in ters:
            detector_name = f"virustotal_ter_{ter}"
            
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

            for h, orig_data in original_samples.items():
                # Baseline
                orig_bypassed = (orig_data['det_ratio'] <= ter)
                
                if orig_bypassed:
                    orig_undet += 1
                    was_detected = False
                else:
                    orig_det += 1
                    was_detected = True
                    
                # Adversarial
                if h in adv_samples:
                    adv_data = adv_samples[h]
                    adv_bypassed = (adv_data['det_ratio'] <= ter)
                    
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

            vt_rows.append([
                standard_tech, detector_name, total_original_malware, orig_det, orig_undet, success_gen_orig_det, success_gen_orig_undet,
                # Orig_Det block
                orig_det_adv_detected, orig_det_adv_undetected, orig_det_dqeaf_evasion, orig_det_fiam_evasion,
                f"{orig_det_evasion_rate * 100:.2f}%", f"{orig_det_dqeaf_rate * 100:.2f}%", f"{orig_det_fiam_rate * 100:.2f}%",
                # Orig_UnDet block
                orig_undet_adv_detected, orig_undet_adv_undetected, orig_undet_dqeaf_evasion, orig_undet_fiam_evasion,
                f"{orig_undet_evasion_rate * 100:.2f}%", f"{orig_undet_dqeaf_rate * 100:.2f}%", f"{orig_undet_fiam_rate * 100:.2f}%"
            ])
            
    # Append and Save
    all_rows = existing_rows + vt_rows
    with open(args.csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_row_1)
        writer.writerow(header_row_2)
        writer.writerows(all_rows)
        
    print(f"Generated combined strict table WITH VirusTotal TER at: {args.csv_out}")

if __name__ == '__main__':
    main()
