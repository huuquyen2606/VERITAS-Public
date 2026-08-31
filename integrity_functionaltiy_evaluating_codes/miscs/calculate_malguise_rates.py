import json
import os
import csv
import glob
import re

def extract_hash(filename):
    match = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if match:
        return match.group(1).lower()
    return None

def is_bypassed(final_label, true_class, true_binary, strict_bypass=True):
    if not final_label:
        return True

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
    # 1. Load data
    with open('malware_detector_results_filtered.json', 'r') as f:
        ml_data = json.load(f)
    original_data = ml_data.get('Test_full', {})
    
    # Extract live hashes
    with open('malguise/dqeaf_integrity_check.json', 'r') as f:
        integ_data = json.load(f).get('malguise_full.npz', {})
    live_hashes = set(extract_hash(k) for k, v in integ_data.items() if extract_hash(k) and v.get('is_alive') == True)
    
    with open('malguise/per_file_evasion_results.json', 'r') as f:
        per_file = json.load(f)
    raw_malguise_data = per_file.get('malguise_full', {})
    
    malguise_data = {}
    for k, v in raw_malguise_data.items():
        h = extract_hash(k)
        if h and h in live_hashes:
            malguise_data[k] = v
    
    with open('dqeaf_FIR.json', 'r') as f:
        dqeaf_fir = json.load(f)
    with open('fiam_FIR.json', 'r') as f:
        fiam_fir = json.load(f)
        
    dqeaf_hashes = {}
    for filename, attrs in dqeaf_fir.get('malguise', {}).items():
        h = extract_hash(filename)
        if h: dqeaf_hashes[h] = (attrs.get('combined_status') == 'BOTH')
        
    fiam_hashes = {}
    for filename, attrs in fiam_fir.get('malguise', {}).items():
        h = extract_hash(filename)
        if h: fiam_hashes[h] = (attrs.get('is_functional') == True and attrs.get('is_alive') == True)
        
    with open('malguise/clamav_heuristic.json', 'r') as f:
        clamav_data = json.load(f)
    with open('malguise/defender_heuristic.json', 'r') as f:
        defender_data = json.load(f)
        
    clamav_lookup = {}
    for filepath, info in clamav_data.get('datasets', {}).get('malguise', {}).get('files', {}).items():
        h = extract_hash(os.path.basename(filepath.replace('\\', '/')))
        if h: clamav_lookup[h] = "Malware" if info.get('status') == "detected" else "Benign"
        
    defender_lookup = {}
    for filepath, info in defender_data.get('datasets', {}).get('malguise', {}).get('files', {}).items():
        h = extract_hash(os.path.basename(filepath.replace('\\', '/')))
        if h: defender_lookup[h] = "Malware" if info.get('status') == "detected" else "Benign"

    # For original heuristics, we load from the main filtered heuristic file
    with open('heuristic_rulebase_results_filtered.json', 'r') as f:
        heur_data = json.load(f)
    heur_original = heur_data.get('Test_full', {})

    # Load VT Original data
    vt_original_samples = {}
    vt_test_dir = 'virus_total_result/VirusTotal-CLI-Tool/results/test'
    test_jsons = glob.glob(os.path.join(vt_test_dir, '**', '*.json'), recursive=True)
    for f in test_jsons:
        if os.path.basename(f).startswith('failed_'): continue
        if 'Benign' in f.split(os.sep): continue # Only care about malware baseline for evasion
        try:
            with open(f, 'r') as fp: vt_data = json.load(fp)
            stats = vt_data.get('data', {}).get('attributes', {}).get('stats') or vt_data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
            malicious = stats.get('malicious', 0)
            total = sum(stats.values())
            h = extract_hash(os.path.basename(f))
            if h and total > 0:
                vt_original_samples[h] = {'det_ratio': (malicious / total) * 100}
        except: pass

    # Load VT Malguise data
    vt_malguise_samples = {}
    vt_malguise_jsons = glob.glob('malguise/malguise_vt_reports/**/*.json', recursive=True)
    for f in vt_malguise_jsons:
        if os.path.basename(f).startswith('failed_'): continue
        try:
            with open(f, 'r') as fp: vt_data = json.load(fp)
            stats = vt_data.get('data', {}).get('attributes', {}).get('stats') or vt_data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
            malicious = stats.get('malicious', 0)
            total = sum(stats.values())
            h = extract_hash(os.path.basename(f))
            if h and total > 0:
                vt_malguise_samples[h] = {'det_ratio': (malicious / total) * 100}
        except: pass

    ml_detectors = [
        "binary_malconv", "imcfn", "m_attn_health", "m_attn_health_multi", 
        "malconv", "multiview_cnn", "rtf_bert", "rtf_cannie", "seqconvattn"
    ]
    ml_names = {
        "binary_malconv": "binary_malconv",
        "imcfn": "imcfn",
        "m_attn_health": "m_attn_health",
        "m_attn_health_multi": "m_attn_health_multi",
        "malconv": "malconv",
        "multiview_cnn": "multiview_cnn",
        "rtf_bert": "rtf_bert",
        "rtf_cannie": "rtf_cannie",
        "seqconvattn": "seqconvattn"
    }
    
    total_original_malware = sum(1 for v in original_data.values() if v.get('true_label_binary') == 'Malware')
    rows = []

    def compute_row(detector_id, detector_name, detector_type):
        orig_det, orig_undet = 0, 0
        success_gen_orig_det, success_gen_orig_undet = 0, 0
        orig_det_adv_det, orig_det_adv_undet, orig_det_dqeaf, orig_det_fiam = 0, 0, 0, 0
        orig_undet_adv_det, orig_undet_adv_undet, orig_undet_dqeaf, orig_undet_fiam = 0, 0, 0, 0
        
        for orig_file, orig_info in original_data.items():
            if orig_info.get('true_label_binary') != 'Malware': continue
            h = extract_hash(orig_file)
            if not h: continue
            
            tc = orig_info.get('true_label_class')
            tb = orig_info.get('true_label_binary')
            
            orig_bypassed = False
            
            if detector_type == 'ml':
                res = orig_info.get('results', {}).get(detector_id, {})
                fl = res.get('final_label')
                orig_bypassed = is_bypassed(fl, tc, tb, True)
            elif detector_type == 'heur':
                h_info = heur_original.get(orig_file, {})
                res = h_info.get('results', {}).get(detector_id, {})
                fl = res.get('final_label')
                orig_bypassed = is_bypassed(fl, tc, tb, True)
            elif detector_type == 'vt':
                if h not in vt_original_samples: continue
                ter = int(detector_id.split('_')[-1])
                orig_bypassed = (vt_original_samples[h]['det_ratio'] <= ter)
                
            was_detected = not orig_bypassed
            if was_detected: orig_det += 1
            else: orig_undet += 1
            
            # Adv check
            is_valid_dqeaf = dqeaf_hashes.get(h, False)
            is_valid_fiam = fiam_hashes.get(h, False)
            
            adv_bypassed = None
            if detector_type == 'ml':
                adv_info = next((v for k,v in malguise_data.items() if extract_hash(k) == h), None)
                if adv_info:
                    res = adv_info.get('results', {}).get(detector_id, {})
                    fl = res.get('final_label')
                    adv_bypassed = is_bypassed(fl, tc, tb, True)
            elif detector_type == 'heur':
                if h in live_hashes:
                    if detector_id == 'clamav': fl = clamav_lookup.get(h)
                    else: fl = defender_lookup.get(h)
                    if fl: adv_bypassed = is_bypassed(fl, tc, tb, True)
            elif detector_type == 'vt':
                if h in live_hashes and h in vt_malguise_samples:
                    ter = int(detector_id.split('_')[-1])
                    adv_bypassed = (vt_malguise_samples[h]['det_ratio'] <= ter)
                    
            if adv_bypassed is not None:
                if was_detected:
                    success_gen_orig_det += 1
                    if adv_bypassed:
                        orig_det_adv_undet += 1
                        if is_valid_dqeaf: orig_det_dqeaf += 1
                        if is_valid_fiam: orig_det_fiam += 1
                    else:
                        orig_det_adv_det += 1
                else:
                    success_gen_orig_undet += 1
                    if adv_bypassed:
                        orig_undet_adv_undet += 1
                        if is_valid_dqeaf: orig_undet_dqeaf += 1
                        if is_valid_fiam: orig_undet_fiam += 1
                    else:
                        orig_undet_adv_det += 1
                        
        er_orig_det = orig_det_adv_undet / orig_det if orig_det > 0 else 0
        dqeaf_orig_det = orig_det_dqeaf / orig_det if orig_det > 0 else 0
        fiam_orig_det = orig_det_fiam / orig_det if orig_det > 0 else 0
        
        er_orig_undet = orig_undet_adv_undet / orig_undet if orig_undet > 0 else 0
        dqeaf_orig_undet = orig_undet_dqeaf / orig_undet if orig_undet > 0 else 0
        fiam_orig_undet = orig_undet_fiam / orig_undet if orig_undet > 0 else 0
        
        return [
            'malguise_full_lived', detector_name, total_original_malware, orig_det, orig_undet, success_gen_orig_det, success_gen_orig_undet,
            orig_det_adv_det, orig_det_adv_undet, orig_det_dqeaf, orig_det_fiam,
            f"{er_orig_det * 100:.2f}%", f"{dqeaf_orig_det * 100:.2f}%", f"{fiam_orig_det * 100:.2f}%",
            orig_undet_adv_det, orig_undet_adv_undet, orig_undet_dqeaf, orig_undet_fiam,
            f"{er_orig_undet * 100:.2f}%", f"{dqeaf_orig_undet * 100:.2f}%", f"{fiam_orig_undet * 100:.2f}%"
        ]

    for det_id in ml_detectors:
        rows.append(compute_row(det_id, ml_names[det_id], 'ml'))
        
    rows.append(compute_row('clamav', 'clamav', 'heur'))
    rows.append(compute_row('defender', 'defender', 'heur'))
    
    for ter in range(10, 101, 10):
        rows.append(compute_row(f'virustotal_ter_{ter}', f'virustotal_ter_{ter}', 'vt'))
        
    # Append to CSV
    csv_file = 'final_result/fixed_csvs/combined_strict_rates_with_vt.csv'
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        
    print(f"Appended {len(rows)} malguise rows to {csv_file}")

if __name__ == '__main__':
    main()
