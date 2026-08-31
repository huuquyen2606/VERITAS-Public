import json
import glob
import os
import re

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
    with open('heuristic_rulebase_results_filtered.json', 'r') as f:
        data = json.load(f)

    test_full = data.get('Test_full', {})
    
    # Assemblyline, clamav, defender
    detectors = ['assemblyline', 'clamav', 'defender']
    metrics = {d: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for d in detectors}
    
    for h, attrs in test_full.items():
        tc = attrs.get('true_label_class')
        tb = attrs.get('true_label_binary')
        
        for d in detectors:
            res = attrs.get('results', {}).get(d, {})
            fl = res.get('final_label')
            
            bypassed = is_bypassed(fl, tc, tb, strict_bypass=True)
            
            if tb == 'Malware':
                if bypassed:
                    metrics[d]['FN'] += 1 # Malware that evaded -> FN
                else:
                    metrics[d]['TP'] += 1 # Malware correctly detected -> TP
            elif tb == 'Benign':
                if bypassed:
                    metrics[d]['TN'] += 1 # Benign correctly "evading" detection -> TN
                else:
                    metrics[d]['FP'] += 1 # Benign falsely detected -> FP

    print("--- Heuristic ML Detectors ---")
    for d in detectors:
        tp = metrics[d]['TP']
        tn = metrics[d]['TN']
        fp = metrics[d]['FP']
        fn = metrics[d]['FN']
        
        # Class Malware (c=1)
        p_malware = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_malware = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Class Benign (c=2)
        p_benign = tn / (tn + fn) if (tn + fn) > 0 else 0
        r_benign = tn / (tn + fn) if (tn + fn) > 0 else 0
        # Wait, Precision for benign is tn / (tn + fn) ?
        # Precision = True Positives / (True Positives + False Positives)
        # For Benign class: "Positive" means predicting Benign.
        # True Benign Predictions = TN
        # False Benign Predictions = FN (Malware predicted as Benign)
        # So P_benign = tn / (tn + fn)
        # Recall = True Positives / (True Positives + False Negatives)
        # For Benign class: "True Positives" = TN, "False Negatives" = FP (Benign predicted as Malware)
        # So R_benign = tn / (tn + fp)
        
        p_benign = tn / (tn + fn) if (tn + fn) > 0 else 0
        r_benign = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        macro_p = (p_malware + p_benign) / 2
        macro_r = (r_malware + r_benign) / 2
        
        f1_malware = 2 * p_malware * r_malware / (p_malware + r_malware) if (p_malware + r_malware) > 0 else 0
        f1_benign = 2 * p_benign * r_benign / (p_benign + r_benign) if (p_benign + r_benign) > 0 else 0
        
        macro_f1 = (f1_malware + f1_benign) / 2
        
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        evasion_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"{d}: Acc={acc*100:.2f} | P={macro_p*100:.2f} | R={macro_r*100:.2f} | F1={macro_f1*100:.2f} | ER={evasion_rate*100:.2f}")

    # VirusTotal
    print("\n--- VirusTotal TERs ---")
    vt_test_dir = 'virus_total_result/VirusTotal-CLI-Tool/results/test'
    test_jsons = glob.glob(os.path.join(vt_test_dir, '**', '*.json'), recursive=True)
    
    vt_original_malware_count = 0
    vt_evasion_counts = {ter: 0 for ter in range(10, 101, 10)}
    
    for f in test_jsons:
        if os.path.basename(f).startswith('failed_'):
            continue
        try:
            with open(f, 'r') as fp:
                vt_data = json.load(fp)
            stats = vt_data.get('data', {}).get('attributes', {}).get('stats', {})
            if not stats:
                stats = vt_data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
                
            malicious = stats.get('malicious', 0)
            total_scanned = sum(stats.values())
            
            h = extract_hash(os.path.basename(f))
            if h and total_scanned > 0:
                det_ratio = (malicious / total_scanned) * 100
                vt_original_malware_count += 1
                
                for ter in range(10, 101, 10):
                    if det_ratio <= ter:
                        vt_evasion_counts[ter] += 1
        except Exception:
            pass

    for ter in range(10, 101, 10):
        er = vt_evasion_counts[ter] / vt_original_malware_count if vt_original_malware_count > 0 else 0
        print(f"virustotal_ter_{ter}: ER={er*100:.2f}%")

if __name__ == '__main__':
    main()
