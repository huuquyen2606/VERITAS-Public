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
    # 1. Parse standard heuristics
    with open('heuristic_rulebase_results_filtered.json', 'r') as f:
        data = json.load(f)

    test_full = data.get('Test_full', {})
    
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
                    metrics[d]['FN'] += 1
                else:
                    metrics[d]['TP'] += 1
            elif tb == 'Benign':
                if bypassed:
                    metrics[d]['TN'] += 1
                else:
                    metrics[d]['FP'] += 1

    # 2. Parse VirusTotal TERs
    for ter in range(10, 101, 10):
        detector_name = f'virustotal_ter_{ter}'
        detectors.append(detector_name)
        metrics[detector_name] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    vt_test_dir = 'virus_total_result/VirusTotal-CLI-Tool/results/test'
    test_jsons = glob.glob(os.path.join(vt_test_dir, '**', '*.json'), recursive=True)
    
    for f in test_jsons:
        if os.path.basename(f).startswith('failed_'):
            continue
            
        # Is it benign or malware?
        # Check if the path contains 'test/Benign/'
        is_benign = 'Benign' in f.split(os.sep)
        
        try:
            with open(f, 'r') as fp:
                vt_data = json.load(fp)
            stats = vt_data.get('data', {}).get('attributes', {}).get('stats', {})
            if not stats:
                stats = vt_data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
                
            malicious = stats.get('malicious', 0)
            total_scanned = sum(stats.values())
            
            h = extract_hash(os.path.basename(f))
            if total_scanned > 0:
                det_ratio = (malicious / total_scanned) * 100
                
                for ter in range(10, 101, 10):
                    bypassed = (det_ratio <= ter)
                    d_name = f'virustotal_ter_{ter}'
                    
                    if not is_benign: # Malware
                        if bypassed:
                            metrics[d_name]['FN'] += 1
                        else:
                            metrics[d_name]['TP'] += 1
                    else: # Benign
                        if bypassed:
                            metrics[d_name]['TN'] += 1
                        else:
                            metrics[d_name]['FP'] += 1
        except Exception:
            pass

    # 3. Calculate final percentages
    print(metrics["virustotal_ter_10"]); results = {}
    for d in detectors:
        tp = metrics[d]['TP']
        tn = metrics[d]['TN']
        fp = metrics[d]['FP']
        fn = metrics[d]['FN']
        
        p_malware = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_malware = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        p_benign = tn / (tn + fn) if (tn + fn) > 0 else 0
        r_benign = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        macro_p = (p_malware + p_benign) / 2
        macro_r = (r_malware + r_benign) / 2
        
        f1_malware = 2 * p_malware * r_malware / (p_malware + r_malware) if (p_malware + r_malware) > 0 else 0
        f1_benign = 2 * p_benign * r_benign / (p_benign + r_benign) if (p_benign + r_benign) > 0 else 0
        
        macro_f1 = (f1_malware + f1_benign) / 2
        
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        evasion_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        results[d] = {
            'Accuracy': acc * 100,
            'Macro-Precision': macro_p * 100,
            'Macro-Recall': macro_r * 100,
            'Macro-F1': macro_f1 * 100,
            'Evasion Rate': evasion_rate * 100
        }

    # 4. Rank metrics for top 3 styling
    def get_ranks(metric_name, reverse=True):
        vals = [(d, results[d][metric_name]) for d in detectors]
        vals.sort(key=lambda x: x[1], reverse=reverse)
        
        # assign ranks dealing with ties
        ranks = {}
        current_rank = 1
        current_val = -1
        for i, (d, val) in enumerate(vals):
            if i == 0:
                ranks[d] = 1
                current_val = val
            else:
                if val == current_val:
                    ranks[d] = current_rank
                else:
                    current_rank += 1
                    current_val = val
                    ranks[d] = current_rank
        return ranks

    rank_acc = get_ranks('Accuracy', reverse=True)
    rank_p = get_ranks('Macro-Precision', reverse=True)
    rank_r = get_ranks('Macro-Recall', reverse=True)
    rank_f1 = get_ranks('Macro-F1', reverse=True)
    rank_er = get_ranks('Evasion Rate', reverse=False)

    def format_val(d, metric_name, ranks_dict):
        val = results[d][metric_name]
        val_str = f"{val:.2f}"
        rank = ranks_dict[d]
        if rank == 1:
            return "\\topI{" + val_str + "}"
        elif rank == 2:
            return "\\topII{" + val_str + "}"
        elif rank == 3:
            return "\\topIII{" + val_str + "}"
        else:
            return val_str

    # 5. Output Latex rows
    label_map = {
        'assemblyline': '\\textbf{Assemblyline} \\cite{assemblyline4}',
        'clamav': '\\textbf{ClamAV} \\cite{clamav}',
        'defender': '\\textbf{Windows Defender} \\cite{microsoftdefender}'
    }
    for ter in range(10, 101, 10):
        label_map[f'virustotal_ter_{ter}'] = f'\\textbf{{VirusTotal TER $\\le$ {ter}\\%}} \\cite{{virustotal}}'
        
    print("        \\multirow{13}{0.12\\textwidth}{\\textbf{Heuristics}} ")
    for d in detectors:
        name = label_map[d]
        v_acc = format_val(d, 'Accuracy', rank_acc)
        v_p = format_val(d, 'Macro-Precision', rank_p)
        v_r = format_val(d, 'Macro-Recall', rank_r)
        v_f1 = format_val(d, 'Macro-F1', rank_f1)
        v_er = format_val(d, 'Evasion Rate', rank_er)
        
        print(f" & {name} & {v_acc} & {v_p} & {v_r} & {v_f1} & {v_er} \\\\ ")

if __name__ == '__main__':
    main()
