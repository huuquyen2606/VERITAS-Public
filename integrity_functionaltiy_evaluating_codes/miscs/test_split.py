import glob, os, json
metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
for fn in glob.glob('virus_total_result/VirusTotal-CLI-Tool/results/test/**/*.json', recursive=True):
    is_b = 'Benign' in fn.split(os.sep)
    if 'failed_' in os.path.basename(fn): continue
    with open(fn, 'r') as fp:
        vt_data = json.load(fp)
    stats = vt_data.get('data', {}).get('attributes', {}).get('stats', {})
    if not stats: stats = vt_data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
    malicious = stats.get('malicious', 0)
    total_scanned = sum(stats.values())
    if total_scanned > 0:
        det_ratio = (malicious / total_scanned) * 100
        bypassed = det_ratio <= 10
        if not is_b: # Malware
            if bypassed: metrics['FN'] += 1
            else: metrics['TP'] += 1
        else: # Benign
            if bypassed: metrics['TN'] += 1
            else: metrics['FP'] += 1

print(metrics)
