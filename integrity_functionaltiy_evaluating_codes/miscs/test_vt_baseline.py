import json
import os
import glob
from sklearn.metrics import accuracy_score, f1_score
import re

def extract_hash(filename):
    match = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if match: return match.group(1).lower()
    return None

def get_vt_ter(filepath):
    try:
        with open(filepath, 'r') as f:
            vt_data = json.load(f)
        stats = vt_data.get('data', {}).get('attributes', {}).get('stats') or vt_data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
        malicious = stats.get('malicious', 0)
        total = sum(stats.values())
        if total > 0:
            return (malicious / total) * 100
        return None
    except:
        return None

with open('malware_detector_results_filtered.json', 'r') as f:
    orig_data = json.load(f).get('Test_full', {})

valid_hashes = set()
for orig_file, orig_info in orig_data.items():
    if orig_info.get('true_label_binary') == 'Malware':
        h = extract_hash(orig_file)
        if h: valid_hashes.add(h)

# Load malware (Test)
malware_jsons = glob.glob('virus_total_result/VirusTotal-CLI-Tool/results/test/**/*.json', recursive=True)
malware_ters = []
for f in malware_jsons:
    if os.path.basename(f).startswith('failed_'): continue
    if 'Benign' in f.split(os.sep): continue
    
    h = extract_hash(os.path.basename(f))
    if h not in valid_hashes: continue # Must be in original_data
    
    ter = get_vt_ter(f)
    if ter is not None:
        malware_ters.append(ter)

# Load benign
benign_jsons = glob.glob('virus_total_result/VirusTotal-CLI-Tool/results/test/Benign/**/*.json', recursive=True)
benign_ters = []
for f in benign_jsons:
    if os.path.basename(f).startswith('failed_'): continue
    ter = get_vt_ter(f)
    if ter is not None:
        benign_ters.append(ter)

print(f"Loaded {len(malware_ters)} malware and {len(benign_ters)} benign")

true_labels = [1]*len(malware_ters) + [0]*len(benign_ters)
all_ters = malware_ters + benign_ters

print("Tau | Acc   | Macro-F1 | N_tau")
for tau in range(10, 81, 10):
    preds = [1 if t > tau else 0 for t in all_ters]
    acc = accuracy_score(true_labels, preds) * 100
    f1 = f1_score(true_labels, preds, average='macro') * 100
    n_tau = sum(1 for t in malware_ters if t > tau)
    print(f"{tau:2d}% | {acc:5.2f} | {f1:5.2f}    | {n_tau}")
