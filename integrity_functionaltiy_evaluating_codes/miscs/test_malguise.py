import json
import re

def extract_hash(filename):
    match = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if match: return match.group(1).lower()
    return None

with open('malguise/per_file_evasion_results.json', 'r') as f:
    per_file = json.load(f).get('malguise_full', {})

with open('malguise/dqeaf_integrity_check.json', 'r') as f:
    integ = json.load(f).get('malguise_full.npz', {})

live_hashes = set(extract_hash(k) for k, v in integ.items() if extract_hash(k) and v.get('is_alive') == True)

with open('malware_detector_results_filtered.json', 'r') as f:
    orig_data = json.load(f).get('Test_full', {})

benign_count = 0
live_benign_count = 0
orig_det_live_benign_count = 0

for k, v in per_file.items():
    h = extract_hash(k)
    res = v.get('results', {}).get('imcfn', {})
    if res.get('final_label') == 'benign':
        benign_count += 1
        if h in live_hashes:
            live_benign_count += 1
            
            # check orig info
            orig_info = next((ov for ok, ov in orig_data.items() if extract_hash(ok) == h), None)
            if orig_info:
                orig_fl = orig_info.get('results', {}).get('imcfn', {}).get('final_label', '')
                if orig_fl != 'benign':
                    orig_det_live_benign_count += 1

print(f"Total benign: {benign_count}")
print(f"Live benign: {live_benign_count}")
print(f"Orig Det Live benign: {orig_det_live_benign_count}")
