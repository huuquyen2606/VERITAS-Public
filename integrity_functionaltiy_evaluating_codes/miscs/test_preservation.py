import json
import re

def extract_hash(filename):
    match = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if match:
        return match.group(1).lower()
    return None

generated_hashes = set()
with open('malguise/per_file_evasion_results.json', 'r') as f:
    det_results = json.load(f)
    for k in det_results['malguise_full'].keys():
        h = extract_hash(k)
        if h: generated_hashes.add(h)

# MAB
with open('malguise/mab_detailed_reports.json', 'r') as f:
    mab_raw = json.load(f).get('malguise_full.npz', {})
mab_map = {extract_hash(k): v.get('is_functional', False) for k, v in mab_raw.items() if extract_hash(k)}

mab_b_cnt = 0
for h in generated_hashes:
    b_m = mab_map.get(h, False)
    if b_m: mab_b_cnt += 1
    
total = len(generated_hashes)
print(f"Generated: {total}")
print(f"MAB: Beh {mab_b_cnt/total*100:.2f}")
