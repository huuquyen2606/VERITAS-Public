import json
with open('malware_detector_results_raw.json', 'r') as f:
    raw_data = json.load(f)
with open('dqeaf_handles/fiam_FIR.json', 'r') as f:
    fiam_report = json.load(f)
    
fiam_alive = set()
for ds in fiam_report.keys():
    for sample, data in fiam_report.get(ds, {}).items():
        if data.get('is_functional') and data.get('is_alive'):
             fiam_alive.add(sample)

print("Alive hashes in FIAM:", len(fiam_alive))
print("Sample of alive fiam hashes:", list(fiam_alive)[:5])

count = 0
for raw_sample in raw_data.get('fiam_1500_filtered', {}).keys():
    parts = raw_sample.split('__')
    if len(parts) >= 2:
        sample_hash = parts[1]
        if sample_hash in fiam_alive:
            count += 1
            
print("Matching fiam raw count:", count)
