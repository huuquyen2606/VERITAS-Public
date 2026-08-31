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

raw_keys = list(raw_data.get('fiam_1500_filtered', {}).keys())
print("Sample of raw fiam keys:", raw_keys[:5])

# Find a match
count = 0
for raw_sample in raw_keys:
    if raw_sample in fiam_alive:
        count += 1
        
print("Exact matches:", count)
