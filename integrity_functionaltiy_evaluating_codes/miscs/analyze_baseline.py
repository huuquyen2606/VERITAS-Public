import json

with open('heuristic_rulebase_results_filtered.json', 'r') as f:
    data = json.load(f)

test_full = data.get('Test_full', {})
malware_count = 0
benign_count = 0

for h, attrs in test_full.items():
    tb = attrs.get('true_label_binary')
    if tb == 'Malware':
        malware_count += 1
    elif tb == 'Benign':
        benign_count += 1

print(f"Heuristic dataset: {malware_count} Malware, {benign_count} Benign")
