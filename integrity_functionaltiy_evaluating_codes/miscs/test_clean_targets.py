import json

with open('malware_detector_results_raw.json', 'r') as f:
    ml_data = json.load(f)
    
test_benign = ml_data.get('Test_benign', {})
print("Total benign:", len(test_benign))
if len(test_benign) > 0:
    sample = next(iter(test_benign.values()))
    print(sample.keys())
    print(sample.get('results', {}).get('binary_malconv', {}))
    
