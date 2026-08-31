import json

with open('malware_detector_results.json', 'r') as f:
    data = json.load(f)

# Inspect labels for first file in Test_full
first_file = list(data['Test_full'].keys())[0]
sample = data['Test_full'][first_file]
print(f"True Class: {sample['true_label_class']}")
print(f"True Binary: {sample['true_label_binary']}")
for det, res in sample['results'].items():
    print(f"Detector: {det}, Final Label: {res.get('final_label')}")
