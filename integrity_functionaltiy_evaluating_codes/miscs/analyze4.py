import json

with open('malware_detector_results.json') as f:
    data = json.load(f)

test_samples = data.get('Test_full', {})
true_bins = set()
true_classes = set()
pred_labels = set()

for val in test_samples.values():
    true_bins.add(val.get('true_label_binary'))
    true_classes.add(val.get('true_label_class'))
    for det, res in val.get('results', {}).items():
        pred_labels.add(res.get('final_label'))

print(f"True binary: {true_bins}")
print(f"True classes: {true_classes}")
print(f"Pred labels: {pred_labels}")
