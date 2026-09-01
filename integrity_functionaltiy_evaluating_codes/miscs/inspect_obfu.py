import json

with open('malware_detector_results.json', 'r') as f:
    data = json.load(f)

obfu = data.get('OBFU-mal_full', {})

labels = set()
for h, sample in obfu.items():
    res = sample.get('results', {}).get('binary_malconv', {})
    labels.add(res.get('final_label'))

print(labels)
