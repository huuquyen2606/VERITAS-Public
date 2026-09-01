import json

with open('malware_detector_results.json', 'r') as f:
    data = json.load(f)

obfu = data.get('OBFU-mal_full', {})

ignored = 0
benign = 0
malware = 0
for h, sample in obfu.items():
    res = sample.get('results', {}).get('binary_malconv', {})
    lbl = res.get('final_label')
    if lbl == 'IGNORED': ignored += 1
    elif lbl == 'Benign': benign += 1
    elif lbl == 'Malware': malware += 1

print(f"IGNORED: {ignored}, Benign: {benign}, Malware: {malware}")
