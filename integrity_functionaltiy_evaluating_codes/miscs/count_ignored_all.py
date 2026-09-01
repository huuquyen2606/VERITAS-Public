import json

with open('malware_detector_results.json', 'r') as f:
    data = json.load(f)

obfu = data.get('OBFU-mal_full', {})

for det in ['m_attn_health', 'm_attn_health_multi', 'multiview_cnn']:
    counts = {}
    for h, sample in obfu.items():
        lbl = sample.get('results', {}).get(det, {}).get('final_label', 'NONE')
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"{det}: {counts}")
