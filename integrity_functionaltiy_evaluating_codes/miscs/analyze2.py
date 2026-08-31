import json

with open('malware_detector_results.json') as f:
    data = json.load(f)

for tech, content in data.items():
    if isinstance(content, dict):
        print(f"Technique: {tech}, Number of samples: {len(content)}")
        print(f"Sample keys: {list(content.keys())[:2]}")
        sample_val = content[list(content.keys())[0]]
        print(f"Sample value structure keys: {list(sample_val.keys())}")
        if 'results' in sample_val:
            print(f"Sample detectors: {list(sample_val['results'].keys())}")
    else:
        print(f"Technique: {tech} is not a dict")
