import json
import re

with open('malware_detector_results.json') as f:
    data = json.load(f)

test_samples = data.get('Test_full', {})
def extract_hash(filename):
    m = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if m:
        return m.group(1).lower()
    return filename

test_hashes = {extract_hash(k): k for k in test_samples.keys()}
print(f"Total Test_full hashes: {len(test_hashes)}")

for tech, content in data.items():
    if tech == 'Test_full' or not isinstance(content, dict): continue
    
    matched = 0
    unmatched = []
    for k in content.keys():
        h = extract_hash(k)
        if h in test_hashes:
            matched += 1
        else:
            if len(unmatched) < 3:
                unmatched.append(k)
                
    print(f"Technique: {tech}, Matched: {matched}/{len(content)}")
    if unmatched:
        print(f"  Unmatched sample keys: {unmatched}")
