import json
import re

with open('malware_detector_results_filtered.json') as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

techniques = set()
for key in data.keys():
    # Extract technique pattern if any. The original ones are just hashes.
    if re.match(r'^[a-fA-F0-9]{40,64}(\.exe)?$', key):
        techniques.add('original')
    else:
        # e.g., shard_1__<hash>__evaded.exe
        m = re.match(r'^(.*?)__([a-fA-F0-9]{40,64})__(.*?)\.exe$', key)
        if m:
            techniques.add(f"{m.group(1)}__...__{m.group(3)}")
        else:
            techniques.add("other: " + key[:20])

print(f"Found patterns: {techniques}")
