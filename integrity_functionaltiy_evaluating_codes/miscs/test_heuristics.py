import json
with open('malguise/clamav_heuristic.json', 'r') as f:
    clamav = json.load(f)
print("Keys in datasets:", list(clamav['datasets'].keys()))
