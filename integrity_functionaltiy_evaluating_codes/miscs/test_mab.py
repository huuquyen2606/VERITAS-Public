import json

with open('malguise/mab_detailed_reports.json', 'r') as f:
    mab = json.load(f)

if 'malguise_full.npz' in mab:
    data = mab['malguise_full.npz']
    print(f"Total elements: {len(data)}")
    if len(data) > 0:
        first_key = list(data.keys())[0]
        print(f"First key: {first_key}")
        print(f"First value: {data[first_key]}")
