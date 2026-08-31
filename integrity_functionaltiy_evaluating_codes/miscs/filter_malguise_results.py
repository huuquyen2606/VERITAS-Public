import json
import os

def main():
    # 1. Load ML detector results for malguise_full
    with open('malguise/malware_detector_results_filtered.json', 'r') as f:
        malguise_results = json.load(f)
        
    malguise_full = malguise_results.get('malguise_full', {})
    
    # 2. Load live hashes
    with open('malguise/malguise_filtered_live_samples.json', 'r') as f:
        live_samples = json.load(f)
        
    live_hashes = live_samples.get('malguise', {})
    
    # 3. Filter to malguise_full_lived
    malguise_full_lived = {}
    for filename, attrs in malguise_full.items():
        # The key in malguise_full is the filename. We need to check if the hash is in live_hashes
        # Assuming the hash is embedded in the filename or the filename IS the hash
        import re
        match = re.search(r'([a-fA-F0-9]{32,64})', filename)
        h = match.group(1).lower() if match else None
        
        if h and h in live_hashes:
            malguise_full_lived[filename] = attrs
            
    print(f"Filtered malguise_full down to {len(malguise_full_lived)} live samples.")
    
    # 4. Load the main malware_detector_results_filtered.json
    with open('malware_detector_results_filtered.json', 'r') as f:
        main_results = json.load(f)
        
    # 5. Append malguise_full_lived
    main_results['malguise_full_lived'] = malguise_full_lived
    
    # 6. Save new file
    out_file = 'malware_detector_resulst_filtered_malguise_included.json'
    with open(out_file, 'w') as f:
        json.dump(main_results, f, indent=4)
        
    print(f"Saved merged ML detector results to {out_file}")

if __name__ == '__main__':
    main()
