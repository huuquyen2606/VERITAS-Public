import json
import csv
import os

def main():
    print("Loading malware detector references...")
    with open('malware_detector_results_raw.json', 'r') as f:
        raw_ref = json.load(f)
        
    with open('malware_detector_results_filtered.json', 'r') as f:
        filtered_ref = json.load(f)
        
    print("Loading heuristic results...")
    with open('heuristic analysis/clamav.json', 'r') as f:
        clamav_data = json.load(f)
        
    with open('heuristic analysis/defender.json', 'r') as f:
        defender_data = json.load(f)
        
    # Build lookup dictionaries by filename
    clamav_lookup = {}
    for ds_name, ds_data in clamav_data.get('datasets', {}).items():
        for filepath, file_info in ds_data.get('files', {}).items():
            filename = os.path.basename(filepath.replace('\\', '/'))
            status = file_info.get('status')
            label = "Malware" if status == "detected" else "Benign"
            clamav_lookup[filename] = label
            
    defender_lookup = {}
    for ds_name, ds_data in defender_data.get('datasets', {}).items():
        for filepath, file_info in ds_data.get('files', {}).items():
            filename = os.path.basename(filepath.replace('\\', '/'))
            status = file_info.get('status')
            label = "Malware" if status == "detected" else "Benign"
            defender_lookup[filename] = label
            
    assemblyline_lookup = {}
    with open('heuristic analysis/assemblyline_summary_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('Filename')
            score_str = row.get('Max_Score', '0')
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0
            label = "Malware" if score >= 1000 else "Benign"
            assemblyline_lookup[filename] = label
            
    def process_reference(ref_data):
        new_data = {}
        for tech, samples in ref_data.items():
            new_data[tech] = {}
            for sample_name, sample_info in samples.items():
                new_sample = {
                    "true_label_class": sample_info.get("true_label_class"),
                    "true_label_binary": sample_info.get("true_label_binary"),
                    "results": {}
                }
                
                # clamav
                clamav_res = clamav_lookup.get(sample_name, "IGNORED")
                new_sample["results"]["clamav"] = {"final_label": clamav_res}
                
                # defender
                defender_res = defender_lookup.get(sample_name, "IGNORED")
                new_sample["results"]["defender"] = {"final_label": defender_res}
                
                # assemblyline
                al_res = assemblyline_lookup.get(sample_name, "IGNORED")
                new_sample["results"]["assemblyline"] = {"final_label": al_res}
                
                new_data[tech][sample_name] = new_sample
        return new_data

    print("Processing raw...")
    raw_res = process_reference(raw_ref)
    with open('heuristic_rulebase_results_raw.json', 'w') as f:
        json.dump(raw_res, f, indent=4)
        
    print("Processing filtered...")
    filtered_res = process_reference(filtered_ref)
    with open('heuristic_rulebase_results_filtered.json', 'w') as f:
        json.dump(filtered_res, f, indent=4)
        
    print("Done!")

if __name__ == '__main__':
    main()
