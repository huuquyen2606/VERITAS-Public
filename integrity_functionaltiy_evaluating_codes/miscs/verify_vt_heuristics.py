import csv
from collections import defaultdict

file_path = 'final_result/fixed_csvs/combined_strict_rates_with_vt.csv'

detectors = [
    "clamav", "defender", 
    "virustotal_ter_20", "virustotal_ter_30", "virustotal_ter_40", 
    "virustotal_ter_50", "virustotal_ter_60", "virustotal_ter_70", "virustotal_ter_80"
]

tech_map = {
    'MAB_full_lived': 'MAB-Malware',
    'aimed_full': 'AIMED-RL',
    'dqef_full_lived': 'DQEAF',
    'OBFU-mal_full': 'OBFU-mal',
    'gapgan_full': 'GAPGAN',
    'Malgpt_full': 'MalGPT',
    'gamma_adv_full': 'GAMMA',
    'malguise_full_lived': 'MalGuise',
    'fiam_1500_filtered': 'VERITAS'
}

order = [
    'MAB-Malware', 'AIMED-RL', 'DQEAF', 'OBFU-mal', 
    'GAPGAN', 'MalGPT', 'GAMMA', 'MalGuise', 'VERITAS'
]

results = defaultdict(lambda: defaultdict(dict))

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    next(reader)
    
    for row in reader:
        if not row: continue
        tech = row[0]
        det = row[1]
        
        if tech not in tech_map: continue
        if det not in detectors: continue
        
        framework = tech_map[tech]
        
        # Valid evasion rate (fiam) is index 13 in the CSV
        try:
            fiam_asr = float(row[13].strip('%'))
        except:
            continue
            
        results[framework][det] = fiam_asr

print("Framework\t| ClamAV | Defend | 20%  | 30%  | 40%  | 50%  | 60%  | 70%  | 80%")
print("-" * 80)

for fw in order:
    if fw not in results: continue
    
    vals = []
    for d in detectors:
        val = results[fw].get(d, 0.0)
        vals.append(f"{val:5.2f}")
        
    print(f"{fw:15s}\t| " + " | ".join(vals))

# Check Nd values
nd_map = {}
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    next(reader)
    for row in reader:
        if not row: continue
        tech = row[0]
        if tech == 'fiam_1500_filtered': # Use any tech since orig_det is same
            det = row[1]
            if det in detectors:
                orig_det = int(row[3])
                nd_map[det] = orig_det

print("\nNd values:")
nd_vals = [f"{nd_map.get(d, 0)}" for d in detectors]
print(" | ".join(nd_vals))

