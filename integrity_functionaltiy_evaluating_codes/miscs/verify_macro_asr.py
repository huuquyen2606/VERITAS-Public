import csv
from collections import defaultdict

file_path = 'final_result/fixed_csvs/combined_strict_rates_with_vt.csv'

ml_detectors = [
    "binary_malconv", "imcfn", "m_attn_health", "m_attn_health_multi", 
    "malconv", "multiview_cnn", "rtf_bert", "rtf_cannie", "seqconvattn"
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

results = defaultdict(lambda: defaultdict(dict))

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    next(reader) # header 1
    next(reader) # header 2
    
    for row in reader:
        if not row: continue
        tech = row[0]
        det = row[1]
        if tech not in tech_map: continue
        if det not in ml_detectors: continue
        
        framework = tech_map[tech]
        
        # We need Orig_Det block (columns 11, 12, 13)
        # 11: Evasion_Rate
        # 12: Valid Evasion rate (dqeaf)
        # 13: Valid evasion rate (fiam)
        try:
            raw_asr = float(row[11].strip('%'))
            dqeaf_asr = float(row[12].strip('%'))
            fiam_asr = float(row[13].strip('%'))
        except:
            continue
            
        results[framework][det] = {
            'raw': raw_asr,
            'dqeaf': dqeaf_asr,
            'fiam': fiam_asr
        }

for fw in results.keys():
    if len(results[fw]) != 9:
        print(f"Warning: {fw} has {len(results[fw])} detectors instead of 9!")
        
    avg_raw = sum(v['raw'] for v in results[fw].values()) / 9
    avg_fiam = sum(v['fiam'] for v in results[fw].values()) / 9
    avg_dqeaf = sum(v['dqeaf'] for v in results[fw].values()) / 9
    
    print(f"{fw:15s}: Raw-ASR: {avg_raw:5.2f} | V-ASR^VER(FIAM): {avg_fiam:5.2f} | V-ASR^DQEAF: {avg_dqeaf:5.2f}")
    
    # Print individual detector raw rates
    det_rates = [f"{results[fw][d]['raw']:5.2f}" for d in ml_detectors]
    print("  Detectors:", " | ".join(det_rates))
