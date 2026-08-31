import csv
from collections import defaultdict

file_path = 'final_result/fixed_csvs/combined_strict_rates_with_vt.csv'

ml_detectors = [
    # Binary
    "binary_malconv", "m_attn_health", "multiview_cnn",
    # Multiclass
    "malconv", "imcfn", "seqconvattn", "rtf_bert", "rtf_cannie", "m_attn_health_multi"
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

results = defaultdict(lambda: defaultdict(float))

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    next(reader)
    for row in reader:
        if not row: continue
        tech = row[0]
        det = row[1]
        if tech not in tech_map or det not in ml_detectors: continue
        framework = tech_map[tech]
        try:
            fiam_asr = float(row[13].strip('%'))
        except: continue
        results[framework][det] = fiam_asr

print("Framework   | Bin_MC | Bin_MAH | Bin_MVC | Mul_MC | Mul_IMC | Mul_SCA | Mul_BERT| Mul_CAN | Mul_MAH | Macro")
for fw in order:
    if fw not in results: continue
    
    vals = [results[fw][d] for d in ml_detectors]
    macro = sum(vals) / len(vals)
    
    val_strs = [f"{v:5.2f}" for v in vals]
    print(f"{fw:11s} | " + " | ".join(val_strs) + f" | {macro:5.2f}")

