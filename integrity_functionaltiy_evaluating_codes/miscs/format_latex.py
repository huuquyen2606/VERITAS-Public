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

order = [
    'MAB-Malware', 'AIMED-RL', 'DQEAF', 'OBFU-mal', 
    'GAPGAN', 'MalGPT', 'GAMMA', 'MalGuise', 'VERITAS'
]

transform_cost = {
    'MAB-Malware': '2.58 / 6 & 108.46',
    'AIMED-RL': '3.24 / 5 & 30.22',
    'DQEAF': '4.94 / 10 & 28.07',
    'OBFU-mal': '1.01 / 5 & 901.52',
    'GAPGAN': '-- & 176.69',
    'MalGPT': '-- & 0.23',
    'GAMMA': '-- & 321.21',
    'MalGuise': '1.00/6 & 0.00',
    'VERITAS': '\\textbf{1.39 / 5} & \\textbf{51.89}'
}

results = defaultdict(lambda: defaultdict(dict))

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
            raw_asr = float(row[11].strip('%'))
            dqeaf_asr = float(row[12].strip('%'))
            fiam_asr = float(row[13].strip('%'))
        except: continue
        results[framework][det] = {'raw': raw_asr, 'dqeaf': dqeaf_asr, 'fiam': fiam_asr}

print(r"""\begin{table*}[ht]
\centering
\caption{
Validity sensitivity and transformation characteristics across the evaluated
attack frameworks. Attack-effectiveness values are macro averages (\%) over
the nine learning-based targets.
}
\label{tab:validity_cost}
\setlength{\tabcolsep}{3.2pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{lrrr|rrrrrrrrr|rr}
\toprule
&
\multicolumn{12}{c|}{\textbf{Attack Effectiveness}}
&
\multicolumn{2}{c}{\textbf{Transformation Cost}}
\\
\cmidrule(lr){2-13}
\cmidrule(lr){14-15}

\textbf{Framework}
& \textbf{Raw-ASR}
& $\mathbf{V\mbox{-}ASR^{VER}}$
& $\mathbf{V\mbox{-}ASR^{DQEAF}}$
& \textbf{MC(B)}
& \textbf{IMCFN}
& \textbf{MAH(B)}
& \textbf{MAH(M)}
& \textbf{MC(M)}
& \textbf{MVCNN}
& \textbf{RTFB}
& \textbf{RTFC}
& \textbf{SCA}
& \textbf{\makecell{Steps\\Avg./Max.}}
& \textbf{\makecell{Size\\Growth(\%)}}
\\
\midrule""")

for fw in order:
    if fw not in results: continue
    
    avg_raw = sum(v['raw'] for v in results[fw].values()) / 9
    avg_fiam = sum(v['fiam'] for v in results[fw].values()) / 9
    avg_dqeaf = sum(v['dqeaf'] for v in results[fw].values()) / 9
    
    det_str = " & ".join(f"{results[fw][d]['raw']:0.2f}" for d in ml_detectors)
    
    if fw == 'VERITAS':
        print(f"\\textbf{{{fw}}}\n& \\textbf{{{avg_raw:0.2f}}} & \\textbf{{{avg_fiam:0.2f}}} & \\textbf{{{avg_dqeaf:0.2f}}}")
        bold_det = " & ".join(f"\\textbf{{{results[fw][d]['raw']:0.2f}}}" for d in ml_detectors)
        print(f"& {bold_det}")
        print(f"& {transform_cost[fw]}\n\\\\")
    else:
        print(f"{fw}\n& {avg_raw:0.2f} & {avg_fiam:0.2f} & {avg_dqeaf:0.2f}")
        print(f"& {det_str}")
        print(f"& {transform_cost[fw]} \\\\")
        
    print()

print(r"""\bottomrule
\end{tabular}}
\end{table*}""")
