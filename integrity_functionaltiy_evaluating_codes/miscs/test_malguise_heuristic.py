import csv

file_path = 'final_result/fixed_csvs/combined_strict_rates_with_vt.csv'

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    next(reader)
    for row in reader:
        if not row: continue
        if row[0] == 'malguise_full_lived':
            if row[1] in ['clamav', 'defender'] or row[1].startswith('virustotal_ter_80'):
                print(f"{row[1]:15s} | OrigDet: {row[3]} | OrigUnDet: {row[4]} | SuccessGenOrigDet: {row[5]} | AdvDet: {row[7]} | FIAM Valid: {row[10]} | FIAM Rate: {row[13]}")
