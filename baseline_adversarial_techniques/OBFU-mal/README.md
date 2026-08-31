# OBFU-mal Reproduction

This repository contains the reproduction of the OBFU-mal framework.

## Getting Started

1. Install:
   ```bash
   pip install -e .
   ```

2. Prepare samples (gym-malware style):
   - Either place hash-named PE files in `data/samples/`, or let scripts auto-prepare from raw datasets:
     - `python scripts/train.py --train_dir ./Adv_agent`
     - `python scripts/evaluate.py --test_dir ./Test --model artifacts/models/dqn_latest.pt`
     - For a fully clean sample folder each run: add `--recreate_sample_dir`

3. (Optional) Override paths:
   - Edit `config.ini` for input/output/model paths.

4. Check readiness:
   ```bash
   python scripts/check_ready.py
   ```

5. Train:
   ```bash
   python scripts/train.py
   ```

6. Evaluate:
   ```bash
   python scripts/evaluate.py
   ```

7. Export action sequences:
   ```bash
   python scripts/export_sequences.py --history artifacts/runs/eval_history_YYYYMMDD_HHMMSS.json
   ```

See `USAGE.md` for a full guide.
