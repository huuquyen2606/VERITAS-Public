# Usage Guide

## 1. Install
Option A (requirements):
```bash
pip install -r requirements.txt
```

Option B (editable):
```bash
pip install -e .
```

## 2. Configure Paths (Optional)
Edit `config.ini` to override paths without touching YAML configs.
```
[paths]
sample_dir = data/samples
output_dir = artifacts/evaded/blackbox
lightgbm_model = gym-malware/result.txt
dqn_model = artifacts/models/dqn_latest.pt
runs_dir = artifacts/runs
```

## 3. Prepare Samples
You can still prepare manually:
```bash
python prepare_samples.py --source-dir ./Adv_agent --dest-dir ./data/samples --clear-dest
```
Or let training/evaluation scripts auto-prepare from raw datasets via `--train_dir` / `--test_dir`.

## 4. Check Readiness
```bash
python scripts/check_ready.py
```
This verifies:
- Required Python packages.
- LightGBM model path.
- Sample directory.
- External tools (UPX, Darkarmour) if used.

## 5. Train
```bash
python scripts/train.py
```

Auto-prepare from a raw training dataset (scan subfolders, skip `Benign`, rename to SHA256 in `data/samples`):
```bash
python scripts/train.py --train_dir ./Adv_agent
```

## 6. Evaluate
```bash
python scripts/evaluate.py
```

Auto-prepare from a raw test dataset before generating AEs:
```bash
python scripts/evaluate.py --test_dir ./Test --model artifacts/models/dqn_latest.pt
```

Nếu muốn chắc chắn sạch hoàn toàn thư mục mẫu trước khi nạp mới:
```bash
python scripts/train.py --train_dir ./Adv_agent --recreate_sample_dir
python scripts/evaluate.py --test_dir ./Test --recreate_sample_dir --model artifacts/models/dqn_latest.pt
```

## 7. Export Action Sequences
```bash
python scripts/export_sequences.py --history artifacts/runs/eval_history_YYYYMMDD_HHMMSS.json
```

## 8. Reset Run Artifacts
```bash
bash reset_training.sh
```

## Notes
- `UPXPack` and `DarkarmourXOR` actions require external tools (`upx`, `darkarmour`). If missing, those actions no-op.
- The LightGBM surrogate model uses `gym-malware/result.txt`.
- Feature extraction uses gym-malware PEFeatureExtractor or ember if installed.

## External Tools (UPX, Darkarmour)
### UPX
- Install a UPX binary and ensure it is in your PATH as `upx`.
- Verify:
  ```bash
  upx -V
  ```

### Darkarmour (per paper usage)
- Obtain the Darkarmour tool used in the paper.
- If you cloned it locally in this repo, set in `config.ini`:
  ```ini
  darkarmour_path = darkarmour-master/darkarmour-master/darkarmour.py
  ```
- If installed globally, ensure it is in PATH as `darkarmour`.
- The code expects the CLI syntax shown in the paper (Figure 1):
  ```bash
  darkarmour -f <input.exe> --encrypt xor --loop <1|2|3> -o <output.exe>
  ```
- When using `darkarmour.py` directly:
  ```bash
  python darkarmour-master/darkarmour-master/darkarmour.py -f <input.exe> --encrypt xor --loop 1 -o <output.exe>
  ```
- Verify:
  ```bash
  python darkarmour-master/darkarmour-master/darkarmour.py --help
  ```

### Check readiness
```bash
python scripts/check_ready.py
```
This prints clear status for `upx` and `darkarmour` and tells you which actions will no-op if missing.





---

cd OBFU-mal
python3 -m venv obfu-venv
source obfu-venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config configs/experiment.yaml
python scripts/evaluate.py --config configs/experiment.yaml --model artifacts/models/dqn_latest.pt

python3 scripts/evaluate.py --config configs/experiment.yaml --model artifacts/models/dqn_latest.pt --plain-log

python3 scripts/train.py --train_dir ./Adv_agent
python3 scripts/evaluate.py --test_dir ./Test --model artifacts/models/dqn_latest.pt