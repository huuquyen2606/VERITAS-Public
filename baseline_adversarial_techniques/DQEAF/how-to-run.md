# DQEAF: How to Run

## 1. Environment Setup

```bash
# Create and activate Python virtual environment
python3 -m venv dqeaf-venv
source dqeaf-venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run preflight sanity check
python3 preflight_check.py
```

## 2. Configuration

Inspect and adjust `configure/config.ini` to point to your datasets and surrogate model paths.

## 3. Training (Phase 1)

Train the DQEAF reinforcement learning agent:

```bash
python3 scripts/train.py --config configure/config.ini
```

Trained models and meta artifacts (`q_target.pt`, `classifier_meta.json`) are saved under `outputs/`.

## 4. Evaluation / Adversarial Generation (Phase 2)

### Option A: Standard Evaluation
```bash
python3 scripts/evaluate.py --config configure/config.ini --model-dir outputs/dqeaf
```

### Option B: Direct AE Generation
```bash
python3 generate_ae.py \
    --model_dir outputs/dqeaf \
    --test_dir data/malware \
    --output_dir Evaded_Malware \
    --device cpu
```

Adversarial binaries are generated and organized by malware family under `Evaded_Malware/`.

## 5. Deactivate Environment

```bash
deactivate
```
