# AIMED-RL: How to Run

## 1. Environment Setup

```bash
# Create and activate Python virtual environment
python3 -m venv fame-env
source fame-env/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 2. Configuration

Inspect and adjust parameters in `src/config.ini` under `[aimedrl]`:
- `episodes = 1500`
- `maxTurns = 10`
- `threshold = 0.9`
- `detectorPath = data/lgbm_ember.pkl`

---

## 3. Training (Phase 1)

Train a new AIMED-RL reinforcement learning agent on malware samples:

```bash
python3 main.py aimed-rl \
    --train \
    --no-evaluate \
    --train-data data/malware \
    --detector-path data/lgbm_ember.pkl \
    --threshold 0.9 \
    --episodes 1500 \
    --max-turns 10 \
    --recursive-data
```

Trained agents and training reports are saved under `samples/rl/agent/last/` and `db/rl/training_reports/last/`.

---

## 4. Evaluation / Adversarial Example Generation (Phase 2)

Evaluate an AIMED-RL agent against test malware samples to generate evasive adversarial examples:

### Option A: Evaluate Pretrained Agent
```bash
python3 main.py aimed-rl \
    --no-train \
    --evaluate \
    --eval-data Test \
    --detector-path data/lgbm_ember.pkl \
    --threshold 0.9 \
    --max-turns 10 \
    --recursive-data \
    --save-ae-dir AEs
```

### Option B: Evaluate Specific Family Agent
```bash
python3 main.py aimed-rl \
    --no-train \
    --evaluate \
    --agent-dir Adv_agent/Locker/agent/last \
    --eval-data Test/Locker \
    --detector-path data/lgbm_ember.pkl \
    --threshold 0.9 \
    --max-turns 10 \
    --save-ae-dir AEs/Locker
```

Generated adversarial binaries are stored under `AEs/` and evaluation reports under `db/rl/evaluating_reports/last/`.

---

## 5. End-to-End Pipeline (Train + Evaluate)

```bash
python3 main.py aimed-rl \
    --train \
    --evaluate \
    --train-data data/malware \
    --eval-data Test \
    --detector-path data/lgbm_ember.pkl \
    --threshold 0.9 \
    --episodes 1500 \
    --max-turns 10 \
    --recursive-data \
    --save-ae-dir AEs
```

---

## 6. Deactivate Environment

```bash
deactivate
```
