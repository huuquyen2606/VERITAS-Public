# AIMED-RL Baseline Reproduction

Paper: `AIMED-RL: Exploring Adversarial Malware Examples with Reinforcement Learning` (ECML PKDD 2021)

This repository contains the reproduction of the **AIMED-RL** baseline model. It utilizes Reinforcement Learning (DQN / Distributional DQN with Prioritized Experience Replay) to iteratively apply semantics-preserving perturbations to Windows Portable Executable (PE) malware binaries in order to evade static ML-based malware detectors.

---

## 1. Overview

- **Observation Space**: 2,350-dimensional PE feature vector extracted via `pefeatures.py` (EMBER feature representation).
- **Action Space**: PE mutation transformations based on OpenAI Gym Malware (Section appending, imports injection, overlay manipulation, header modifications).
- **Agent Architecture**: Distributional DQN / Double DQN with Prioritized Experience Replay.
- **Surrogate Detector**: LightGBM classifier trained on EMBER PE features (`data/lgbm_ember.pkl`, default threshold $\tau = 0.9$).

---

## 2. Quick Start

For detailed step-by-step instructions, see [`how-to-run.md`](how-to-run.md).

```bash
# Setup environment
python3 -m venv fame-env
source fame-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Run Training (1500 episodes, max 10 turns)
python3 main.py aimed-rl --train --no-evaluate --train-data data/malware --episodes 1500 --max-turns 10

# Run Evaluation & Generate Adversarial Examples
python3 main.py aimed-rl --no-train --evaluate --eval-data Test --max-turns 10 --save-ae-dir AEs
```

---

## 3. Directory Structure

```text
├── main.py                     ← Main execution entry point
├── src/                        ← Core AIMED-RL source code & config
│   ├── config.ini              ← Hyperparameter and path configurations
│   ├── rl.py                   ← Reinforcement learning environment & DQN agent
│   ├── implementation.py       ← High-level attack pipeline implementation
│   └── functions.py            ← PE analysis, parsing & reporting utilities
├── data/                       ← PE feature extractors & pretrained surrogate models
│   ├── lgbm_ember.pkl          ← LightGBM EMBER surrogate detector
│   ├── pefeatures.py           ← Feature extractor
│   └── manipulate.py           ← PE binary mutation actions
├── Adv_agent/                  ← Pretrained RL agents for each malware family
├── Test/                       ← Evaluation malware dataset organized by family
├── AEs/                        ← Generated adversarial malware binaries
├── how-to-run.md               ← Comprehensive reviewer guide
└── requirements.txt            ← Python dependencies
```
