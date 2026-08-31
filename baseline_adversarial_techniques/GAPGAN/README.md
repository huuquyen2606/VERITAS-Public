# GAPGAN - Generative Adversarial Payloads GAN

> **Paper:** *Black-box Adversarial Attacks Against Deep Learning Based Malware Binaries Detection with GAN*
> (Yuan et al., ECAI 2020)

GAPGAN generates adversarial payloads (byte-level attack blocks) to append to the end of malware PE binaries, bypassing deep-learning-based malware detection systems in a complete **black-box** setting at the **byte-level**.

---

## Overview Architecture

```
                ┌──────────────┐
  x_mal ───────►│  Generator G │──► a_adv (adversarial payloads)
  (normalized   │  2 Conv1d    │        │
   malware)     │  FC resize   │        │  concat
                │  2 DeConv    │        ▼
                │  1×1 Conv    │   x_adv = [x_mal, a_adv]
                │  Tanh output │        │
                └──────────────┘        │
                                        ▼
                              ┌───────────────────┐
                              │ Discriminator D   │◄── f(x) labels
                              │ 2 Conv1d          │    (from Black-box)
                              │ AdaptiveAvgPool   │
                              │ FC → Sigmoid      │
                              └───────────────────┘
                                        ▲
              ┌─────────────────────────┘
              │  query
              ▼
     ┌───────────────────┐
     │ Black-box f       │  MalConv (target detector)
     │ Embedding(256,8)  │  Pretrained separately
     │ Gated Conv1d      │  on labeled dataset
     └───────────────────┘
```

**Workflow:**

1. **Discriminator D** learns to *mimic* (distill) the classification behavior of the black-box detector **f** — trained using labels queried from f, NOT ground-truth labels.
2. **Generator G** receives normalized malware binaries, generates adversarial payload bytes, and appends them to create adversarial candidates `x_adv`.
3. G is optimized with an **automatic weighted balance β** (Eq. 4) between whole-file effectiveness (`x_adv`) and standalone payload effectiveness (`a_adv`).
4. **Dynamic Threshold ε** (Eq. 6) progressively zeroes out small payload values during training to reduce quantization loss when mapping back to discrete bytes.

---

## Data Requirements

Place datasets in the following directories:

| Directory | Content | Format |
|:---|:---|:---|
| `data/malware/` | Malware binaries | Raw PE binary (`.exe`, `.dll`, raw binary) |
| `data/benign/` | Benign binaries | Raw PE binary |

- Each file must be a **raw binary**, without encoding/compression.
- The pipeline automatically reads bytes, pads zeros up to length `t` (default `2,000,000`), and normalizes values to `[-1, 1]`.

---

## Target Detector Models

**MalConv (Target Detector f)** is integrated at `detector/target_model.py`.

- Train MalConv separately on labeled data (malware/benign) before running GAPGAN.
- Save weights as `.pth` and pass the path via `--blackbox_path`.
- If no weights are provided, `train.py` initializes random weights for pipeline validation.

---

## Quick Start

### 1. Environment Setup

```bash
# Create and activate Python virtual environment
python3 -m venv gapgan-venv
source gapgan-venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Smoke Test (Pipeline Verification)

```bash
python3 smoke_test.py
```

Expected output: `ALL SMOKE TESTS PASSED ✓`

### 3. Training GAPGAN (Algorithm 1)

```bash
python3 training/train.py \
    --malware_dir  data/malware \
    --benign_dir   data/benign \
    --blackbox_path checkpoints/malconv.pth \
    --input_length 2000000 \
    --payload_rate 0.025 \
    --max_iter 1000 \
    --batch_size 32 \
    --lr_g 0.0002 \
    --lr_d 0.0002 \
    --epsilon 0.06 \
    --save_dir checkpoints \
    --device cuda
```

Checkpoints are saved every 200 iterations under `checkpoints/`.

### 4. Evaluation — Attack & ASR

```bash
python3 evaluation/evaluate.py \
    --malware_dir  data/malware \
    --benign_dir   data/benign \
    --generator_path checkpoints/generator_1000.pth \
    --blackbox_path  checkpoints/malconv.pth \
    --input_length 2000000 \
    --payload_rate 0.025 \
    --epsilon 0.06 \
    --output_dir adversarial_samples \
    --device cuda
```

Output: ASR (Attack Success Rate) table and generated adversarial binaries under `adversarial_samples/`.

---

## Directory Structure

```
GAPGAN/
├── data/
│   ├── malware/                ← Raw PE malware binaries
│   └── benign/                 ← Raw PE benign binaries
├── feature_extraction/
│   └── preprocessor.py         ← Padding, normalize, BinaryDataset
├── detector/
│   └── target_model.py         ← MalConv (black-box target)
├── gan/
│   ├── generator.py            ← Generator G
│   └── discriminator.py        ← Discriminator D
├── training/
│   ├── tuning.py               ← Automatic β (Eq. 4)
│   ├── thresholding.py         ← Dynamic threshold ε (Eq. 6)
│   └── train.py                ← Algorithm 1 — training loop
├── evaluation/
│   └── evaluate.py             ← Attack process & ASR (Eq. 7)
├── requirements.txt
└── README.md
```
