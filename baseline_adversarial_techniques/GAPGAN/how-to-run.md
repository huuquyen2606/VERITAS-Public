# GAPGAN: How to Run

## 1. Environment Setup

```bash
# Create and activate Python virtual environment
python3 -m venv gapgan-venv
source gapgan-venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Smoke Test (Pipeline Verification)

```bash
python3 smoke_test.py
```

## 3. Training (Algorithm 1)

```bash
python3 training/train.py \
    --malware_dir data/malware \
    --benign_dir data/benign \
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

## 4. Evaluation (Attack & ASR)

```bash
python3 evaluation/evaluate.py \
    --malware_dir data/malware \
    --benign_dir data/benign \
    --generator_path checkpoints/generator_1000.pth \
    --blackbox_path checkpoints/malconv.pth \
    --input_length 2000000 \
    --payload_rate 0.025 \
    --epsilon 0.06 \
    --output_dir adversarial_samples \
    --device cuda
```

## 5. Deactivate Environment

```bash
deactivate
```
