"""
GAPGAN - Smoke Test
=====================
Verifies that all modules import correctly and that tensor dimensions
flow through the pipeline without errors using dummy data.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extraction.preprocessor import normalize, denormalize, pad_binary
from detector.target_model import MalConv
from gan.generator import Generator
from gan.discriminator import Discriminator
from training.tuning import compute_beta
from training.thresholding import apply_dynamic_threshold

# --- Config ---
INPUT_LENGTH = 4096       # Small size for fast smoke test
PAYLOAD_RATE = 0.1        # 10%
PAYLOAD_LENGTH = int(INPUT_LENGTH * PAYLOAD_RATE)
TOTAL_LENGTH = INPUT_LENGTH + PAYLOAD_LENGTH
BATCH_SIZE = 4
DEVICE = "cpu"

print(f"INPUT_LENGTH={INPUT_LENGTH}, PAYLOAD_LENGTH={PAYLOAD_LENGTH}, "
      f"TOTAL_LENGTH={TOTAL_LENGTH}")

# --- 1. Preprocessor ---
import numpy as np
raw = np.random.randint(0, 256, size=2000, dtype=np.uint8)
padded = pad_binary(raw, INPUT_LENGTH)
assert padded.shape == (INPUT_LENGTH,), f"pad_binary shape mismatch: {padded.shape}"
normed = normalize(padded)
assert normed.min() >= -1.0 and normed.max() <= 1.0, "normalize range error"
recovered = denormalize(normed)
assert recovered.dtype == np.uint8, "denormalize dtype error"
print("[OK] Preprocessor: pad_binary, normalize, denormalize")

# --- 2. MalConv Target Model ---
f = MalConv(input_length=INPUT_LENGTH, embedding_dim=8, channels=32)
byte_input = torch.randint(0, 256, (BATCH_SIZE, INPUT_LENGTH))
prob = f(byte_input)
assert prob.shape == (BATCH_SIZE,), f"MalConv output shape mismatch: {prob.shape}"
labels = f.predict(byte_input)
assert labels.shape == (BATCH_SIZE,), f"MalConv predict shape mismatch"
print(f"[OK] MalConv: output={prob}, labels={labels}")

# --- 3. Generator ---
G = Generator(input_length=INPUT_LENGTH, payload_length=PAYLOAD_LENGTH,
              channels=32)
x_mal = torch.randn(BATCH_SIZE, INPUT_LENGTH).clamp(-1, 1)
a_adv = G(x_mal)
assert a_adv.shape == (BATCH_SIZE, PAYLOAD_LENGTH), \
    f"Generator output shape mismatch: {a_adv.shape}"
assert a_adv.min() >= -1.0 and a_adv.max() <= 1.0, \
    "Generator output out of [-1,1] range"
print(f"[OK] Generator: output shape={a_adv.shape}, "
      f"range=[{a_adv.min():.3f}, {a_adv.max():.3f}]")

# --- 4. Discriminator ---
D = Discriminator(input_length=TOTAL_LENGTH, channels=32)
x_adv = torch.cat([x_mal, a_adv], dim=1)
assert x_adv.shape == (BATCH_SIZE, TOTAL_LENGTH)
d_out = D(x_adv)
assert d_out.shape == (BATCH_SIZE,), \
    f"Discriminator output shape mismatch: {d_out.shape}"
print(f"[OK] Discriminator: output shape={d_out.shape}, "
      f"values={d_out.detach()}")

# --- 5. Dynamic Threshold ---
threshed = apply_dynamic_threshold(a_adv, 500, 1000, epsilon=0.06)
assert threshed.shape == a_adv.shape
zero_count = (threshed == 0).sum().item()
print(f"[OK] Dynamic Threshold: zeroed {zero_count}/{threshed.numel()} elements")

# --- 6. Automatic Beta ---
beta = compute_beta(d_out, d_out * 0.5)
assert 0.0 <= beta <= 1.0, f"Beta out of range: {beta}"
print(f"[OK] Automatic Beta: β={beta:.4f}")

# --- 7. Loss computation (Eq. 3 & 5) ---
bce = torch.nn.BCELoss()
# Discriminator distillation loss (Eq. 5)
f_targets = torch.sigmoid(torch.randn(BATCH_SIZE))
loss_d = bce(d_out.detach(), f_targets)
print(f"[OK] L_D (distillation) = {loss_d.item():.4f}")

# Generator loss (Eq. 3)
d_xadv = D(x_adv)
d_aadv = D(torch.cat([torch.zeros_like(x_mal) - 1.0, a_adv], dim=1))
loss_g = -(1 - beta) * d_xadv.mean() - beta * d_aadv.mean()
print(f"[OK] L_G (generator) = {loss_g.item():.4f}")

print("\n" + "=" * 50)
print("ALL SMOKE TESTS PASSED ✓")
print("=" * 50)
