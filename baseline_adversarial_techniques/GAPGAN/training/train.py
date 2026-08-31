"""
GAPGAN - Training Pipeline
=============================
Paper: "Black-box Adversarial Attacks Against Deep Learning Based Malware
        Binaries Detection with GAN" (ECAI 2020)

Implements Algorithm 1 from the paper.

Training loop overview (per iteration):
  1. Sample m examples from S → training set for D (S_d).
  2. For each x_d in S_d:  query black-box f to get f(x_d).
  3. Update θ_D with distillation loss L_D (Eq. 5).
  4. Sample m malware examples → training set for G (S_adv).
  5. Generate payloads a_adv = G(x_adv).
  6. Apply dynamic threshold to a_adv.
  7. Craft adversarial samples x_adv = [x_mal, a_adv].
  8. Compute automatic β (Eq. 4).
  9. Update θ_G with generator loss L_G (Eq. 3).
"""

import os
import argparse
from collections import deque
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Optional

# ---- Project imports ----
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction.preprocessor import BinaryDataset, normalize, denormalize, normalized_to_byte_input
from detector.target_model import MalConv
from gan.generator import Generator
from gan.discriminator import Discriminator
from training.tuning import compute_beta
from training.thresholding import apply_dynamic_threshold


# =========================================================================
# Helper utilities
# =========================================================================

def split_by_label(dataset):
    """Split dataset indices into malware (y=-1) and benign (y=1)."""
    mal_idx, ben_idx = [], []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        if label.item() == -1:
            mal_idx.append(i)
        else:
            ben_idx.append(i)
    return mal_idx, ben_idx





class DataPool:
    """
    Bounded FIFO data pool for discriminator distillation.

    Paper §3.3 states that both x_adv and x_ben are integrated into a
    data pool, and each iteration samples a mixed batch from that pool to
    query the black-box detector f. The paper does not specify a pool size,
    so we keep a bounded buffer for practicality on long binary sequences.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.samples = deque(maxlen=max_size)

    def add_batch(self, x_batch: torch.Tensor) -> None:
        stored = x_batch.detach().to(device='cpu', dtype=torch.float16)
        for sample in stored:
            self.samples.append(sample.clone())

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if not self.samples:
            raise ValueError('Data pool is empty.')

        pool_list = list(self.samples)
        num = min(batch_size, len(pool_list))
        indices = torch.randperm(len(pool_list))[:num].tolist()
        batch = torch.stack([
            pool_list[i].to(device=device, dtype=torch.float32)
            for i in indices
        ], dim=0)
        return batch

    def __len__(self) -> int:
        return len(self.samples)


def classification_agreement(a: torch.Tensor, b: torch.Tensor,
                             threshold: float = 0.5) -> float:
    """Agreement rate between two probability outputs under a threshold."""
    a_cls = a >= threshold
    b_cls = b >= threshold
    return (a_cls == b_cls).float().mean().item()


def load_checkpoint_state_dict(checkpoint_path: str,
                               device: torch.device) -> dict:
    """Load either a wrapped checkpoint or a raw state dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def collect_files(root: str) -> set[str]:
    """Collect all real file paths under a directory."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    files = set()
    for base, _, names in os.walk(root):
        for name in names:
            path = os.path.realpath(os.path.join(base, name))
            if os.path.isfile(path):
                files.add(path)
    return files


def assert_disjoint(dir_a: str, dir_b: str) -> None:
    """Ensure two directory trees do not share any files."""
    overlap = collect_files(dir_a) & collect_files(dir_b)
    if overlap:
        sample = sorted(overlap)[0]
        raise ValueError(
            "Detector-train and attack sets must be disjoint; "
            f"found overlap: {sample}"
        )


def require_checkpoint(checkpoint_path: str, label: str) -> None:
    """Fail fast if a required checkpoint is missing."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing {label} checkpoint: {checkpoint_path}"
        )


def query_blackbox(f: nn.Module, x_normalized: torch.Tensor,
                   device: torch.device) -> torch.Tensor:
    """
    Query the black-box detector f with normalized continuous samples.
    Returns f's probability output (soft label).
    """
    byte_input = normalized_to_byte_input(x_normalized).to(device)
    with torch.no_grad():
        probs = f(byte_input)
    return probs


# =========================================================================
# Main training function  (Algorithm 1)
# =========================================================================

def train_gapgan(
    malware_dir: str,
    benign_dir: str,
    blackbox_path: str,
    input_length: int = 2_000_000,
    payload_rate: float = 0.025,
    max_iter: int = 1000,
    batch_size: int = 32,
    lr_g: float = 0.0002,
    lr_d: float = 0.0002,
    epsilon: float = 0.06,
    save_dir: str = "checkpoints",
    device_name: str = "cpu",
    disjoint_against_dir: Optional[str] = None,
    data_pool_size: int = 64,
):
    """
    Full GAPGAN training loop (Algorithm 1).

    Parameters
    ----------
    malware_dir : str      Path to directory of raw malware binaries.
    benign_dir : str       Path to directory of raw benign binaries.
    blackbox_path : str    Path to pre-trained MalConv weights.
    input_length : int     Fixed input size t (default 2,000,000).
    payload_rate : float   Rate of payload length to input length.
    max_iter : int         T_max — maximum training iterations.
    batch_size : int       Mini-batch size m.
    lr_g : float           Learning rate for Generator.
    lr_d : float           Learning rate for Discriminator.
    epsilon : float        Maximum threshold ε for dynamic thresholding.
    save_dir : str         Directory to save model checkpoints.
    device_name : str      Device string ('cuda' or 'cpu').
    data_pool_size : int   Maximum number of samples kept in the
                           discriminator data pool.
    """

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # --- Payload length ---
    payload_length = int(input_length * payload_rate)
    total_length = input_length + payload_length  # t + payload

    print(f"[GAPGAN] input_length={input_length}, "
          f"payload_length={payload_length}, "
          f"total_length={total_length}")
    print(f"[GAPGAN] max_iter={max_iter}, batch_size={batch_size}, "
          f"epsilon={epsilon}")
    print(f"[GAPGAN] device={device}")
    print(f"[GAPGAN] data_pool_size={data_pool_size}")

    if disjoint_against_dir is not None:
        assert_disjoint(malware_dir, disjoint_against_dir)
        assert_disjoint(benign_dir, disjoint_against_dir)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset = BinaryDataset(malware_dir, benign_dir,
                            target_length=input_length)
    mal_idx, ben_idx = split_by_label(dataset)
    print(f"[GAPGAN] Dataset: {len(mal_idx)} malware, "
          f"{len(ben_idx)} benign")

    mal_subset = Subset(dataset, mal_idx)
    ben_subset = Subset(dataset, ben_idx)

    # DataLoaders — drop_last=True to keep batch sizes consistent
    mal_loader = DataLoader(mal_subset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
    ben_loader = DataLoader(ben_subset, batch_size=batch_size,
                            shuffle=True, drop_last=True)

    # ------------------------------------------------------------------
    # 2. Load black-box detector f
    # ------------------------------------------------------------------
    f = MalConv(input_length=input_length).to(device)
    require_checkpoint(blackbox_path, "trained MalConv")
    f.load_state_dict(load_checkpoint_state_dict(blackbox_path, device))
    f.eval()
    print("[GAPGAN] Black-box detector f loaded.")

    # ------------------------------------------------------------------
    # 3. Initialize Generator & Discriminator
    # ------------------------------------------------------------------
    G = Generator(input_length=input_length,
                  payload_length=payload_length).to(device)
    D = Discriminator(input_length=total_length).to(device)

    # Optimizers — Adam (framework default for GAN)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_g,
                                   betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_d,
                                   betas=(0.5, 0.999))

    # Loss for discriminator distillation: Binary Cross-Entropy
    # Paper Eq. 5: H(D(x), f(x))  — H is cross-entropy
    bce_loss = nn.BCELoss()

    # Paper Eq. 5 separates expectations over x_adv and x_ben.
    # Keep two pools so D always distills from both distributions explicitly.
    adv_pool = DataPool(max_size=data_pool_size)
    ben_pool = DataPool(max_size=data_pool_size)

    # ------------------------------------------------------------------
    # 4. Training loop  (Algorithm 1)
    # ------------------------------------------------------------------
    mal_iter = iter(mal_loader)
    ben_iter = iter(ben_loader)

    for iteration in range(max_iter):

        # ==============================================================
        # Step A: Update Discriminator D
        # ==============================================================
        D.train()
        G.eval()

        # --- Sample malware batch ---
        try:
            x_mal, _, _ = next(mal_iter)
        except StopIteration:
            mal_iter = iter(mal_loader)
            x_mal, _, _ = next(mal_iter)
        x_mal = x_mal.to(device)  # (batch, t)

        # --- Sample benign batch ---
        try:
            x_ben, _, _ = next(ben_iter)
        except StopIteration:
            ben_iter = iter(ben_loader)
            x_ben, _, _ = next(ben_iter)
        x_ben = x_ben.to(device)  # (batch, t)

        # --- Generate adversarial payloads ---
        with torch.no_grad():
            a_adv = G(x_mal)  # (batch, payload_length)
            # Apply dynamic threshold (Eq. 6)
            a_adv = apply_dynamic_threshold(
                a_adv, iteration, max_iter, epsilon
            )
            # Craft adversarial samples: x_adv = [x_mal, a_adv]
            x_adv = torch.cat([x_mal, a_adv], dim=1)  # (batch, total_length)

        # Pad benign to total_length (append zeros like MalConv padding)
        ben_pad = torch.zeros(x_ben.size(0), payload_length,
                              device=device) - 1.0  # -1.0 = byte 0 normalized
        x_ben_full = torch.cat([x_ben, ben_pad], dim=1)

        # Paper §3.3 / Eq. 5: D learns from both x_adv and x_ben.
        adv_pool.add_batch(x_adv)
        ben_pool.add_batch(x_ben_full)
        x_adv_pool = adv_pool.sample(batch_size, device)
        x_ben_pool = ben_pool.sample(batch_size, device)
        adv_pool_size = len(adv_pool)
        ben_pool_size = len(ben_pool)

        # --- Query black-box f on pooled samples ---
        with torch.no_grad():
            f_adv_pool = query_blackbox(f, x_adv_pool, device)
            f_ben_pool = query_blackbox(f, x_ben_pool, device)

        # --- D forward on pooled samples ---
        d_adv_pool = D(x_adv_pool)
        d_ben_pool = D(x_ben_pool)

        # --- Distillation loss L_D (Eq. 5) ---
        loss_d_adv = bce_loss(d_adv_pool, f_adv_pool)
        loss_d_ben = bce_loss(d_ben_pool, f_ben_pool)
        loss_d = loss_d_adv + loss_d_ben

        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()

        with torch.no_grad():
            mae_adv = (d_adv_pool - f_adv_pool).abs().mean().item()
            mae_ben = (d_ben_pool - f_ben_pool).abs().mean().item()
            agree_adv = classification_agreement(d_adv_pool, f_adv_pool)
            agree_ben = classification_agreement(d_ben_pool, f_ben_pool)

        # ==============================================================
        # Step B: Update Generator G
        # ==============================================================
        G.train()
        D.eval()

        # --- Sample fresh malware batch for G ---
        try:
            x_mal_g, _, _ = next(mal_iter)
        except StopIteration:
            mal_iter = iter(mal_loader)
            x_mal_g, _, _ = next(mal_iter)
        x_mal_g = x_mal_g.to(device)

        # --- Generate adversarial payloads ---
        a_adv_g = G(x_mal_g)
        a_adv_g_threshed = apply_dynamic_threshold(
            a_adv_g, iteration, max_iter, epsilon
        )

        # Craft adversarial samples
        x_adv_g = torch.cat([x_mal_g, a_adv_g_threshed], dim=1)

        # --- D scores ---
        d_xadv = D(x_adv_g)        # D's score on full adversarial sample
        d_aadv = D(
            torch.cat([
                torch.zeros(x_mal_g.size(0), input_length,
                            device=device) - 1.0,
                a_adv_g_threshed
            ], dim=1)
        )  # D's score on payload only (padded)

        # --- Automatic weight tuning β (Eq. 4) ---
        with torch.no_grad():
            beta = compute_beta(d_xadv, d_aadv)

        # --- Generator loss L_G (Eq. 3) ---
        # L_G = -(1-β) * E[D(x_adv)]  -  β * E[D(a_adv)]
        # We want to MAXIMIZE D's output → minimize -D(x)
        loss_g = -(1.0 - beta) * d_xadv.mean() - beta * d_aadv.mean()

        optimizer_G.zero_grad()
        loss_g.backward()
        optimizer_G.step()

        # ==============================================================
        # Logging
        # ==============================================================
        if iteration % 50 == 0 or iteration == max_iter - 1:
            print(f"[Iter {iteration:5d}/{max_iter}]  "
                  f"L_D={loss_d.item():.4f}  L_G={loss_g.item():.4f}  "
                  f"β={beta:.4f}  adv_pool={adv_pool_size:3d}  "
                  f"ben_pool={ben_pool_size:3d}  "
                  f"D(x_adv)={d_xadv.mean().item():.4f}  "
                  f"D(a_adv)={d_aadv.mean().item():.4f}  "
                  f"MAE_adv={mae_adv:.4f}  MAE_ben={mae_ben:.4f}  "
                  f"Agr_adv={agree_adv:.2f}  Agr_ben={agree_ben:.2f}")

        # Save checkpoints periodically
        if (iteration + 1) % 200 == 0 or iteration == max_iter - 1:
            torch.save(G.state_dict(),
                       os.path.join(save_dir, f"generator_{iteration+1}.pth"))
            torch.save(D.state_dict(),
                       os.path.join(save_dir,
                                    f"discriminator_{iteration+1}.pth"))
            print(f"[GAPGAN] Checkpoints saved at iteration {iteration+1}.")

    print("[GAPGAN] Training complete.")
    return G, D


# =========================================================================
# CLI entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GAPGAN Training — Algorithm 1 (ECAI 2020)")

    parser.add_argument("--malware_dir", type=str, required=True,
                        help="Path to malware binaries directory.")
    parser.add_argument("--benign_dir", type=str, required=True,
                        help="Path to benign software binaries directory.")
    parser.add_argument("--blackbox_path", type=str, required=True,
                        help="Path to pre-trained MalConv detector weights.")
    parser.add_argument("--input_length", type=int, default=2_000_000,
                        help="Fixed input size t (default: 2000000).")
    parser.add_argument("--payload_rate", type=float, default=0.025,
                        help="Payload rate (default: 0.025 = 2.5%%).")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Maximum training iterations T_max.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size m.")
    parser.add_argument("--lr_g", type=float, default=0.0002,
                        help="Generator learning rate.")
    parser.add_argument("--lr_d", type=float, default=0.0002,
                        help="Discriminator learning rate.")
    parser.add_argument("--epsilon", type=float, default=0.06,
                        help="Maximum threshold ε (default: 0.06).")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Checkpoint save directory.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cuda/cpu).")
    parser.add_argument("--disjoint_against_dir", type=str, default=None,
                        help="Optional dataset root that must be disjoint "
                             "from the GAPGAN training inputs.")
    parser.add_argument("--data_pool_size", type=int, default=64,
                        help="Maximum number of samples stored in the "
                             "paper-style discriminator data pool.")

    args = parser.parse_args()

    train_gapgan(
        malware_dir=args.malware_dir,
        benign_dir=args.benign_dir,
        blackbox_path=args.blackbox_path,
        input_length=args.input_length,
        payload_rate=args.payload_rate,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        epsilon=args.epsilon,
        save_dir=args.save_dir,
        device_name=args.device,
        disjoint_against_dir=args.disjoint_against_dir,
        data_pool_size=args.data_pool_size,
    )


if __name__ == "__main__":
    main()
