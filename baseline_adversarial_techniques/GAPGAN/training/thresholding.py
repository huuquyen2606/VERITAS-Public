"""
GAPGAN - Dynamic Threshold Strategy
======================================
Paper: "Black-box Adversarial Attacks Against Deep Learning Based Malware
        Binaries Detection with GAN" (ECAI 2020)

Reference: Section 3.3 / Equation 6.
  e = { e,  if |e| > ε * (i / T_max)
      { 0,  else

  "we propose to use a dynamic threshold strategy to limit the minimum
   value of payloads."

  ε (maximum threshold) = 0.06  (from Table 7)
  i = current training iteration
  T_max = maximum training iteration
"""

import torch


def apply_dynamic_threshold(payloads: torch.Tensor,
                            current_iter: int,
                            max_iter: int,
                            epsilon: float = 0.06) -> torch.Tensor:
    """
    Apply the dynamic threshold to adversarial payloads in-place.

    Bytes with absolute value smaller than the current threshold
    are set to zero.  The threshold grows linearly from 0 to ε
    over the course of training.

    Parameters
    ----------
    payloads : Tensor of shape (batch, payload_length)
        Generated adversarial payloads in continuous space [-1, 1].
    current_iter : int
        Current training iteration i (0-indexed).
    max_iter : int
        Maximum training iteration T_max.
    epsilon : float
        Maximum threshold value ε (default 0.06, per Table 7).

    Returns
    -------
    thresholded : Tensor of same shape
        Payloads with small values zeroed out.
    """
    # Dynamic threshold: ε * (i / T_max), grows from 0 to ε
    threshold = epsilon * (current_iter / max_iter)

    # Equation 6: zero out bytes below threshold
    mask = payloads.abs() > threshold
    thresholded = payloads * mask.float()

    return thresholded
