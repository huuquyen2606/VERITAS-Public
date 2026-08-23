"""
GAPGAN - Automatic Weight Tuning (β)
=======================================
Paper: "Black-box Adversarial Attacks Against Deep Learning Based Malware
        Binaries Detection with GAN" (ECAI 2020)

Reference: Section 3.3 / Equation 4.
  β = exp(E_{x~p_{x_adv}}[D(x)]) / (exp(E_{x~p_{x_adv}}[D(x)]) + exp(E_{a~p_{a_adv}}[D(a)]))

  "If x_adv is more effective than a_adv, then the expectation of the
   output of D to x_adv is larger. The automatic tuning mechanism will
   increase β to improve the learning rate of a_adv indirectly."
"""

import torch


def compute_beta(d_scores_xadv: torch.Tensor,
                 d_scores_aadv: torch.Tensor) -> float:
    """
    Compute the automatic weight tuning parameter β.

    Parameters
    ----------
    d_scores_xadv : Tensor of shape (batch,)
        D's output probabilities for adversarial *samples* x_adv.
    d_scores_aadv : Tensor of shape (batch,)
        D's output probabilities for adversarial *payloads* a_adv alone.

    Returns
    -------
    beta : float
        Tuning weight in [0, 1].
    """
    # Expectations
    e_xadv = d_scores_xadv.mean()
    e_aadv = d_scores_aadv.mean()

    # Equation 4
    exp_xadv = torch.exp(e_xadv)
    exp_aadv = torch.exp(e_aadv)

    beta = (exp_xadv / (exp_xadv + exp_aadv)).item()
    return beta
