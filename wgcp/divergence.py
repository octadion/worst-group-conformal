"""Cross-group divergence of true-label conformity scores.

For one (model, score), the worst group's distribution of s(x, y_true) against the pooled rest,
summarised by Wasserstein-1 and the two-sample KS statistic. The score families live on
different scales, so divergences are reported per score and never averaged across them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from conformal.scores import draw_randomization, scores_all, true_label_scores

__all__ = ["true_label_conformity_scores", "cross_group_divergence", "GroupDivergence"]


def true_label_conformity_scores(probs: np.ndarray, y_true: np.ndarray, *, score: str = "APS",
                                 seed: int = 0, randomize: bool = True) -> np.ndarray:
    """``randomize`` draws the APS/RAPS uniform once with a fixed seed; THR ignores it."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true)
    u = draw_randomization(probs.shape[0], seed) if randomize else None
    return true_label_scores(scores_all(score, probs, u=u), y_true)


@dataclass
class GroupDivergence:
    score: str
    worst_group: int
    n_worst: int
    n_rest: int
    wasserstein1: float
    ks_stat: float
    ks_pvalue: float
    worst_mean: float
    rest_mean: float


def cross_group_divergence(scores: np.ndarray, group_id: np.ndarray, worst_group: int, *,
                           score_name: str = "APS") -> GroupDivergence:
    """``worst_group`` is supplied by the caller, so divergence and accuracy name one group."""
    scores = np.asarray(scores, dtype=np.float64)
    group_id = np.asarray(group_id)
    worst_mask = group_id == worst_group
    s_worst = scores[worst_mask]
    s_rest = scores[~worst_mask]
    if s_worst.size == 0 or s_rest.size == 0:
        raise ValueError(f"empty side: |worst|={s_worst.size}, |rest|={s_rest.size} "
                         f"for worst_group={worst_group}")
    ks = ks_2samp(s_worst, s_rest)
    return GroupDivergence(
        score=score_name, worst_group=int(worst_group),
        n_worst=int(s_worst.size), n_rest=int(s_rest.size),
        wasserstein1=float(wasserstein_distance(s_worst, s_rest)),
        ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue),
        worst_mean=float(s_worst.mean()), rest_mean=float(s_rest.mean()),
    )
