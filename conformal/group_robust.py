"""Group-conditional (Mondrian) and total-variation-robust thresholds.

Mondrian: one conformal quantile per stratum, so coverage holds within each stratum. Group ids
are g = 2y + a for label y and binary attribute a, and a candidate label c is admitted at the
quantile of the stratum that label implies, g(c) = c * n_attributes + a, which keeps the set
computable without the test label.

TV-robust: coverage under a test distribution within total variation eps of the calibration
distribution drops by at most eps, so calibrating at level 1 - alpha + eps restores 1 - alpha.
Here eps is the observed distance between calibration and test true-label scores, which uses the
test labels and therefore bounds what such a rule could deliver rather than what it would.
"""
from __future__ import annotations

import numpy as np

from .split_conformal import conformal_quantile


def mondrian_quantiles(cal_scores_true: np.ndarray, cal_group: np.ndarray,
                       alpha: float) -> dict:
    """{group: qhat}; groups absent from the calibration set are omitted."""
    out = {}
    for g in np.unique(cal_group):
        out[int(g)] = conformal_quantile(cal_scores_true[cal_group == g], alpha)
    return out


def mondrian_build_sets(test_scores_all: np.ndarray, test_group: np.ndarray, group_q: dict,
                        *, n_attributes: int = 2) -> np.ndarray:
    """(N, C) membership under per-stratum thresholds; missing strata fall back to +inf."""
    g = np.asarray(test_group).astype(int)
    a = g % int(n_attributes)
    n_classes = test_scores_all.shape[1]
    lookup = np.array([group_q.get(int(gid), float("inf"))
                       for gid in range(n_classes * int(n_attributes))])
    qmat = lookup[np.arange(n_classes)[None, :] * int(n_attributes) + a[:, None]]
    return test_scores_all <= qmat


def score_tv_distance(cal_scores: np.ndarray, test_scores: np.ndarray,
                      n_bins: int = 50) -> float:
    """Total variation between two score distributions on shared bins."""
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    if cal_scores.size == 0 or test_scores.size == 0:
        return 0.0
    lo = min(cal_scores.min(), test_scores.min())
    hi = max(cal_scores.max(), test_scores.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    pc, _ = np.histogram(cal_scores, bins=edges)
    pt, _ = np.histogram(test_scores, bins=edges)
    return float(0.5 * np.abs(pc / pc.sum() - pt / pt.sum()).sum())


def robust_quantile(cal_scores_true: np.ndarray, alpha: float,
                    eps: float) -> tuple[float, float]:
    """Threshold at the inflated level, returned with that level."""
    alpha_robust = max(alpha - eps, 0.0)
    return conformal_quantile(cal_scores_true, alpha_robust), alpha_robust
