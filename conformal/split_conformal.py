"""Marginal split conformal prediction.

    qhat = k-th smallest calibration score, k = ceil((n + 1)(1 - alpha))
    C(x) = {y : s(x, y) <= qhat}

which covers with probability at least 1 - alpha under exchangeability of calibration and test.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """Returns +inf when k > n, i.e. when n is too small for the level; sets are then full."""
    n = cal_scores.shape[0]
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(np.partition(cal_scores, k - 1)[k - 1])


def build_sets(test_scores_all: np.ndarray, qhat: float) -> np.ndarray:
    return test_scores_all <= qhat


def set_sizes(membership: np.ndarray) -> np.ndarray:
    return membership.sum(axis=1)


def covered(membership: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return membership[np.arange(membership.shape[0]), y_true].astype(np.int64)


def marginal_coverage(membership: np.ndarray, y_true: np.ndarray) -> float:
    return float(covered(membership, y_true).mean())


def empirical_coverage_by_group(membership: np.ndarray, y_true: np.ndarray,
                                group_id: np.ndarray) -> dict:
    cov = covered(membership, y_true)
    return {int(g): float(cov[group_id == g].mean()) for g in np.unique(group_id)}
