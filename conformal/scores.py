"""Nonconformity scores THR, APS and RAPS.

Every ``*_scores_all`` returns an (N, C) array of s(x, y) for all labels, with lower meaning
more conforming, so a prediction set is {y : s(x, y) <= qhat}.

    THR   s(x, y) = 1 - p(y | x)
    APS   s(x, y) = sum of p over labels strictly more probable than y, plus u * p(y | x)
    RAPS  APS + lam_reg * max(rank(y) - k_reg, 0)

``u`` is a per-sample uniform draw giving exact rather than conservative coverage; the same
draw must serve calibration and test of a given split.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def thr_scores_all(probs: np.ndarray) -> np.ndarray:
    return 1.0 - probs


def _aps_like_scores(probs: np.ndarray, u: Optional[np.ndarray], lam_reg: float,
                     k_reg: int) -> np.ndarray:
    n, c = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    cum_before = cum - sorted_p

    rand_term = sorted_p if u is None else u[:, None] * sorted_p
    ranks = np.arange(1, c + 1)[None, :]
    penalty = lam_reg * np.maximum(ranks - k_reg, 0)

    scores_sorted = cum_before + rand_term + penalty
    scores = np.empty_like(scores_sorted)
    np.put_along_axis(scores, order, scores_sorted, axis=1)
    return scores


def aps_scores_all(probs: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
    return _aps_like_scores(probs, u, lam_reg=0.0, k_reg=1)


def raps_scores_all(probs: np.ndarray, u: Optional[np.ndarray] = None,
                    lam_reg: float = 0.01, k_reg: int = 1) -> np.ndarray:
    return _aps_like_scores(probs, u, lam_reg=lam_reg, k_reg=k_reg)


SCORE_FNS = {
    "THR": lambda probs, u=None, **kw: thr_scores_all(probs),
    "APS": lambda probs, u=None, **kw: aps_scores_all(probs, u),
    "RAPS": lambda probs, u=None, lam_reg=0.01, k_reg=1, **kw: raps_scores_all(
        probs, u, lam_reg=lam_reg, k_reg=k_reg
    ),
}


def scores_all(name: str, probs: np.ndarray, u: Optional[np.ndarray] = None,
               **kw) -> np.ndarray:
    if name not in SCORE_FNS:
        raise ValueError(f"unknown score {name!r}; choose from {list(SCORE_FNS)}")
    return SCORE_FNS[name](probs, u=u, **kw)


def true_label_scores(scores_all_arr: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return scores_all_arr[np.arange(scores_all_arr.shape[0]), y_true]


def draw_randomization(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)
