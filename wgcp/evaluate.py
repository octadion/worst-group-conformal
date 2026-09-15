"""One conformal evaluation of one fitted head.

Given posteriors on the evaluation pool, this splits the pool into disjoint calibration and test
halves, composites each to its target rho, calibrates on the calibration half's true-label
scores, builds test sets under the chosen policy, and returns one flat record: coverage
(marginal, per-group minimum, mean over groups), coverage gap, set sizes, base accuracy, and the
cross-group score divergence.

The grid iterates this over method x seed x score x rho x split.
"""
from __future__ import annotations

import numpy as np

from conformal.group_robust import (mondrian_build_sets, mondrian_quantiles, robust_quantile,
                                    score_tv_distance)
from conformal.scores import draw_randomization, scores_all, true_label_scores
from conformal.split_conformal import conformal_quantile

from .divergence import cross_group_divergence
from .resampling import resample_to_rho, split_pool

RHO_CAL = 0.95
RHO_SWEEP = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50)
CALIBRATIONS = ("marginal_split", "mondrian", "shift_robust")


def _per_group(values: np.ndarray, group: np.ndarray) -> dict:
    return {int(g): float(values[group == g].mean()) for g in np.unique(group)}


def evaluate(probs: np.ndarray, y: np.ndarray, group: np.ndarray, *, score: str = "APS",
             alpha: float = 0.1, rho_test: float = 0.95, rho_cal: float = RHO_CAL,
             split_seed: int = 0, frac_cal: float = 0.5, n_eval: int | None = None,
             calibration: str = "marginal_split", return_examples: bool = False) -> dict:
    """``group`` is 2y + a. ``return_examples`` additionally returns the per-example arrays this
    already computes, so a figure of individual prediction sets comes from the same code path as
    the tables rather than a second implementation."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y)
    group = np.asarray(group)
    N = probs.shape[0]

    cal_pool, test_pool = split_pool(N, frac_cal=frac_cal, seed=split_seed)
    if n_eval is None:
        n_eval = min(cal_pool.size, test_pool.size)

    cal_rs = resample_to_rho(group[cal_pool], rho_cal, n_eval, seed=split_seed * 2 + 1)
    test_rs = resample_to_rho(group[test_pool], rho_test, n_eval, seed=split_seed * 2 + 2)
    cal_idx = cal_pool[cal_rs.idx]
    test_idx = test_pool[test_rs.idx]

    u_cal = draw_randomization(cal_idx.size, seed=split_seed * 7 + 1)
    cal_scores_all = scores_all(score, probs[cal_idx], u=u_cal)
    cal_true = true_label_scores(cal_scores_all, y[cal_idx])
    cal_group = group[cal_idx]

    u_test = draw_randomization(test_idx.size, seed=split_seed * 7 + 2)
    test_scores_all = scores_all(score, probs[test_idx], u=u_test)
    y_test = y[test_idx]
    g_test = group[test_idx]

    if calibration == "marginal_split":
        qhat = conformal_quantile(cal_true, alpha)
        membership = test_scores_all <= qhat
        thresholds = {"pooled": float(qhat)}
    elif calibration == "mondrian":
        gq = mondrian_quantiles(cal_true, cal_group, alpha)
        membership = mondrian_build_sets(test_scores_all, g_test, gq)
        thresholds = {int(g): float(q) for g, q in dict(gq).items()}
    elif calibration == "shift_robust":
        test_true_tmp = true_label_scores(test_scores_all, y_test)
        eps = score_tv_distance(cal_true, test_true_tmp)
        qhat, _ = robust_quantile(cal_true, alpha, eps)
        membership = test_scores_all <= qhat
        thresholds = {"pooled_robust": float(qhat), "eps": float(eps)}
    else:
        raise ValueError(f"unknown calibration {calibration!r}; choose from {CALIBRATIONS}")

    covered = membership[np.arange(test_idx.size), y_test].astype(np.float64)
    set_size = membership.sum(axis=1).astype(np.float64)

    cov_by_g = _per_group(covered, g_test)
    size_by_g = _per_group(set_size, g_test)
    worst_g = min(cov_by_g, key=lambda g: (cov_by_g[g], g))
    worst_cov = cov_by_g[worst_g]
    marg_cov = float(covered.mean())
    # The unweighted mean over groups is what Mondrian targets; it sits at 1 - alpha even where the
    # minimum does not, and n_cal_worst_group sets the size of that min-over-groups effect.
    mean_group_cov = float(np.mean(list(cov_by_g.values())))
    cov_range = float(max(cov_by_g.values()) - min(cov_by_g.values()))
    n_cal_worst_group = int((cal_group == worst_g).sum())
    cov_gap = float((1.0 - alpha) - worst_cov)
    mean_size = float(set_size.mean())
    worst_g_size = float(size_by_g[worst_g])
    size_disparity = float(max(size_by_g.values()) - min(size_by_g.values()))
    base_top1 = float((np.argmax(probs[test_idx], axis=1) == y_test).mean())

    test_true = true_label_scores(test_scores_all, y_test)
    div = cross_group_divergence(test_true, g_test, worst_group=worst_g, score_name=score)

    out = {
        "score": score, "calibration": calibration, "alpha": alpha, "rho_cal": rho_cal,
        "rho_test": rho_test, "split_seed": split_seed, "n_eval": int(n_eval),
        "rho_cal_realized": cal_rs.rho_realized, "rho_test_realized": test_rs.rho_realized,
        "marginal_cov": marg_cov, "worst_group": int(worst_g), "worst_group_cov": float(worst_cov),
        "mean_group_cov": mean_group_cov, "cov_range": cov_range,
        "n_cal_worst_group": n_cal_worst_group,
        "cov_gap": cov_gap, "mean_set_size": mean_size, "worst_group_set_size": worst_g_size,
        "set_size_disparity": size_disparity, "base_top1": base_top1,
        "div_wasserstein1": div.wasserstein1, "div_ks_stat": div.ks_stat,
        "div_ks_pvalue": div.ks_pvalue,
    }
    if return_examples:
        out["examples"] = {
            "test_idx": test_idx.astype(int),
            "y_test": y_test.astype(int),
            "group_test": g_test.astype(int),
            "membership": membership.astype(bool),
            "scores_all": test_scores_all.astype(np.float64),
            "thresholds": thresholds,
            "worst_group": int(worst_g),
        }
    return out
