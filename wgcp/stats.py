"""Uncertainty for a nested design: calibration splits sit inside training seeds.

    cluster_bootstrap_ci    resample seeds, then splits within each drawn seed
    paired_cluster_diff_ci  the same for a difference measured on identical cells
    tost_equivalence        two one-sided tests against a margin
    spread_equivalence      one-sided bound on the max-min spread of per-method means
    correlation_ci          bootstrap interval for a correlation over few points
    min_of_k_shortfall      E[min] - mu for k Gaussian coverages, closed form
    simulate_min_coverage   the same quantity under the exact conformal law
    group_counts_at_rho     per-group calibration counts at correlation strength rho

All of it is numpy over computed records.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_B = 2000


def _as_cells(values, seeds, splits=None):
    v = np.asarray(values, dtype=np.float64)
    s = np.asarray(seeds)
    if v.shape[0] != s.shape[0]:
        raise ValueError(f"values ({v.shape[0]}) and seeds ({s.shape[0]}) must align")
    sp = np.arange(v.shape[0]) if splits is None else np.asarray(splits)
    return v, s, sp


def cluster_bootstrap_ci(values, seeds, splits=None, *, stat=np.mean, B: int = DEFAULT_B,
                         alpha: float = 0.05, seed: int = 0) -> dict:
    """Stage 1 resamples seeds, stage 2 the splits within them.

    ``method`` reports ``"flat (1 seed)"` when a single seed makes clustering impossible.
    """
    v, s, _ = _as_cells(values, seeds, splits)
    if v.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_seeds": 0, "n_obs": 0, "method": "empty"}
    uniq = np.unique(s)
    by_seed = [np.flatnonzero(s == u) for u in uniq]
    rng = np.random.default_rng(seed)
    point = float(stat(v))

    if len(uniq) < 2:
        draws = [float(stat(v[rng.integers(0, v.size, v.size)])) for _ in range(B)]
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"point": point, "lo": float(lo), "hi": float(hi), "n_seeds": int(len(uniq)),
                "n_obs": int(v.size), "method": "flat (1 seed)"}

    draws = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(by_seed), len(by_seed))
        idx = []
        for j in pick:
            grp = by_seed[j]
            idx.append(grp[rng.integers(0, grp.size, grp.size)])
        draws[b] = stat(v[np.concatenate(idx)])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_seeds": int(len(uniq)),
            "n_obs": int(v.size), "method": "cluster"}


def paired_cluster_diff_ci(a_values, b_values, seeds, *, B: int = DEFAULT_B, alpha: float = 0.05,
                           seed: int = 0) -> dict:
    """mean(a) - mean(b) for conditions measured on the same (seed, split) cells."""
    a = np.asarray(a_values, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")
    out = cluster_bootstrap_ci(a - b, seeds, stat=np.mean, B=B, alpha=alpha, seed=seed)
    out["excludes_zero"] = bool(out["lo"] > 0 or out["hi"] < 0)
    return out


def tost_equivalence(values, seeds, *, margin: float, splits=None, B: int = DEFAULT_B,
                     alpha: float = 0.05, seed: int = 0, center: float = 0.0) -> dict:
    """Equivalence holds when the whole (1 - 2*alpha) interval lies inside center +/- margin."""
    ci = cluster_bootstrap_ci(values, seeds, splits, stat=np.mean, B=B, alpha=2 * alpha, seed=seed)
    lo, hi = ci["lo"], ci["hi"]
    equivalent = bool(lo > center - margin and hi < center + margin)
    return {**ci, "margin": float(margin), "center": float(center), "equivalent": equivalent,
            "conf_level": 1 - 2 * alpha,
            "verdict": "EQUIVALENT" if equivalent else "not equivalent (interval escapes margin)"}


def spread_equivalence(by_method: dict, *, margin: float, B: int = DEFAULT_B, alpha: float = 0.05,
                       seed: int = 0) -> dict:
    """One-sided bound on max-min of per-method means; ``by_method`` maps method to (values, seeds).

    The spread is non-negative, so the test is one-sided: flat only if its upper bound is below
    the margin.
    """
    names = sorted(by_method)
    if len(names) < 2:
        return {"spread": float("nan"), "hi": float("nan"), "margin": float(margin),
                "equivalent": False, "verdict": "need >=2 methods", "methods": names}
    prepared = []
    for m in names:
        v, s = by_method[m]
        v = np.asarray(v, dtype=np.float64)
        s = np.asarray(s)
        prepared.append((v, s, [np.flatnonzero(s == u) for u in np.unique(s)]))
    point = float(np.ptp([v.mean() for v, _, _ in prepared]))

    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        means = []
        for v, _, by_seed in prepared:
            if len(by_seed) < 2:
                means.append(v[rng.integers(0, v.size, v.size)].mean())
                continue
            pick = rng.integers(0, len(by_seed), len(by_seed))
            idx = [by_seed[j][rng.integers(0, by_seed[j].size, by_seed[j].size)] for j in pick]
            means.append(v[np.concatenate(idx)].mean())
        draws[b] = np.ptp(means)
    hi = float(np.percentile(draws, 100 * (1 - alpha)))
    return {"spread": point, "hi": hi, "margin": float(margin), "equivalent": bool(hi < margin),
            "conf_level": 1 - alpha, "methods": names,
            "verdict": ("EQUIVALENT (spread bounded below margin)" if hi < margin
                        else "not equivalent (spread upper bound exceeds margin)")}


def correlation_ci(x, y, *, kind: str = "pearson", B: int = DEFAULT_B, alpha: float = 0.05,
                   seed: int = 0) -> dict:
    """Resamples the (x, y) pairs. ``small_n`` marks n < 8, where r is not a precise estimate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must align: {x.shape} vs {y.shape}")
    n = x.size

    def _r(xa, ya):
        if xa.size < 3 or np.std(xa) == 0 or np.std(ya) == 0:
            return float("nan")
        if kind == "spearman":
            xa = np.argsort(np.argsort(xa)).astype(float)
            ya = np.argsort(np.argsort(ya)).astype(float)
        return float(np.corrcoef(xa, ya)[0, 1])

    point = _r(x, y)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = _r(x[idx], y[idx])
        if not np.isnan(v):
            draws.append(v)
    if len(draws) < B // 10:
        return {"r": point, "lo": float("nan"), "hi": float("nan"), "n": int(n), "kind": kind,
                "small_n": True, "excludes_zero": False,
                "note": "CI unavailable: too many degenerate resamples at this n"}
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"r": point, "lo": float(lo), "hi": float(hi), "n": int(n), "kind": kind,
            "small_n": bool(n < 8), "excludes_zero": bool(lo > 0 or hi < 0),
            "note": ("n is small; treat as a tendency, not a precise estimate" if n < 8 else "")}


def min_of_k_shortfall(k: int, sd: float) -> float:
    """E[min] = mu - c_k * sd for k Gaussian coverages; c_4 = 1.029."""
    if k < 2:
        return 0.0
    rng = np.random.default_rng(0)
    return float(-np.mean(np.min(rng.standard_normal((200000, k)), axis=1)) * sd)


def simulate_min_coverage(n_cal, n_test, *, k_groups: int = 4, alpha: float = 0.1,
                          policy: str = "mondrian", n_draws: int = 4000, seed: int = 0) -> dict:
    """Worst-group coverage of an exactly valid procedure, from the conformal law alone.

    ``n_cal``/``n_test`` take a scalar or per-group counts. Two effects push the reported minimum
    below target: the minimum over k noisy coverages is biased low under any policy, and under
    ``mondrian`` each threshold comes from that group's own, possibly tiny, calibration sample.
    ``policy="marginal"`` pools the calibration scores and so isolates the first effect.
    """
    rng = np.random.default_rng(seed)
    ncal = np.full(k_groups, int(n_cal)) if np.isscalar(n_cal) else np.asarray(n_cal, dtype=int)
    ntest = np.full(k_groups, int(n_test)) if np.isscalar(n_test) else np.asarray(n_test, dtype=int)
    if ncal.size != k_groups or ntest.size != k_groups:
        raise ValueError(f"per-group counts must have length k_groups={k_groups}")
    if policy not in ("mondrian", "marginal"):
        raise ValueError(f"unknown policy {policy!r}")

    def q(s):
        n = s.size
        kk = math.ceil((n + 1) * (1 - alpha))
        return np.inf if kk > n else np.partition(s, kk - 1)[kk - 1]

    mins, means, per_group = [], [], []
    for _ in range(n_draws):
        cal = [rng.random(int(n)) for n in ncal]
        qs = [q(c) for c in cal] if policy == "mondrian" else [q(np.concatenate(cal))] * k_groups
        covs = [float((rng.random(int(ntest[j])) <= qs[j]).mean()) for j in range(k_groups)]
        mins.append(min(covs))
        means.append(float(np.mean(covs)))
        per_group.extend(covs)
    mins = np.asarray(mins)
    sd = float(np.std(per_group))
    return {"policy": policy, "n_cal_per_group": ncal.tolist(), "n_test_per_group": ntest.tolist(),
            "k_groups": int(k_groups), "target": 1 - alpha,
            "mean_over_groups": float(np.mean(means)), "expected_min": float(mins.mean()),
            "min_lo": float(np.percentile(mins, 2.5)), "min_hi": float(np.percentile(mins, 97.5)),
            "shortfall": float((1 - alpha) - mins.mean()), "per_group_sd": sd,
            "shortfall_over_sd": float(((1 - alpha) - mins.mean()) / sd) if sd > 0 else float("nan")}


def group_counts_at_rho(n_total: int, rho: float, k_groups: int = 4) -> np.ndarray:
    """Groups ordered 2y + a; each aligned group holds rho/2 of the pool, each other (1-rho)/2."""
    if k_groups != 4:
        raise ValueError("group_counts_at_rho is defined for the 4-group (y, a) layout")
    maj, mino = rho / 2.0, (1.0 - rho) / 2.0
    return np.round(np.array([maj, mino, mino, maj]) * n_total).astype(int)
