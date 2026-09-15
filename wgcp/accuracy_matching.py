"""Accuracy-matched divergence by pooling (seed, split) points. SUPERSEDED.

This is the matching the published analysis withdraws. It is kept because the H1 verdict fields
in the released records were produced with it; the reported comparison is
``intervals.model_level_delta``, which matches at model level, one point per training seed.

A more accurate model has different score distributions, so a raw divergence reduction can be
accuracy rather than reduced heterogeneity. Here each method's (base accuracy, divergence) points
are pooled across seeds and splits at a fixed (score, rho), restricted to the two methods'
overlapping accuracy support; each method's divergence at a common accuracy a* comes from a local
linear fit, and Delta(a*) = div_ERM(a*) - div_robust(a*), with a bootstrap over points.

Without overlapping support the comparison is infeasible and is reported as ``matched=False``
rather than extrapolated.
"""
from __future__ import annotations

import numpy as np

__all__ = ["raw_divergence", "matched_divergence"]


def _filter(records, method, score, rho):
    pts = [(r["base_top1"], r) for r in records
           if r["method"] == method and r["score"] == score and r["rho_test"] == rho]
    return pts


def raw_divergence(records, method, score, rho, metric="div_wasserstein1") -> dict:
    """Uncontrolled mean divergence + mean base accuracy for one (method, score, rho)."""
    pts = _filter(records, method, score, rho)
    if not pts:
        return {"method": method, "score": score, "rho": rho, "n": 0,
                "divergence_mean": float("nan"), "base_top1_mean": float("nan")}
    acc = np.array([a for a, _ in pts])
    div = np.array([r[metric] for _, r in pts])
    return {"method": method, "score": score, "rho": rho, "n": len(pts),
            "metric": metric, "divergence_mean": float(div.mean()),
            "divergence_std": float(div.std()), "base_top1_mean": float(acc.mean()),
            "base_top1_std": float(acc.std()), "label": "uncontrolled"}


def _div_at(acc: np.ndarray, div: np.ndarray, a_star: float) -> float:
    """Expected divergence at accuracy a* via a local linear fit (falls back to mean)."""
    if acc.size < 2 or np.ptp(acc) < 1e-9:
        return float(div.mean())
    m, c = np.polyfit(acc, div, 1)
    return float(m * a_star + c)


def matched_divergence(records, robust_method, score, rho, *, reference="erm",
                       metric="div_wasserstein1", n_boot=2000, ci=0.95, seed=0) -> dict:
    """Accuracy-matched divergence effect of ``robust_method`` vs ``reference`` at (score, rho).

    Returns a dict with matched flag, a_star, the matched effect Delta(a*) and its bootstrap CI,
    and ``reduces`` = (Delta>0 and CI excludes 0) i.e. the robust method lowers divergence at
    equal accuracy. ``reduces`` is the per-(score,rho) GO signal H1 aggregates over score fns.
    """
    pe = _filter(records, reference, score, rho)
    pr = _filter(records, robust_method, score, rho)
    out = {"robust_method": robust_method, "reference": reference, "score": score, "rho": rho,
           "metric": metric, "n_ref": len(pe), "n_robust": len(pr)}
    if not pe or not pr:
        return {**out, "matched": False, "reason": "missing points for one method"}

    acc_e = np.array([a for a, _ in pe]); div_e = np.array([r[metric] for _, r in pe])
    acc_r = np.array([a for a, _ in pr]); div_r = np.array([r[metric] for _, r in pr])

    lo = max(acc_e.min(), acc_r.min())
    hi = min(acc_e.max(), acc_r.max())
    if lo > hi:
        return {**out, "matched": False,
                "reason": "no overlapping accuracy support (cannot match base accuracy)",
                "ref_acc_range": [float(acc_e.min()), float(acc_e.max())],
                "robust_acc_range": [float(acc_r.min()), float(acc_r.max())]}
    a_star = 0.5 * (lo + hi)

    # point estimate at a*
    delta = _div_at(acc_e, div_e, a_star) - _div_at(acc_r, div_r, a_star)

    # bootstrap over points (resample each method's points independently)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    ie = np.arange(acc_e.size); ir = np.arange(acc_r.size)
    for b in range(n_boot):
        be = rng.choice(ie, size=ie.size, replace=True)
        br = rng.choice(ir, size=ir.size, replace=True)
        deltas[b] = _div_at(acc_e[be], div_e[be], a_star) - _div_at(acc_r[br], div_r[br], a_star)
    tail = (1.0 - ci) / 2.0
    ci_lo, ci_hi = np.quantile(deltas, [tail, 1.0 - tail])
    excludes_zero = bool(ci_lo > 0 or ci_hi < 0)
    return {**out, "matched": True, "a_star": float(a_star),
            "overlap": [float(lo), float(hi)], "delta_matched": float(delta),
            "ci": [float(ci_lo), float(ci_hi)], "ci_level": ci,
            "excludes_zero": excludes_zero, "reduces": bool(delta > 0 and ci_lo > 0)}
