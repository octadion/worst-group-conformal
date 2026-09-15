"""Select AFR's gamma on held-out data, as its published protocol prescribes.

The grid and the ablation fit AFR at a fixed gamma = 2. On CelebA that reweighting inverts the
predicted class prior, so a majority cell becomes the worst group; the effect is monotone in
gamma and gamma = 1 improves over ERM there.

Gamma is chosen on a group-stratified half of the reweighting split, and the evaluation domain is
read only afterwards, for reporting. Selection by worst-group accuracy uses group labels: AFR is
group-label-free in training, and its published protocol likewise selects on a small
group-labelled set.

The selection split is carved from ``d_learn`` ONLY. ``eval_domain`` is never touched during
selection -- it is read once, afterwards, for reporting. Picking gamma from the eval numbers in the
diagnostic would be test-set selection and would invalidate everything downstream.
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .methods import fit_afr

__all__ = ["stratified_split", "select_afr_gamma", "fit_afr_tuned", "report_afr_tuning",
           "SELECTABLE_GAMMAS", "DIAGNOSTIC_GAMMAS"]

# gamma = 0 reduces AFR to an unweighted fit on the reweighting split, which is ERM's own arm, so
# it is a diagnostic control rather than a candidate.
SELECTABLE_GAMMAS = (0.5, 1.0, 2.0, 4.0)
DIAGNOSTIC_GAMMAS = (0.0,) + SELECTABLE_GAMMAS
DEFAULT_GAMMAS = SELECTABLE_GAMMAS


def stratified_split(group, *, val_frac=0.5, seed=0):
    """Split indices group-by-group so the validation half contains minority examples.

    A plain random split of CelebA's reweighting set would leave the 81-example minority group
    represented by chance alone. Stratifying guarantees each group is present in both halves; it
    does not make the minority half large, which is why ``select_afr_gamma`` reports that count.
    """
    g = np.asarray(group)
    rng = np.random.default_rng(seed)
    fit_idx, val_idx = [], []
    for grp in np.unique(g):
        idx = np.where(g == grp)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * idx.size))) if idx.size > 1 else 0
        val_idx.append(idx[:n_val])
        fit_idx.append(idx[n_val:])
    return np.sort(np.concatenate(fit_idx)), np.sort(np.concatenate(val_idx))


def select_afr_gamma(X_rw, y_rw, g_rw, *, gammas=DEFAULT_GAMMAS, val_frac=0.5, seed=0,
                     **afr_kw):
    """Pick gamma by worst-group accuracy on a held-out half of the reweighting split.

    Returns the chosen gamma plus every candidate's validation score and the size of the smallest
    validation group, because that count is what decides whether the choice means anything. An
    earlier attempt at model selection in this project was made on 14 minority examples and the
    noisy argmax made results WORSE; the fix is not to hide the number but to report it.
    """
    X_rw = np.asarray(X_rw)
    y_rw = np.asarray(y_rw)
    g_rw = np.asarray(g_rw)
    fit_i, val_i = stratified_split(g_rw, val_frac=val_frac, seed=seed)

    groups, counts = np.unique(g_rw[val_i], return_counts=True)
    n_val_min = int(counts.min())
    # Binomial SE of a per-group accuracy at its worst case, as a floor on the selection noise.
    se_min = float(np.sqrt(0.25 / n_val_min)) if n_val_min else float("nan")

    per_gamma = {}
    for gam in gammas:
        head = fit_afr(X_rw[fit_i], y_rw[fit_i], gamma=gam, seed=seed, **afr_kw)
        pred = np.argmax(head.predict_proba(X_rw[val_i]), axis=1)
        _, wg = metrics.worst_group_accuracy(pred, y_rw[val_i], g_rw[val_i])
        per_gamma[gam] = float(wg)

    best = max(per_gamma, key=lambda k: (per_gamma[k], -k))   # ties -> smaller gamma (less extreme)
    return {"gamma": best, "val_scores": per_gamma, "n_val_min_group": n_val_min,
            "val_se_min_group": se_min, "n_fit": int(fit_i.size), "n_val": int(val_i.size)}


def fit_afr_tuned(X_rw, y_rw, g_rw, *, gammas=DEFAULT_GAMMAS, val_frac=0.5, seed=0, **afr_kw):
    """Select gamma on held-out data, then refit AFR on the FULL reweighting split with it.

    Refitting on everything after selection is the standard two-stage protocol: the split exists to
    choose gamma, not to shrink the training set the final head sees.
    """
    sel = select_afr_gamma(X_rw, y_rw, g_rw, gammas=gammas, val_frac=val_frac, seed=seed, **afr_kw)
    head = fit_afr(X_rw, y_rw, gamma=sel["gamma"], seed=seed, **afr_kw)
    return head, sel


def report_afr_tuning(gd, *, gammas=DEFAULT_GAMMAS, seeds=(0, 1, 2), val_frac=0.5, verbose=True):
    """Fixed-gamma vs selected-gamma AFR for one GridData, with the gate floor for context.

    ``eval_domain`` is read only here, after gamma is already fixed by the validation split.
    """
    from .grid import worst_group_floor

    Xrw, yrw, grw = gd.reweight
    Xev, yev, gev = gd.eval_domain
    floor = worst_group_floor(gd.dataset, "afr")

    def wg_on_eval(head):
        return float(metrics.worst_group_accuracy(
            np.argmax(head.predict_proba(Xev), axis=1), yev, gev)[1])

    rows = []
    for seed in seeds:
        fixed = wg_on_eval(fit_afr(Xrw, yrw, gamma=2.0, seed=seed))
        head, sel = fit_afr_tuned(Xrw, yrw, grw, gammas=gammas, val_frac=val_frac, seed=seed)
        rows.append({"seed": seed, "fixed_gamma_2.0": fixed, "selected_gamma": sel["gamma"],
                     "tuned_wg_eval": wg_on_eval(head), "val_scores": sel["val_scores"],
                     "n_val_min_group": sel["n_val_min_group"],
                     "val_se_min_group": sel["val_se_min_group"]})

    # Oracle bound: each gamma fitted on all of d_learn and scored on the evaluation domain, then
    # the best taken. This is test-set selection and is never AFR's reported result; it bounds what
    # any selection rule could reach. The validation selector is noisy here by construction, with
    # about 40 minority examples (SE ~0.079) against candidate gaps of ~0.05.
    oracle = {}
    for gam in gammas:
        oracle[gam] = wg_on_eval(fit_afr(Xrw, yrw, gamma=gam, seed=seeds[0]))
    best_gamma = max(oracle, key=oracle.get)

    out = {"backbone": gd.backbone, "dataset": gd.dataset, "floor": float(floor), "rows": rows,
           "fixed_mean": float(np.mean([r["fixed_gamma_2.0"] for r in rows])),
           "tuned_mean": float(np.mean([r["tuned_wg_eval"] for r in rows])),
           "oracle_by_gamma": oracle, "oracle_best_gamma": best_gamma,
           "oracle_best_wg": float(oracle[best_gamma])}
    out["fixed_passes_floor"] = bool(out["fixed_mean"] >= floor)
    out["tuned_passes_floor"] = bool(out["tuned_mean"] >= floor)
    out["oracle_passes_floor"] = bool(out["oracle_best_wg"] >= floor)

    if verbose:
        r0 = rows[0]
        print(f"\n=== AFR gamma selection: {gd.backbone} / {gd.dataset} ===")
        print(f"  reweight split: {r0['n_val_min_group']} examples in the smallest VALIDATION "
              f"group (binomial SE <= {r0['val_se_min_group']:.3f})")
        if r0["val_se_min_group"] > 0.05:
            print("  [warn] that SE is large; the selection is noisy and must be reported as such")
        print(f"  {'seed':>4s} {'chosen':>7s} {'wg eval @gamma=2':>17s} {'wg eval @chosen':>16s}"
              f"   validation scores by gamma")
        for r in rows:
            vs = ", ".join(f"{g}:{v:.3f}" for g, v in sorted(r["val_scores"].items()))
            print(f"  {r['seed']:4d} {r['selected_gamma']:7.1f} {r['fixed_gamma_2.0']:17.3f} "
                  f"{r['tuned_wg_eval']:16.3f}   {vs}")
        print(f"  mean: fixed={out['fixed_mean']:.3f} -> tuned={out['tuned_mean']:.3f}   "
              f"gate floor={floor:.2f}   "
              f"passes: fixed={out['fixed_passes_floor']} tuned={out['tuned_passes_floor']}")
        print(f"  ORACLE upper bound (best gamma chosen ON EVAL -- not reportable as a result, "
              f"only as a bound):")
        print(f"    " + ", ".join(f"{g}:{v:.3f}" for g, v in sorted(out["oracle_by_gamma"].items()))
              + f"  -> best gamma={out['oracle_best_gamma']} wg={out['oracle_best_wg']:.3f}"
              f"  passes floor={out['oracle_passes_floor']}")
    return out
