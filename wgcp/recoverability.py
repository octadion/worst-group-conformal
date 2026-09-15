"""Does the representation that delivers worst-group coverage still carry the attribute?

Two measurements on the same frozen features, per (backbone, dataset): the test AUROC of a
standardised probe recovering the spurious attribute, and the worst-group coverage per-group
calibration reaches there (APS, rho = 0.95).

    tension_dead   coverage near target while the attribute stays recoverable, AUROC >= 0.65
    tension_alive  coverage only where recoverability falls to chance, AUROC <= 0.55
    ambiguous      otherwise

Part B, run when a cell is ambiguous or alive, projects out the top-k attribute-predictive
directions for k in {0, 1, 5, 20} and traces coverage against AUROC along that path: coverage
rising only as AUROC falls toward 0.5 is the tension surviving.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from conformal.group_robust import mondrian_build_sets, mondrian_quantiles
from conformal.scores import draw_randomization, scores_all, true_label_scores
from .resampling import resample_to_rho, split_pool

from .heads import assert_l2_normalized, fit_species_head, head_probs

RECOVER_HIGH = 0.65        # AUROC >= -> spurious attr is recoverable
RECOVER_CHANCE = 0.55      # AUROC <= -> recoverability near chance
COV_MARGIN = 0.03          # "near target" = worst-group cov >= (1-alpha) - COV_MARGIN
PART_B_KS = (0, 1, 5, 20)

__all__ = ["spurious_from_group", "recoverability_auroc", "mondrian_worst_group_coverage",
           "classify_tension", "run_part_a", "run_part_b", "write_recoverability_md",
           "make_part_b_figure", "needs_part_b"]


def spurious_from_group(group: np.ndarray) -> np.ndarray:
    """group_id = 2*y + spurious  ->  spurious = group % 2 (place for Waterbirds, Male for CelebA)."""
    return np.asarray(group).astype(int) % 2


def _ci95(x: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))


# ---------------------------------------------------------------------------------------
# recoverability: standardized probe predicting the spurious attribute
# ---------------------------------------------------------------------------------------
def recoverability_auroc(X_train, s_train, X_test, s_test, *, seeds=(0, 1, 2), n_boot=1000) -> dict:
    """In-domain train->test test-AUROC of a standardized probe predicting the spurious attribute.

    >=3 training seeds (bootstrapped train resamples -> refit) give per-seed AUROCs; the 95% CI is a
    bootstrap over the TEST set of the seed-averaged positive-class scores (finite-test uncertainty).
    """
    s_train = np.asarray(s_train).astype(int)
    s_test = np.asarray(s_test).astype(int)
    if len(np.unique(s_test)) < 2:
        return {"auroc_mean": float("nan"), "auroc_per_seed": [], "ci": (float("nan"), float("nan")),
                "note": "spurious attribute single-valued in test -> AUROC undefined"}
    per_seed, scores = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        bi = rng.choice(len(s_train), size=len(s_train), replace=True)   # bootstrap train
        clf = fit_species_head(X_train[bi], s_train[bi], seed=sd)        # StandardScaler -> LR
        pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
        p = clf.predict_proba(X_test)[:, pos_col]
        scores.append(p)
        per_seed.append(float(roc_auc_score(s_test, p)))
    mean_scores = np.mean(scores, axis=0)
    rng = np.random.default_rng(12345)
    boot = []
    n = len(s_test)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(s_test[idx])) < 2:
            continue
        boot.append(roc_auc_score(s_test[idx], mean_scores[idx]))
    return {"auroc_mean": float(np.mean(per_seed)), "auroc_per_seed": per_seed,
            "auroc_seed_std": float(np.std(per_seed)), "ci": _ci95(np.array(boot))}


# ---------------------------------------------------------------------------------------
# worst-group coverage on the representation (Mondrian / group-conditional, APS, rho=0.95)
# ---------------------------------------------------------------------------------------
def mondrian_worst_group_coverage(probs, y, group, *, alpha=0.1, rho=0.95, n_splits=10,
                                  score="APS") -> dict:
    """Best worst-group coverage achievable via group-conditional (Mondrian) calibration on the
    representation's posteriors. Cal/test resampled to rho from disjoint pools (reuses the harness)."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y); group = np.asarray(group)
    N = len(y)
    worst, marg = [], []
    for sp in range(n_splits):
        cal_pool, test_pool = split_pool(N, frac_cal=0.5, seed=sp)
        m = min(cal_pool.size, test_pool.size)
        cal_idx = cal_pool[resample_to_rho(group[cal_pool], rho, m, seed=sp * 2 + 1).idx]
        test_idx = test_pool[resample_to_rho(group[test_pool], rho, m, seed=sp * 2 + 2).idx]
        u_cal = draw_randomization(cal_idx.size, sp * 7 + 1)
        cal_true = true_label_scores(scores_all(score, probs[cal_idx], u=u_cal), y[cal_idx])
        gq = mondrian_quantiles(cal_true, group[cal_idx], alpha)
        u_test = draw_randomization(test_idx.size, sp * 7 + 2)
        memb = mondrian_build_sets(scores_all(score, probs[test_idx], u=u_test), group[test_idx], gq)
        covered = memb[np.arange(test_idx.size), y[test_idx]].astype(float)
        gt = group[test_idx]
        per_g = {int(g): float(covered[gt == g].mean()) for g in np.unique(gt)}
        worst.append(min(per_g.values()))
        marg.append(float(covered.mean()))
    worst = np.array(worst)
    return {"worst_group_cov_mean": float(worst.mean()), "worst_group_cov_ci": _ci95(worst),
            "worst_group_cov_std": float(worst.std()), "marginal_cov_mean": float(np.mean(marg)),
            "target": 1.0 - alpha}


def classify_tension(worst_cov_mean: float, auroc_mean: float, *, alpha=0.1) -> str:
    target = 1.0 - alpha
    good_cov = worst_cov_mean >= (target - COV_MARGIN)
    if good_cov and auroc_mean >= RECOVER_HIGH:
        return "tension_dead"
    if good_cov and auroc_mean <= RECOVER_CHANCE:
        return "tension_alive"
    return "ambiguous"


# ---------------------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------------------
def run_part_a(data_by_key: dict, *, seeds=(0, 1, 2), alpha=0.1, rho=0.95, n_splits=10) -> dict:
    """Per (backbone, dataset): recoverability AUROC + best worst-group Mondrian coverage + verdict."""
    cells = {}
    for key, gd in data_by_key.items():
        Xtr, ytr, gtr = gd.train
        Xev, yev, gev = gd.eval_domain
        assert_l2_normalized(Xtr, tag=f"{key} train (recoverability)")   # deployed features must be L2 (§2a)
        rec = recoverability_auroc(Xtr, spurious_from_group(gtr), Xev, spurious_from_group(gev),
                                   seeds=seeds)
        head = fit_species_head(Xtr, ytr, seed=0)                        # representation's class head
        probs = head_probs(head, Xev, gd.n_classes)
        cov = mondrian_worst_group_coverage(probs, yev, gev, alpha=alpha, rho=rho, n_splits=n_splits)
        verdict = classify_tension(cov["worst_group_cov_mean"], rec["auroc_mean"], alpha=alpha)
        cells[key] = {"recoverability": rec, "coverage": cov, "verdict": verdict,
                      "backbone": gd.backbone, "dataset": gd.dataset}
    return {"part": "A", "alpha": alpha, "rho": rho, "cells": cells}


def needs_part_b(part_a: dict) -> bool:
    return any(c["verdict"] in ("ambiguous", "tension_alive") for c in part_a["cells"].values())


# ---------------------------------------------------------------------------------------
# Part B — invariant-ize by projecting out top-k spurious-predictive directions
# ---------------------------------------------------------------------------------------
def _project_out(X_train_fit, s_train, X_list, k: int, *, seed=0):
    """Remove the top-k spurious-predictive linear directions (fit on train, applied to all). Each
    direction = normalized logistic weight on the deflated features (iterative deflation)."""
    Xt = np.asarray(X_train_fit, dtype=np.float64).copy()
    applied = [np.asarray(x, dtype=np.float64).copy() for x in X_list]
    removed = 0
    for _ in range(k):
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Xt, s_train)
        w = clf.coef_[0]
        nrm = np.linalg.norm(w)
        if nrm < 1e-12:
            break
        d = w / nrm
        Xt = Xt - np.outer(Xt @ d, d)
        applied = [a - np.outer(a @ d, d) for a in applied]
        removed += 1
    return applied, removed


def _deflation_path(X_train_fit, s_train, X_eval, ks, *, seed=0, chunk=8192):
    """Yield ``(k, removed, X_train_k, X_eval_k)`` for each k in ``ks``, deflating once.

    The directions and projected features are the ones ``_project_out`` gives when called
    separately for each k: direction j is fit on the features after the first j-1 have been
    removed, so the path to k=5 is the first five steps of the path to k=20. What changes is memory.
    ``_project_out`` restarts from scratch for every k, keeps two float64 copies of the training
    features that are deflated identically, and builds a full n-by-d outer product at every step.
    On CelebA's 162,770 x 2048 ResNet features those are 2.7 GB arrays, several alive at once, and
    that exhausted a 53 GB runtime. Here one working copy per split is updated in place, a row
    chunk at a time. The yielded arrays ARE the working copies: use them before advancing.
    """
    Xt = np.array(X_train_fit, dtype=np.float64, copy=True)
    Xe = np.array(X_eval, dtype=np.float64, copy=True)
    wanted = sorted({int(k) for k in ks})
    removed, stopped = 0, False
    for step in range(0, wanted[-1] + 1):
        if step > 0 and not stopped:
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Xt, s_train)
            w = clf.coef_[0]
            nrm = np.linalg.norm(w)
            del clf
            if nrm < 1e-12:
                stopped = True           # as in _project_out: nothing left to remove
            else:
                d = w / nrm
                for X in (Xt, Xe):
                    for i in range(0, X.shape[0], chunk):
                        blk = X[i:i + chunk]
                        blk -= np.outer(blk @ d, d)
                removed += 1
        if step in wanted:
            yield step, removed, Xt, Xe


def run_part_b(data_by_key: dict, *, ks=PART_B_KS, seeds=(0, 1, 2), alpha=0.1, rho=0.95,
               n_splits=10) -> dict:
    """For each cell and each k, measure recoverability AUROC + worst-group Mondrian coverage on the
    invariant-ized features, to trace the coverage-vs-auditability tradeoff."""
    import gc
    cells = {}
    for key, gd in data_by_key.items():
        Xtr, ytr, gtr = gd.train
        Xev, yev, gev = gd.eval_domain
        s_tr, s_ev = spurious_from_group(gtr), spurious_from_group(gev)
        by_k = {}
        for k, removed, Xtr_k, Xev_k in _deflation_path(Xtr, s_tr, Xev, ks):
            rec = recoverability_auroc(Xtr_k, s_tr, Xev_k, s_ev, seeds=seeds)
            head = fit_species_head(Xtr_k, ytr, seed=0)
            probs = head_probs(head, Xev_k, gd.n_classes)
            del head
            cov = mondrian_worst_group_coverage(probs, yev, gev, alpha=alpha, rho=rho, n_splits=n_splits)
            by_k[k] = {"k": k, "removed": removed, "auroc": rec["auroc_mean"],
                       "auroc_ci": rec["ci"], "worst_group_cov": cov["worst_group_cov_mean"],
                       "worst_group_cov_ci": cov["worst_group_cov_ci"]}
            gc.collect()
        curve = [by_k[int(k)] for k in ks]       # in the caller's order, as before
        gc.collect()
        aurocs = [c["auroc"] for c in curve]
        covs = [c["worst_group_cov"] for c in curve]
        # Tension ALIVE = you must DESTROY recoverability to GAIN coverage: coverage materially RISES
        # as AUROC falls toward chance. If coverage is already good at k=0 (high recoverability),
        # there is NO tension regardless of what happens to AUROC (Mondrian holds coverage anyway).
        neg_tradeoff = bool(covs[-1] > covs[0] + 0.02 and aurocs[-1] < aurocs[0] - 0.05)
        cells[key] = {"curve": curve, "negative_tradeoff": neg_tradeoff,
                      "good_cov_at_full_recoverability": bool(covs[0] >= (1.0 - alpha) - COV_MARGIN),
                      "backbone": gd.backbone, "dataset": gd.dataset}
    return {"part": "B", "ks": list(ks), "alpha": alpha, "rho": rho, "cells": cells}


# ---------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------
def make_part_b_figure(part_b: dict, outpath: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key, c in part_b["cells"].items():
        xs = [p["auroc"] for p in c["curve"]]
        ys = [p["worst_group_cov"] for p in c["curve"]]
        ax.plot(xs, ys, marker="o", label=f"{c['backbone']}/{c['dataset']}")
        for p in c["curve"]:
            ax.annotate(f"k={p['k']}", (p["auroc"], p["worst_group_cov"]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
    ax.axvline(0.5, color="k", ls=":", lw=0.8, label="chance AUROC")
    ax.axhline(1.0 - part_b["alpha"], color="grey", ls="--", lw=0.8, label="coverage target")
    ax.set_xlabel("recoverability AUROC (spurious attr)  [lower = less auditable]")
    ax.set_ylabel("worst-group coverage")
    ax.set_title("Part B — coverage vs auditability tradeoff (sweep k)")
    ax.legend(fontsize=7); plt.tight_layout()
    fig.savefig(outpath, dpi=120); plt.close(fig)
    return outpath


def write_recoverability_md(part_a: dict, path: str, *, part_b: dict | None = None,
                            fig_path: str | None = None, synthetic: bool = False):
    L = []
    if synthetic:
        L += ["> **SYNTHETIC LOGIC-VALIDATION ONLY — NOT a scientific result.** Numbers are from a toy",
              "> generator; they validate the machinery, not the phenomenon.\n"]
    L.append("# RECOVERABILITY.md — coverage–auditability tension\n")
    L.append(f"Target spurious attribute: place (Waterbirds) / Male (CelebA). Probe: StandardScaler→"
             f"LogisticRegression, in-domain train→test. Coverage: Mondrian APS @ρ={part_a['rho']}, "
             f"target 1-α={1-part_a['alpha']:.2f}. Verdict bars: recoverable AUROC≥{RECOVER_HIGH}, "
             f"chance≤{RECOVER_CHANCE}, coverage 'near target' within {COV_MARGIN}.\n")
    L.append("## Part A — per cell\n")
    L.append("| backbone | dataset | recoverability AUROC [95% CI] | worst-group cov [95% CI] | verdict |")
    L.append("|---|---|---|---|---|")
    for key, c in part_a["cells"].items():
        r = c["recoverability"]; cov = c["coverage"]
        rci, cci = r["ci"], cov["worst_group_cov_ci"]
        L.append(f"| {c['backbone']} | {c['dataset']} | {r['auroc_mean']:.3f} "
                 f"[{rci[0]:.3f},{rci[1]:.3f}] | {cov['worst_group_cov_mean']:.3f} "
                 f"[{cci[0]:.3f},{cci[1]:.3f}] | **{c['verdict']}** |")
    L.append("")
    verds = [c["verdict"] for c in part_a["cells"].values()]
    n_dead = verds.count("tension_dead"); n_alive = verds.count("tension_alive"); n_amb = verds.count("ambiguous")
    L.append(f"**Per-cell tally:** tension_dead={n_dead}, tension_alive={n_alive}, ambiguous={n_amb}.")
    overall = ("DEAD (every cell: good worst-group coverage coexists with recoverable spurious info)"
               if n_dead == len(verds) else
               "ALIVE (some cell: good coverage only at near-chance recoverability)"
               if n_alive and not n_dead else "MIXED / inconclusive — see per-cell + Part B")
    L.append(f"**Overall (Part A): coverage–auditability tension is {overall}.**\n")
    if not part_b:
        L.append("_Part B not run (no ambiguous/tension_alive cell), or pending human review after the "
                 "Part-A STOP._\n")
    else:
        L.append("## Part B — coverage vs recoverability across invariant-ization (project out top-k)\n")
        L.append(f"k swept over {part_b['ks']}. Tension = a clear negative tradeoff (coverage holds "
                 f"only as recoverability falls toward 0.5).\n")
        for key, c in part_b["cells"].items():
            L.append(f"### {c['backbone']} / {c['dataset']} — negative tradeoff: **{c['negative_tradeoff']}**\n")
            L.append("| k | dirs removed | recoverability AUROC | worst-group cov |")
            L.append("|---|---|---|---|")
            for p in c["curve"]:
                L.append(f"| {p['k']} | {p['removed']} | {p['auroc']:.3f} | {p['worst_group_cov']:.3f} |")
            L.append("")
        if fig_path:
            L.append(f"![Part B tradeoff]({fig_path})\n")
        any_neg = any(c["negative_tradeoff"] for c in part_b["cells"].values())
        L.append(f"**Overall (Part B): {'a negative coverage↔auditability tradeoff IS present (tension alive)' if any_neg else 'no negative tradeoff — coverage holds at high recoverability (tension dead)'}.**\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
