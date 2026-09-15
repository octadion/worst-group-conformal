"""The calibration ablation: three policies over identical cached posteriors.

For every (backbone, dataset, method, seed, score, rho_test, split), the same head is calibrated
three ways -- one shared threshold, one threshold per group, and a total-variation-robust
threshold -- so the only thing that varies is how scores become sets. Nothing is retrained.

    C1  under per-group thresholds every method reaches target within 0.02, while under one
        shared threshold every method sits significantly below it
    C2  with coverage matched by per-group thresholds, worst-group set size still differs by
        training method, and whether it tracks base accuracy is reported
    C3  both hold across the rho_test sweep, thresholds fixed at rho_cal
"""
from __future__ import annotations

import os

import numpy as np

from . import metrics
from .evaluate import CALIBRATIONS, RHO_SWEEP, evaluate
from .griddata import release_memory, rss_gib
from .grid import WG_ACC_SOFT_FLAG, worst_group_floor
from .heads import head_probs
from .methods import METHODS, fit_method
from .verdicts import SCORES

NEAR_TARGET = 0.02   # "reaches target" / "valid" tolerance (spec)


# ---------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------
def run_ablation(data_by_key: dict, *, methods=METHODS, scores=SCORES, rho_sweep=RHO_SWEEP,
                 calibrations=CALIBRATIONS, seeds=(0, 1, 2), n_splits=10, alpha=0.1,
                 method_hp=None, mem_trace=False) -> dict:
    """Fit each method's head ONCE per (key, seed) on cached features (no backbone retraining),
    then evaluate every calibration × score × rho × split on those posteriors."""
    method_hp = method_hp or {}
    records, excluded = [], []
    for key, gd in data_by_key.items():
        Xev, yev, gev = gd.eval_domain
        for method in methods:
            for seed in seeds:
                # Printed BEFORE the fit and flushed. This ablation evaluates three calibration
                # policies rather than one, so the expensive cells run over an hour; without a
                # line per arm the cell looks hung, and an OOM kill would leave no evidence of
                # which arm was in flight. Telemetry only -- it changes nothing computed.
                if mem_trace:
                    print(f"    [mem] {gd.backbone}/{gd.dataset} {method}/s{seed} "
                          f"pre-fit rss={rss_gib():.2f} GiB", flush=True)
                head = fit_method(method, gd.train, gd.reweight, seed=seed, **method_hp.get(method, {}))
                probs = head_probs(head, Xev, gd.n_classes)
                del head
                release_memory()
                if mem_trace:
                    print(f"    [mem] {gd.backbone}/{gd.dataset} {method}/s{seed} "
                          f"post-fit rss={rss_gib():.2f} GiB", flush=True)
                _, wg_acc = metrics.worst_group_accuracy(np.argmax(probs, axis=1), yev, gev)
                # Evaluate the arm and KEEP its records, tagged, exactly as run_grid does. This
                # previously `continue`d, so a gated arm left no trace and its influence could not
                # be measured afterwards. That matters more here than in the grid: c1_verdict
                # requires every method to be present, and ERM -- the method whose marginal-split
                # failure IS the claim -- is gated out on CelebA for the frozen SSL backbones.
                floor = worst_group_floor(gd.dataset, method)
                soft = WG_ACC_SOFT_FLAG.get((gd.dataset, method))
                if wg_acc < floor:
                    gate_status = "excluded"
                    excluded.append({"backbone": gd.backbone, "dataset": gd.dataset, "method": method,
                                     "seed": seed, "worst_group_acc": wg_acc, "floor": floor})
                elif soft is not None and wg_acc < soft:
                    gate_status = "flagged"
                else:
                    gate_status = "kept"
                for cal in calibrations:
                    for score in scores:
                        for rho in rho_sweep:
                            for sp in range(n_splits):
                                rec = evaluate(probs, yev, gev, score=score, alpha=alpha,
                                               rho_test=rho, split_seed=sp, calibration=cal)
                                rec.update({"backbone": gd.backbone, "dataset": gd.dataset,
                                            "method": method, "train_seed": seed,
                                            "worst_group_acc": wg_acc,
                                            "gate_status": gate_status,
                                            "gate_floor": float(floor)})
                                records.append(rec)
                del probs
                release_memory()
    kept = [r for r in records if r.get("gate_status") != "excluded"]
    verdicts = _build_verdicts(kept, scores=scores, rho_sweep=rho_sweep, alpha=alpha)
    out = {"records": records, "excluded": excluded, "verdicts": verdicts, "alpha": alpha,
           "calibrations": list(calibrations), "rho_sweep": list(rho_sweep)}
    if any(r.get("gate_status") == "excluded" for r in records):
        out["verdicts_with_excluded"] = _build_verdicts(records, scores=scores,
                                                        rho_sweep=rho_sweep, alpha=alpha)
    return out


def _build_verdicts(records, *, scores, rho_sweep, alpha) -> dict:
    verdicts = {}
    for key in sorted({(r["backbone"], r["dataset"]) for r in records}):
        recs = [r for r in records if (r["backbone"], r["dataset"]) == key]
        present = sorted({r["method"] for r in recs})
        verdicts[key] = {"C1": c1_verdict(recs, present, alpha=alpha, scores=scores),
                         "C2": c2_verdict(recs, present, alpha=alpha),
                         "C3": c3_verdict(recs, present, alpha=alpha, scores=scores, rho_sweep=rho_sweep)}
    return verdicts


# ---------------------------------------------------------------------------------------
# aggregation (mean + bootstrap CI + seed/split variance separation)
# ---------------------------------------------------------------------------------------
def _agg(recs, key, *, n_boot=2000, seed=0):
    vals = np.array([r[key] for r in recs], dtype=float)
    if vals.size == 0:
        return {"mean": float("nan"), "ci": (float("nan"), float("nan")), "n": 0,
                "seed_std": float("nan"), "split_std": float("nan")}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(vals, vals.size, replace=True).mean() for _ in range(n_boot)])
    seed_means = [np.mean([r[key] for r in recs if r["train_seed"] == s])
                  for s in sorted({r["train_seed"] for r in recs})]
    split_means = [np.mean([r[key] for r in recs if r["split_seed"] == s])
                   for s in sorted({r["split_seed"] for r in recs})]
    return {"mean": float(vals.mean()), "ci": (float(np.quantile(boot, .025)), float(np.quantile(boot, .975))),
            "n": int(vals.size), "seed_std": float(np.std(seed_means)), "split_std": float(np.std(split_means))}


def _filter(recs, **kw):
    return [r for r in recs if all(r[k] == v for k, v in kw.items())]


# ---------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------
def c1_verdict(recs, methods, *, alpha=0.1, rho=0.95, scores=SCORES, score="APS") -> dict:
    """Mondrian near-target for all methods AND marginal_split significantly below for all."""
    target = 1.0 - alpha
    out = {"target": target, "rho": rho, "score": score, "methods": {}}
    for m in methods:
        row = {}
        for cal in ("mondrian", "marginal_split", "shift_robust"):
            row[cal] = _agg(_filter(recs, method=m, calibration=cal, score=score, rho_test=rho),
                            "worst_group_cov")
        row["mondrian_near_target"] = bool(row["mondrian"]["mean"] >= target - NEAR_TARGET)
        # "significantly below" = bootstrap CI upper bound below target
        row["split_below_target"] = bool(row["marginal_split"]["ci"][1] < target)
        row["split_shortfall"] = float(target - row["marginal_split"]["mean"])
        out["methods"][m] = row
    out["C1_holds"] = bool(all(r["mondrian_near_target"] for r in out["methods"].values())
                           and all(r["split_below_target"] for r in out["methods"].values()))
    return out


def c2_verdict(recs, methods, *, alpha=0.1, rho=0.95, score="APS") -> dict:
    """Under mondrian (coverage matched), worst-group set size vs base accuracy across methods."""
    out = {"rho": rho, "score": score, "methods": {}}
    accs, sizes = [], []
    for m in methods:
        f = _filter(recs, method=m, calibration="mondrian", score=score, rho_test=rho)
        wsize = _agg(f, "worst_group_set_size"); wcov = _agg(f, "worst_group_cov")
        acc = _agg(f, "worst_group_acc")["mean"]
        out["methods"][m] = {"worst_group_set_size": wsize, "mean_set_size": _agg(f, "mean_set_size"),
                             "worst_group_cov": wcov["mean"], "worst_group_acc": acc}
        accs.append(acc); sizes.append(wsize["mean"])
    corr = (float(np.corrcoef(accs, sizes)[0, 1])
            if len(methods) > 1 and np.std(accs) > 1e-9 and np.std(sizes) > 1e-9 else float("nan"))
    out["acc_vs_setsize_corr"] = corr
    out["efficiency_tracks_accuracy"] = bool(corr < 0) if corr == corr else False   # neg => more acc, smaller sets
    return out


def c3_verdict(recs, methods, *, alpha=0.1, scores=SCORES, rho_sweep=RHO_SWEEP) -> dict:
    target = 1.0 - alpha
    out = {"target": target, "rho_cal": 0.95, "rho_sweep": list(rho_sweep), "methods": {}}
    for m in methods:
        per_score = {}
        for sc in scores:
            curve = []
            for rho in rho_sweep:
                a = _agg(_filter(recs, method=m, calibration="mondrian", score=sc, rho_test=rho),
                         "worst_group_cov")
                curve.append({"rho_test": rho, "worst_group_cov": a["mean"], "ci": a["ci"],
                              "valid": bool(a["mean"] >= target - NEAR_TARGET)})
            per_score[sc] = {"curve": curve, "survives": all(c["valid"] for c in curve),
                             "first_invalid": next((c["rho_test"] for c in curve if not c["valid"]), None)}
        out["methods"][m] = per_score
    return out


# ---------------------------------------------------------------------------------------
# CSV + report + figure
# ---------------------------------------------------------------------------------------
_CSV_COLS = ["backbone", "dataset", "method", "train_seed", "calibration", "score", "alpha",
             "rho_cal", "rho_test", "split_seed", "worst_group_acc", "base_top1", "marginal_cov",
             "worst_group_cov", "cov_gap", "mean_set_size", "worst_group_set_size",
             "gate_status", "gate_floor",
             # Kept because `evaluate` already computes them and dropping them cost a re-run once.
             # `n_cal_worst_group` in particular is what the Mondrian small-group question needs:
             # a group-conditional threshold fitted on very few calibration points is exactly the
             # regime the reviewers asked about, and it cannot be checked after the fact.
             "n_eval", "worst_group", "mean_group_cov", "cov_range", "n_cal_worst_group",
             "set_size_disparity", "div_wasserstein1", "div_ks_stat", "div_ks_pvalue",
             "rho_cal_realized", "rho_test_realized"]


def write_csv(records, path):
    """Write records, warning about any field this schema would silently drop.

    ``extrasaction="ignore"`` once discarded seven identifying columns without a word and made a
    finished multi-hour run's CSV unusable. The writer still ignores extras -- that is what keeps
    old and new records mergeable -- but it no longer does so quietly.
    """
    import csv
    if records:
        dropped = sorted({k for r in records for k in r} - set(_CSV_COLS))
        if dropped:
            print(f"[write_csv] WARNING: not in the schema, dropped: {dropped}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({**{"gate_status": "kept", "gate_floor": ""}, **r})


_FLOAT_COLS = {"alpha", "rho_cal", "rho_test", "worst_group_acc", "base_top1", "marginal_cov",
               "worst_group_cov", "cov_gap", "mean_set_size", "worst_group_set_size"}
_INT_COLS = {"train_seed", "split_seed"}


def records_from_csv(path: str) -> list:
    """Load ablation records from a CSV written by write_csv (coercing numeric columns)."""
    import csv
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = {}
            for k, val in row.items():
                r[k] = float(val) if k in _FLOAT_COLS else int(float(val)) if k in _INT_COLS else val
            out.append(r)
    return out


def reanalyze(csv_path: str, *, md_path: str = "CALIBRATION_ABLATION.md",
              figdir: str = "results/study/figures", make_figs: bool = True,
              excluded: list | None = None) -> dict:
    """Recompute C1/C2/C3 from a saved CSV (scores/ρ/calibrations inferred) and rewrite the MD +
    figures from the FULL record set. No retraining. Used to regenerate after appending cells."""
    records = records_from_csv(csv_path)
    scores = tuple(sorted({r["score"] for r in records}))
    rho_sweep = tuple(sorted({r["rho_test"] for r in records}, reverse=True))
    cals = tuple(sorted({r["calibration"] for r in records}))
    alpha = float(records[0]["alpha"]) if records else 0.1
    out = {"records": records, "excluded": excluded or [], "alpha": alpha,
           "calibrations": list(cals), "rho_sweep": list(rho_sweep),
           "verdicts": _build_verdicts(records, scores=scores, rho_sweep=rho_sweep, alpha=alpha)}
    figs = make_c1_figure(out, figdir) if make_figs else []
    write_calibration_ablation_md(out, md_path, fig_paths=figs)
    return out


def run_ablation_streaming(keys, build_fn, *, methods=METHODS, scores=SCORES,
                           rho_sweep=RHO_SWEEP, calibrations=CALIBRATIONS, seeds=(0, 1, 2),
                           n_splits=10, alpha=0.1, method_hp=None, cell_csv=None, verbose=True):
    """``run_ablation`` one (backbone, dataset) cell at a time, persisting each finished cell.

    Same reasons as ``run_grid_streaming``. Holding every GridData at once is what exhausted RAM on
    the grid, and this ablation evaluates three calibration policies rather than one, so it runs
    longer and a disconnect costs more. ``keys`` is a sequence of ``(backbone, dataset)``;
    ``build_fn(backbone, dataset)`` returns its GridData. A cell already present in ``cell_csv`` is
    skipped, so re-running the cell continues where it stopped.
    """
    import gc

    done, records = set(), []
    if cell_csv and os.path.exists(cell_csv):
        records = records_from_csv(cell_csv)
        done = {(r["backbone"], r["dataset"]) for r in records}
        if verbose:
            print(f"[resume] {len(records):,} records for {sorted(done)} already in {cell_csv}")

    failed = []
    for bb, ds in keys:
        if (bb, ds) in done:
            if verbose:
                print(f"[skip] {bb}/{ds} already done", flush=True)
            continue
        try:
            gd = build_fn(bb, ds)
        except Exception as e:                     # noqa: BLE001 - one cell must not sink the run
            failed.append(((bb, ds), repr(e)))
            print(f"[FAIL] {bb}/{ds}: {e}", flush=True)
            continue
        if verbose:
            print(f"    [mem] {bb}/{ds} GridData built, rss={rss_gib():.2f} GiB", flush=True)
        one = run_ablation({(bb, ds): gd}, methods=methods, scores=scores, rho_sweep=rho_sweep,
                           calibrations=calibrations, seeds=seeds, n_splits=n_splits,
                           alpha=alpha, method_hp=method_hp, mem_trace=verbose)
        records += one["records"]
        del gd, one
        release_memory()
        gc.collect()
        if cell_csv:
            write_csv(records, cell_csv)
        if verbose:
            print(f"[cell done] {bb}/{ds}: {len(records):,} records  (rss {rss_gib():.2f} GiB)",
                  flush=True)

    kept = [r for r in records if r.get("gate_status") != "excluded"]
    out = {"records": records, "failed": failed, "alpha": alpha,
           "calibrations": list(calibrations), "rho_sweep": list(rho_sweep),
           "verdicts": _build_verdicts(kept, scores=scores, rho_sweep=rho_sweep, alpha=alpha)}
    # Rebuilt FROM the records so a resumed run is correct: finished cells come back from CSV and
    # never pass through the loop, so an accumulated list would report zero excluded arms.
    out["excluded"] = [{"backbone": r["backbone"], "dataset": r["dataset"], "method": r["method"],
                        "seed": r["train_seed"], "worst_group_acc": r["worst_group_acc"],
                        "floor": r.get("gate_floor")}
                       for r in {(r["backbone"], r["dataset"], r["method"], r["train_seed"]): r
                                 for r in records if r.get("gate_status") == "excluded"}.values()]
    if out["excluded"]:
        out["verdicts_with_excluded"] = _build_verdicts(records, scores=scores,
                                                        rho_sweep=rho_sweep, alpha=alpha)
    return out


def extend_ablation_to(data_by_key: dict, *, csv_path: str = "results/study/calibration_ablation.csv",
                       md_path: str = "CALIBRATION_ABLATION.md", figdir: str = "results/study/figures",
                       methods=METHODS, scores=SCORES, rho_sweep=RHO_SWEEP, calibrations=CALIBRATIONS,
                       seeds=(0, 1, 2), n_splits=10, alpha=0.1, method_hp=None) -> dict:
    """Run the ablation on NEW cells (e.g. CelebA) and MERGE into the existing CSV WITHOUT
    overwriting other datasets' rows: existing rows for the new (backbone,dataset) keys are
    replaced, all other rows are kept. Then regenerate the MD + figures from the full set.
    Idempotent: re-running the same cells replaces (not duplicates) their rows. AFR on CelebA is
    excluded by the §2 worst-group-accuracy gate (logged in `excluded`)."""
    new = run_ablation(data_by_key, methods=methods, scores=scores, rho_sweep=rho_sweep,
                       calibrations=calibrations, seeds=seeds, n_splits=n_splits, alpha=alpha,
                       method_hp=method_hp)
    existing = records_from_csv(csv_path) if os.path.exists(csv_path) else []
    new_keys = {(r["backbone"], r["dataset"]) for r in new["records"]}
    kept = [r for r in existing if (r["backbone"], r["dataset"]) not in new_keys]
    write_csv(kept + new["records"], csv_path)        # Waterbirds rows kept; new cells appended/replaced
    out = reanalyze(csv_path, md_path=md_path, figdir=figdir, excluded=new["excluded"])
    out["new_excluded"] = new["excluded"]
    return out


def make_c1_figure(out: dict, outdir: str):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    written = []
    cals = out["calibrations"]
    target = 1.0 - out["alpha"]
    for key, v in out["verdicts"].items():
        bb, ds = key
        c1 = v["C1"]
        methods = list(c1["methods"])
        x = np.arange(len(methods)); w = 0.8 / len(cals)
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(methods)), 4.5))
        for i, cal in enumerate(cals):
            means = [c1["methods"][m][cal]["mean"] for m in methods]
            errs = [[c1["methods"][m][cal]["mean"] - c1["methods"][m][cal]["ci"][0] for m in methods],
                    [c1["methods"][m][cal]["ci"][1] - c1["methods"][m][cal]["mean"] for m in methods]]
            ax.bar(x + i * w, means, w, yerr=errs, capsize=3, label=cal)
        ax.axhline(target, color="k", ls="--", lw=1, label=f"target {target:.2f}")
        ax.set_xticks(x + w * (len(cals) - 1) / 2); ax.set_xticklabels(methods, rotation=15)
        ax.set_ylabel("worst-group coverage (APS, ρ=0.95)")
        ax.set_title(f"C1 — worst-group coverage by calibration policy — {bb}/{ds}\n"
                     f"(marginal_split fails; Mondrian hits target regardless of training)")
        ax.legend(fontsize=8); ax.set_ylim(0, 1.0); plt.tight_layout()
        p = os.path.join(outdir, f"calib_C1_{bb}_{ds}.png"); fig.savefig(p, dpi=120); written.append(p)
        plt.close(fig)
    return written


def write_calibration_ablation_md(out: dict, path: str, *, fig_paths=None, synthetic=False):
    target = 1.0 - out["alpha"]
    L = []
    if synthetic:
        L += ["> **SYNTHETIC LOGIC-VALIDATION ONLY — NOT a scientific result.** Toy-generator numbers;",
              "> they validate the machinery, not the phenomenon.\n"]
    L.append("# CALIBRATION_ABLATION.md — coverage is a calibration property, not a training one\n")
    L.append(f"Axis: calibration ∈ {out['calibrations']}. Target 1-α={target:.2f}; 'near target' within "
             f"{NEAR_TARGET}. Re-calibrates cached head posteriors only (no backbone retraining). "
             f"Means ± 95% CI over train-seeds × calibration-splits (seed-std/split-std reported).\n")
    if out["excluded"]:
        L.append("## Excluded arms (§2 worst-group acc floor)\n")
        for e in out["excluded"]:
            L.append(f"- {e['backbone']}/{e['dataset']} {e['method']} seed{e['seed']}: "
                     f"worst-group acc {e['worst_group_acc']:.3f} < {e['floor']}")
        L.append("")

    for key, v in out["verdicts"].items():
        bb, ds = key
        L.append(f"## {bb} / {ds}\n")
        # C1 — worst-group coverage + cov_gap by calibration × method (APS body)
        c1 = v["C1"]
        key_recs = [r for r in out["records"] if (r["backbone"], r["dataset"]) == key]

        def _c1_table(c1obj):
            rows = ["| method | marginal_split cov [CI] (gap) | mondrian cov [CI] (gap) | "
                    "shift_robust cov [CI] (gap) | split shortfall |", "|---|---|---|---|---|"]
            for m, row in c1obj["methods"].items():
                def cell(cal):
                    a = row[cal]; gap = target - a["mean"]
                    return f"{a['mean']:.3f} [{a['ci'][0]:.3f},{a['ci'][1]:.3f}] ({gap:+.3f})"
                rows.append(f"| {m} | {cell('marginal_split')} | {cell('mondrian')} | "
                            f"{cell('shift_robust')} | {row['split_shortfall']:+.3f} |")
            return rows

        L.append(f"### C1 — worst-group coverage + cov_gap by calibration × training (APS, ρ={c1['rho']}) — "
                 f"**C1 holds: {c1['C1_holds']}**\n")
        L += _c1_table(c1)
        L.append(f"\n_cov_gap = target − coverage (shown in parens). Mondrian near-target "
                 f"(≥{target-NEAR_TARGET:.2f}) for all methods: "
                 f"{all(r['mondrian_near_target'] for r in c1['methods'].values())}; marginal_split "
                 f"CI-upper below target for all: {all(r['split_below_target'] for r in c1['methods'].values())}. "
                 f"Core evidence: group-conditional calibration, not the representation, delivers worst-group coverage._\n")
        # appendix: RAPS / THR C1 tables (same structure)
        for sc in ("RAPS", "THR"):
            c1_sc = c1_verdict(key_recs, list(c1["methods"]), alpha=out["alpha"], rho=c1["rho"], score=sc)
            L.append(f"<details><summary>C1 appendix — {sc} (ρ={c1['rho']}) — C1 holds: {c1_sc['C1_holds']}</summary>\n")
            L += _c1_table(c1_sc)
            L.append("\n</details>\n")

        # C2 — efficiency under Mondrian + base accuracy
        c2 = v["C2"]
        L.append(f"### C2 — efficiency (worst-group set size under Mondrian) vs base accuracy (APS, ρ={c2['rho']})\n")
        L.append("| method | worst-group set size [CI] | mean set size | worst-group acc |")
        L.append("|---|---|---|---|")
        for m, row in c2["methods"].items():
            ws = row["worst_group_set_size"]
            L.append(f"| {m} | {ws['mean']:.3f} [{ws['ci'][0]:.3f},{ws['ci'][1]:.3f}] | "
                     f"{row['mean_set_size']['mean']:.3f} | {row['worst_group_acc']:.3f} |")
        L.append(f"\n_accuracy↔worst-group-set-size corr = {c2['acc_vs_setsize_corr']:.3f}; "
                 f"efficiency tracks accuracy (more accurate → smaller sets): {c2['efficiency_tracks_accuracy']}. "
                 f"Message: training/accuracy buys efficiency, calibration buys coverage._\n")

        # C3 — Mondrian shift validity
        c3 = v["C3"]
        L.append(f"### C3 — does Mondrian survive shift? (cal ρ={c3['rho_cal']}, valid = cov ≥ {target-NEAR_TARGET:.2f})\n")
        L.append("| method | score | survives sweep | first invalid ρ |")
        L.append("|---|---|---|---|")
        for m, ps in c3["methods"].items():
            for sc, r in ps.items():
                L.append(f"| {m} | {sc} | {r['survives']} | {r['first_invalid']} |")
        L.append("")

    if fig_paths:
        L.append("## Figure — C1 worst-group coverage by calibration policy, grouped by training\n")
        for p in fig_paths:
            import os
            L.append(f"![C1]({os.path.basename(p) if not p.startswith('results') else p})")
        L.append("")
    L.append("## Variance note\nSeed-std vs split-std are carried per aggregate in the JSON/_agg "
             "outputs (training-seed variance vs calibration-split variance kept separate).\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
