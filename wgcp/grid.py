"""The main grid: one conformal evaluation per (backbone, dataset, method, seed, score, rho, split).

Records are written long, one row per evaluation, and the pre-specified verdicts are then run per
(backbone, dataset). Features arrive pre-extracted in a ``GridData``, so nothing here needs torch.

Every arm passes a per-method worst-group accuracy gate before it is used: an arm below its floor
is excluded with its reason recorded, never silently dropped.
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .evaluate import RHO_SWEEP, evaluate
from .griddata import GridData, release_memory, rss_gib
from .heads import head_probs
from .methods import METHODS, fit_method
from .verdicts import SCORES, h1_verdict, h2_verdict, h3_verdict

# Worst-group accuracy floors per (dataset, method). Floors are hard; the soft flag marks an arm
# that is kept but reported as below an advisory reference.
_DEFAULT_FLOOR = {"erm": 0.30, "dfr": 0.45, "afr": 0.40, "groupdro_ll": 0.45,
                  "balanced_subsample": 0.40}
_CELEBA_FLOOR = {"erm": 0.30, "dfr": 0.80, "afr": 0.75, "groupdro_ll": 0.75,
                 "balanced_subsample": 0.75}
WG_ACC_FLOOR = {"waterbirds": _DEFAULT_FLOOR, "celeba": _CELEBA_FLOOR}
WG_ACC_SOFT_FLAG = {("celeba", "dfr"): 0.85}   # (dataset, method) -> warn-if-below (no exclusion)


def worst_group_floor(dataset: str, method: str) -> float:
    return WG_ACC_FLOOR.get(dataset, _DEFAULT_FLOOR).get(method, 0.0)


def run_grid(data_by_key: dict, *, methods=METHODS, scores=SCORES, rho_sweep=RHO_SWEEP,
             seeds=(0, 1, 2), n_splits=10, alpha=0.1, method_hp=None,
             mem_trace=False) -> dict:
    """Run the full last-layer grid over the provided (backbone, dataset) GridData objects.

    Returns {"records": [...], "excluded": [...], "verdicts": {key: {h1,h2,h3}}}. ``method_hp``
    optionally maps method name -> dict of hyperparameters.
    """
    method_hp = method_hp or {}
    records, excluded, flagged = [], [], []

    for key, gd in data_by_key.items():
        Xev, yev, gev = gd.eval_domain
        for method in methods:
            for seed in seeds:
                hp = method_hp.get(method, {})
                # Printed BEFORE the fit and flushed: an OOM kill leaves no traceback, so the last
                # line on stdout is the only evidence of which arm was in flight and how much was
                # already resident. Telemetry only -- it never affects what is computed.
                if mem_trace:
                    print(f"    [mem] {gd.backbone}/{gd.dataset} {method}/s{seed} "
                          f"pre-fit rss={rss_gib():.2f} GiB", flush=True)
                head = fit_method(method, gd.train, gd.reweight, seed=seed, **hp)
                if mem_trace:
                    print(f"    [mem] {gd.backbone}/{gd.dataset} {method}/s{seed} "
                          f"post-fit rss={rss_gib():.2f} GiB", flush=True)
                probs = head_probs(head, Xev, gd.n_classes)
                # The head is dead once `probs` exists -- every record below is derived from
                # `probs`, `yev` and `gev` alone. Releasing it here rather than at the next
                # reassignment is what keeps the arms from accumulating.
                del head
                release_memory()
                if mem_trace:
                    print(f"    [mem] {gd.backbone}/{gd.dataset} {method}/s{seed} "
                          f"post-probs rss={rss_gib():.2f} GiB", flush=True)
                y_pred = np.argmax(probs, axis=1)
                _, wg_acc = metrics.worst_group_accuracy(y_pred, yev, gev)

                # §2 per-method worst-group accuracy gate. The arm is still EVALUATED and its
                # records kept, tagged ``gate_status="excluded"``; the verdicts drop them, but they
                # are in the CSV so the sensitivity analysis reviewers asked for (R2.4) is possible
                # without a re-run. Previously this `continue`d, so a failed arm left no trace in
                # the records at all and its influence could not be measured after the fact.
                floor = worst_group_floor(gd.dataset, method)
                soft = WG_ACC_SOFT_FLAG.get((gd.dataset, method))
                if wg_acc < floor:
                    gate_status = "excluded"
                    excluded.append({"backbone": gd.backbone, "dataset": gd.dataset,
                                     "method": method, "seed": seed, "worst_group_acc": wg_acc,
                                     "floor": floor, "reason": "below per-method worst-group acc floor"})
                elif soft is not None and wg_acc < soft:
                    gate_status = "flagged"          # soft warning band: reported, NOT excluded
                    flagged.append({"backbone": gd.backbone, "dataset": gd.dataset,
                                    "method": method, "seed": seed, "worst_group_acc": wg_acc,
                                    "expected_min": soft,
                                    "reason": "below expected worst-group acc (kept; flag for review)"})
                else:
                    gate_status = "kept"

                for score in scores:
                    for rho in rho_sweep:
                        for sp in range(n_splits):
                            rec = evaluate(probs, yev, gev, score=score, alpha=alpha,
                                           rho_test=rho, split_seed=sp)
                            rec.update({"backbone": gd.backbone, "dataset": gd.dataset,
                                        "method": method, "train_seed": seed,
                                        "worst_group_acc": wg_acc, "gate_status": gate_status,
                                        "gate_floor": float(floor)})
                            records.append(rec)
                del probs
                release_memory()

    # Verdicts keep their original meaning -- gated-out arms do not count -- but now by an
    # explicit filter rather than by the records never existing. ``verdicts_with_excluded`` is the
    # R2.4 sensitivity analysis: the same verdicts computed WITH the failed arms included.
    kept = [r for r in records if r.get("gate_status") != "excluded"]
    verdicts = build_verdicts(kept, scores=scores, rho_sweep=rho_sweep)
    out = {"records": records, "excluded": excluded, "flagged": flagged, "verdicts": verdicts}
    if excluded:
        out["verdicts_with_excluded"] = build_verdicts(records, scores=scores,
                                                       rho_sweep=rho_sweep)
    return out


def gate_lists_from_records(records):
    """Rebuild the (excluded, flagged) arm lists from tidy records.

    The records carry ``gate_status`` and ``gate_floor`` per arm, so the lists are a function of the
    records and do not depend on which pass produced them. That is what makes them correct after a
    resume, where finished cells are read back from CSV rather than recomputed.
    """
    seen, excluded, flagged = set(), [], []
    for r in records:
        status = r.get("gate_status")
        if status not in ("excluded", "flagged"):
            continue
        arm = (r["backbone"], r["dataset"], r["method"], r["train_seed"])
        if arm in seen:
            continue
        seen.add(arm)
        entry = {"backbone": r["backbone"], "dataset": r["dataset"], "method": r["method"],
                 "seed": r["train_seed"], "worst_group_acc": r["worst_group_acc"]}
        if status == "excluded":
            entry["floor"] = r.get("gate_floor")
            entry["reason"] = "below per-method worst-group acc floor"
            excluded.append(entry)
        else:
            entry["expected_min"] = WG_ACC_SOFT_FLAG.get((r["dataset"], r["method"]))
            entry["reason"] = "below expected worst-group acc (kept; flag for review)"
            flagged.append(entry)
    return excluded, flagged


def run_grid_streaming(keys, build_fn, *, methods=METHODS, scores=SCORES, rho_sweep=RHO_SWEEP,
                       seeds=(0, 1, 2), n_splits=10, alpha=0.1, method_hp=None,
                       cell_csv=None, verbose=True) -> dict:
    """Same grid, one (backbone, dataset) cell at a time, with each cell persisted.

    ``run_grid`` takes a dict of every GridData at once. At four backbones that is 3.3 GB of
    features held for the whole run, on top of the ~3.0 GB an arm transiently needs to fit a head
    on CelebA's 162,770 x 2048 train split -- together enough to exhaust a standard Colab runtime.
    Here only one cell is live at a time, and the loop reports its own peak RSS per cell.

    It is also resumable, which ``run_grid`` is not: the full grid measures ~5.5 h and a disconnect
    at hour five would otherwise lose everything. Each finished cell's records are written to
    ``cell_csv``, and a cell already present there is skipped on the next pass.

    ``keys`` is a sequence of ``(backbone, dataset)``; ``build_fn(backbone, dataset)`` returns its
    GridData.
    """
    import gc
    import os
    import threading
    import time

    # Memory telemetry. Three separate RAM crashes on the CelebA cells were each diagnosed by
    # reasoning about which arrays were live, and the first two diagnoses were WRONG -- the real
    # cost was a float64 upcast inside the L2 guard and a temporary inside numpy's std. Guessing
    # cost several multi-hour runs, so the loop now reports what it actually used.
    peak = [rss_gib()]
    _stop = threading.Event()

    def _sample():
        while not _stop.is_set():
            peak[0] = max(peak[0], rss_gib())
            _stop.wait(0.25)

    _sampler = threading.Thread(target=_sample, daemon=True)
    _sampler.start()

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
                print(f"[skip] {bb}/{ds} already done")
            continue
        try:
            gd = build_fn(bb, ds)
        except Exception as e:                     # noqa: BLE001 - one cell must not sink the run
            failed.append(((bb, ds), repr(e)))
            print(f"[FAIL] {bb}/{ds}: {e}", flush=True)
            continue
        if verbose:
            print(f"    [mem] {bb}/{ds} GridData built, rss={rss_gib():.2f} GiB", flush=True)
        one = run_grid({(bb, ds): gd}, methods=methods, scores=scores, rho_sweep=rho_sweep,
                       seeds=seeds, n_splits=n_splits, alpha=alpha, method_hp=method_hp,
                       mem_trace=verbose)
        records += one["records"]
        del gd, one
        gc.collect()
        if cell_csv:                               # persist after every cell, not at the end
            write_csv(records, cell_csv)
        if verbose:
            print(f"[cell done] {bb}/{ds}: {len(records):,} records total  "
                  f"(rss {rss_gib():.2f} GiB, peak {peak[0]:.2f} GiB)", flush=True)
        peak[0] = rss_gib()                       # reset so the next cell reports its own peak

    _stop.set()
    _sampler.join(timeout=2)

    # Derive the gate lists FROM THE RECORDS, not by accumulating them as cells finish. On a
    # resumed run the finished cells come back from the CSV and never pass through the loop, so an
    # accumulated list would report zero excluded arms -- which would silently break the R2.4
    # sensitivity analysis exactly when the run took more than one session.
    excluded, flagged = gate_lists_from_records(records)
    kept = [r for r in records if r.get("gate_status") != "excluded"]
    out = {"records": records, "excluded": excluded, "flagged": flagged, "failed": failed,
           "verdicts": build_verdicts(kept, scores=scores, rho_sweep=rho_sweep)}
    if any(r.get("gate_status") == "excluded" for r in records):
        out["verdicts_with_excluded"] = build_verdicts(records, scores=scores,
                                                       rho_sweep=rho_sweep)
    return out
def build_verdicts(records, *, scores=SCORES, rho_sweep=RHO_SWEEP) -> dict:
    """Compute H1/H2/H3 per (backbone, dataset) from tidy records. Used by run_grid AND reanalyze
    (so Tasks A/B re-run on an existing CSV with no retraining)."""
    keys = sorted({(r["backbone"], r["dataset"]) for r in records})
    verdicts = {}
    for key in keys:
        recs = [r for r in records if (r["backbone"], r["dataset"]) == key]
        present = sorted({r["method"] for r in recs})
        robust = [m for m in present if m != "erm"]
        if "erm" not in present or not robust:
            verdicts[key] = {"note": "ERM and/or robust arms missing/excluded; verdicts skipped",
                             "present_methods": present}
            continue
        verdicts[key] = {
            "h1": h1_verdict(recs, robust, scores=scores),
            "h2": h2_verdict(recs, present),
            "h3": h3_verdict(recs, robust, scores=scores, rho_sweep=rho_sweep),
        }
    return verdicts


# --------------------------------------------------------------------------------------
# tidy CSV + RESULTS_study.md emission
# --------------------------------------------------------------------------------------
# Every field a downstream re-analysis needs. Omissions here are silent data loss: the writer uses
# extrasaction="ignore", so a key absent from this list is dropped without a word. That is how a
# persisted representation run lost its representation/head/calibration columns.
_CSV_COLS = ["backbone", "dataset", "method", "train_seed", "gate_status", "gate_floor",
             "score", "calibration", "alpha", "rho_cal", "rho_test", "split_seed", "n_eval",
             "worst_group_acc", "base_top1", "marginal_cov",
             "worst_group", "worst_group_cov", "mean_group_cov", "cov_range",
             "n_cal_worst_group", "cov_gap", "mean_set_size",
             "worst_group_set_size", "set_size_disparity", "div_wasserstein1", "div_ks_stat",
             "div_ks_pvalue", "rho_cal_realized", "rho_test_realized"]


def write_csv(records, path):
    import csv
    dropped = sorted(set(records[0]) - set(_CSV_COLS)) if records else []
    blank = sorted(set(_CSV_COLS) - set(records[0])) if records else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    if dropped:                       # loud: a dropped column cannot be recovered from the file
        print(f"[warn] {path}: record fields NOT written (add them to _CSV_COLS): {dropped}")
    if blank:
        print(f"[warn] {path}: columns written blank (absent from the records): {blank}")


def write_results_md(out: dict, path: str, *, synthetic: bool = False):
    """Write RESULTS_study.md: accuracy-matched comparison UP FRONT, uncontrolled labeled,
    H2 rankings, H3 survival. Negatives stated plainly."""
    L = []
    if synthetic:
        L += ["> **SYNTHETIC LOGIC-VALIDATION ONLY — NOT a scientific result.** Numbers below come",
              "> from a toy generator and validate the reporting/analysis machinery, not the",
              "> phenomenon. Real numbers come from the Colab grid run.\n"]
    L.append("# RESULTS_study.md — Conformal Burden v2 (H1/H2/H3, last-layer arms)\n")
    if out["excluded"]:
        L.append("## Excluded arms (§2 worst-group accuracy gate)\n")
        for e in out["excluded"]:
            L.append(f"- {e['backbone']}/{e['dataset']} {e['method']} seed{e['seed']}: "
                     f"worst-group acc {e['worst_group_acc']:.3f} < floor {e['floor']} — {e['reason']}")
        L.append("")

    for key, v in out["verdicts"].items():
        bb, ds = key
        L.append(f"## {bb} / {ds}\n")
        if "note" in v:
            L.append(f"_{v['note']}_ (present: {v.get('present_methods')})\n")
            continue

        # H1 — accuracy-matched FIRST, uncontrolled labeled
        h1 = v["h1"]
        L.append(f"### H1 (Transfer) — bar: {h1['bar']} @ rho={h1['rho']}\n")
        L.append("**Accuracy-matched divergence (PRIMARY — confound-controlled):**\n")
        L.append("| method | GO | scores reducing | per-score Delta(a*) [CI] |")
        L.append("|---|---|---|---|")
        for m, mr in h1["methods"].items():
            cells = []
            for sc, r in mr["per_score"].items():
                if r.get("matched"):
                    cells.append(f"{sc}: {r['delta_matched']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]"
                                 f"{'*' if r['reduces'] else ''}")
                else:
                    cells.append(f"{sc}: unmatched ({r.get('reason','')})")
            L.append(f"| {m} | {'GO' if mr['GO'] else 'no'} | {mr['n_scores_reduce']}/{mr['n_scores']} | "
                     + "<br>".join(cells) + " |")
        L.append("\n_* = reduction CI excludes 0. Delta>0 = robust method lowers divergence at matched accuracy._\n")
        L.append("**Uncontrolled (raw) divergence — reported with base accuracy, NOT the verdict:**\n")
        L.append("| method | score | raw divergence | base top-1 |")
        L.append("|---|---|---|---|")
        for m, mr in h1["methods"].items():
            for sc, r in mr["per_score"].items():
                rr = r.get("raw_robust", {})
                L.append(f"| {m} | {sc} | {rr.get('divergence_mean', float('nan')):.4f} | "
                         f"{rr.get('base_top1_mean', float('nan')):.3f} |")
        # ERM reference row
        any_m = next(iter(h1["methods"].values()))
        for sc, r in any_m["per_score"].items():
            re = r.get("raw_ref", {})
            L.append(f"| erm (ref) | {sc} | {re.get('divergence_mean', float('nan')):.4f} | "
                     f"{re.get('base_top1_mean', float('nan')):.3f} |")
        L.append("")

        # H2 — ranking inversion WITH CIs (Task B)
        h2 = v["h2"]
        L.append("### H2 (Ranking inversion — headline)\n")
        L.append(f"- by worst-group **accuracy** (higher=better): {' > '.join(h2['ranking_by_accuracy'])} "
                 f"(top: **{h2['top_by_accuracy']}**)")
        L.append(f"- by worst-group **burden** ({h2['burden_key']}, lower=better): "
                 f"{' > '.join(h2['ranking_by_burden'])} (top: **{h2['top_by_burden']}**)")
        L.append("")
        L.append(f"| method | worst-group acc | {h2['burden_key']} [95% CI] |")
        L.append("|---|---|---|")
        for m in h2["ranking_by_accuracy"]:
            ci = h2["burden_ci"][m]
            L.append(f"| {m} | {h2['worst_group_acc'][m]:.3f} | "
                     f"{h2['burden'][m]:.4f} [{ci[0]:.4f},{ci[1]:.4f}] |")
        diff_ci = h2["inversion_diff_ci"]
        L.append("")
        L.append(f"- point-estimate inversion: **{h2['inversion_point']}**")
        L.append(f"- inversion REAL (burden of acc-top minus burden-top separated, CI excludes 0): "
                 f"**{h2['inversion_real']}** "
                 f"(Δ{h2['burden_key']}={h2['inversion_diff']:+.4f} [{diff_ci[0]:+.4f},{diff_ci[1]:+.4f}])")
        if h2['inversion_point'] and not h2['inversion_real']:
            L.append("  - point inversion but CIs OVERLAP → treat as noise, not a real inversion.")
        L.append("")

        # H3 — burden survival (Task A): divergence channel + set-size disparity trend, coverage separate
        h3 = v["h3"]
        L.append(f"### H3 (Shift survival) — calibrate ρ={h3['rho_cal']}, sweep {h3['rho_sweep']}\n")
        L.append(f"_What H3 measures (NOT coverage): {h3['criterion']}_\n")
        L.append("**(1) Divergence survival — does the H1 accuracy-matched reduction hold across ρ?**\n")
        L.append("| method | score | label |")
        L.append("|---|---|---|")
        for m, mm in h3["methods"].items():
            for sc, r in mm["per_score"].items():
                L.append(f"| {m} | {sc} | {r['failure_type']} |")
        L.append("\n_Labels: survived / never_held / held_then_broke@ρ / undefined@ρ. "
                 "'undefined@ρ' = matched comparison undefined where accuracy supports don't overlap "
                 "at that ρ — report as undefined, NOT as collapse._\n")
        L.append("**(2) Set-size disparity vs ρ — OBSERVED trend (data-driven, no assumed relocation):**\n")
        L.append("| method | disparity @ρ=0.95 | disparity @ρ=0.50 | trend | (robust − ERM) @ρ=0.50 |")
        L.append("|---|---|---|---|---|")
        for m, mm in h3["methods"].items():
            sd = {d["rho_test"]: d for d in mm["setsize_disparity_curve"]}
            lo_rho = min(h3["rho_sweep"]); hi_rho = max(h3["rho_sweep"])
            L.append(f"| {m} | {sd[hi_rho]['robust']:.3f} | {sd[lo_rho]['robust']:.3f} | "
                     f"{mm['setsize_trend']} | {sd[lo_rho]['robust_minus_erm']:+.3f} |")
        L.append("\n_'grows' = sets enlarge under shift; 'eases' = disparity is maximal at ρ_cal and "
                 "shrinks as ρ→0.5; 'flat' = unchanged. Where the trend is 'eases'/'flat', there is "
                 "NO relocation to report — do not impose the 'relocate, not remove' framing._\n")
        L.append("**Coverage stability (reported, NOT the H3 criterion):**\n")
        L.append("| method | mean worst-group cov | target 1-α | range over ρ | flat (<0.05) | sub-target |")
        L.append("|---|---|---|---|---|---|")
        for m, cs in h3["coverage_stability"].items():
            L.append(f"| {m} | {cs['mean']:.3f} | {cs['target']:.3f} | {cs['range']:.3f} | "
                     f"{cs['flat']} | {cs['sub_target']} |")
        L.append("\n_Read H3 from the data: e.g. worst-group coverage stable but sub-target, set-size "
                 "disparity maximal at ρ_cal and easing as ρ→0.5, matched-divergence survival undefined "
                 "→ no shift collapse and no relocation. State whatever the columns show._\n")

    if out.get("flagged"):
        L.append("## Flagged arms (soft worst-group warning — kept, not excluded)\n")
        for fl in out["flagged"]:
            L.append(f"- {fl['backbone']}/{fl['dataset']} {fl['method']} seed{fl['seed']}: "
                     f"worst-group acc {fl['worst_group_acc']:.3f} < expected {fl['expected_min']} "
                     f"— {fl['reason']}")
        L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------------------
# reanalysis from a saved CSV (Tasks A/B with NO retraining)
# --------------------------------------------------------------------------------------
_FLOAT_COLS = {"alpha", "rho_cal", "rho_test", "rho_test_realized", "worst_group_acc", "base_top1",
               "marginal_cov", "worst_group_cov", "cov_gap", "mean_set_size",
               "worst_group_set_size", "set_size_disparity", "div_wasserstein1", "div_ks_stat",
               "div_ks_pvalue", "gate_floor", "mean_group_cov", "cov_range",
               "rho_cal_realized"}
_INT_COLS = {"train_seed", "split_seed", "worst_group", "n_cal_worst_group", "n_eval"}
_STR_COLS = {"backbone", "dataset", "method", "score", "calibration", "gate_status"}


def records_from_csv(path: str) -> list:
    """Load tidy grid records from a CSV written by write_csv (coercing numeric columns)."""
    import csv
    recs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = {}
            for k, val in row.items():
                if val == "" or val is None:
                    r[k] = None
                elif k in _STR_COLS:
                    r[k] = val
                elif k in _FLOAT_COLS:
                    r[k] = float(val)
                elif k in _INT_COLS:
                    r[k] = int(float(val))
                else:
                    r[k] = val
            recs.append(r)
    return recs


def reanalyze(csv_path: str, *, results_md="RESULTS.md", scores=SCORES,
              rho_sweep=None) -> dict:
    """Recompute the verdicts from saved records; no retraining.

    The rho sweep is inferred from the records, so it matches whatever was run.
    """
    records = records_from_csv(csv_path)
    if rho_sweep is None:
        rho_sweep = tuple(sorted({r["rho_test"] for r in records}, reverse=True))
    out = {"records": records, "excluded": [], "flagged": [],
           "verdicts": build_verdicts(records, scores=scores, rho_sweep=rho_sweep)}
    write_results_md(out, results_md, synthetic=False)
    return out
