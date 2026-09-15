"""The representation-level test: fine-tune the backbone, then re-run the calibration comparison.

Two levers are put head to head. The representation lever is the best worst-group coverage
reachable under one shared threshold, maximising over representations and heads; the calibration
lever is the worst reached under per-group thresholds, minimising over the same. If the worst cell
of the calibration lever still beats the best cell of the representation lever, with intervals
separated, the dissociation holds at the representation level; if a robust representation closes
the shared-threshold gap on its own, it does not. ``representation_verdict`` returns whichever the
data give.

``build_repr_griddata`` needs a GPU; everything below it is numpy over records.
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .evaluate import evaluate
from .griddata import GridData
from .heads import head_probs
from .methods import fit_method
from .stats import (cluster_bootstrap_ci, correlation_ci, paired_cluster_diff_ci,
                    spread_equivalence)

REPR_OBJECTIVES = ("erm", "groupdro", "reweight")
REPR_HEADS = ("erm", "dfr", "groupdro_ll")
REPR_CALIBRATIONS = ("marginal_split", "mondrian")
REPR_SCORES = ("APS", "RAPS", "THR")

# Flatness margin for the Mondrian equivalence test (R3). 0.03 is three times the largest
# cross-training spread the paper reports under Mondrian (0.024) and below the smallest ERM lift
# it reports (0.06), so it is a margin that the claim can actually fail.
FLATNESS_MARGIN = 0.03


def build_repr_griddata(dataset: str, objective: str, cfg: dict, *, ft_seed: int = 0,
                        data_seed: int = 0, **ft_kwargs) -> GridData:
    """Fine-tune a backbone end-to-end under ``objective`` and wrap its features as GridData.

    ``data_seed`` fixes the split assignment while ``ft_seed`` varies the fine-tune, so seed-to-seed
    spread here is representation variance, not resampling variance. Requires torch + GPU.
    """
    from .griddata import _load_bundle
    from .finetune import finetune_features

    b = _load_bundle(dataset, cfg, data_seed)
    paths = {sp: b.meta["paths"][sp] for sp in ("train", "d_learn", "d_cal", "d_test")}
    y = {sp: np.asarray(b.y[sp]).astype(int) for sp in paths}
    grp = {sp: np.asarray(b.group_id[sp]).astype(int) for sp in paths}

    rcfg = dict(cfg.get("finetune", {}))
    rcfg.update(ft_kwargs)
    feats = finetune_features(paths, y, grp, objective=objective, tag=dataset, seed=ft_seed, **rcfg)

    eval_X = np.concatenate([feats["d_cal"], feats["d_test"]], axis=0)
    eval_y = np.concatenate([y["d_cal"], y["d_test"]])
    eval_g = np.concatenate([grp["d_cal"], grp["d_test"]])
    return GridData(
        backbone=f"ft_{objective}", dataset=dataset,
        train=(feats["train"], y["train"], grp["train"]),
        reweight=(feats["d_learn"], y["d_learn"], grp["d_learn"]),
        eval_domain=(eval_X, eval_y, eval_g), n_classes=int(b.n_classes),
    )


def records_for(gd: GridData, key: tuple, *, heads=REPR_HEADS, scores=REPR_SCORES,
                calibrations=REPR_CALIBRATIONS, head_seed=0, n_splits=10, alpha=0.1,
                rho_cal=0.95, rho_test=0.95, method_hp=None) -> list:
    """All records for ONE fine-tuned representation. Holds no reference to ``gd`` on return, so
    the caller can free a 1.5 GB CelebA GridData before building the next one."""
    method_hp = method_hp or {}
    dataset, repr_name, ft_seed = key
    Xev, yev, gev = gd.eval_domain
    out = []
    for head in heads:
        fitted = fit_method(head, gd.train, gd.reweight, seed=head_seed,
                            **method_hp.get(head, {}))
        probs = head_probs(fitted, Xev, gd.n_classes)
        _, wg_acc = metrics.worst_group_accuracy(np.argmax(probs, axis=1), yev, gev)
        for score in scores:
            for calibration in calibrations:
                for sp in range(n_splits):
                    rec = evaluate(probs, yev, gev, score=score, alpha=alpha,
                                   rho_cal=rho_cal, rho_test=rho_test, split_seed=sp,
                                   calibration=calibration)
                    rec.update({"dataset": dataset, "representation": repr_name,
                                "ft_seed": int(ft_seed), "head": head,
                                "backbone": gd.backbone, "worst_group_acc": float(wg_acc)})
                    out.append(rec)
    return out


def _finish(records, scores):
    return {"records": records,
            "verdicts": representation_verdict(records, scores=scores),
            "manipulation": manipulation_check(records)}


def run_representation_experiment(data_by_repr: dict, *, heads=REPR_HEADS, scores=REPR_SCORES,
                                  calibrations=REPR_CALIBRATIONS, head_seeds=(0,), n_splits=10,
                                  alpha=0.1, rho_cal=0.95, rho_test=0.95, method_hp=None) -> dict:
    """Run heads x calibrations x scores on each fine-tuned representation.

    ``data_by_repr`` maps ``(dataset, repr_name, ft_seed) -> GridData``. The fine-tune seed is the
    clustering unit for every interval downstream, so it is carried on every record.

    This form holds every GridData in memory at once. That is fine for Waterbirds (~0.1 GB each)
    but not for CelebA, where nine runs are ~14 GB -- use ``run_representation_streaming`` there.
    """
    records = []
    for key, gd in data_by_repr.items():
        records += records_for(gd, key, heads=heads, scores=scores, calibrations=calibrations,
                               head_seed=head_seeds[0], n_splits=n_splits, alpha=alpha,
                               rho_cal=rho_cal, rho_test=rho_test, method_hp=method_hp)
    return _finish(records, scores)


def run_representation_streaming(keys, build_fn, *, heads=REPR_HEADS, scores=REPR_SCORES,
                                 calibrations=REPR_CALIBRATIONS, head_seeds=(0,), n_splits=10,
                                 alpha=0.1, rho_cal=0.95, rho_test=0.95, method_hp=None,
                                 prior_records=(), verbose=True) -> dict:
    """Same experiment, but building and discarding one representation at a time.

    ``keys`` is a sequence of ``(dataset, repr_name, ft_seed)``; ``build_fn(*key)`` returns its
    GridData. Peak memory is one GridData rather than all of them, which is what makes CelebA
    feasible: nine CelebA GridData held together are ~14 GB, and the features are never needed
    again once their records exist.

    ``prior_records`` lets an earlier dataset's records (e.g. Waterbirds) be folded in without
    holding that dataset's features too. A build failure is logged and skipped, not raised, so one
    bad arm does not discard the runs that already succeeded.
    """
    import gc

    records = list(prior_records)
    failed = []
    for key in keys:
        try:
            gd = build_fn(*key)
        except Exception as e:                     # noqa: BLE001 - one arm must not sink the run
            failed.append((key, repr(e)))
            print(f"[FAIL] {key}: {e}", flush=True)
            continue
        records += records_for(gd, key, heads=heads, scores=scores, calibrations=calibrations,
                               head_seed=head_seeds[0], n_splits=n_splits, alpha=alpha,
                               rho_cal=rho_cal, rho_test=rho_test, method_hp=method_hp)
        del gd
        gc.collect()
        if verbose:
            print(f"[records] {key}: {len(records)} total", flush=True)
    out = _finish(records, scores)
    out["failed"] = failed
    return out


def _sel(records, **eq):
    return [r for r in records if all(r.get(k) == v for k, v in eq.items())]


def _vals(recs, field="worst_group_cov"):
    return (np.array([r[field] for r in recs], dtype=float),
            np.array([r["ft_seed"] for r in recs]))


def manipulation_check(records, *, reference: str = "erm", primary_head: str = "erm",
                       weak_margin: float = 0.05) -> dict:
    """Did the robust objectives actually produce more robust representations?

    This gates the interpretation of everything else. A null result ("changing the representation
    does not improve worst-group coverage") is evidence only if the representation really changed in
    the intended direction; if the robust arm is no better -- or worse -- than ERM on worst-group
    *accuracy*, the manipulation failed and the null says nothing about representations.

    Compares eval worst-group accuracy per (representation, head) against the reference
    representation, clustered on fine-tune seeds. Worst-group accuracy is a property of
    (representation, head, seed) alone, so records are de-duplicated to one value per cell first --
    otherwise the repeated splits/scores/calibrations in the record list would inflate n and
    manufacture significance.
    """
    out = {}
    for dataset in sorted({r["dataset"] for r in records}):
        sub = [r for r in records if r["dataset"] == dataset]
        heads = sorted({r["head"] for r in sub})
        reprs = sorted({r["representation"] for r in sub})
        if reference not in reprs:
            out[dataset] = {"verdict": f"no {reference!r} reference representation", "per_head": {}}
            continue

        def cell(rp, h):                       # one wg-acc per (repr, head, seed)
            by_seed = {}
            for r in sub:
                if r["representation"] == rp and r["head"] == h:
                    by_seed[r["ft_seed"]] = r["worst_group_acc"]

            seeds = np.array(sorted(by_seed))
            return np.array([by_seed[s] for s in seeds], dtype=float), seeds

        per_head, improved = {}, []
        for h in heads:
            ref_v, ref_s = cell(reference, h)
            row = {"reference_wg_acc": float(ref_v.mean()) if ref_v.size else float("nan")}
            for rp in reprs:
                if rp == reference:
                    continue
                v, s = cell(rp, h)
                if v.size and ref_v.size and v.size == ref_v.size and (s == ref_s).all():
                    d = paired_cluster_diff_ci(v, ref_v, s)
                    row[rp] = {"wg_acc": float(v.mean()), **d}
                    if d["point"] > 0 and d["excludes_zero"]:
                        improved.append((h, rp))
            per_head[h] = row
        # "Some head improved" is too weak a gate. The primary lever holds the head at
        # ``primary_head``, so the manipulation has to have worked THERE for that lever's null to
        # say anything about representations. On CelebA the best gain was +0.019 at the DFR head
        # while the plain head moved -0.002 (ns) -- a pass under the old rule, but the primary
        # lever was then comparing representations that are statistically identical.
        at_primary = [(h, rp) for (h, rp) in improved if h == primary_head]
        best_gain = max((v["point"] for row in per_head.values() for v in row.values()
                         if isinstance(v, dict)), default=float("nan"))
        primary_gain = max((v["point"] for k, v in per_head.get(primary_head, {}).items()
                            if isinstance(v, dict)), default=float("nan"))
        if at_primary:
            status = "PASS"
            note = (f"the manipulation works at the primary head ({primary_head}); the primary "
                    f"lever compares genuinely different representations")
        elif improved:
            status = "WEAK"
            note = (f"a robust objective beats the reference only at head(s) "
                    f"{sorted({h for h, _ in improved})}, NOT at the primary head "
                    f"{primary_head!r} (best gain there {primary_gain:+.3f}). The primary lever "
                    f"therefore compares near-identical representations, so its null is weak "
                    f"evidence about representations and must be reported as such")
        else:
            status = "FAIL"
            note = ("no robust objective beat the reference on eval worst-group accuracy; a null "
                    "coverage result here is UNINFORMATIVE about representations, not evidence "
                    "for the thesis")
        if status != "FAIL" and np.isfinite(best_gain) and best_gain < weak_margin:
            note += (f". Note the effect is small in absolute terms (best gain {best_gain:+.3f} "
                     f"< {weak_margin:.2f}), so the representation axis was barely moved")
        out[dataset] = {
            "per_head": per_head, "reference": reference, "primary_head": primary_head,
            "significantly_more_robust": [list(x) for x in improved],
            "improved_at_primary_head": bool(at_primary),
            "best_gain": float(best_gain), "primary_head_gain": float(primary_gain),
            "status": status, "verdict": f"{status} — {note}",
        }
    return out


def representation_verdict(records, *, scores=REPR_SCORES, margin: float = FLATNESS_MARGIN) -> dict:
    """Per (dataset, score): flatness within each representation, and the two-lever comparison."""
    out = {}
    datasets = sorted({r["dataset"] for r in records})
    for dataset in datasets:
        for score in scores:
            base = _sel(records, dataset=dataset, score=score)
            if not base:
                continue
            reprs = sorted({r["representation"] for r in base})
            heads = sorted({r["head"] for r in base})

            # --- per representation: is Mondrian flat across heads, and is marginal not?
            per_repr = {}
            for rp in reprs:
                by_head_m, by_head_g = {}, {}
                for h in heads:
                    mo = _sel(base, representation=rp, head=h, calibration="mondrian")
                    ma = _sel(base, representation=rp, head=h, calibration="marginal_split")
                    if mo:
                        by_head_m[h] = _vals(mo)
                    if ma:
                        by_head_g[h] = _vals(ma)
                per_repr[rp] = {
                    "mondrian_flatness": spread_equivalence(by_head_m, margin=margin) if by_head_m else None,
                    "marginal_spread": (float(np.ptp([v.mean() for v, _ in by_head_g.values()]))
                                        if len(by_head_g) > 1 else float("nan")),
                    "mondrian_by_head": {h: float(v.mean()) for h, (v, _) in by_head_m.items()},
                    "marginal_by_head": {h: float(v.mean()) for h, (v, _) in by_head_g.items()},
                }

            # --- the two levers.
            # The lever comparison must vary ONE axis at a time. Maximising the marginal side over
            # (representation x head) jointly lets a robust *head* stand in for the representation
            # -- and empirically it does: the best marginal cell on Waterbirds is the ERM
            # representation with a DFR head, i.e. the last-layer axis the original paper already
            # studied, not the representation axis the reviewers asked about. So the primary test
            # holds the head FIXED and varies only the representation; the joint version is kept as
            # a strictly harder secondary bound.
            def _lever(cells_marg, cells_mond, label):
                if not (cells_marg and cells_mond):
                    return {"verdict": "undetermined", "note": "need both calibration policies"}
                best_marg = max(cells_marg, key=lambda k: cells_marg[k][0].mean())
                worst_mond = min(cells_mond, key=lambda k: cells_mond[k][0].mean())
                bm_v, bm_s = cells_marg[best_marg]
                wm_v, wm_s = cells_mond[worst_mond]
                if best_marg == worst_mond:          # same cell -> paired difference is exact
                    diff = paired_cluster_diff_ci(wm_v, bm_v, wm_s)
                else:                                # different cells -> unpaired cluster CIs
                    a = cluster_bootstrap_ci(wm_v, wm_s)
                    b = cluster_bootstrap_ci(bm_v, bm_s)
                    diff = {"point": a["point"] - b["point"], "lo": a["lo"] - b["hi"],
                            "hi": a["hi"] - b["lo"], "n_seeds": min(a["n_seeds"], b["n_seeds"]),
                            "n_obs": a["n_obs"] + b["n_obs"], "method": a["method"] + " (unpaired)"}
                    diff["excludes_zero"] = bool(diff["lo"] > 0 or diff["hi"] < 0)
                # Three outcomes, not two. An earlier version tested only ``point > 0`` and labelled
                # everything else "competitive", which silently disguised the case that actually
                # contradicts the thesis: a CI-separated NEGATIVE difference, i.e. marginal
                # calibration beating Mondrian. That happens on CelebA once the head is robust, and
                # it has to be reported as a loss rather than a tie.
                sep = bool(diff.get("excludes_zero"))
                if diff["point"] > 0 and sep:
                    direction, verdict = "calibration_dominates", (
                        "CALIBRATION LEVER DOMINATES (worst Mondrian beats best marginal, "
                        "CI excludes 0)")
                elif diff["point"] < 0 and sep:
                    direction, verdict = "marginal_wins", (
                        "MARGINAL CALIBRATION WINS (CI excludes 0) — the dominance claim does NOT "
                        "hold in this cell")
                else:
                    direction, verdict = "indistinguishable", (
                        "INDISTINGUISHABLE (CI includes 0) — no dominance either way; the "
                        "representation lever is competitive here")
                return {
                    "scope": label,
                    "best_marginal_cell": list(best_marg) if isinstance(best_marg, tuple) else [best_marg],
                    "best_marginal_cov": float(bm_v.mean()),
                    "worst_mondrian_cell": list(worst_mond) if isinstance(worst_mond, tuple) else [worst_mond],
                    "worst_mondrian_cov": float(wm_v.mean()),
                    "diff_worst_mondrian_minus_best_marginal": diff,
                    "direction": direction,
                    "verdict": verdict,
                }

            marg_cells, mond_cells = {}, {}
            by_head = {}
            for h in heads:
                hm, hd = {}, {}
                for rp in reprs:
                    ma = _sel(base, representation=rp, head=h, calibration="marginal_split")
                    mo = _sel(base, representation=rp, head=h, calibration="mondrian")
                    if ma:
                        marg_cells[(rp, h)] = hm[rp] = _vals(ma)
                    if mo:
                        mond_cells[(rp, h)] = hd[rp] = _vals(mo)
                by_head[h] = _lever(hm, hd, f"representation-only (head fixed = {h})")

            lever_joint = _lever(marg_cells, mond_cells, "joint over representation x head")
            # The plain ERM head is the primary readout: no last-layer intervention, so the only
            # thing that differs between its cells is the representation itself.
            primary_head = "erm" if "erm" in by_head else (heads[0] if heads else None)
            lever = dict(by_head.get(primary_head, lever_joint))
            lever["primary_head"] = primary_head
            lever["by_head"] = by_head
            lever["joint"] = lever_joint

            # --- does a robust REPRESENTATION fix marginal calibration on its own?
            repr_effect = {}
            if "erm" in reprs:
                for rp in reprs:
                    if rp == "erm":
                        continue
                    for h in heads:
                        a = _sel(base, representation=rp, head=h, calibration="marginal_split")
                        b = _sel(base, representation="erm", head=h, calibration="marginal_split")
                        if a and b and len(a) == len(b):
                            av, asd = _vals(a)
                            bv, _ = _vals(b)
                            repr_effect[f"{rp}_vs_erm/{h}"] = paired_cluster_diff_ci(av, bv, asd)

            out[f"{dataset}/{score}"] = {"per_representation": per_repr, "levers": lever,
                                         "marginal_repr_effect": repr_effect}
    return out


# The grid module's write_csv has a FIXED column list built for the last-layer grid, and it drops
# unknown keys silently. Routing representation records through it lost representation / head /
# calibration / ft_seed / mean_group_cov -- i.e. every field needed to map a row back to its cell --
# so a persisted run could not be re-analysed without re-reading the features. These columns are
# the record's own identity plus the quantities the verdicts consume.
REPR_CSV_COLS = ["dataset", "representation", "ft_seed", "head", "backbone", "calibration",
                 "score", "alpha", "rho_cal", "rho_test", "split_seed", "n_eval",
                 "worst_group_acc", "base_top1", "marginal_cov", "worst_group",
                 "worst_group_cov", "mean_group_cov", "cov_range", "cov_gap",
                 "n_cal_worst_group", "mean_set_size", "worst_group_set_size",
                 "set_size_disparity", "div_wasserstein1", "div_ks_stat", "div_ks_pvalue",
                 "rho_cal_realized", "rho_test_realized"]
_REPR_INT = {"ft_seed", "split_seed", "worst_group", "n_cal_worst_group", "n_eval"}
_REPR_STR = {"dataset", "representation", "head", "backbone", "calibration", "score"}


def write_representation_csv(records, path: str) -> str:
    """Persist representation records with every identifying field intact."""
    import csv
    missing = set(REPR_CSV_COLS) - set(records[0]) if records else set()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REPR_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    if missing:                                  # loud, not silent: a dropped column is data loss
        print(f"[warn] {path}: columns absent from the records and written blank: "
              f"{sorted(missing)}")
    return path


def records_from_representation_csv(path: str) -> list:
    """Read back what ``write_representation_csv`` wrote, restoring types."""
    import csv
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = {}
            for k, v in row.items():
                if v == "" or v is None:
                    r[k] = None
                elif k in _REPR_STR:
                    r[k] = v
                elif k in _REPR_INT:
                    r[k] = int(float(v))
                else:
                    r[k] = float(v)
            out.append(r)
    return out


def reanalyze_representation(csv_path: str, *, md_path: str = "REPRESENTATION.md",
                             scores=REPR_SCORES) -> dict:
    """Recompute every verdict and rewrite the report from a saved CSV.

    Pure re-analysis: no GPU, no fine-tuning, and no re-reading of the cached features (8 GB on
    CelebA). This is the path to use whenever the analysis code changes but the runs have not.
    """
    records = records_from_representation_csv(csv_path)
    present = sorted({r["dataset"] for r in records})
    print(f"loaded {len(records)} records from {csv_path} | datasets: {present}")
    out = _finish(records, scores)
    write_representation_md(out, md_path)
    return out


def write_representation_md(out: dict, path: str = "REPRESENTATION.md") -> str:
    """Emit the representation-level report."""
    recs, V = out["records"], out["verdicts"]
    L = ["# Representation-level dissociation test",
         "",
         "End-to-end fine-tuned backbones (ERM / GroupDRO / group-balanced), each carrying the full",
         "head x calibration comparison. Answers ACML R1.2, R2.1, R3.1: the representation itself",
         "varies here, not only the last layer. Intervals are two-stage cluster bootstraps over",
         "fine-tune seeds (splits nested within seed), per R2.3.",
         ""]

    # Gate first: a null coverage result only counts as evidence if the manipulation worked.
    man = out.get("manipulation") or {}
    if man:
        L += ["## Manipulation check (READ FIRST)", "",
              "Did the robust objectives actually produce more robust representations? If not, the",
              "null coverage results below say nothing about representations and must not be read",
              "as support for the thesis.", ""]
        for dataset, d in sorted(man.items()):
            L.append(f"**{dataset}: {d['verdict']}**")
            L.append("")
            if d.get("status") == "WEAK":
                L.append(f"> The representation axis was barely moved on **{dataset}**: best gain "
                         f"{d.get('best_gain', float('nan')):+.3f} across heads, but only "
                         f"{d.get('primary_head_gain', float('nan')):+.3f} at the primary head "
                         f"`{d.get('primary_head')}`. Read this dataset's primary-lever null as "
                         f"weak evidence about representations, and say so in the write-up.")
                L.append("")
            if d.get("per_head"):
                L.append("| head | representation | eval wg acc | Δ vs " + d.get("reference", "erm")
                         + " [95% CI] | more robust |")
                L.append("|---|---|---|---|---|")
                for h, row in sorted(d["per_head"].items()):
                    ref = row.get("reference_wg_acc", float("nan"))
                    L.append(f"| {h} | {d.get('reference','erm')} (ref) | {ref:.3f} | — | — |")
                    for rp, v in sorted(row.items()):
                        if not isinstance(v, dict):
                            continue
                        sig = "yes" if (v["point"] > 0 and v["excludes_zero"]) else "no"
                        L.append(f"| {h} | {rp} | {v['wg_acc']:.3f} | {v['point']:+.3f} "
                                 f"[{v['lo']:+.3f}, {v['hi']:+.3f}] | {sig} |")
                L.append("")
    for key in sorted(V):
        v = V[key]
        L.append(f"## {key}")
        L.append("")
        L.append("| representation | Mondrian spread (across heads) | equivalence @ margin | marginal spread |")
        L.append("|---|---|---|---|")
        for rp, d in sorted(v["per_representation"].items()):
            f = d["mondrian_flatness"]
            if f is None:
                L.append(f"| {rp} | — | — | {d['marginal_spread']:.3f} |")
            else:
                L.append(f"| {rp} | {f['spread']:.3f} (hi {f['hi']:.3f}) | "
                         f"{'EQUIVALENT' if f['equivalent'] else 'not equivalent'} "
                         f"(±{f['margin']:.3f}) | {d['marginal_spread']:.3f} |")
        L.append("")
        lev = v["levers"]
        if "best_marginal_cell" in lev:
            d = lev["diff_worst_mondrian_minus_best_marginal"]
            L.append(f"**Primary lever comparison — representation only** (head held fixed at "
                     f"`{lev.get('primary_head')}`, so the only axis that varies is the "
                     f"representation). Best marginal representation "
                     f"`{lev['best_marginal_cell']}` = {lev['best_marginal_cov']:.3f}; "
                     f"worst Mondrian representation `{lev['worst_mondrian_cell']}` = "
                     f"{lev['worst_mondrian_cov']:.3f}. Difference "
                     f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] ({d['method']}, "
                     f"{d['n_seeds']} seeds).")
            L.append("")
            L.append(f"**Verdict: {lev['verdict']}**")
            L.append("")
            if lev.get("by_head"):
                L.append("Same comparison with each head held fixed:")
                L.append("")
                L.append("| head held fixed | best marginal | worst Mondrian | difference [CI] | verdict |")
                L.append("|---|---|---|---|---|")
                LABEL = {"calibration_dominates": "Mondrian dominates",
                         "marginal_wins": "**marginal WINS**",
                         "indistinguishable": "indistinguishable"}
                for h, hv in sorted(lev["by_head"].items()):
                    if "best_marginal_cell" not in hv:
                        continue
                    hd_ = hv["diff_worst_mondrian_minus_best_marginal"]
                    L.append(f"| {h} | {hv['best_marginal_cov']:.3f} "
                             f"({hv['best_marginal_cell'][0]}) | {hv['worst_mondrian_cov']:.3f} "
                             f"({hv['worst_mondrian_cell'][0]}) | {hd_['point']:+.3f} "
                             f"[{hd_['lo']:+.3f}, {hd_['hi']:+.3f}] | "
                             f"{LABEL.get(hv.get('direction'), '?')} |")
                L.append("")
            j = lev.get("joint")
            if j and "best_marginal_cell" in j:
                jd = j["diff_worst_mondrian_minus_best_marginal"]
                L.append(f"_Secondary (strictly harder) bound, maximising the marginal side jointly "
                         f"over representation x head: best marginal `{j['best_marginal_cell']}` = "
                         f"{j['best_marginal_cov']:.3f} vs worst Mondrian "
                         f"`{j['worst_mondrian_cell']}` = {j['worst_mondrian_cov']:.3f}, "
                         f"difference {jd['point']:+.3f} [{jd['lo']:+.3f}, {jd['hi']:+.3f}] "
                         f"-> **{j.get('direction')}**. This lets a robust **head** substitute for "
                         f"the representation, so it tests a different question than the reviewers "
                         f"asked -- but where it reads `marginal_wins`, the unconditional dominance "
                         f"claim fails and the paper must say so._")
                L.append("")
        else:
            L.append(f"_{lev.get('note', lev['verdict'])}_")
        L.append("")
        if v["marginal_repr_effect"]:
            L.append("**Does a robust representation fix marginal calibration by itself?** "
                     "(paired, vs the ERM representation, same head)")
            L.append("")
            L.append("| comparison | Δ worst-group cov [95% CI] | significant |")
            L.append("|---|---|---|")
            for k, d in sorted(v["marginal_repr_effect"].items()):
                L.append(f"| {k} | {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | "
                         f"{'yes' if d['excludes_zero'] else 'no'} |")
            L.append("")

    # Did the robust objectives actually produce robust representations? Without this the whole
    # comparison could be vacuous, so it is reported on the EVALUATION distribution (train
    # worst-group accuracy is not evidence -- a 10-epoch fine-tune memorises the training split).
    L += ["## Did the fine-tune change the representation? (eval worst-group accuracy)", ""]
    ds_list = sorted({r["dataset"] for r in recs})
    for dataset in ds_list:
        sub = [r for r in recs if r["dataset"] == dataset]
        reprs = sorted({r["representation"] for r in sub})
        heads = sorted({r["head"] for r in sub})
        L.append(f"**{dataset}** — worst-group accuracy, mean over fine-tune seeds")
        L.append("")
        L.append("| representation | " + " | ".join(heads) + " |")
        L.append("|---" * (len(heads) + 1) + "|")
        for rp in reprs:
            row = []
            for h in heads:
                vals = [r["worst_group_acc"] for r in sub
                        if r["representation"] == rp and r["head"] == h]
                row.append(f"{np.mean(vals):.3f}" if vals else "—")
            L.append(f"| {rp} | " + " | ".join(row) + " |")
        L.append("")

    # min-over-groups diagnostic: the sub-target level, measured rather than argued (R1.1, R3)
    mond = [r for r in recs if r.get("calibration") == "mondrian"]
    if mond:
        wg = np.array([r["worst_group_cov"] for r in mond])
        mg = np.array([r["mean_group_cov"] for r in mond if "mean_group_cov" in r])
        L += ["## Sub-target diagnostic (min-over-groups selection effect)", "",
              f"- mean worst-group coverage: **{wg.mean():.4f}**"]
        if mg.size:
            L.append(f"- mean *over-groups* coverage: **{mg.mean():.4f}** "
                     f"(this is the quantity Mondrian targets; it should sit at 1-α)")
            L.append(f"- gap between them: **{mg.mean() - wg.mean():.4f}** — the min-over-k effect, "
                     "not a validity failure")
        L.append("")
    text = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text
