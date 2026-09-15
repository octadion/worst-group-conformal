"""The intervals of the calibration table and of the model-level matched comparison.

Both are two-stage cluster bootstraps: resample training seeds, then the calibration splits
within each drawn seed.

Model-level matching gives each training seed one (base accuracy, divergence) point, the mean
over its calibration splits. ERM's reference accuracy a* is its own mean, since its solver is
deterministic and its seeds coincide; each robust method's divergence at a* is read off a
least-squares line through its per-seed points.

    python -m wgcp.intervals
"""
from __future__ import annotations

import collections
import csv
import os

import numpy as np

from .stats import DEFAULT_B, cluster_bootstrap_ci, spread_equivalence

RECORDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "records")
BACKBONES = ("resnet50_erm", "clip_vitb32", "dinov2_vitb14", "vit_b16_in1k")
DATASETS = ("waterbirds", "celeba")
RETAINED = ("kept", "flagged")


def _load(name):
    with open(os.path.join(RECORDS, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _two_stage_spread_ci(by_method: dict, *, B: int = DEFAULT_B, alpha: float = 0.05,
                         seed: int = 0):
    prepared = []
    for v, s in by_method.values():
        v, s = np.asarray(v, float), np.asarray(s)
        prepared.append((v, [np.flatnonzero(s == u) for u in np.unique(s)]))
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        means = []
        for v, by_seed in prepared:
            pick = rng.integers(0, len(by_seed), len(by_seed))
            idx = [by_seed[j][rng.integers(0, by_seed[j].size, by_seed[j].size)] for j in pick]
            means.append(v[np.concatenate(idx)].mean())
        draws[b] = np.ptp(means)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def table2(records=None, *, B: int = DEFAULT_B):
    """Calibration comparison at APS, rho_cal = rho_test = 0.95.

    Lift is the same ERM model under both rules, paired per (seed, split); it is reported even
    where ERM is below its accuracy floor, which the table marks with a dagger.
    """
    recs = records if records is not None else _load("calibration_ablation_4bb.csv")
    recs = [r for r in recs if r["score"] == "APS" and r["rho_test"] == "0.95"]
    out = []
    for ds in DATASETS:
        for bb in BACKBONES:
            rows = [r for r in recs if r["dataset"] == ds and r["backbone"] == bb]
            pair = collections.defaultdict(dict)
            for r in rows:
                if r["method"] == "erm":
                    pair[(r["train_seed"], r["split_seed"])][r["calibration"]] = \
                        float(r["worst_group_cov"])
            keys = sorted(k for k, v in pair.items() if {"mondrian", "marginal_split"} <= set(v))
            diff = np.array([pair[k]["mondrian"] - pair[k]["marginal_split"] for k in keys])
            lift = cluster_bootstrap_ci(diff, np.array([k[0] for k in keys]),
                                        np.array([k[1] for k in keys]), B=B)
            erm_retained = any(r["method"] == "erm" and r["gate_status"] in RETAINED for r in rows)
            spreads = {}
            for cal in ("marginal_split", "mondrian"):
                bym = collections.defaultdict(lambda: ([], []))
                for r in rows:
                    if r["calibration"] == cal and r["gate_status"] in RETAINED:
                        bym[r["method"]][0].append(float(r["worst_group_cov"]))
                        bym[r["method"]][1].append(r["train_seed"])
                spreads[cal] = dict(bym)
            pg = spreads["mondrian"]
            out.append({
                "dataset": ds, "backbone": bb, "erm_retained": erm_retained,
                "shared": float(np.mean([pair[k]["marginal_split"] for k in keys])),
                "per_group": float(np.mean([pair[k]["mondrian"] for k in keys])),
                "lift": float(diff.mean()), "lift_ci": (lift["lo"], lift["hi"]),
                "spread_shared": float(np.ptp([np.mean(v[0])
                                               for v in spreads["marginal_split"].values()])),
                "spread_per_group": float(np.ptp([np.mean(v[0]) for v in pg.values()])),
                "spread_per_group_ci": _two_stage_spread_ci(pg, B=B),
                "spread_per_group_upper_one_sided": spread_equivalence(pg, margin=0.05, B=B)["hi"],
            })
    return out


def model_level_delta(records=None, *, dataset="waterbirds", backbone="resnet50_erm",
                      method="groupdro_ll", score="APS", metric="div_wasserstein1",
                      B: int = DEFAULT_B, alpha: float = 0.05, seed: int = 0):
    """Delta(a*) = D_ERM(a*) - D_method(a*), with a two-stage cluster bootstrap.

    ``supports_overlap`` says whether a* lies inside the robust method's per-seed accuracy range;
    where it does not, the comparison is an extrapolation and the paper reports it as undefined.
    """
    recs = records if records is not None else _load("grid_records.csv")

    def cloud(m):
        d = collections.defaultdict(list)
        for r in recs:
            if (r["dataset"] == dataset and r["backbone"] == backbone and r["method"] == m
                    and r["score"] == score and r["rho_test"] == "0.95"
                    and r["gate_status"] in RETAINED):
                d[r["train_seed"]].append((float(r["base_top1"]), float(r[metric])))
        return list(d.values())

    def delta(erm, rob):
        a_star = np.mean([a for seed_pts in erm for a, _ in seed_pts])
        d_erm = np.mean([x for seed_pts in erm for _, x in seed_pts])
        acc = np.array([np.mean([a for a, _ in p]) for p in rob])
        div = np.array([np.mean([x for _, x in p]) for p in rob])
        slope, icpt = np.polyfit(acc, div, 1)
        return d_erm - (icpt + slope * a_star), a_star, acc

    erm, rob = cloud("erm"), cloud(method)
    point, a_star, acc = delta(erm, rob)
    overlap = bool(acc.min() <= a_star <= acc.max())
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        e = [erm[j] for j in rng.integers(0, len(erm), len(erm))]
        g = [rob[j] for j in rng.integers(0, len(rob), len(rob))]
        e = [[p[k] for k in rng.integers(0, len(p), len(p))] for p in e]
        g = [[p[k] for k in rng.integers(0, len(p), len(p))] for p in g]
        draws[b] = delta(e, g)[0]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta": float(point), "ci": (float(lo), float(hi)), "a_star": float(a_star),
            "per_seed_acc": acc.tolist(), "supports_overlap": overlap}


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", np.exceptions.RankWarning)
    print("Calibration comparison (Table 2)")
    for row in table2():
        lo, hi = row["lift_ci"]
        slo, shi = row["spread_per_group_ci"]
        print(f"  {row['dataset'][:2]}/{row['backbone']:14s} shared {row['shared']:.3f}  "
              f"per-group {row['per_group']:.3f}  lift {row['lift']:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"spread {row['spread_shared']:.3f} / {row['spread_per_group']:.3f} "
              f"[{slo:.3f},{shi:.3f}]  one-sided upper "
              f"{row['spread_per_group_upper_one_sided']:.4f}"
              + ("" if row["erm_retained"] else "  (ERM below floor)"))
    print("Model-level matched comparison, GroupDRO-LL against ERM on ResNet-50/Waterbirds")
    for sc in ("APS", "RAPS", "THR"):
        d = model_level_delta(score=sc)
        print(f"  {sc:4s} a*={d['a_star']:.4f}  Delta {d['delta']:+.3f} "
              f"[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]  supports overlap: {d['supports_overlap']}")
