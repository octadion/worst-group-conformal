"""Attribute recoverability beside worst-group coverage, on the same frozen features.

    python scripts/run_recoverability.py --out records/recoverability_partB.json

Part A measures both quantities per cell and labels the tension. Part B runs only where a cell
comes back ambiguous or alive: it projects out the top-k attribute-predictive directions and
traces coverage against probe AUROC along that path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wgcp.config import BACKBONES, DATASETS, DEFAULT_CACHE, build_cell  # noqa: E402
from wgcp.recoverability import (PART_B_KS, needs_part_b, run_part_a, run_part_b,  # noqa: E402
                                 write_recoverability_md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="records/recoverability_partB.json")
    ap.add_argument("--md", default="RECOVERABILITY.md")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    data = {(bb, ds): build_cell(bb, ds, device=args.device, cache_dir=args.cache_dir)
            for ds in args.datasets for bb in args.backbones}

    t = time.time()
    part_a = run_part_a(data, seeds=tuple(args.seeds), n_splits=args.splits)
    for key, cell in part_a["cells"].items():
        r, cov = cell["recoverability"], cell["coverage"]
        print(f"{key}: AUROC {r['auroc_mean']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}] | "
              f"worst-group coverage {cov['worst_group_cov_mean']:.3f} | {cell['verdict']}")

    part_b = None
    if needs_part_b(part_a):
        part_b = {"part": "B", "ks": list(PART_B_KS), "alpha": 0.1, "rho": 0.95, "cells": {}}
        for key, gd in data.items():
            one = run_part_b({key: gd}, ks=PART_B_KS, seeds=tuple(args.seeds),
                             n_splits=args.splits)
            part_b["cells"].update(one["cells"])
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"ks": part_b["ks"], "alpha": part_b["alpha"], "rho": part_b["rho"],
                           "cells": {"/".join(k) if isinstance(k, tuple) else str(k): v
                                     for k, v in part_b["cells"].items()}}, f, indent=1,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o))
            print(f"[part B] {key} done, saved -> {args.out}", flush=True)
    else:
        print("every cell came back tension_dead; part B is not needed")

    write_recoverability_md(part_a, args.md, part_b=part_b)
    print(f"\nfinished in {(time.time() - t) / 60:.0f} min -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
