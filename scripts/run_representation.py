"""Fine-tune the backbone end to end, then re-run the calibration comparison on it.

    python scripts/run_representation.py --datasets waterbirds --out records/representation_records.csv

This is the only stage that needs a GPU. Fine-tuned features are cached, so a re-run over a warm
cache is numpy only. Waterbirds uses five fine-tuning seeds and CelebA three, and each objective
keeps its own recipe (wgcp.config.FT_RECIPE).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wgcp.config import (DEFAULT_CACHE, FT_RECIPE, FT_SEEDS, SELECT_BY, cfg_for)  # noqa: E402
from wgcp.representation import (build_repr_griddata, records_from_representation_csv,  # noqa: E402
                                 run_representation_streaming, write_representation_csv,
                                 write_representation_md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="records/representation_records.csv")
    ap.add_argument("--md", default="REPRESENTATION.md")
    ap.add_argument("--datasets", nargs="+", default=["waterbirds", "celeba"])
    ap.add_argument("--objectives", nargs="+", default=["erm", "groupdro", "reweight"])
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def build(dataset, objective, ft_seed):
        cfg = cfg_for(dataset, device=args.device, cache_dir=args.cache_dir)
        cfg["finetune"] = {"device": args.device, "batch_size": 128, "num_workers": 4, "amp": True,
                           "cache_dir": os.path.join(args.cache_dir, "finetune"),
                           "ckpt_dir": args.ckpt_dir, "select_by": SELECT_BY,
                           **FT_RECIPE[dataset][objective]}
        return build_repr_griddata(dataset, objective, cfg, ft_seed=ft_seed)

    keys = [(ds, obj, s) for ds in args.datasets for obj in args.objectives
            for s in FT_SEEDS[ds]]
    prior = records_from_representation_csv(args.out) if os.path.exists(args.out) else ()

    t = time.time()
    out = run_representation_streaming(keys, build, n_splits=args.splits, prior_records=prior)
    write_representation_csv(out["records"], args.out)
    write_representation_md(out, args.md)
    print(f"\n{len(out['records']):,} records in {(time.time() - t) / 60:.0f} min -> {args.out}")
    if out["failed"]:
        print(f"failed: {out['failed']}")
    return 0 if not out["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
