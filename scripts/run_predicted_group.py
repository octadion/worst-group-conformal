"""Per-group calibration with the attribute predicted instead of given.

    python scripts/run_predicted_group.py --out records/predicted_group_mondrian_4bb.csv

Resumable per (backbone, dataset) cell. Coverage is always scored against the true groups.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wgcp.config import BACKBONES, DATASETS, DEFAULT_CACHE, build_cell  # noqa: E402
from wgcp.methods import METHODS  # noqa: E402
from wgcp.predicted_group_mondrian import run_predicted_group_streaming  # noqa: E402
from wgcp.verdicts import SCORES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="records/predicted_group_mondrian_4bb.csv")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keys = [(bb, ds) for ds in args.datasets for bb in args.backbones]

    t = time.time()
    out = run_predicted_group_streaming(
        keys, lambda bb, ds: build_cell(bb, ds, device=args.device, cache_dir=args.cache_dir),
        methods=METHODS, scores=SCORES, seeds=tuple(args.seeds), n_splits=args.splits,
        alpha=args.alpha, cell_csv=args.out)

    done = sorted({(r["backbone"], r["dataset"]) for r in out["records"]})
    print(f"\n{len(out['records']):,} records in {(time.time() - t) / 60:.0f} min")
    print(f"cells complete: {len(done)}/{len(keys)} -> {args.out}")
    if out["failed"]:
        print(f"failed: {out['failed']}")
    return 0 if len(done) == len(keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
