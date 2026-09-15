"""Fetch Waterbirds and check CelebA, then print the roots to export.

    python scripts/prepare_data.py --root data

Waterbirds downloads and extracts itself (about 1.5 GB). CelebA must be obtained separately, from
Kaggle or the project page, and extracted so that list_attr_celeba.txt, list_eval_partition.txt
and img_align_celeba/ sit under one directory.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wgcp.config import WATERBIRDS_URL  # noqa: E402
from wgcp.data.base import SplitSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data", help="directory to hold both datasets")
    args = ap.parse_args()

    wb = os.path.abspath(os.path.join(args.root, "waterbirds"))
    ce = os.path.abspath(os.path.join(args.root, "celeba"))
    os.makedirs(wb, exist_ok=True)

    from wgcp.data.waterbirds import load_waterbirds
    b = load_waterbirds({"root": wb, "n_classes": 2, "download": True, "url": WATERBIRDS_URL},
                        seed=0, split_spec=SplitSpec(), build_datasets=False)
    print(f"waterbirds ready: {sum(len(v) for v in b.meta['paths'].values())} images under {wb}")

    have = os.path.isdir(ce) and any("list_attr_celeba.txt" in fs for _, _, fs in os.walk(ce))
    if have:
        from wgcp.data.celeba import load_celeba
        c = load_celeba({"root": ce, "n_classes": 2}, seed=0, split_spec=SplitSpec())
        print(f"celeba ready: {sum(len(v) for v in c.meta['paths'].values())} images under {ce}")
    else:
        print(f"celeba missing: put list_attr_celeba.txt, list_eval_partition.txt and "
              f"img_align_celeba/ under {ce}")

    print("\nexport these before running anything else:")
    print(f"  export WATERBIRDS_ROOT={wb}")
    print(f"  export CELEBA_ROOT={ce}")
    return 0 if have else 1


if __name__ == "__main__":
    raise SystemExit(main())
