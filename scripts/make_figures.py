"""Redraw the paper's figures from the records.

    python scripts/make_figures.py --out figures
    python scripts/make_figures.py --only teaser --images path/to/waterbirds/images

Every figure prints the numbers it plots, so it can be checked against the tables. The teaser's
first panel needs four Waterbirds images, named in ``wgcp.figures.teaser.GRID``; without them,
pass --skip teaser.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wgcp.figures import FIGURES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", default="records")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--images", default="images", help="directory holding the teaser's images")
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=sorted(FIGURES))
    ap.add_argument("--skip", nargs="+", choices=sorted(FIGURES), default=[])
    args = ap.parse_args()

    failed = []
    for name in [n for n in args.only if n not in args.skip]:
        print(f"=== {name} ===")
        draw = FIGURES[name]
        try:
            if name == "teaser":
                paths = draw(args.records, args.images, args.out)
            else:
                paths = draw(args.records, args.out)
            print(f"  -> {', '.join(os.path.basename(p) for p in paths)}" if paths else "  skipped")
        except Exception as e:
            failed.append((name, repr(e)))
            print(f"  FAILED: {e}")
    if failed:
        print(f"\n{len(failed)} figure(s) failed: {[n for n, _ in failed]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
