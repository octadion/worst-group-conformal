#!/bin/sh
# Rebuild every table and figure from the shipped records, then check them against the
# reported numbers. No GPU, no datasets; a few minutes.
set -e

python scripts/make_tables.py --records records --out tables/appendix-tables.tex
python scripts/make_figures.py --records records --out figures --skip teaser
python -m wgcp.intervals
python -m pytest tests -q

echo
echo "tables in tables/, figures in figures/"
echo "the teaser also needs four Waterbirds images: scripts/make_figures.py --only teaser --images DIR"
echo "to rebuild the records themselves, see the second half of README.md"
