# worst-group-conformal

Worst-group conformal coverage depends far more on how scores are turned into prediction sets
than on how the model was trained. Under one threshold shared by all groups, an ordinary model
covers 89% of test images but only 51% of those in its worst group; giving each group its own
threshold closes most of that gap under every training method tried, on the same model and the
same scores. Fine-tuning the representation end to end does not change it.

This repository holds the code and the records behind those numbers: about 116,000 conformal
evaluations over two benchmarks, four backbones, five last-layer training methods, three
conformity scores, three calibration policies and a correlation-strength sweep.

## Install

```bash
python -m pip install -r requirements.txt
```

Python 3.8 or newer. The analysis, table and figure paths need only numpy, scipy, scikit-learn,
pandas and matplotlib. Extracting features or fine-tuning also needs torch, torchvision and
open_clip, and a GPU.

## Two ways to use it

**From the shipped records (minutes, no GPU).** Everything the paper reports is recomputed from
`records/`:

```bash
python scripts/make_tables.py --out tables/appendix-tables.tex
python scripts/make_figures.py --out figures --skip teaser
python -m wgcp.intervals          # the calibration table and the matched comparison
python -m pytest tests            # the records still produce the reported numbers
```

The teaser's first panel needs four Waterbirds images; pass `--images` pointing at a directory
holding the files named in `wgcp/figures/teaser.py` to draw it.

**From the data (hours, GPU for two stages).**

```bash
python scripts/prepare_data.py --root data      # Waterbirds downloads; CelebA you supply
export WATERBIRDS_ROOT=... CELEBA_ROOT=...

python scripts/run_grid.py                      # five seeds, one shared threshold, rho sweep
python scripts/run_calibration_ablation.py      # three calibration policies
python scripts/run_predicted_group.py           # per-group calibration with a predicted attribute
python scripts/run_recoverability.py            # attribute recoverability against coverage
python scripts/run_representation.py            # end-to-end fine-tuning (GPU)
```

Each run is resumable: a `(backbone, dataset)` cell already present in the output CSV is skipped.
Features are cached under `caches/`, so the first pass over a backbone is the expensive one and
everything after it is disk reads.

## Layout

```
conformal/   scores (THR, APS, RAPS), the split-conformal threshold, and the group-conditional
             and TV-robust variants
wgcp/        heads and training methods, the conformal evaluation, the experiment drivers,
             the statistics, and the figures
wgcp/data/   Waterbirds and CelebA loaders: paths, labels, attribute, group, seeded splits
scripts/     one command per stage, plus the table and figure generators
records/     the evaluations behind every table and figure
tests/       coverage guarantees on synthetic data, and the records against the reported numbers
```

## Records

| file | what it holds |
| --- | --- |
| `grid_records.csv` | the five-seed grid, one shared threshold, across the rho sweep |
| `calibration_ablation_4bb.csv` | three calibration policies over the same posteriors |
| `predicted_group_mondrian_4bb.csv` | per-group calibration with true and predicted attributes |
| `representation_records.csv` | the end-to-end fine-tuning study |
| `per_group_coverage.csv` | all four groups' coverage, from a re-run at the headline setting |
| `group_unique_support.csv` | distinct images behind the composited group counts |
| `afr_tuning.json` | AFR's gamma selection and its oracle bound |
| `recoverability_partB.json` | coverage against probe AUROC as attribute directions are removed |

Each row of a records CSV is one conformal evaluation: a `(backbone, dataset, method, seed,
score, rho_test, calibration, split)` cell with its coverage, set sizes, base accuracy and
cross-group divergence. `gate_status` marks runs that fell below their worst-group accuracy
floor; the analysis keeps `kept` and `flagged` and drops `excluded`.

## Citation

`CITATION.cff` carries the machine-readable entry. The manuscript it accompanies is under review;
cite the software by its archived version until the paper appears.

## License

MIT, see `LICENSE`. Waterbirds and CelebA are redistributed by their own authors under their own
terms and are not included here.
