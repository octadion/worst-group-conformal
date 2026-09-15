"""Write the appendix tables as LaTeX, straight from the records.

    python scripts/make_tables.py --records records --out tables/appendix-tables.tex

Four tables, and the per-setting disparities printed alongside so they can be checked by eye:

    cross-group divergences per setting, D_KS for each score and W1 for the headline one
    the fine-tuning study, with its across-seed spread
    predicted groups, all three conditions, for APS and THR
    the variance decomposition, across-seed against across-split

Per-group coverage for all four groups is not built here: the records carry the mean over groups,
the minimum and the range, so the four separately come from re-running the evaluation
(scripts/run_calibration_ablation.py with return_examples) rather than from re-aggregating.
"""
import argparse
import csv
import io
import os
from collections import defaultdict

import numpy as np

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--records", default="records")
_ap.add_argument("--out", default="tables/appendix-tables.tex")
_args = _ap.parse_args()
R = _args.records
OUT = _args.out
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)

NAME = {"resnet50_erm": "ResNet-50", "clip_vitb32": "CLIP",
        "dinov2_vitb14": "DINOv2", "vit_b16_in1k": "ViT-B/16"}
MNAME = {"erm": "ERM", "dfr": "DFR", "afr": "AFR",
         "balanced_subsample": "Bal.\\ sub.", "groupdro_ll": "GroupDRO-LL"}
PNAME = {"marginal_split": "shared", "mondrian": "per-group", "shift_robust": "shift-robust"}
BB = ["resnet50_erm", "clip_vitb32", "dinov2_vitb14", "vit_b16_in1k"]
DS = ["waterbirds", "celeba"]
SC = ["APS", "RAPS", "THR"]
# The score-by-score tables print the headline score and the one whose behaviour is least regular.
# RAPS sits on top of APS wherever both are computed, so a RAPS row is a repeated row.
SC_SHOWN = ["APS", "THR"]
MS = ["erm", "dfr", "afr", "balanced_subsample", "groupdro_ll"]

abl = list(csv.DictReader(open(f"{R}/calibration_ablation_4bb.csv")))
rep = list(csv.DictReader(open(f"{R}/representation_records.csv")))
pg = list(csv.DictReader(open(f"{R}/predicted_group_mondrian_4bb.csv")))
out = []


def w(rows, col):
    return np.mean([float(r[col]) for r in rows]) if rows else float("nan")


def f(x, d=3):
    return "---" if np.isnan(x) else f"{x:.{d}f}"


def head(title, label, spec, cols, small="footnotesize"):
    out.append(r"\begin{table}[!ht]" "\n" r"\centering")
    out.append(f"\\caption{{{title}}}")
    out.append(f"\\label{{{label}}}")
    out.append(f"\\{small}")
    out.append(r"\setlength{\tabcolsep}{3.5pt}")
    out.append(f"\\begin{{tabular}}{{{spec}}}")
    out.append(r"\toprule")
    out.append(" & ".join(cols) + r" \\")
    out.append(r"\midrule")


def foot():
    out.append(r"\botrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}" "\n")


# ───────────────────────────────────────────────────────── G2 divergences
# W1 is given for the headline score only. Appendix A shows it carries no bound, so it is here to
# be comparable with the summary Burden et al. report and not because three columns of it are read.
head(r"Cross-group conformity-score divergences under one shared threshold "
     r"($\rhocal=\rhotest=0.95$), pooled over all non-excluded runs, including the seven flagged "
     r"DFR runs on CelebA that Figure~\ref{fig:mech} leaves out. "
     r"$D_{\mathrm{KS}}$ is the quantity bounded in Eq.~\eqref{eq:ksbound}; $\Wone$, given for "
     r"APS alone, is a descriptive summary and enters no bound.",
     "tab:div", "l" + "c" * 4,
     ["Setting"] + [f"$D_{{\\mathrm{{KS}}}}$ {sc}" for sc in SC] + [r"$\Wone$ APS"])
for ds in DS:
    out.append(r"\multicolumn{5}{l}{\emph{" + ("Waterbirds" if ds == "waterbirds"
                                               else "CelebA") + r"}} \\")
    for bb in BB:
        cells, w1 = [], float("nan")
        for sc in SC:
            sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                   and r["score"] == sc and r["rho_test"] == "0.95"
                   and r["calibration"] == "marginal_split" and r["gate_status"] != "excluded"]
            cells.append(f(w(sel, "div_ks_stat")))
            if sc == "APS":
                w1 = w(sel, "div_wasserstein1")
        out.append(" & ".join([NAME[bb]] + cells + [f(w1)]) + r" \\")
foot()

# ───────────────────────────────────────────────────────── G4 fine-tuning in full
# The RAPS rows are dropped, so the caption states how far RAPS gets from APS. Computed here
# rather than typed, because a caption that asserts a bound has to be re-derived when the records
# are re-read; the constant went stale once already.
def _ft(ds, rp, sc, pol):
    sel = [r for r in rep if r["dataset"] == ds and r["representation"] == rp
           and r["score"] == sc and r["rho_test"] == "0.95" and r["head"] == "erm"
           and r["calibration"] == pol]
    return w(sel, "worst_group_cov") if sel else float("nan")


# Rounded before differencing, so the bound is what a reader gets from the printed precision
# rather than 0.001 less than the columns appear to differ by.
_gaps = [abs(round(_ft(ds, rp, "APS", pol), 3) - round(_ft(ds, rp, "RAPS", pol), 3))
         for ds in DS for rp in sorted({r["representation"] for r in rep})
         for pol in ("marginal_split", "mondrian")]
RAPS_FT_GAP = f"{np.nanmax(_gaps):.3f}"

head(r"The fine-tuning study of Table~\ref{tab:repr} for APS and THR ($\rhocal=\rhotest=0.95$): "
     r"means over five fine-tuning seeds on Waterbirds and three on CelebA, with the across-seed standard deviation in brackets. "
     r"Worst-group accuracy does not depend on the score. RAPS is within $" + RAPS_FT_GAP + r"$ of APS "
     r"throughout.",
     "tab:reprfull", "ll" + "ccc",
     ["Objective", "Score", "wg acc", "wg cov shared", "wg cov per-group"])
# The dataset is already a section header inside the table, so a Dataset column would be blank in
# every row. Objectives are named as the body's Table 3 names them, not as the raw record values.
RPNAME = {"erm": "ERM", "groupdro": "GroupDRO", "reweight": "Reweighting"}
reps = sorted({r["representation"] for r in rep})
for ds in DS:
    out.append(r"\multicolumn{5}{l}{\emph{" + ("Waterbirds" if ds == "waterbirds"
                                               else "CelebA") + r"}} \\")
    for rp in reps:
        for sc in SC_SHOWN:
            # Hold the head at plain ERM, exactly as Table 3 in the body does. Without this
            # the table averages over all three heads and disagrees with the body.
            base = [r for r in rep if r["dataset"] == ds and r["representation"] == rp
                    and r["score"] == sc and r["rho_test"] == "0.95"
                    and r["head"] == "erm"]
            if not base:
                continue
            acc = w(base, "worst_group_acc")
            cells = []
            for pol in ("marginal_split", "mondrian"):
                p = [r for r in base if r["calibration"] == pol]
                per_seed = [np.mean([float(x["worst_group_cov"]) for x in p
                                     if x["ft_seed"] == s])
                            for s in sorted({r["ft_seed"] for r in p})]
                cells.append(f"{np.mean(per_seed):.3f}\\,[{np.std(per_seed):.3f}]"
                             if per_seed else "---")
            out.append(" & ".join([RPNAME.get(rp, rp) if sc == SC_SHOWN[0] else "", sc, f(acc)]
                                  + cells) + r" \\")
foot()

# ───────────────────────────────────────────────────────── G5 predicted groups in full
COND = sorted({r["condition"] for r in pg})
def _pgc(bb, ds, sc, c):
    sel = [r for r in pg if r["backbone"] == bb and r["dataset"] == ds
           and r["score"] == sc and r["condition"] == c and r["gate_status"] != "excluded"]
    return w(sel, "worst_group_cov")


RAPS_PG_GAP = "{:.3f}".format(np.nanmax(
    [abs(round(_pgc(bb, ds, "APS", c), 3) - round(_pgc(bb, ds, "RAPS", c), 3))
     for bb in BB for ds in DS for c in COND]))

head(r"Predicted-group calibration: worst-group coverage under the three conditions of "
     r"Table~\ref{tab:predgroup} ($\rhocal=\rhotest=0.95$), averaged over the kept training "
     r"methods, for APS and THR. Coverage is always scored against the true groups. RAPS is "
     r"within $" + RAPS_PG_GAP + r"$ of APS in every cell.",
     "tab:pgfull", "ll" + "c" * (len(COND) * 1) + "c",
     ["Setting", "Score"] + [c.replace("_", " ") for c in COND] + ["probe AUROC"])
for ds in DS:
    out.append(r"\multicolumn{" + str(2 + len(COND) + 1) + r"}{l}{\emph{"
               + ("Waterbirds" if ds == "waterbirds" else "CelebA") + r"}} \\")
    for bb in BB:
        for sc in SC_SHOWN:
            cells = []
            for c in COND:
                sel = [r for r in pg if r["backbone"] == bb and r["dataset"] == ds
                       and r["score"] == sc and r["condition"] == c
                       and r["gate_status"] != "excluded"]
                cells.append(f(w(sel, "worst_group_cov")))
            au = [r for r in pg if r["backbone"] == bb and r["dataset"] == ds
                  and r["gate_status"] != "excluded"]
            out.append(" & ".join([NAME[bb] if sc == SC_SHOWN[0] else "", sc] + cells
                                  + [f(w(au, "probe_auroc"))]) + r" \\")
foot()

# ───────────────────────────────────────────────────────── G6 variance decomposition
head(r"Standard deviation of worst-group coverage across training seeds and across calibration "
     r"splits (APS, $\rhocal=\rhotest=0.95$), averaged over the kept training methods: "
     r"per-group thresholds in the first two columns, one shared threshold in the last. ERM "
     r"and AFR are omitted because their solver ignores the seed.",
     "tab:variance", "lccc",
     ["Setting", "SD across seeds", "SD across splits", "SD across splits"])
# Two columns sharing a heading is resolved only by the sub-line below, which does not visibly
# cover the first of them; spell the rule out in the header instead.
# The shared-threshold column exists because Section 5.5 compares the two rules' split-to-split
# stability and withdraws an earlier claim that Mondrian was the steadier of the two. Without it
# the marginal numbers appear in the prose and in no table, so no reader can check the comparison.
out[-2] = " & ".join(["Setting", "SD across seeds",
                          "SD across splits", "SD across splits"]) + r" \\"
out.insert(len(out) - 1, r" & (per-group) & (per-group) & (shared) \\")
for ds in DS:
    out.append(r"\multicolumn{4}{l}{\emph{" + ("Waterbirds" if ds == "waterbirds"
                                               else "CelebA") + r"}} \\")
    for bb in BB:
        seed_sd, split_sd, marg_sd = [], [], []
        for m in ("dfr", "balanced_subsample", "groupdro_ll"):
            for pol, sink in (("mondrian", split_sd), ("marginal_split", marg_sd)):
                sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                       and r["method"] == m and r["score"] == "APS" and r["rho_test"] == "0.95"
                       and r["calibration"] == pol and r["gate_status"] != "excluded"]
                if not sel:
                    continue
                by_seed = defaultdict(list)
                for r_ in sel:
                    by_seed[r_["train_seed"]].append(float(r_["worst_group_cov"]))
                if pol == "mondrian":
                    means = [np.mean(v) for v in by_seed.values()]
                    if len(means) > 1:
                        seed_sd.append(np.std(means))
                sink.append(np.mean([np.std(v) for v in by_seed.values()]))
        out.append(" & ".join([NAME[bb], f(np.mean(seed_sd), 4) if seed_sd else "---",
                               f(np.mean(split_sd), 4) if split_sd else "---",
                               f(np.mean(marg_sd), 4) if marg_sd else "---"]) + r" \\")
foot()

# Set-size disparity used to have a table of its own, whose two coverage columns repeated the
# per-group table's mean column exactly. It is now a column of that table; this prints the numbers
# so they can be checked against what the manuscript carries.
print("\n   disparity, to be carried by the per-group table:")
for ds in DS:
    for bb in BB:
        cells = []
        for pol in ("marginal_split", "mondrian"):
            sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                   and r["score"] == "APS" and r["rho_test"] == "0.95"
                   and r["calibration"] == pol and r["gate_status"] != "excluded"]
            cells.append(f"{f(w(sel, 'mean_group_cov'))}/{f(w(sel, 'set_size_disparity'))}")
        print(f"     {ds:11s} {NAME[bb]:10s} shared {cells[0]}  per-group {cells[1]}")

# ── the dissociation for all three scores at once, which heads the generated file ────────────
spread = [r"\begin{table}[!ht]", r"\centering",
          r"\caption{The dissociation across all three conformity scores "
          r"($\rhocal=\rhotest=0.95$): the spread in worst-group coverage across the kept "
          r"training methods, under one shared threshold and under per-group thresholds.}",
          r"\label{tab:scorespread}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
          r"\begin{tabular}{@{}l" + "cc" * 3 + r"@{}}", r"\toprule",
          r" & \multicolumn{2}{c}{APS} & \multicolumn{2}{c}{RAPS} & \multicolumn{2}{c}{THR} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
          r"Setting & shared & per-gr. & shared & per-gr. & shared & per-gr. \\", r"\midrule"]
for ds in DS:
    spread.append(r"\multicolumn{7}{@{}l}{\emph{"
                  + ("Waterbirds" if ds == "waterbirds" else "CelebA") + r"}} \\")
    for bb in BB:
        cells = []
        for sc in SC:
            for pol in ("marginal_split", "mondrian"):
                per = []
                for meth in MS:
                    sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                           and r["method"] == meth and r["score"] == sc
                           and r["rho_test"] == "0.95" and r["calibration"] == pol
                           and r["gate_status"] != "excluded"]
                    if sel:
                        per.append(w(sel, "worst_group_cov"))
                cells.append(f"{max(per) - min(per):.3f}" if len(per) > 1 else "---")
        spread.append(" & ".join([NAME[bb]] + cells) + r" \\")
spread += [r"\botrule", r"\end{tabular}", r"\end{table}", ""]
out = spread + out

io.open(OUT, "w", encoding="utf-8", newline="").write("\n".join(out) + "\n")
n_tab = sum(1 for l in out if l.startswith(r"\begin{table}"))
n_row = sum(1 for l in out if l.endswith(r"\\") and "multicolumn" not in l
            and "toprule" not in l)
print(f"-> {OUT}")
print(f"   {n_tab} tabel, ~{n_row} baris data, {len(out)} baris LaTeX")
