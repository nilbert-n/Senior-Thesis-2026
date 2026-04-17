"""
Figure 8 – Apparent Kd across ASO design variants (ranked).

The Kd here is the COM-12 Å two-tier estimate; error bars are the
standard deviation across four contact-distance cutoffs (10 / 12 / 15 /
20 Å), which we interpret as the *systematic* uncertainty from the
contact-distance definition.  Because the error bars are comparable to
the Kd itself, the plot is best read as a qualitative ranking rather
than an absolute comparison.

Concentration-series points (unmodified 10-mer at 25 / 50 / 200 ASO)
are shown in blue; sequence variants (all at 100 ASO) are shown in red.

Data source:
    Week 10/aso_project 3/kd_plots/kd_ranked_summary.csv
"""
from __future__ import annotations
import os, sys, csv
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote
from _preview_data import make_kd_summary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

apply_style()

CSV_PATH = ("/scratch/gpfs/JERELLE/nilbert/Single_Chain/"
            "Week 10/aso_project 3/kd_plots/kd_ranked_summary.csv")

def _load_rows():
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH, newline="") as fh:
            return list(csv.DictReader(fh))
    print(f"  [PREVIEW] {CSV_PATH} not found — using fabricated data.",
          file=sys.stderr)
    return make_kd_summary()

rows = _load_rows()
names  = np.array([r["name"] for r in rows])
kd_mM  = np.array([float(r["kd_main_mM"]) for r in rows])
kd_err = np.array([float(r["kd_std_across_cutoffs_mM"]) for r in rows])

order = np.argsort(kd_mM)
names, kd_mM, kd_err = names[order], kd_mM[order], kd_err[order]

LABEL_MAP = {
    "25ASO_unmod":                       "25 ASO  (unmod.)",
    "50ASO_unmod":                       "50 ASO  (unmod.)",
    "200ASO_unmod":                      "200 ASO  (unmod.)",
    "100ASO_truncated_unmodified":       "Truncated (7-mer)",
    "100ASO_unmodified_AAtoCC":          "AA→CC",
    "100ASO_unmodified_loop_GG_to_AA":   "Loop GG→AA",
    "100ASO_mismatch_G6A_unmodified":    "Mismatch G6A",
    "100ASO_all_purine_unmodified":      "All-purine",
    "100ASO_scrambled_unmodified":       "Scrambled",
    "100ASO_mismatch_U5C_unmodified":    "Mismatch U5C",
    "100ASO_extended_12mer_unmodified":  "Extended 12-mer",
    "100ASO_extended_14mer_unmodified":  "Extended 14-mer",
}
disp_labels = [LABEL_MAP.get(n, n) for n in names]

CONC_NAMES = {"25ASO_unmod", "50ASO_unmod", "200ASO_unmod"}
bar_cols   = [COLORS["blue"] if n in CONC_NAMES else COLORS["red"] for n in names]

n = len(kd_mM)
y = np.arange(n)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.xaxis.grid(True, alpha=0.35)

ax.barh(y, kd_mM, xerr=kd_err,
        color=bar_cols, edgecolor="white", linewidth=0.4,
        height=0.64,
        error_kw=dict(ecolor=COLORS["gray"], capsize=2.5,
                      elinewidth=0.8, capthick=0.8))

for yi, (kd, err) in enumerate(zip(kd_mM, kd_err)):
    ax.text(kd + err + kd_mM.max() * 0.015, yi,
            f"{kd:.0f}", va="center", fontsize=7.5)

ax.set_yticks(y)
ax.set_yticklabels(disp_labels, fontsize=9)
ax.set_xlabel("Apparent Kd  (mM)")
ax.set_title("Apparent binding affinity across variants")
ax.set_xlim(0, (kd_mM + kd_err).max() * 1.10)
ax.invert_yaxis()   # strongest binder on top

ax.legend(handles=[
    Patch(color=COLORS["blue"], label="Concentration series (unmod. 10-mer)"),
    Patch(color=COLORS["red"],  label="Sequence variant (100 ASO)"),
], loc="lower right", ncol=1, fontsize=8)

# Reference note: 100ASO_unmod is analysed separately in Week 8 pipeline
if "100ASO_unmod" not in set(names):
    ax.text(0.98, 0.97,
            "100 ASO unmod. reference:\nanalysed separately (Week 8)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=COLORS["gray"], style="italic")

add_footnote(fig, "Kd: COM-12 Å two-tier estimator.  Error bars: σ across "
                  "cutoffs 10 / 12 / 15 / 20 Å (systematic uncertainty from "
                  "contact definition).  All sequence variants: 100 ASOs, "
                  "160 Å cubic box.")

save_fig(fig, "fig8_kd_comparison")
