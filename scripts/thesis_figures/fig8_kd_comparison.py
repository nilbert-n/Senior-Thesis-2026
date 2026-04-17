"""
Figure 8 – Apparent Kd across ASO design variants (ranked horizontal bar).

Two sub-groups are distinguished by colour:
  - Concentration series (25/50/200 ASO, unmodified 10-mer):  blue palette
  - Sequence variants    (all at 100 ASO):                    red/orange palette

The Kd is the COM-12 Å two-tier estimate (f_site / [V/N_avogadro]).
Error bars represent the standard deviation of Kd across the four
distance cutoffs (10, 12, 15, 20 Å), reflecting systematic uncertainty
in the contact-distance definition.

Lower Kd = higher apparent binding affinity.

Data source:
    kd_ranked_summary.csv  (Week 10/aso_project 3/kd_plots/)
Output:
    Figures/thesis_results/fig8_kd_comparison.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt
import csv

apply_style()

CSV_PATH = ("/scratch/gpfs/JERELLE/nilbert/Single_Chain/"
            "Week 10/aso_project 3/kd_plots/kd_ranked_summary.csv")

# ── Load CSV with stdlib (no pandas dependency) ───────────────────────────────
names, labels_raw, kd_mM, kd_err = [], [], [], []

with open(CSV_PATH, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        names.append(row["name"])
        labels_raw.append(row["label"])
        kd_mM.append(float(row["kd_main_mM"]))
        kd_err.append(float(row["kd_std_across_cutoffs_mM"]))

names     = np.array(names)
kd_mM     = np.array(kd_mM)
kd_err    = np.array(kd_err)

# ── Sort by Kd (ascending = best binders first) ───────────────────────────────
order  = np.argsort(kd_mM)
names  = names[order]
kd_mM  = kd_mM[order]
kd_err = kd_err[order]

# Clean display labels
LABEL_MAP = {
    "25ASO_unmod":                       "25 ASO  (unmodified)",
    "50ASO_unmod":                       "50 ASO  (unmodified)",
    "200ASO_unmod":                      "200 ASO  (unmodified)",
    "100ASO_truncated_unmodified":       "Truncated  (7-mer)",
    "100ASO_unmodified_AAtoCC":          "AA→CC mutation",
    "100ASO_unmodified_loop_GG_to_AA":   "Loop GG→AA mutation",
    "100ASO_mismatch_G6A_unmodified":    "Mismatch G6A",
    "100ASO_all_purine_unmodified":      "All-purine",
    "100ASO_scrambled_unmodified":       "Scrambled sequence",
    "100ASO_mismatch_U5C_unmodified":    "Mismatch U5C",
    "100ASO_extended_12mer_unmodified":  "Extended 12-mer",
    "100ASO_extended_14mer_unmodified":  "Extended 14-mer",
}
disp_labels = [LABEL_MAP.get(n, n) for n in names]

# ── Colour coding ─────────────────────────────────────────────────────────────
CONC_NAMES = {"25ASO_unmod", "50ASO_unmod", "200ASO_unmod"}

def _bar_color(name):
    return COLORS["blue"] if name in CONC_NAMES else COLORS["red"]

bar_cols = [_bar_color(n) for n in names]

# ── Plot ──────────────────────────────────────────────────────────────────────
n = len(kd_mM)
y = np.arange(n)

fig, ax = plt.subplots(figsize=(8.0, 5.5))

bars = ax.barh(y, kd_mM, xerr=kd_err,
               color=bar_cols, edgecolor="white", linewidth=0.5,
               height=0.62,
               error_kw=dict(ecolor=COLORS["gray"], capsize=3.5,
                             elinewidth=1.0, capthick=1.0))

# Value labels
for yi, (kd, err) in enumerate(zip(kd_mM, kd_err)):
    ax.text(kd + err + 15, yi,
            f"{kd:.0f} mM", va="center", fontsize=8.5)

ax.set_yticks(y)
ax.set_yticklabels(disp_labels, fontsize=9.5)
ax.set_xlabel("Apparent Kd  (mM)")
ax.set_title("Apparent binding affinity across ASO design variants\n"
             "(lower Kd = stronger binding)", pad=10)

ax.set_xlim(0, kd_mM.max() + kd_err[np.argmax(kd_mM)] + 350)

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=COLORS["blue"], label="Concentration series  (unmodified 10-mer)"),
    Patch(color=COLORS["red"],  label="Sequence variant  (100 ASO)"),
], fontsize=9, loc="lower right")

# Reference line at the 100ASO_unmod Kd if available among concentration entries
# (no explicit 100ASO_unmod in this CSV — note this gap)
ax.axvline(kd_mM[0], color=COLORS["gray"], ls=":", lw=0.8, alpha=0.5)

fig.text(0.01, 0.01,
         "Kd: COM-12 Å two-tier estimator.  Error bars: σ across 4 cutoff distances "
         "(10/12/15/20 Å).  All sequence-variant simulations: 100 ASOs, 160 Å cubic box.",
         fontsize=7, color=COLORS["gray"], style="italic")

save_fig(fig, "fig8_kd_comparison")
