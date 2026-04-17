"""
Figure 9 – Hairpin loop RMSF across ASO design variants (horizontal bar).

The mean RMSF of the 8 loop residues (relative positions 13–20 of the
32-nt 1YMO hairpin) is compared across all simulated ASO variants.  A
higher loop RMSF indicates that the ASO design leaves the recognition
loop more conformationally disordered.

Variants are sorted from lowest to highest loop RMSF.  The unmodified
10-mer at 100 ASO (reference design) is highlighted.

Data source:
    rmsf_campaign_pdbref/rmsf_campaign_summary.csv
        (Week 10/aso_project 3/rmsf_campaign_pdbref/)
Output:
    Figures/thesis_results/fig9_loop_rmsf_variants.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt
import csv

apply_style()

CSV_PATH = ("/scratch/gpfs/JERELLE/nilbert/Single_Chain/"
            "Week 10/aso_project 3/rmsf_campaign_pdbref/rmsf_campaign_summary.csv")

names, loop_rmsf, loop_max = [], [], []

with open(CSV_PATH, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        names.append(row["name"])
        loop_rmsf.append(float(row["loop_mean_rmsf_A"]))
        loop_max.append(float(row["loop_max_rmsf_A"]))

names      = np.array(names)
loop_rmsf  = np.array(loop_rmsf)
loop_max   = np.array(loop_max)

# Sort ascending
order     = np.argsort(loop_rmsf)
names     = names[order]
loop_rmsf = loop_rmsf[order]
loop_max  = loop_max[order]

LABEL_MAP = {
    "100ASO_unmod":                        "100 ASO  (reference: unmodified 10-mer)",
    "25ASO_unmod":                         "25 ASO  (unmodified 10-mer)",
    "50ASO_unmod":                         "50 ASO  (unmodified 10-mer)",
    "200ASO_unmod":                        "200 ASO  (unmodified 10-mer)",
    "100ASO_truncated_unmodified":         "Truncated  (7-mer)",
    "100ASO_unmodified_AAtoCC":            "AA→CC mutation",
    "100ASO_unmodified_loop_GG_to_AA":     "Loop GG→AA mutation",
    "100ASO_mismatch_G6A_unmodified":      "Mismatch G6A",
    "100ASO_all_purine_unmodified":        "All-purine",
    "100ASO_scrambled_unmodified":         "Scrambled sequence",
    "100ASO_mismatch_U5C_unmodified":      "Mismatch U5C",
    "100ASO_extended_12mer_unmodified":    "Extended 12-mer",
    "100ASO_extended_14mer_unmodified":    "Extended 14-mer",
}
disp_labels = [LABEL_MAP.get(n, n) for n in names]

# Colour: highlight reference (100ASO_unmod) in orange; concentration series blue;
# sequence variants gray/red
CONC_NAMES = {"25ASO_unmod", "50ASO_unmod", "200ASO_unmod"}
REF_NAME   = "100ASO_unmod"

def _color(name):
    if name == REF_NAME:      return COLORS["orange"]
    if name in CONC_NAMES:    return COLORS["blue"]
    return COLORS["red"]

bar_cols = [_color(n) for n in names]

n = len(loop_rmsf)
y = np.arange(n)

fig, ax = plt.subplots(figsize=(8.0, 5.5))

bars = ax.barh(y, loop_rmsf, color=bar_cols,
               edgecolor="white", linewidth=0.5, height=0.62)

# Range markers (loop_max)
ax.plot(loop_max, y, "|", color=COLORS["gray"], ms=7, mew=1.5,
        label="Loop RMSF max")

# Reference line: reference design
ref_mask = (names == REF_NAME)
if ref_mask.any():
    ref_rmsf = loop_rmsf[ref_mask][0]
    ax.axvline(ref_rmsf, color=COLORS["orange"], ls="--", lw=1.2,
               label=f"Reference (100 ASO unmod): {ref_rmsf:.1f} Å")

# Value labels
for yi, rmsf_val in enumerate(loop_rmsf):
    ax.text(rmsf_val + 0.1, yi, f"{rmsf_val:.1f}", va="center", fontsize=8.5)

ax.set_yticks(y)
ax.set_yticklabels(disp_labels, fontsize=9.5)
ax.set_xlabel("Mean loop RMSF (Å)  [residues 13–20]")
ax.set_title("Hairpin loop flexibility across ASO design variants\n"
             "(higher RMSF = more disordered recognition loop)", pad=10)
ax.set_xlim(0, loop_rmsf.max() + 5)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=COLORS["orange"], label="Reference (100 ASO, unmodified 10-mer)"),
    Patch(color=COLORS["blue"],   label="Concentration series"),
    Patch(color=COLORS["red"],    label="Sequence variants (100 ASO)"),
], fontsize=9, loc="lower right")

fig.text(0.01, 0.01,
         "Loop residues: relative positions 13–20 of 32-nt 1YMO hairpin.  "
         "RMSF computed after Kabsch alignment to stem residues.  "
         "All simulations: ~1 µs, NVT ensemble.",
         fontsize=7, color=COLORS["gray"], style="italic")

save_fig(fig, "fig9_loop_rmsf_variants")
