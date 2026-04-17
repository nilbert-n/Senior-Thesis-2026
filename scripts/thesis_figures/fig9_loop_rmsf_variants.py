"""
Figure 9 – Hairpin loop RMSF across ASO design variants.

The mean RMSF of the 8 recognition-loop residues (relative positions
13–20 of the 32-nt 1YMO hairpin) is compared across every simulated
ASO variant.  Lower loop RMSF means the ASO design keeps the loop
more structurally ordered.

The reference design (100 ASO, unmodified 10-mer) is highlighted in
yellow.  A faint grey "|" marker on each bar shows that variant's
*maximum* per-residue RMSF across the loop — i.e., the most flexible
residue in that variant's loop.

Data source:
    Week 10/aso_project 3/rmsf_campaign_pdbref/
    rmsf_campaign_summary.csv
"""
from __future__ import annotations
import os, sys, csv
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote
from _preview_data import make_loop_rmsf_summary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

apply_style()

CSV_PATH = ("/scratch/gpfs/JERELLE/nilbert/Single_Chain/"
            "Week 10/aso_project 3/rmsf_campaign_pdbref/"
            "rmsf_campaign_summary.csv")

def _load_rows():
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH, newline="") as fh:
            return list(csv.DictReader(fh))
    print(f"  [PREVIEW] {CSV_PATH} not found — using fabricated data.",
          file=sys.stderr)
    return make_loop_rmsf_summary()

rows = _load_rows()
names      = np.array([r["name"] for r in rows])
loop_rmsf  = np.array([float(r["loop_mean_rmsf_A"]) for r in rows])
loop_max   = np.array([float(r["loop_max_rmsf_A"])  for r in rows])

order     = np.argsort(loop_rmsf)
names     = names[order]
loop_rmsf = loop_rmsf[order]
loop_max  = loop_max[order]

LABEL_MAP = {
    "100ASO_unmod":                        "100 ASO  (reference)",
    "25ASO_unmod":                         "25 ASO  (unmod.)",
    "50ASO_unmod":                         "50 ASO  (unmod.)",
    "200ASO_unmod":                        "200 ASO  (unmod.)",
    "100ASO_truncated_unmodified":         "Truncated (7-mer)",
    "100ASO_unmodified_AAtoCC":            "AA→CC",
    "100ASO_unmodified_loop_GG_to_AA":     "Loop GG→AA",
    "100ASO_mismatch_G6A_unmodified":      "Mismatch G6A",
    "100ASO_all_purine_unmodified":        "All-purine",
    "100ASO_scrambled_unmodified":         "Scrambled",
    "100ASO_mismatch_U5C_unmodified":      "Mismatch U5C",
    "100ASO_extended_12mer_unmodified":    "Extended 12-mer",
    "100ASO_extended_14mer_unmodified":    "Extended 14-mer",
}
disp_labels = [LABEL_MAP.get(n, n) for n in names]

CONC_NAMES = {"25ASO_unmod", "50ASO_unmod", "200ASO_unmod"}
REF_NAME   = "100ASO_unmod"

def _color(name):
    if name == REF_NAME:     return COLORS["yellow"]
    if name in CONC_NAMES:   return COLORS["blue"]
    return COLORS["red"]

bar_cols = [_color(n) for n in names]

n = len(loop_rmsf)
y = np.arange(n)

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.xaxis.grid(True, alpha=0.35)

ax.barh(y, loop_rmsf, color=bar_cols,
        edgecolor="white", linewidth=0.4, height=0.64)

# Max-RMSF tick marks as '|' outside the bar (so they never overlap labels)
ax.plot(loop_max, y, "|", color=COLORS["gray"], ms=6, mew=1.2, zorder=4)

# Value labels to the right of the max tick
xmax_text = loop_rmsf.max() + 5
for yi, (mean_v, max_v) in enumerate(zip(loop_rmsf, loop_max)):
    ax.text(max_v + 0.4, yi, f"{mean_v:.1f}",
            va="center", fontsize=7.5)

# Reference line at the reference design's value (if present in data)
ref_mask = (names == REF_NAME)
if ref_mask.any():
    ref_rmsf = float(loop_rmsf[ref_mask][0])
    ax.axvline(ref_rmsf, color=COLORS["yellow"], ls="--", lw=1.0, alpha=0.9,
               zorder=1)

ax.set_yticks(y)
ax.set_yticklabels(disp_labels, fontsize=9)
ax.set_xlabel("Loop RMSF (Å)  —  residues 13–20")
ax.set_title("Hairpin loop flexibility across variants")
ax.set_xlim(0, xmax_text)

# Place legend below the plot to avoid overlapping the lowest bars
ax.legend(handles=[
    Patch(color=COLORS["yellow"], label="Reference (100 ASO, unmod. 10-mer)"),
    Patch(color=COLORS["blue"],   label="Concentration series"),
    Patch(color=COLORS["red"],    label="Sequence variant"),
    Line2D([0], [0], marker="|", color=COLORS["gray"], mew=1.2, ms=8,
           ls="", label="per-variant loop max"),
], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
    fontsize=7.5, frameon=False)

fig.subplots_adjust(bottom=0.28)
add_footnote(fig, "Loop residues: relative positions 13–20 of the 32-nt "
                  "1YMO hairpin.  RMSF computed after Kabsch alignment to "
                  "stem residues.  All simulations: ~1 µs, NVT ensemble.",
             y=0.0)

save_fig(fig, "fig9_loop_rmsf_variants")
