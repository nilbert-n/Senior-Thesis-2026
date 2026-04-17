"""
Figure 5 – ASO–hairpin contact probability map.

Heatmap of the time-averaged contact probability P(contact) for every
(ASO bead × hairpin bead) pair at a 16.56 Å cutoff.  TBP (triple-base-
pair) contacts from the literature are marked with open red squares.
The hairpin loop region (residues 13–20) is shaded in pale red.

The colour scale is normalised to the data maximum (≈ 10⁻³) because
overall contact probabilities are low in this short NVE run; the
scientific content is the spatial pattern, not the absolute magnitude.

Data source:
    Week 8/analysis/contact_results.npz  → prob10, tbp_names, tbp_probs
    Week 8/analysis/parsed_data.npz      → aso_labels, hp_labels
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote
from _preview_data import (load_or_fake, make_contact_results,
                           make_parsed_data)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

apply_style()

CON = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/contact_results.npz"
PRS = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/parsed_data.npz"
c = load_or_fake(CON, make_contact_results)
p = load_or_fake(PRS, make_parsed_data)

prob10     = np.asarray(c["prob10"])         # (10, 32)
aso_labels = np.asarray(p["aso_labels"])
hp_labels  = np.asarray(p["hp_labels"])

# TBP contact positions (ASO bead idx, HP bead idx) — from literature
TBP = [
    ("U5·A21",  4,  6),
    ("U5·A31",  4, 16),
    ("G6·C20",  5,  5),
    ("G6·A32",  5, 17),
    ("U10·A36", 9, 21),
]

fig, ax = plt.subplots(figsize=(6.2, 3.4))

cmap = mcolors.LinearSegmentedColormap.from_list(
    "contact_blues",
    ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])

vmax = max(prob10.max(), 1e-4)
im = ax.imshow(prob10, aspect="auto", origin="lower",
               cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
cbar.set_label(f"P(contact)   ×10⁻³  (max = {vmax*1e3:.2f})", fontsize=8)
cbar.ax.tick_params(labelsize=7)
# Ticks shown as the raw value in units of 10⁻³ — keeps the label
# and the tick scale consistent (avoids the matplotlib ScalarFormatter
# stamping an independent "1e-4" offset on the colour bar).
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*1e3:.1f}"))

n_aso, n_hp = prob10.shape
ax.set_yticks(range(n_aso))
ax.set_yticklabels(aso_labels, fontsize=7)
hp_ticks = list(range(0, n_hp, 4))
ax.set_xticks(hp_ticks)
ax.set_xticklabels([hp_labels[i] for i in hp_ticks], fontsize=7,
                   rotation=40, ha="right")

# Loop region (residues 13–20 = HP bead indices 2–9)
ax.axvspan(1.5, 9.5, color=COLORS["lred"], alpha=0.18, zorder=0)

for name, ai, hi in TBP:
    ax.plot(hi, ai, "s", ms=8, mfc="none",
            mec=COLORS["red"], mew=1.5, zorder=5)

ax.set_xlabel("Hairpin bead")
ax.set_ylabel("Docked ASO bead")
ax.set_title("ASO–hairpin contact map")

ax.legend(handles=[
    Line2D([0], [0], marker="s", ms=7, mfc="none",
           mec=COLORS["red"], mew=1.5, lw=0, label="TBP contact (lit.)"),
    Rectangle((0, 0), 1, 1, fc=COLORS["lred"], alpha=0.4,
              label="Hairpin loop (13–20)"),
], loc="upper right", ncol=1)

add_footnote(fig, "Contact cutoff: 16.56 Å (CG model).  Colourscale "
                  "normalised to data maximum.  TBP = triple-base-pair "
                  "contacts from published 1YMO–ASO structures.",
             reserve=0.36)

save_fig(fig, "fig5_contact_map")
