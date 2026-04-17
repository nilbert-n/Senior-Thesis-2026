"""
Figure 5 – ASO–hairpin contact probability map.

Heatmap of the time-averaged contact probability P(contact) for every
ASO bead × hairpin bead pair (cutoff 16.56 Å).  TBP (triple-base-pair)
contacts identified from the literature are marked with open squares.

The colour scale is normalised to the data maximum (not to 1.0) because
the overall contact probabilities in this CG run are low; this choice
preserves the spatial pattern of the binding footprint.

Data source:
    contact_results.npz → prob10, tbp_names, tbp_probs
    parsed_data.npz     → aso_labels, hp_labels
Output:
    Figures/thesis_results/fig5_contact_map.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

apply_style()

CON = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/contact_results.npz"
PRS = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/parsed_data.npz"

c      = np.load(CON)
p      = np.load(PRS, allow_pickle=True)

prob10     = c["prob10"]       # (10, 32)
tbp_names  = c["tbp_names"]   # ['U5·A21', ...]
hp_labels  = p["hp_labels"]   # ['11(G)', '12(C)', ...]
aso_labels = p["aso_labels"]  # ['1(G)', ...]

# TBP contact positions (ASO bead idx, HP bead idx) — matches 03_contact_analysis.py
TBP = [
    ("U5·A21",  4,  6),
    ("U5·A31",  4, 16),
    ("G6·C20",  5,  5),
    ("G6·A32",  5, 17),
    ("U10·A36", 9, 21),
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.0))

# Custom colormap: white → light blue → blue → deep blue
cmap = mcolors.LinearSegmentedColormap.from_list(
    "contact_blues",
    ["#f7fbff", "#9ecae1", "#3182bd", "#08306b"])

vmax = max(prob10.max(), 1e-4)
im = ax.imshow(prob10, aspect="auto", origin="lower",
               cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("Contact probability  (cutoff 16.56 Å)", fontsize=10)

# Tick labels
n_aso, n_hp = prob10.shape
ax.set_yticks(range(n_aso))
ax.set_yticklabels(aso_labels, fontsize=8)
# Show every 4th hp label to avoid crowding
hp_ticks = list(range(0, n_hp, 4))
ax.set_xticks(hp_ticks)
ax.set_xticklabels([hp_labels[i] for i in hp_ticks], fontsize=8,
                   rotation=40, ha="right")

# Mark loop region on x-axis (HP beads 2–9 in 0-based relative idx = residues 13–20)
# From rmsf_campaign_summary: loop_residues = 13,14,15,16,17,18,19,20
# Relative to HP bead 0 (residue 11) → loop bead indices 2-9
ax.axvspan(1.5, 9.5, color=COLORS["lred"], alpha=0.10, zorder=0, label="Hairpin loop")

# Mark TBP contacts
for name, ai, hi, in TBP:
    ax.plot(hi, ai, "s", ms=9, mfc="none",
            mec=COLORS["red"], mew=1.8, zorder=5)

ax.set_xlabel("Hairpin bead")
ax.set_ylabel("Docked ASO bead")
ax.set_title("ASO–hairpin contact probability map  (docked ASO, 3.64 µs NVE)")

# Legend
legend_handles = [
    Line2D([0], [0], marker="s", ms=8, mfc="none",
           mec=COLORS["red"], mew=1.8, lw=0, label="TBP contact site"),
    plt.Rectangle((0, 0), 1, 1, fc=COLORS["lred"], alpha=0.3,
                  label="Hairpin loop  (res. 13–20)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

fig.text(0.01, 0.01,
         "Colorscale normalised to data maximum (not 1.0) to preserve "
         "spatial pattern.  TBP: triple-base-pair contacts from literature.",
         fontsize=7, color=COLORS["gray"], style="italic")

save_fig(fig, "fig5_contact_map")
