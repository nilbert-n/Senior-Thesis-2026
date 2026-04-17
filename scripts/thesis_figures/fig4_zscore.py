"""
Figure 4 – Z-score of docked-ASO RMSF relative to the 99 free-ASO ensemble.

Z = (RMSF_docked − μ_free) / σ_free

A negative z-score indicates a bead that is dynamically suppressed relative
to the free-state ensemble — i.e., its motion is restrained by the hairpin.
Dashed ±2 lines mark the conventional significance threshold.

Data source:
    dynamics_results.npz → z_score
    parsed_data.npz      → aso_labels
Output:
    Figures/thesis_results/fig4_zscore.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt

apply_style()

DYN = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/dynamics_results.npz"
PRS = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/parsed_data.npz"

d      = np.load(DYN)
p      = np.load(PRS, allow_pickle=True)
z      = d["z_score"]          # (10,)
labels = p["aso_labels"]

x = np.arange(len(z))

# Color: blue = constrained (z < 0), red = more flexible (z > 0)
bar_colors = [COLORS["blue"] if zi < 0 else COLORS["red"] for zi in z]

fig, ax = plt.subplots(figsize=(6.5, 4.0))

bars = ax.bar(x, z, color=bar_colors, edgecolor="white", linewidth=0.6, zorder=3)

# Significance thresholds
ax.axhline( 2, color=COLORS["gray"], ls="--", lw=1.1, label="|z| = 2  (significance)")
ax.axhline(-2, color=COLORS["gray"], ls="--", lw=1.1)
ax.axhline( 0, color="black", lw=0.8)

# Value labels on bars
for bar, zi in zip(bars, z):
    yoff = 0.15 if zi >= 0 else -0.35
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + yoff,
            f"{zi:+.2f}", ha="center", va="bottom" if zi >= 0 else "top",
            fontsize=8.5, fontweight="bold",
            color=COLORS["blue"] if zi < 0 else COLORS["red"])

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_xlabel("ASO bead")
ax.set_ylabel("RMSF z-score  (docked vs free-ASO ensemble)")
ax.set_title("Binding-induced dynamic suppression per ASO bead")
ax.legend(loc="upper right")
ax.set_xlim(-0.6, len(x) - 0.4)

# Legend patches
from matplotlib.patches import Patch
leg2 = ax.legend(
    handles=[
        Patch(color=COLORS["blue"], label="Constrained  (z < 0)"),
        Patch(color=COLORS["red"],  label="More flexible  (z > 0)"),
    ],
    loc="lower right", fontsize=9)
ax.add_artist(leg2)
ax.legend([ax.lines[0]], ["|z| = 2  threshold"], loc="upper right")

fig.text(0.01, 0.01,
         "z = (RMSF_docked − μ_free) / σ_free.  "
         "n_free = 99 ASOs; 3.64 µs NVE run.",
         fontsize=7, color=COLORS["gray"], style="italic")

save_fig(fig, "fig4_zscore")
