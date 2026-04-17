"""
Figure 3 – Per-bead RMSF: docked ASO vs the 99 free ASO ensemble.

The docked ASO's bead-by-bead RMSF is compared to the mean ± 2σ envelope
from 99 free (unbound) ASOs.  Beads whose RMSF falls below the free-ASO
mean are dynamically constrained by the hairpin; beads above the mean are
transiently disordered at the binding interface.

Data source:
    dynamics_results.npz → aso_rmsf, free_rmsf_mean, free_rmsf_std
    parsed_data.npz      → aso_labels
Output:
    Figures/thesis_results/fig3_rmsf_comparison.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt

apply_style()

DYN  = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/dynamics_results.npz"
PRS  = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/parsed_data.npz"

d    = np.load(DYN)
p    = np.load(PRS, allow_pickle=True)

aso_rmsf  = d["aso_rmsf"]           # (10,)
mu        = d["free_rmsf_mean"]      # (10,)
sigma     = d["free_rmsf_std"]       # (10,)
labels    = p["aso_labels"]          # ['1(G)', '2(G)', ...]

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(6.5, 4.2))

# ±2σ and ±1σ bands for free-ASO ensemble
ax.fill_between(x, mu - 2*sigma, mu + 2*sigma,
                color=COLORS["lgray"], alpha=0.50, label="Free-ASO ±2σ")
ax.fill_between(x, mu - sigma, mu + sigma,
                color=COLORS["lgray"], alpha=0.80, label="Free-ASO ±1σ")

# Free-ASO mean
ax.plot(x, mu, color=COLORS["gray"], lw=1.5, ls="--", label="Free-ASO mean (n=99)")

# Docked ASO
ax.plot(x, aso_rmsf, "o-",
        color=COLORS["red"], lw=2.0, ms=6,
        markerfacecolor="white", markeredgewidth=1.8,
        label="Docked ASO")

# Shade beads that are significantly constrained (below −1σ)
for i, (yr, ym, ys) in enumerate(zip(aso_rmsf, mu, sigma)):
    if yr < ym - ys:
        ax.axvspan(i - 0.4, i + 0.4, color=COLORS["lblue"], alpha=0.18, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_xlabel("ASO bead")
ax.set_ylabel("RMSF (Å)")
ax.set_title("Thermal flexibility: docked ASO vs free-ASO ensemble")
ax.legend(loc="upper left")
ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylim(0)

# Annotate constrained region
below = aso_rmsf < mu - sigma
if below.any():
    first_b, last_b = np.where(below)[0][[0, -1]]
    ax.annotate("Constrained\n(bound interface)",
                xy=(first_b, aso_rmsf[first_b]), xycoords="data",
                xytext=(first_b + 1.0, aso_rmsf[first_b] - 2.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=1.0),
                fontsize=8, color=COLORS["blue"], ha="center")

fig.text(0.01, 0.01,
         "Blue shading: beads constrained >1σ below free-ASO mean.  "
         "Data: 3.64 µs NVE run, 100-ASO system (PDB 1YMO, chain A).",
         fontsize=7, color=COLORS["gray"], style="italic")

save_fig(fig, "fig3_rmsf_comparison")
