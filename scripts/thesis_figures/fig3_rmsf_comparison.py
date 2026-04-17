"""
Figure 3 – Per-bead RMSF: docked ASO vs the 99 free-ASO ensemble.

The free-ASO RMSF values here are *internal* fluctuations, i.e. computed
after per-frame Kabsch alignment of each free ASO onto its own
trajectory-averaged mean structure.  The earlier version of this figure
compared the docked ASO's Kabsch-aligned RMSF against a free-ASO RMSF
that was only COM-centred, which mixed in rigid-body rotational
tumbling and inflated the free-ASO RMSF by roughly an order of
magnitude.  The corrected free-ASO RMSF is produced by the updated
``02_dynamics_analysis.py`` in ``Single_Chain/Week 8/analysis``.

Data source:
    Week 8/analysis/dynamics_results.npz  → aso_rmsf, free_rmsf_mean,
                                            free_rmsf_std
    Week 8/analysis/parsed_data.npz       → aso_labels
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote
from _preview_data import (load_or_fake, make_dynamics_results,
                           make_parsed_data)

import numpy as np
import matplotlib.pyplot as plt

apply_style()

DYN = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/dynamics_results.npz"
PRS = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/parsed_data.npz"
d = load_or_fake(DYN, make_dynamics_results)
p = load_or_fake(PRS, make_parsed_data)

aso_rmsf = np.asarray(d["aso_rmsf"], dtype=float)
mu       = np.asarray(d["free_rmsf_mean"], dtype=float)
sigma    = np.asarray(d["free_rmsf_std"],  dtype=float)
labels   = np.asarray(p["aso_labels"])

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(5.8, 3.4))

ax.fill_between(x, mu - 2*sigma, mu + 2*sigma,
                color=COLORS["lgray"], alpha=0.6, label="Free-ASO ±2σ")
ax.fill_between(x, mu - sigma, mu + sigma,
                color=COLORS["gray"],  alpha=0.35, label="Free-ASO ±1σ")
ax.plot(x, mu, color=COLORS["gray"], lw=1.2, ls="--",
        label="Free-ASO mean (n = 99)")

ax.plot(x, aso_rmsf, "o-", color=COLORS["red"], lw=1.6, ms=5,
        markerfacecolor="white", markeredgewidth=1.3, label="Docked ASO")

# Highlight beads that are significantly constrained (docked RMSF < μ−1σ)
constrained = np.where(aso_rmsf < mu - sigma)[0]
for i in constrained:
    ax.axvspan(i - 0.4, i + 0.4, color=COLORS["lblue"], alpha=0.22, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_xlabel("ASO bead")
ax.set_ylabel("RMSF (Å)")
ax.set_title("Thermal flexibility: docked vs free ASOs")
ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylim(0)
ax.legend(loc="upper center", ncol=2)

if len(constrained):
    first = int(constrained[0])
    ax.annotate("constrained\n(bound interface)",
                xy=(first, aso_rmsf[first]),
                xytext=(first + 1.5, aso_rmsf[first] + 0.9),
                arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=0.8),
                fontsize=7.5, color=COLORS["blue"], ha="left")

add_footnote(fig, "Blue shading: beads where docked RMSF < free-ASO mean "
                  "− 1σ.  Free-ASO RMSF is computed after per-ASO Kabsch "
                  "alignment to remove rigid-body tumbling.")

save_fig(fig, "fig3_rmsf_comparison")
