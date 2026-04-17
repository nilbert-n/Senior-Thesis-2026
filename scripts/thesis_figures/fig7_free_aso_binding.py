"""
Figure 7 – Free-ASO binding at high ASO loading (2-panel).

Left panel (A): binding probability for each of the 99 free ASOs, sorted
from highest to lowest.  The docked ASO (mol 1) is not included.

Right panel (B): space–time distance heatmap — minimum distance between
each free ASO and the hairpin, for all 99 ASOs over the full trajectory.
ASOs are sorted by binding probability (highest at bottom).  The dashed
line separates ever-bound from never-bound ASOs.

Data source:
    binding_results.npz → bind_prob, dist_to_hp, ts
Output:
    Figures/thesis_results/fig7_free_aso_binding.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt

apply_style()

BND = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/binding_results.npz"
b   = np.load(BND)

bind_prob  = b["bind_prob"]   # (99,)
dist_to_hp = b["dist_to_hp"]  # (99, F)
ts         = b["ts"] * 0.01   # µs

BIND_CUTOFF = 16.56  # Å — from original script

n_free       = len(bind_prob)
sorted_idx   = np.argsort(bind_prob)[::-1]
bp_sorted    = bind_prob[sorted_idx] * 100       # percent
dist_sorted  = dist_to_hp[sorted_idx]
n_ever_bound = int((bind_prob > 0).sum())

# ── Layout ────────────────────────────────────────────────────────────────────
fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(11, 4.5),
                                       gridspec_kw={"width_ratios": [1, 1.8]})
fig.subplots_adjust(wspace=0.30)

# ── A: Binding probability bar ─────────────────────────────────────────────
bar_colors = [COLORS["red"] if b > 0 else COLORS["lgray"] for b in bp_sorted]
ax_bar.bar(np.arange(n_free), bp_sorted, color=bar_colors,
           width=1.0, edgecolor="none")

ax_bar.set_xlabel("Free ASO (sorted by binding probability)")
ax_bar.set_ylabel("Binding probability (%)")
ax_bar.set_title(f"A  –  Free-ASO Binding Probability\n"
                 f"({n_ever_bound}/{n_free} ever contacted hairpin)")

# Annotate bound fraction
ax_bar.axvline(n_ever_bound - 0.5, color=COLORS["gray"], ls="--", lw=1.1)
ax_bar.text(n_ever_bound + 1, bp_sorted.max() * 0.6,
            f"Never bound\n({n_free - n_ever_bound} ASOs)",
            fontsize=8.5, color=COLORS["gray"])
ax_bar.set_xlim(0, n_free)

from matplotlib.patches import Patch
ax_bar.legend(handles=[
    Patch(color=COLORS["red"],   label=f"Contacted (≤{BIND_CUTOFF:.0f} Å)"),
    Patch(color=COLORS["lgray"], label="Never bound"),
], fontsize=9, loc="upper right")

# ── B: Distance heatmap ────────────────────────────────────────────────────
# Cap distances at 200 Å for visualisation (free ASOs in a ~160 Å box)
D_cap = np.clip(dist_sorted, 0, 200)

im = ax_heat.imshow(
    D_cap, aspect="auto", origin="lower",
    cmap="RdYlGn_r",
    vmin=0, vmax=200,
    extent=[ts[0], ts[-1], 0, n_free])

cbar = fig.colorbar(im, ax=ax_heat, shrink=0.88, pad=0.02)
cbar.set_label("Min distance to hairpin (Å)", fontsize=9)

# Bound / unbound separator
ax_heat.axhline(n_ever_bound + 0.5, color="white", ls="--", lw=1.5)
ax_heat.text(ts[-1] * 0.02, n_ever_bound + 1.5,
             f"Bound ({n_ever_bound})", color="white", fontsize=8.5)
ax_heat.text(ts[-1] * 0.02, n_ever_bound - 4,
             f"Never bound ({n_free - n_ever_bound})", color="white", fontsize=8.5)


ax_heat.set_xlabel("Simulation time (µs)")
ax_heat.set_ylabel("Free ASO rank (sorted by binding probability)")
ax_heat.set_title(f"B  –  Distance Heatmap: All {n_free} Free ASOs\n"
                  f"(distances capped at 200 Å for display)")

fig.text(0.01, 0.01,
         f"Binding cutoff: {BIND_CUTOFF} Å (CG model).  "
         "Colour scale: green = far, red = close to hairpin.  "
         "100-ASO NVE run, 3.64 µs.",
         fontsize=7, color=COLORS["gray"], style="italic")

save_fig(fig, "fig7_free_aso_binding")
