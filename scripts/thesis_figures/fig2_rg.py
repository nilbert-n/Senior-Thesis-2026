"""
Figure 2 – Radius of gyration vs simulation time.

Two traces: hairpin-alone Rg and full complex (hairpin + docked ASO) Rg.
The gap between the two curves reflects the spatial extent contributed
by the docked ASO.

Data source:
    dynamics_results.npz  → keys: ts, rog_hp, rog_complex
Output:
    Figures/thesis_results/fig2_rg.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

apply_style()

NPZ     = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/dynamics_results.npz"
d       = np.load(NPZ)
time_us = d["ts"] * 0.01          # µs
rog_hp  = d["rog_hp"]
rog_cx  = d["rog_complex"]

W    = 50
hp_sm = uniform_filter1d(rog_hp.astype(float), W)
cx_sm = uniform_filter1d(rog_cx.astype(float), W)

fig, ax = plt.subplots(figsize=(6.5, 4.0))

# Fill between to visualise ASO contribution
ax.fill_between(time_us, hp_sm, cx_sm,
                color=COLORS["lblue"], alpha=0.25, label="ASO contribution")

# Raw
ax.plot(time_us, rog_hp, color=COLORS["orange"], lw=0.5, alpha=0.30)
ax.plot(time_us, rog_cx, color=COLORS["blue"],   lw=0.5, alpha=0.30)

# Smoothed
ax.plot(time_us, hp_sm, color=COLORS["orange"], lw=2.0, label=f"Hairpin  (mean {rog_hp.mean():.1f} Å)")
ax.plot(time_us, cx_sm, color=COLORS["blue"],   lw=2.0, label=f"Complex  (mean {rog_cx.mean():.1f} Å)")

# Mean reference lines
ax.axhline(rog_hp.mean(),  color=COLORS["orange"], ls="--", lw=1.0, alpha=0.55)
ax.axhline(rog_cx.mean(),  color=COLORS["blue"],   ls="--", lw=1.0, alpha=0.55)

ax.set_xlabel("Simulation time (µs)")
ax.set_ylabel("Radius of gyration (Å)")
ax.set_title("Compactness of the ASO–hairpin complex over the NVE run")
ax.legend(loc="upper right")
ax.set_xlim(0, time_us[-1])
ax.set_ylim(0)

fig.text(0.01, 0.01,
         "Shaded region: Rg increase attributed to docked ASO.  "
         "50-frame rolling mean (~25 ns window).",
         fontsize=7, color=COLORS["gray"], style="italic")

save_fig(fig, "fig2_rg")
