"""
Figure 1 – RMSD of hairpin and docked ASO vs the PDB 1YMO reference.

The raw per-frame trace is shown as a faint background, the 50-frame
(~25 ns) rolling mean as the main trace. Time-averaged means are
annotated in the legend rather than painted onto the traces.

Data source:
    Week 8/analysis/dynamics_results.npz  (ts, hp_rmsd, aso_rmsd)
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote
from _preview_data import load_or_fake, make_dynamics_results

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

apply_style()

NPZ = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/dynamics_results.npz"
d = load_or_fake(NPZ, make_dynamics_results)

time_us  = np.asarray(d["ts"]) * 0.01    # ×10⁶ steps → µs  (10 fs/step)
hp_rmsd  = np.asarray(d["hp_rmsd"]).astype(float)
aso_rmsd = np.asarray(d["aso_rmsd"]).astype(float)

W = 50                                     # ~25 ns rolling window
hp_sm  = uniform_filter1d(hp_rmsd,  W)
aso_sm = uniform_filter1d(aso_rmsd, W)

fig, ax = plt.subplots(figsize=(5.2, 3.2))

ax.plot(time_us, hp_rmsd,  color=COLORS["lblue"], lw=0.4, alpha=0.5)
ax.plot(time_us, aso_rmsd, color=COLORS["lred"],  lw=0.4, alpha=0.5)

ax.plot(time_us, hp_sm,  color=COLORS["blue"], lw=1.6,
        label=f"Hairpin (32 nt)  μ={hp_rmsd.mean():.1f} Å")
ax.plot(time_us, aso_sm, color=COLORS["red"], lw=1.6,
        label=f"Docked ASO (10 nt)  μ={aso_rmsd.mean():.1f} Å")

ax.set_xlabel("Simulation time (µs)")
ax.set_ylabel("RMSD to 1YMO (Å)")
ax.set_title("RMSD vs NMR reference")
ax.set_xlim(0, time_us[-1])
ax.set_ylim(0, max(hp_rmsd.max(), aso_rmsd.max()) * 1.08)
ax.legend(loc="upper right")

add_footnote(fig, "Pale trace: raw frames.  Solid trace: 50-frame rolling "
                  "mean (~25 ns).  Kabsch-aligned to PDB 1YMO chain A, "
                  "residues 15–46.")

save_fig(fig, "fig1_rmsd")
