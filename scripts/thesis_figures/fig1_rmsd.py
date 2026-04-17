"""
Figure 1 – RMSD vs simulation time.

Two traces: hairpin RMSD and docked ASO RMSD vs the PDB 1YMO NMR model 1
reference structure (Kabsch alignment).  Raw per-frame data shown in pale
background; 50-frame rolling mean shown as the main trace.

Data source:
    /scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/dynamics_results.npz
        keys used: ts, hp_rmsd, aso_rmsd
Output:
    Figures/thesis_results/fig1_rmsd.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

apply_style()

# ── Load data ────────────────────────────────────────────────────────────────
NPZ = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/dynamics_results.npz"
d   = np.load(NPZ)

# Convert timestep (×10⁶ steps) → simulation time (µs)
# LAMMPS: timestep 10 fs, real units → 1×10⁶ steps = 10 ns = 0.01 µs
time_us  = d["ts"] * 0.01
hp_rmsd  = d["hp_rmsd"]
aso_rmsd = d["aso_rmsd"]

# 50-frame smoothing (≈ 25 ns window)
W = 50
hp_sm  = uniform_filter1d(hp_rmsd.astype(float),  W)
aso_sm = uniform_filter1d(aso_rmsd.astype(float), W)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.0))

# Raw traces (faint)
ax.plot(time_us, hp_rmsd,  color=COLORS["lblue"], lw=0.5, alpha=0.35)
ax.plot(time_us, aso_rmsd, color=COLORS["lred"],  lw=0.5, alpha=0.35)

# Smoothed traces
ax.plot(time_us, hp_sm,  color=COLORS["blue"], lw=2.0, label="Hairpin (32 nt)")
ax.plot(time_us, aso_sm, color=COLORS["red"],  lw=2.0, label="Docked ASO (10 nt)")

# Reference lines: time-averaged mean
ax.axhline(hp_rmsd.mean(),  color=COLORS["blue"], ls="--", lw=1.0, alpha=0.55)
ax.axhline(aso_rmsd.mean(), color=COLORS["red"],  ls="--", lw=1.0, alpha=0.55)

ax.set_xlabel("Simulation time (µs)")
ax.set_ylabel("RMSD vs PDB 1YMO (Å)")
ax.set_title("Structural deviation from NMR reference during NVE production run")
ax.legend(loc="upper right")

# Annotation: mean values
ax.text(time_us[-1]*0.98, hp_rmsd.mean() + 0.4,
        f"mean = {hp_rmsd.mean():.1f} Å", ha="right",
        fontsize=9, color=COLORS["blue"])
ax.text(time_us[-1]*0.98, aso_rmsd.mean() - 0.8,
        f"mean = {aso_rmsd.mean():.1f} Å", ha="right",
        fontsize=9, color=COLORS["red"])

ax.set_xlim(0, time_us[-1])
ax.set_ylim(0)

fig.text(0.01, 0.01,
         "Raw trace (pale) + 50-frame rolling mean (~25 ns window). "
         "Reference: PDB 1YMO chain A, residues 15–46.",
         fontsize=7, color=COLORS["gray"], style="italic")

save_fig(fig, "fig1_rmsd")
