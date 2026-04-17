"""
Figure 2 – Radius of gyration of hairpin alone and of the ASO–hairpin
complex, computed with the nearest-image convention (see the updated
``radius_of_gyration_complex`` in ``Single_Chain/Week 8/analysis/
02_dynamics_analysis.py``).

The earlier version of this figure showed ``Rg_complex ≈ 2300 Å`` — an
artefact of concatenating ASO and hairpin coordinates without
periodic-boundary unwrapping. After the fix the complex Rg is slightly
larger than the hairpin-alone Rg, as expected.

Data source:
    Week 8/analysis/dynamics_results.npz  (ts, rog_hp, rog_complex)
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

time_us = np.asarray(d["ts"]) * 0.01
rog_hp  = np.asarray(d["rog_hp"]).astype(float)
rog_cx  = np.asarray(d["rog_complex"]).astype(float)

# Sanity check: warn (but still plot) if we detect the old unwrapped-
# coords artefact (Rg > 2× box size ≈ 320 Å would be suspicious).
if rog_cx.max() > 200:
    print("  [warn] rog_complex exceeds 200 Å — likely unwrapped-coords "
          "bug; re-run 02_dynamics_analysis.py with the nearest-image fix.")

W = 50
hp_sm = uniform_filter1d(rog_hp, W)
cx_sm = uniform_filter1d(rog_cx, W)

fig, ax = plt.subplots(figsize=(5.2, 3.2))

ax.fill_between(time_us, hp_sm, cx_sm,
                color=COLORS["lblue"], alpha=0.35, label="ASO contribution")
ax.plot(time_us, rog_hp, color=COLORS["yellow"], lw=0.4, alpha=0.5)
ax.plot(time_us, rog_cx, color=COLORS["blue"],   lw=0.4, alpha=0.5)
ax.plot(time_us, hp_sm,  color=COLORS["yellow"], lw=1.6,
        label=f"Hairpin  μ={rog_hp.mean():.1f} Å")
ax.plot(time_us, cx_sm,  color=COLORS["blue"],   lw=1.6,
        label=f"Complex  μ={rog_cx.mean():.1f} Å")

ax.set_xlabel("Simulation time (µs)")
ax.set_ylabel("Radius of gyration (Å)")
ax.set_title("Complex compactness")
ax.set_xlim(0, time_us[-1])
ax.legend(loc="lower right")

add_footnote(fig, "Rg computed after nearest-image reconstruction of the "
                  "ASO around the hairpin COM (periodic cubic box, 160 Å).")

save_fig(fig, "fig2_rg")
