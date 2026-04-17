"""
Figure 4 – Z-score of docked-ASO RMSF vs the free-ASO ensemble.

``z = (RMSF_docked − μ_free) / σ_free``

A negative z-score marks a bead that is dynamically suppressed relative
to the free-state ensemble, i.e. restrained by the hairpin.  Blue bars
fall below zero, red bars above; dashed lines at |z| = 2 mark the
conventional significance threshold.

With the corrected free-ASO RMSF reference (post-Kabsch alignment; see
``02_dynamics_analysis.py``), z-scores sit in the ±5 range typical for
single-simulation comparisons, not the ±700 range produced by the
earlier un-aligned reference.

Data source:
    Week 8/analysis/dynamics_results.npz  → z_score
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
from matplotlib.patches import Patch

apply_style()

DYN = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/dynamics_results.npz"
PRS = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/parsed_data.npz"
d = load_or_fake(DYN, make_dynamics_results)
p = load_or_fake(PRS, make_parsed_data)

z      = np.asarray(d["z_score"], dtype=float)
labels = np.asarray(p["aso_labels"])
x      = np.arange(len(z))

# Guard: if the pipeline hasn't been rerun with the RMSF fix yet,
# z magnitudes will be enormous.  Clip for display and note in footnote.
RAW_MAX = float(np.max(np.abs(z)))
clipped = RAW_MAX > 10
if clipped:
    z_display = np.clip(z, -10, 10)
    print(f"  [warn] |z|max = {RAW_MAX:.1f} > 10 — clipping to ±10 for display; "
          f"re-run 02_dynamics_analysis.py with the Kabsch-aligned free-ASO RMSF.")
else:
    z_display = z

bar_colors = [COLORS["blue"] if zi < 0 else COLORS["red"] for zi in z_display]

fig, ax = plt.subplots(figsize=(5.8, 3.4))
ax.yaxis.grid(True, alpha=0.4)               # gridlines help read bar heights

bars = ax.bar(x, z_display, color=bar_colors, edgecolor="white",
              linewidth=0.6, zorder=3)

ax.axhline( 2, color=COLORS["gray"], ls="--", lw=1.0)
ax.axhline(-2, color=COLORS["gray"], ls="--", lw=1.0)
ax.axhline( 0, color="black", lw=0.8)

for bar, zi in zip(bars, z_display):
    off = 0.12 if zi >= 0 else -0.12
    va  = "bottom" if zi >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width() / 2, zi + off,
            f"{zi:+.1f}", ha="center", va=va,
            fontsize=7.5, fontweight="regular",
            color=COLORS["blue"] if zi < 0 else COLORS["red"])

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_xlabel("ASO bead")
ax.set_ylabel("RMSF z-score  (docked vs free)")
ax.set_title("Binding-induced flexibility change")
ax.set_xlim(-0.6, len(x) - 0.4)

# Legend in the empty upper region — bars are all negative in the
# corrected pipeline, so the top half of the axes is free space.
ax.legend(handles=[
    Patch(color=COLORS["blue"], label="constrained  (z < 0)"),
    Patch(color=COLORS["red"],  label="more flexible  (z > 0)"),
], loc="upper right", ncol=1, fontsize=7.5)

footnote = ("z = (RMSF_docked − μ_free) / σ_free.  n_free = 99 ASOs.  "
            "Dashed lines at |z| = 2 mark the usual significance threshold.")
if clipped:
    footnote += f"  Raw |z| up to {RAW_MAX:.0f}; clipped to ±10 for display."
add_footnote(fig, footnote)

save_fig(fig, "fig4_zscore")
