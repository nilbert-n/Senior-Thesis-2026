"""
Figure 7 – Free-ASO binding at high loading (2-panel).

Left  (A): sorted binding probability for the 99 free ASOs.  Since most
           ASOs never contact the hairpin, we draw only the ever-bound
           ASOs as bars and annotate the never-bound count in text
           (a bar chart with 88 zero-height bars carries no information).

Right (B): minimum ASO–hairpin distance through time for the 11
           ever-bound ASOs (zoomed-in rank axis) plus a small inset
           showing the full 99-ASO matrix for completeness.  The 16.56 Å
           contact threshold is drawn as a horizontal line in the inset.

Data source:
    Week 8/analysis/binding_results.npz  → bind_prob, dist_to_hp, ts
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote
from _preview_data import load_or_fake, make_binding_results

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

apply_style()

BND = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/binding_results.npz"
b = load_or_fake(BND, make_binding_results)

bind_prob  = np.asarray(b["bind_prob"],  dtype=float)
dist_to_hp = np.asarray(b["dist_to_hp"], dtype=float)
ts         = np.asarray(b["ts"],         dtype=float) * 0.01   # µs

BIND_CUTOFF = 16.56   # Å

n_free       = len(bind_prob)
order_desc   = np.argsort(bind_prob)[::-1]
bp_sorted    = bind_prob[order_desc] * 100
dist_sorted  = dist_to_hp[order_desc]
n_bound      = int((bind_prob > 0).sum())

fig, (ax_bar, ax_heat) = plt.subplots(
    1, 2, figsize=(9.2, 3.6),
    gridspec_kw={"width_ratios": [1.0, 1.6]})
fig.subplots_adjust(wspace=0.32)

# ── A: Binding probability (only the ever-bound ASOs) ───────────────────────
bp_bound = bp_sorted[:n_bound]
ax_bar.xaxis.grid(True, alpha=0.35)
bars = ax_bar.barh(np.arange(n_bound), bp_bound,
                   color=COLORS["red"], edgecolor="white", linewidth=0.5,
                   height=0.62)
for i, bp in enumerate(bp_bound):
    ax_bar.text(bp * 1.03, i, f"{bp:.3f}%", va="center", fontsize=7)

ax_bar.set_yticks(np.arange(n_bound))
ax_bar.set_yticklabels([f"ASO #{idx+2}" for idx in order_desc[:n_bound]],
                        fontsize=7)
ax_bar.set_xlabel("P(binding) (%)")
# The sparsity info goes into the title itself — no floating text needed.
ax_bar.set_title(
    f"A  Ever-bound ASOs\n"
    f"({n_bound} of {n_free} ASOs entered the 16.56 Å cutoff)",
    fontsize=10, loc="left")
ax_bar.invert_yaxis()
ax_bar.set_xlim(0, max(bp_bound) * 1.25 if n_bound else 0.05)

# ── B: Distance heatmap (ever-bound only, with mini-inset for full matrix) ──
D_bound_cap = np.clip(dist_sorted[:n_bound], 0, 200)
extent = [ts[0], ts[-1], n_bound, 0]   # row 0 = strongest binder
im = ax_heat.imshow(D_bound_cap, aspect="auto", origin="upper",
                    cmap="RdYlGn_r", vmin=0, vmax=200,
                    extent=extent, interpolation="nearest")
cbar = fig.colorbar(im, ax=ax_heat, shrink=0.9, pad=0.02)
cbar.set_label("min. distance to hairpin (Å)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Mark the binding cutoff on the colour bar
cbar.ax.axhline(BIND_CUTOFF, color="white", lw=1.0)
cbar.ax.text(1.05, BIND_CUTOFF, f"  {BIND_CUTOFF:g}  Å",
             transform=cbar.ax.get_yaxis_transform(),
             fontsize=7, va="center", color=COLORS["gray"])

ax_heat.set_yticks(np.arange(n_bound) + 0.5)
ax_heat.set_yticklabels(
    [f"#{idx+2}" for idx in order_desc[:n_bound]], fontsize=7)
ax_heat.set_xlabel("Simulation time (µs)")
ax_heat.set_ylabel("Ever-bound ASO (rank)")
ax_heat.set_title("B  Distance-to-hairpin, ever-bound ASOs")

add_footnote(fig, f"Binding cutoff: {BIND_CUTOFF} Å (CG model).  "
                  "Panel A lists only the ASOs that ever entered the "
                  "cutoff; panel B shows their distance trace over the "
                  "3.64 µs NVE run.  Inset: full 99-ASO distance matrix.",
             reserve=0.22)

# Inset: full 99-ASO matrix at low resolution, placed manually via
# fig.add_axes so we don't get spurious wrapper axes (inset_axes from
# axes_grid1 renders a phantom axes that picks up the shared colormap).
# Position is computed AFTER add_footnote so the axes position is final.
pos = ax_heat.get_position()
inset_w = pos.width  * 0.30
inset_h = pos.height * 0.40
ax_inset = fig.add_axes([pos.x0 + 0.008,
                          pos.y0 + 0.008,
                          inset_w, inset_h])
ax_inset.imshow(np.clip(dist_sorted, 0, 200),
                aspect="auto", origin="upper",
                cmap="RdYlGn_r", vmin=0, vmax=200,
                extent=[ts[0], ts[-1], n_free, 0], interpolation="nearest")
ax_inset.axhline(n_bound, color="white", lw=1.0)
ax_inset.set_xticks([]); ax_inset.set_yticks([])
for spine in ax_inset.spines.values():
    spine.set_edgecolor("white"); spine.set_linewidth(0.8)
ax_inset.set_title(f"all {n_free} ASOs", fontsize=7, pad=2,
                   color="white")

save_fig(fig, "fig7_free_aso_binding")
