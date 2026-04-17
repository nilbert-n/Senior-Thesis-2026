"""
Figure 6 – Triple-base-pair (TBP) contact probabilities.

Horizontal bar chart of the contact probability for each of the five
key ASO–hairpin base-pairing interactions identified in the published
1YMO literature.  The dashed line marks the mean contact probability
averaged over all 10 × 32 ASO × hairpin bead pairs in this run.

Pairs whose contact probability is exactly zero (e.g. U10·A36 in the
current Week 8 run) are shown as hollow markers on the axis baseline so
the reader can see they were measured rather than silently omitted.

Data source:
    Week 8/analysis/contact_results.npz  → tbp_probs, tbp_names, prob10
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote
from _preview_data import load_or_fake, make_contact_results

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

apply_style()

CON = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/contact_results.npz"
c = load_or_fake(CON, make_contact_results)

tbp_probs = np.asarray(c["tbp_probs"], dtype=float)
tbp_names = np.asarray(c["tbp_names"])
prob10    = np.asarray(c["prob10"],    dtype=float)

mean_all  = prob10.mean()

order      = np.argsort(tbp_probs)[::-1]
names_sort = tbp_names[order]
probs_sort = tbp_probs[order]

# Colour by "anchor" residue of the ASO
def _contact_color(name):
    if name.startswith("U5"):  return COLORS["red"]
    if name.startswith("G6"):  return COLORS["blue"]
    return COLORS["green"]     # U10

bar_cols = [_contact_color(n) for n in names_sort]

fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.xaxis.grid(True, alpha=0.35)

y = np.arange(len(names_sort))

# Show zero-probability pairs as hollow markers on the axis so they
# aren't silently absent from the figure.
nonzero_mask = probs_sort > 0
nz_bars = ax.barh(y[nonzero_mask], probs_sort[nonzero_mask] * 100,
                  color=[bar_cols[i] for i in np.where(nonzero_mask)[0]],
                  edgecolor="white", linewidth=0.6, height=0.55)
for i in np.where(~nonzero_mask)[0]:
    ax.plot(0, y[i], marker="o", mfc="none", mec=bar_cols[i],
            mew=1.2, ms=7, zorder=3)
    ax.text(max(probs_sort) * 100 * 0.02, y[i],
            "not observed", va="center", fontsize=7,
            color=COLORS["gray"])

ax.axvline(mean_all * 100, color=COLORS["gray"], ls="--", lw=1.0,
           label=f"All-pairs mean: {mean_all*100:.3f}%")

for bar, p in zip(nz_bars, probs_sort[nonzero_mask]):
    ax.text(bar.get_width() * 1.03,
            bar.get_y() + bar.get_height() / 2,
            f"{p*100:.3f}%", va="center", fontsize=8)

ax.set_yticks(y)
ax.set_yticklabels(names_sort, fontsize=9)
ax.set_xlabel("P(contact)  (%)")
ax.set_title("Key TBP contact probabilities")
ax.set_xlim(0, max(probs_sort) * 100 * 1.55 if probs_sort.max() else 0.05)

# Combined legend: colour groups AND the all-pairs mean reference line
from matplotlib.lines import Line2D
legend_items = [
    Patch(color=COLORS["red"],   label="U5 contacts"),
    Patch(color=COLORS["blue"],  label="G6 contacts"),
    Patch(color=COLORS["green"], label="U10 anchor"),
    Line2D([0], [0], color=COLORS["gray"], ls="--", lw=1.0,
           label=f"All-pairs mean ({mean_all*100:.3f}%)"),
]
# Legend below the plot — avoids covering the bars
ax.legend(handles=legend_items, loc="upper center",
          bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=7.5, frameon=False)

add_footnote(fig, "TBP pairs follow 1YMO chain-A residue numbering.  "
                  "Contact cutoff: 16.56 Å (CG model).  'not observed' = "
                  "zero contacts in this 3.64 µs NVE run.",
             reserve=0.40)

save_fig(fig, "fig6_tbp_contacts")
