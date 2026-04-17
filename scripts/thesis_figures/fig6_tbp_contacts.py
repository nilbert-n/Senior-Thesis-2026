"""
Figure 6 – Triple-base-pair (TBP) contact probabilities.

Horizontal bar chart of the contact probability for each of the five
key base-pairing interactions identified in the literature between the
ASO and the 1YMO RNA hairpin loop.  Error bars are not available for a
single simulation; the dashed line marks the overall mean contact probability
averaged over all ASO × hairpin bead pairs.

Data source:
    contact_results.npz → tbp_probs, tbp_names, prob10
Output:
    Figures/thesis_results/fig6_tbp_contacts.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt

apply_style()

CON = "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 8/analysis/contact_results.npz"
c   = np.load(CON)

tbp_probs = c["tbp_probs"]   # (5,)
tbp_names = c["tbp_names"]   # ['U5·A21', ...]
prob10    = c["prob10"]       # (10, 32) – all-pairs

mean_all  = prob10.mean()
max_all   = prob10.max()

# Sort by probability for clarity
order      = np.argsort(tbp_probs)[::-1]
names_sort = tbp_names[order]
probs_sort = tbp_probs[order]

# Colour by contact group: U5 contacts (red), G6 contacts (blue), U10 (green)
def _contact_color(name):
    if name.startswith("U5"):  return COLORS["red"]
    if name.startswith("G6"):  return COLORS["blue"]
    return COLORS["green"]

bar_cols = [_contact_color(n) for n in names_sort]

fig, ax = plt.subplots(figsize=(6.5, 3.8))

y = np.arange(len(names_sort))
bars = ax.barh(y, probs_sort * 100, color=bar_cols,
               edgecolor="white", linewidth=0.6, height=0.55)

# Mean all-pair reference line
ax.axvline(mean_all * 100, color=COLORS["gray"], ls="--", lw=1.2,
           label=f"Mean contact prob. (all pairs): {mean_all*100:.4f}%")

# Value labels
for bar, p in zip(bars, probs_sort):
    ax.text(bar.get_width() + max(probs_sort)*0.02*100,
            bar.get_y() + bar.get_height() / 2,
            f"{p*100:.4f}%", va="center", fontsize=9)

ax.set_yticks(y)
ax.set_yticklabels(names_sort, fontsize=10)
ax.set_xlabel("Contact probability (%)")
ax.set_title("Key TBP contact probabilities  (docked ASO, 3.64 µs NVE)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, max(probs_sort) * 100 * 1.45)

# Colour legend
from matplotlib.patches import Patch
col_legend = [
    Patch(color=COLORS["red"],   label="U5 contacts"),
    Patch(color=COLORS["blue"],  label="G6 contacts"),
    Patch(color=COLORS["green"], label="U10 anchor"),
]
ax.legend(handles=col_legend, loc="lower right", fontsize=9)

fig.text(0.01, 0.01,
         "TBP pairs follow literature residue numbering for 1YMO (chain A).  "
         "Contact cutoff: 16.56 Å (CG model).",
         fontsize=7, color=COLORS["gray"], style="italic")

save_fig(fig, "fig6_tbp_contacts")
