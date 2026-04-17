"""
Shared publication style for all thesis figures.
Import at the top of each figure script:
    import sys, os; sys.path.insert(0, os.path.dirname(__file__))
    from _style import apply_style, COLORS, OUTDIR, save_fig
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Output directory ─────────────────────────────────────────────────────────
OUTDIR = "/scratch/gpfs/JERELLE/nilbert/Figures/thesis_results"
os.makedirs(OUTDIR, exist_ok=True)

# ── Consistent color palette (colorbrewer-inspired) ──────────────────────────
COLORS = {
    "blue":   "#2166ac",
    "red":    "#d6604d",
    "green":  "#4dac26",
    "orange": "#e08214",
    "purple": "#762a83",
    "gray":   "#636363",
    "lblue":  "#92c5de",   # light blue (fill / band)
    "lred":   "#f4a582",   # light red  (fill / band)
    "lgray":  "#d9d9d9",   # light gray (fill / band)
}

# ── Seaborn-paper-like rcParams (pure matplotlib, no seaborn dependency) ─────
def apply_style():
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,
        "legend.frameon":     True,
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "#cccccc",
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.color":         "#cccccc",
        "grid.alpha":         0.5,
        "grid.linewidth":     0.5,
        "lines.linewidth":    1.6,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "figure.dpi":         150,       # screen preview
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":       42,        # embed fonts in PDF
        "ps.fonttype":        42,
    })

def save_fig(fig, stem):
    """Save PNG + PDF into OUTDIR."""
    for ext in ("png", "pdf"):
        path = os.path.join(OUTDIR, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  Saved → {path}")
    plt.close(fig)
