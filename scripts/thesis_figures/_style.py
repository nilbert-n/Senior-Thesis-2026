"""
Shared publication style for all thesis figures.

Design principles
-----------------
* Paul-Tol "bright" palette (colour-blind-safe).
* Regular-weight, short titles (the LaTeX ``\\caption{}`` carries the
  descriptive text in the thesis writeup).
* No gridlines by default — individual figures may turn them on locally
  where they help reading (e.g. bar / z-score plots).
* Footer "methods/caveat" text is rendered safely below the plotting
  axes via :func:`add_footnote` so it never collides with the x-axis
  label (the old pattern ``fig.text(0.01, 0.01, ...)`` + ``bbox='tight'``
  placed the text on top of the x-label).
* PDF output is the default — vector, font-embedded, writeup-ready.
* Output directory is resolved in this order:
    1. ``$THESIS_FIG_OUT``  (environment variable, explicit override)
    2. ``/scratch/gpfs/JERELLE/nilbert/Figures/thesis_results``
       (Princeton cluster, when present)
    3. ``<repo>/Figures/thesis_results``  (always works from a laptop)

Import at the top of each figure script::

    import sys, os; sys.path.insert(0, os.path.dirname(__file__))
    from _style import apply_style, COLORS, save_fig, add_footnote
"""
from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Output directory resolution ──────────────────────────────────────────────
def _resolve_outdir() -> str:
    explicit = os.environ.get("THESIS_FIG_OUT")
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit

    cluster = "/scratch/gpfs/JERELLE/nilbert/Figures/thesis_results"
    if os.path.isdir(os.path.dirname(cluster)):
        os.makedirs(cluster, exist_ok=True)
        return cluster

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    repo_out = os.path.join(repo_root, "Figures", "thesis_results")
    os.makedirs(repo_out, exist_ok=True)
    return repo_out


OUTDIR = _resolve_outdir()


# ── Colour palette: Paul-Tol "bright" (colour-blind safe) ────────────────────
#  Reference: https://personal.sron.nl/~pault/  (§3 "Bright qualitative scheme")
COLORS = {
    "blue":   "#4477AA",
    "red":    "#EE6677",
    "green":  "#228833",
    "yellow": "#CCBB44",
    "cyan":   "#66CCEE",
    "purple": "#AA3377",
    "gray":   "#BBBBBB",
    # Lighter variants for fills / bands (hand-mixed 40 % towards white)
    "lblue":  "#A8BED4",
    "lred":   "#F6B4BD",
    "lgreen": "#A7CBAE",
    "lgray":  "#E0E0E0",
    # Backwards-compat aliases used by a handful of older scripts
    "orange": "#CCBB44",   # maps onto Tol yellow, readable alongside blue/red
}


# ── Global rcParams — journal-style defaults ─────────────────────────────────
def apply_style():
    mpl.rcParams.update({
        # Fonts
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans",
                                "Liberation Sans"],
        "font.size":           9,
        "axes.labelsize":      9,
        "axes.titlesize":      10,
        "axes.titleweight":    "regular",   # NOT bold — journal convention
        "axes.titlelocation":  "left",      # short labels sit top-left
        "axes.titlepad":       6,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "legend.fontsize":     8,
        "legend.frameon":      False,       # no boxed legends
        "legend.handlelength": 1.6,
        "legend.borderpad":    0.2,
        # Spines
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        # Ticks — short inward ticks, journal style
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.size":    3.0,
        "ytick.major.size":    3.0,
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        # Grid — OFF by default, enabled per-figure when helpful
        "axes.grid":           False,
        "grid.color":          "#dddddd",
        "grid.alpha":          0.6,
        "grid.linewidth":      0.5,
        # Lines
        "lines.linewidth":     1.4,
        "lines.markersize":    4.5,
        # Figure backgrounds
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        # Output
        "figure.dpi":          120,         # screen preview only
        "savefig.dpi":         300,         # unused for PDF but harmless
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.03,
        "pdf.fonttype":        42,          # embed TrueType in PDF
        "ps.fonttype":         42,
    })


# ── Helpers ──────────────────────────────────────────────────────────────────
def add_footnote(fig, text: str, *, y: float | None = None,
                 color: str | None = None, reserve: float = 0.28):
    """
    No-op.

    The thesis writeup uses the LaTeX ``\\caption{}`` for methods /
    caveat text, so the old grey italic footer printed inside the
    figure has been retired.  Kept as a no-op (rather than deleted)
    so existing figure scripts that still call ``add_footnote(fig,
    "...")`` continue to work unchanged.

    ``y``, ``color`` and ``reserve`` are accepted for backwards
    compatibility and ignored.
    """
    return None


def save_fig(fig, stem: str, *, formats=("pdf",)):
    """
    Save the figure into :data:`OUTDIR`. By default only PDF is emitted
    (vector, font-embedded, thesis-ready). Pass ``formats=("pdf", "png")``
    to also write a bitmap for slides or web.
    """
    for ext in formats:
        path = os.path.join(OUTDIR, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  Saved → {path}")
    plt.close(fig)
