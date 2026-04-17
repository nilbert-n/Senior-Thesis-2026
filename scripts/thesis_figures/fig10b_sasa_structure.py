"""
Figure 10b – Structural rendering of the 1YMO-derived hairpin, cartoon
coloured by per-residue SASA.

Two outputs are produced:

* ``sasa_structure.{png,pdf}`` — PyMOL-rendered cartoon (standalone, for
  use in slides or alongside the bar plot).
* ``sasa_figure.{png,pdf}`` — composite panel with the 3D cartoon on
  the left and the bar plot on the right, for the thesis figure.

The SASA colour scale matches the bar plot in :mod:`fig10_sasa_profile`
(blue → yellow → red; loop residues 13–20 are always rendered on top
of the stem so the apex is visible).

Data sources:
    * ``Single_Chain/Week 10/aso_project 3/1YMO.pdb1``
    * ``Single_Chain/Week 10/aso_project 3/sasa_fig46/fig_4_6_sasa_vs_residue.csv``
      (or a live FreeSASA call if the CSV is absent).
"""
from __future__ import annotations
import os, sys, csv, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.image import imread
from matplotlib.ticker import MultipleLocator

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
PDB_PATH = os.path.abspath(os.path.join(
    HERE, "..", "..", "Single_Chain", "Week 10", "aso_project 3", "1YMO.pdb1"))
CSV_PATH = os.path.abspath(os.path.join(
    HERE, "..", "..", "Single_Chain", "Week 10", "aso_project 3",
    "sasa_fig46", "fig_4_6_sasa_vs_residue.csv"))

CHAIN        = "A"
ABS_START    = 15
ABS_END      = 46
LOOP_ABS     = set(range(27, 35))   # absolute residues 27-34  == rel 13-20
LOOP_REL     = set(range(13, 21))


# ── Load SASA per residue ────────────────────────────────────────────────────
def _load_sasa() -> dict[int, float]:
    """Return ``{absolute_residue: sasa_total_A2}`` for residues 15-46."""
    if os.path.isfile(CSV_PATH):
        out: dict[int, float] = {}
        with open(CSV_PATH, newline="") as fh:
            for row in csv.DictReader(fh):
                out[int(row["absolute_residue"])] = float(row["sasa_total_A2"])
        return out
    try:
        import freesasa
        structure = freesasa.Structure(PDB_PATH)
        result = freesasa.calc(structure)
        areas = result.residueAreas()[CHAIN]
        return {int(k): areas[k].total for k in areas
                if ABS_START <= int(k) <= ABS_END}
    except Exception as exc:
        raise SystemExit(
            f"No SASA data found.  Expected CSV at {CSV_PATH} or a working "
            f"FreeSASA install on {PDB_PATH}.  ({exc})"
        )


sasa_by_abs = _load_sasa()
abs_residues = sorted(sasa_by_abs)
sasa_values  = np.array([sasa_by_abs[r] for r in abs_residues])
rel_residues = np.array([r - ABS_START + 1 for r in abs_residues])
is_loop_rel  = np.array([r in LOOP_REL for r in rel_residues])


# ── SASA colour map (matches fig10 bar scheme) ───────────────────────────────
SASA_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "sasa_bright",
    [COLORS["blue"], COLORS["lblue"], COLORS["yellow"], COLORS["red"]],
)
norm = mcolors.Normalize(vmin=float(sasa_values.min()),
                         vmax=float(sasa_values.max()))


# ── PyMOL render of the cartoon ──────────────────────────────────────────────
def _render_structure_png(png_path: str, width: int = 1600, height: int = 1600):
    """Run PyMOL headless, colour by SASA, save ray-traced PNG."""
    import pymol
    pymol.finish_launching(["pymol", "-cq"])    # -c: headless, -q: quiet
    from pymol import cmd

    cmd.reinitialize()
    cmd.load(PDB_PATH, "hp")
    cmd.remove(f"not chain {CHAIN}")
    cmd.remove(f"not resi {ABS_START}-{ABS_END}")
    cmd.remove("solvent")
    cmd.remove("resn HOH+NA+CL+MG")
    cmd.hide("everything")
    cmd.show("cartoon")
    cmd.set("cartoon_ring_mode", 3)
    cmd.set("cartoon_ring_finder", 1)
    cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.set("cartoon_ladder_mode", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("ambient", 0.35)
    cmd.set("specular", 0.2)
    cmd.set("ray_trace_mode", 1)        # soft black outlines
    cmd.set("ray_trace_color", "gray30")
    cmd.set("antialias", 2)
    cmd.bg_color("white")

    cmd.color("gray70", "hp")

    for abs_res, sasa in sasa_by_abs.items():
        rgb = SASA_CMAP(norm(sasa))[:3]
        cname = f"sasa_c_{abs_res}"
        cmd.set_color(cname, [float(x) for x in rgb])
        cmd.color(cname, f"hp and resi {abs_res}")

    cmd.orient("hp")
    cmd.zoom("hp", buffer=2.0)
    cmd.turn("x", -15)
    cmd.turn("y", 10)

    cmd.ray(width, height)
    cmd.png(png_path, dpi=300)
    cmd.delete("all")


# ── Composite figure: cartoon + bar plot ─────────────────────────────────────
def _compose_figure(structure_png: str, out_stem: str):
    fig = plt.figure(figsize=(10.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25],
                          left=0.03, right=0.98, top=0.95, bottom=0.13,
                          wspace=0.08)

    # Left: rendered cartoon
    ax_struct = fig.add_subplot(gs[0, 0])
    ax_struct.imshow(imread(structure_png))
    ax_struct.axis("off")
    ax_struct.set_title("A  Hairpin cartoon (1YMO-derived)", loc="left")

    # Right: SASA bars
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.yaxis.grid(True, alpha=0.35)
    ax_bar.set_axisbelow(True)
    ax_bar.axvspan(min(LOOP_REL) - 0.5, max(LOOP_REL) + 0.5,
                   color=COLORS["lred"], alpha=0.45, zorder=0, linewidth=0)

    bar_colors = [SASA_CMAP(norm(v)) for v in sasa_values]
    ax_bar.bar(rel_residues, sasa_values,
               color=bar_colors, edgecolor="white", linewidth=0.4,
               width=0.82, zorder=2)

    top3 = np.argsort(sasa_values)[-3:][::-1]
    pad = 0.02 * sasa_values.max()
    for idx in top3:
        ax_bar.text(rel_residues[idx], sasa_values[idx] + pad,
                    f"{rel_residues[idx]}",
                    ha="center", va="bottom",
                    fontsize=7.5, color=COLORS["gray"])

    ax_bar.set_xlim(0.5, 32.5)
    ax_bar.set_ylim(0, sasa_values.max() * 1.12)
    ax_bar.xaxis.set_major_locator(MultipleLocator(4))
    ax_bar.set_xlabel("Hairpin residue (renumbered 1–32)")
    ax_bar.set_ylabel("SASA  (Å²)")
    ax_bar.set_title("B  Per-residue SASA")

    # Shared colorbar tying the structure colours to the bar scale
    sm = plt.cm.ScalarMappable(cmap=SASA_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_bar, shrink=0.7, pad=0.02)
    cbar.set_label("SASA  (Å²)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Legend patch so the loop shading is named
    ax_bar.legend(handles=[
        Patch(color=COLORS["lred"], alpha=0.45, label="Loop (13–20)"),
    ], loc="upper right", fontsize=8)

    save_fig(fig, out_stem, formats=("pdf", "png"))


# ── Main ─────────────────────────────────────────────────────────────────────
if not os.path.isfile(PDB_PATH):
    raise SystemExit(f"PDB file not found: {PDB_PATH}")

with tempfile.TemporaryDirectory() as tmpdir:
    struct_png = os.path.join(tmpdir, "hairpin_sasa.png")
    _render_structure_png(struct_png)

    # Also publish the standalone cartoon so it can be used on its own
    from shutil import copyfile
    from _style import OUTDIR
    for ext in ("png", "pdf"):
        dst = os.path.join(OUTDIR, f"sasa_structure.{ext}")
        if ext == "png":
            copyfile(struct_png, dst)
        else:
            # Re-wrap the PNG inside a PDF so LaTeX \includegraphics has
            # a vector file to link — matplotlib can carry a bitmap page.
            fig_only = plt.figure(figsize=(5.4, 5.4))
            ax_only = fig_only.add_axes([0, 0, 1, 1])
            ax_only.imshow(imread(struct_png))
            ax_only.axis("off")
            fig_only.savefig(dst, bbox_inches="tight", pad_inches=0)
            plt.close(fig_only)
        print(f"  Saved → {dst}")

    _compose_figure(struct_png, "sasa_figure")
