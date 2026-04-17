"""
Figure 10 – Solvent-accessible surface area per residue on the folded
1YMO-derived hairpin reference.

Publication rewrite of ``Single_Chain/Week 10/aso_project 3/CGSASA.py``
and ``SASA.py`` — same FreeSASA analysis, consistent Paul-Tol bright
palette, short title, and PDF + PNG output.

Data source (one of, in priority order):
    1.  ``sasa_fig46/fig_4_6_sasa_vs_residue.csv`` next to this script's
        parent Week-10 directory, if the campaign has been run.
    2.  Live FreeSASA call on ``1YMO.pdb1`` if that file and the
        ``freesasa`` python package are both importable.
    3.  Fabricated preview data (marked ``[PREVIEW]`` in stderr).
"""
from __future__ import annotations
import os, sys, csv
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote
from _preview_data import make_sasa_profile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

apply_style()

# 1YMO PDB chain-A residues 15-46 renumbered 1-32; loop = relative 13-20
LOOP_RES = set(range(13, 21))

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "Single_Chain", "Week 10", "aso_project 3",
    "sasa_fig46", "fig_4_6_sasa_vs_residue.csv",
)


def _load_rows() -> list[dict]:
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH, newline="") as fh:
            return list(csv.DictReader(fh))

    # Optional live FreeSASA call — avoids recomputing if CSV already present.
    try:
        import freesasa
        pdb = os.path.join(os.path.dirname(CSV_PATH), "..", "1YMO.pdb1")
        pdb = os.path.abspath(pdb)
        if os.path.isfile(pdb):
            structure = freesasa.Structure(pdb)
            result = freesasa.calc(structure)
            areas = result.residueAreas()["A"]
            rows = []
            for k in sorted(areas, key=int):
                absn = int(k)
                if 15 <= absn <= 46:
                    rel = absn - 15 + 1
                    rows.append(dict(
                        relative_residue=rel,
                        absolute_residue=absn,
                        region="loop" if rel in LOOP_RES else "non-loop",
                        sasa_total_A2=float(areas[k].total),
                    ))
            return rows
    except Exception:
        pass

    print(f"  [PREVIEW] {CSV_PATH} not found — using fabricated data.",
          file=sys.stderr)
    return make_sasa_profile()


rows = _load_rows()
rel    = np.array([int(r["relative_residue"])    for r in rows])
absres = np.array([int(r["absolute_residue"])    for r in rows])
sasa   = np.array([float(r["sasa_total_A2"])     for r in rows])
is_loop = np.array([int(r_) in LOOP_RES for r_ in rel])

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.4, 3.8))
ax.yaxis.grid(True, alpha=0.35)
ax.set_axisbelow(True)

# Shade the loop region lightly for context
ax.axvspan(min(LOOP_RES) - 0.5, max(LOOP_RES) + 0.5,
           color=COLORS["lred"], alpha=0.45, zorder=0, linewidth=0)

bar_cols = np.where(is_loop, COLORS["red"], COLORS["blue"])
ax.bar(rel, sasa,
       color=bar_cols, edgecolor="white", linewidth=0.4,
       width=0.82, zorder=2)

# Label the three most-exposed residues
top3 = np.argsort(sasa)[-3:][::-1]
pad = 0.02 * sasa.max()
for idx in top3:
    ax.text(rel[idx], sasa[idx] + pad, f"{rel[idx]}",
            ha="center", va="bottom", fontsize=7.5,
            color=COLORS["gray"])

ax.set_xlim(0.5, 32.5)
ax.set_ylim(0, sasa.max() * 1.12)
ax.set_xticks(np.arange(1, 33, 2))
ax.set_xlabel("Hairpin residue (1YMO-derived, renumbered 1–32)")
ax.set_ylabel("SASA  (Å²)")
ax.set_title("Per-residue SASA")

ax.legend(handles=[
    Patch(color=COLORS["red"],  label="Loop (13–20)"),
    Patch(color=COLORS["blue"], label="Stem / flanking"),
], loc="upper right", ncol=2, fontsize=8)

loop_mean    = sasa[is_loop].mean()
nonloop_mean = sasa[~is_loop].mean()
peak_res     = int(rel[int(np.argmax(sasa))])

add_footnote(
    fig,
    f"FreeSASA on 1YMO-derived folded reference (PDB chain A, residues 15–46).  "
    f"Loop-mean SASA = {loop_mean:.0f} Å²;  stem-mean = {nonloop_mean:.0f} Å².  "
    f"Peak: residue {peak_res} ({sasa.max():.0f} Å²).",
    reserve=0.24,
)

save_fig(fig, "sasa_profile", formats=("pdf", "png"))
