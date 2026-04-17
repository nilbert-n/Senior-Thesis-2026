"""
Figure 11 – Loop-core RMSF summary across the ASO campaign.

Publication rewrite of the ``make_split_summary_plot`` routine in
``Single_Chain/Week 10/aso_project 3/RMSF2.py``.

Left panel : loop-core mean RMSF vs ASO concentration (unmodified 10-mer
             at 25 / 50 / 100 / 200 ASO).
Right panel: loop-core mean RMSF across 100-ASO design variants, sorted.

Data source:
    ``Single_Chain/Week 10/aso_project 3/rmsf_campaign_pdbref/`` ``rmsf_campaign_summary.csv``
Fall-back preview data is used when the CSV is not reachable.
"""
from __future__ import annotations
import os, re, sys, csv
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote

import numpy as np
import matplotlib.pyplot as plt

apply_style()

from _preview_data import make_rmsf_campaign_summary

CSV_CANDIDATES = [
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "Single_Chain", "Week 10", "aso_project 3",
        "rmsf_campaign_pdbref", "rmsf_campaign_summary.csv",
    ),
    "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 10/"
    "aso_project 3/rmsf_campaign_pdbref/rmsf_campaign_summary.csv",
]


def _load_rows() -> list[dict]:
    for p in CSV_CANDIDATES:
        if os.path.isfile(p):
            with open(p, newline="") as fh:
                return list(csv.DictReader(fh))
    print("  [PREVIEW] rmsf_campaign_summary.csv not found — "
          "using fabricated data.", file=sys.stderr)
    return make_rmsf_campaign_summary()


# ── Row classification helpers (match RMSF2.py) ──────────────────────────────
_COUNT_RE = re.compile(r"^(\d+)ASO_unmod$", re.IGNORECASE)


def _aso_count(name: str) -> int | None:
    m = _COUNT_RE.match(name.strip())
    return int(m.group(1)) if m else None


_LABEL_MAP = {
    "100ASO_unmod":                      "Unmod. (ref.)",
    "100ASO_scrambled_unmodified":       "Scrambled",
    "100ASO_unmodified_AAtoCC":          "AA→CC",
    "100ASO_unmodified_loop_GG_to_AA":   "Loop GG→AA",
    "100ASO_mismatch_U5C_unmodified":    "Mismatch U5C",
    "100ASO_mismatch_G6A_unmodified":    "Mismatch G6A",
    "100ASO_truncated_unmodified":       "Truncated (7-mer)",
    "100ASO_all_purine_unmodified":      "All-purine",
    "100ASO_extended_12mer_unmodified":  "Extended 12-mer",
    "100ASO_extended_14mer_unmodified":  "Extended 14-mer",
}


rows = _load_rows()

count_rows = []
for r in rows:
    n = _aso_count(r["name"])
    if n is not None:
        count_rows.append((n, float(r["loop_mean_rmsf_A"])))
count_rows.sort(key=lambda t: t[0])

variant_rows = []
for r in rows:
    if r["name"].startswith("100ASO") and _aso_count(r["name"]) is None \
            and r["name"] in _LABEL_MAP:
        variant_rows.append((_LABEL_MAP[r["name"]], float(r["loop_mean_rmsf_A"]),
                             r["name"] == "100ASO_unmod"))
# Also include 100ASO_unmod reference in the variants panel as the anchor.
for r in rows:
    if r["name"] == "100ASO_unmod":
        variant_rows.append(("Unmod. (ref.)", float(r["loop_mean_rmsf_A"]), True))
        break
# De-dupe: keep first entry per label
seen = set()
variant_rows = [t for t in variant_rows if not (t[0] in seen or seen.add(t[0]))]
variant_rows.sort(key=lambda t: t[1])

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11.0, 4.3),
    gridspec_kw={"width_ratios": [1.0, 1.6]},
)

# Panel A — concentration series ─────────────────────────────────────────────
xs = np.array([t[0] for t in count_rows])
ys = np.array([t[1] for t in count_rows])

ax1.yaxis.grid(True, alpha=0.35)
ax1.set_axisbelow(True)
ax1.plot(xs, ys, "o-", color=COLORS["blue"], markersize=6,
         markeredgecolor="white", markeredgewidth=0.6, linewidth=1.6,
         zorder=3)

pad = 0.04 * (ys.max() - ys.min() + 1.0)
for xi, yi in zip(xs, ys):
    ax1.text(xi, yi + pad, f"{yi:.1f}",
             ha="center", va="bottom", fontsize=7.5, color=COLORS["gray"])

ax1.set_xticks(xs)
ax1.set_xlabel("Number of ASOs")
ax1.set_ylabel("Loop-core mean RMSF  (Å)")
ax1.set_title("A  RMSF vs ASO count")
ax1.set_ylim(ys.min() - 1.2, ys.max() + 1.8)

# Panel B — 100-ASO design variants ──────────────────────────────────────────
labels = [t[0] for t in variant_rows]
vals   = np.array([t[1] for t in variant_rows])
is_ref = np.array([t[2] for t in variant_rows])

ax2.xaxis.grid(True, alpha=0.35)
ax2.set_axisbelow(True)

bar_cols = [COLORS["red"] if ref else COLORS["blue"] for ref in is_ref]
ax2.barh(labels, vals,
         color=bar_cols, edgecolor="white", linewidth=0.4,
         height=0.68)

xmax = vals.max()
for yi, v in enumerate(vals):
    ax2.text(v + xmax * 0.01, yi, f"{v:.1f}",
             va="center", fontsize=7.5, color=COLORS["gray"])

ax2.set_xlabel("Loop-core mean RMSF  (Å)")
ax2.set_title("B  RMSF across 100-ASO variants")
ax2.set_xlim(0, xmax * 1.12)
ax2.invert_yaxis()  # stiffest loop on top

# Reference annotation
if any(is_ref):
    ax2.text(0.98, 0.03,
             "Red = 100 ASO unmodified reference",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=7, color=COLORS["gray"], style="italic")

add_footnote(
    fig,
    "Loop-core residues 13–20;  RMSF computed after Kabsch alignment on "
    "the stem (residues 1–12, 21–32) against the folded 1YMO reference.  "
    "Higher RMSF = more mobile loop.",
    reserve=0.22,
)

save_fig(fig, "rmsf_summary", formats=("pdf", "png"))
