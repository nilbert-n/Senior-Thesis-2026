"""
Figure 12 – Apparent Kd summary across ASO campaign (two-panel).

Publication rewrite of ``Single_Chain/Week 10/aso_project 3/plot_kd_campaign.py``.

Left panel : Kd vs ASO concentration (unmodified 10-mer at 25 / 50 /
             100 / 200 ASO).  NaNs are skipped — at COM ≤ 12 Å the
             100-ASO unmodified point is not bound and is absent.
Right panel: Kd across 100-ASO design variants, ranked (log-x so the
             14-mer outlier does not flatten the rest).

Data source:
    ``Single_Chain/Week 10/aso_project 3/analysis3/kd_two_tier_campaign_summary.csv``
Fall-back preview data is used when the CSV is not reachable.

The COM cutoff column is selected by ``--com`` (default 12, matching
fig 8); Kd is converted from molar to mM for readability.
"""
from __future__ import annotations
import os, re, sys, csv, argparse
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS, save_fig, add_footnote
from _preview_data import make_kd_campaign_summary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

apply_style()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--com", type=int, default=12,
                    help="COM cutoff in Å (default: 12)")
args, _ = parser.parse_known_args()

CSV_CANDIDATES = [
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "Single_Chain", "Week 10", "aso_project 3",
        "analysis3", "kd_two_tier_campaign_summary.csv",
    ),
    "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 10/"
    "aso_project 3/analysis3/kd_two_tier_campaign_summary.csv",
]


def _load_rows() -> list[dict]:
    for p in CSV_CANDIDATES:
        if os.path.isfile(p):
            with open(p, newline="") as fh:
                return list(csv.DictReader(fh))
    print("  [PREVIEW] kd_two_tier_campaign_summary.csv not found — "
          "using fabricated data.", file=sys.stderr)
    return make_kd_campaign_summary()


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


def _kd_mM(row: dict, com: int) -> float:
    """Return Kd in mM from the preview rows or the real CSV."""
    if "kd_mM" in row:
        try:
            return float(row["kd_mM"])
        except (TypeError, ValueError):
            return float("nan")
    key = f"kd_molar_com_{com}"
    if key in row and row[key] not in ("", "nan", None):
        try:
            return float(row[key]) * 1000.0
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


rows = _load_rows()

count_rows = []
for r in rows:
    n = _aso_count(r["name"])
    if n is None:
        continue
    kd = _kd_mM(r, args.com)
    if np.isfinite(kd):
        count_rows.append((n, kd))
count_rows.sort(key=lambda t: t[0])

variant_rows = []
for r in rows:
    if r["name"] in _LABEL_MAP:
        kd = _kd_mM(r, args.com)
        if np.isfinite(kd):
            variant_rows.append((_LABEL_MAP[r["name"]], kd,
                                 r["name"] == "100ASO_unmod"))
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

if len(xs):
    ax1.plot(xs, ys, "o-", color=COLORS["blue"], markersize=6,
             markeredgecolor="white", markeredgewidth=0.6, linewidth=1.6,
             zorder=3)
    pad = 0.04 * (ys.max() - ys.min() + 1.0)
    for xi, yi in zip(xs, ys):
        ax1.text(xi, yi + pad, f"{yi:.0f}",
                 ha="center", va="bottom", fontsize=7.5, color=COLORS["gray"])
    ax1.set_xticks(sorted(set([25, 50, 100, 200]) | set(xs.tolist())))
    ax1.set_ylim(0, ys.max() * 1.25)

# Call out missing 100-ASO point
missing = [c for c in (25, 50, 100, 200) if c not in set(xs.tolist())]
if missing:
    ax1.text(0.98, 0.96,
             f"{', '.join(str(m) for m in missing)} ASO: not bound\n"
             f"at COM ≤ {args.com} Å",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=7, color=COLORS["gray"], style="italic")

ax1.set_xlabel("Number of ASOs")
ax1.set_ylabel("Apparent Kd  (mM)")
ax1.set_title(f"A  Kd vs ASO count  (COM ≤ {args.com} Å)")

# Panel B — 100-ASO design variants ──────────────────────────────────────────
labels = [t[0] for t in variant_rows]
vals   = np.array([t[1] for t in variant_rows])
is_ref = np.array([t[2] for t in variant_rows])

ax2.xaxis.grid(True, alpha=0.35, which="both")
ax2.set_axisbelow(True)

bar_cols = [COLORS["red"] if ref else COLORS["blue"] for ref in is_ref]
ax2.barh(labels, vals,
         color=bar_cols, edgecolor="white", linewidth=0.4,
         height=0.68)

use_log = bool(len(vals)) and (vals.max() / max(vals.min(), 1e-6) > 6) and (vals.min() > 0)
if use_log:
    ax2.set_xscale("log")
    xmax = vals.max()
    label_pad = lambda v: v * 1.08
    ax2.set_xlim(vals.min() * 0.7, xmax * 1.8)
else:
    xmax = vals.max() if len(vals) else 1.0
    label_pad = lambda v: v + xmax * 0.01
    ax2.set_xlim(0, xmax * 1.14)

for yi, v in enumerate(vals):
    ax2.text(label_pad(v), yi, f"{v:.0f}",
             va="center", fontsize=7.5, color=COLORS["gray"])

ax2.set_xlabel("Apparent Kd  (mM)"
               + ("  — log scale" if use_log else ""))
ax2.set_title(f"B  Kd across 100-ASO variants  (COM ≤ {args.com} Å)")
ax2.invert_yaxis()  # strongest binder on top

if any(is_ref):
    ax2.text(0.98, 0.03,
             "Red = 100 ASO unmodified reference",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=7, color=COLORS["gray"], style="italic")

add_footnote(
    fig,
    f"Two-tier Kd at COM ≤ {args.com} Å (see Methods §2.4).  "
    "Values expressed in mM; single-trajectory point estimates — the "
    "cross-cutoff systematic uncertainty is shown separately in fig 8.  "
    "Lower Kd = tighter apparent binding.",
    reserve=0.22,
)

save_fig(fig, "kd_summary", formats=("pdf", "png"))
