#!/usr/bin/env python3
"""
Make side-by-side Kd plots:
1) Kd vs ASO count (25/50/100/200)
2) Kd across 100ASO design variants

Example:
    python plot_kd_split_panels.py \
        --csv kd_analysis/kd_two_tier_campaign_summary.csv \
        --com 12 \
        --outdir kd_split_plots
"""

from __future__ import annotations

import argparse
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finite_or_nan(x):
    try:
        x = float(x)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return np.nan


def discover_com_cols(df: pd.DataFrame):
    """
    Find columns like kd_molar_com_10, kd_molar_com_12, ...
    Returns dict: {10: "kd_molar_com_10", 12: "kd_molar_com_12", ...}
    """
    out = {}
    pat = re.compile(r"^kd_molar_com_(\d+(?:\.\d+)?)$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            val = float(m.group(1))
            if val.is_integer():
                val = int(val)
            out[val] = c
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def extract_aso_count(name: str):
    m = re.match(r"^(\d+)ASO(?:_|$)", str(name))
    return int(m.group(1)) if m else None


def is_unmodified_baseline(name: str) -> bool:
    """
    Count-series entries:
      25ASO_unmod
      50ASO_unmod
      100ASO_unmod
      200ASO_unmod
    or variants with unmodified / no extra suffix
    """
    s = str(name)
    m = re.match(r"^(\d+)ASO(.*)$", s)
    if not m:
        return False
    suffix = m.group(2).strip("_").lower()
    return suffix in {"", "unmod", "unmodified"}


def clean_design_label(name: str) -> str:
    s = str(name)

    # remove leading 100ASO
    s = re.sub(r"^100ASO_?", "", s)

    # baseline
    if s.lower() in {"", "unmod", "unmodified"}:
        return "Unmodified"

    s = s.replace("_unmodified", "")
    s = s.replace("_unmod", "")
    s = s.replace("AAtoCC", "AA→CC")
    s = s.replace("GG_to_AA", "GG→AA")
    s = s.replace("_", " ").strip()

    # prettier capitalization
    replacements = {
        "extended 12mer": "Extended 12mer",
        "extended 14mer": "Extended 14mer",
        "mismatch g6a": "Mismatch G6A",
        "mismatch u5c": "Mismatch U5C",
        "all purine": "All purine",
        "scrambled": "Scrambled",
        "truncated": "Truncated",
        "loop gg→aa": "Loop GG→AA",
    }
    return replacements.get(s.lower(), s.title())


def main():
    p = argparse.ArgumentParser(description="Make split Kd figure: ASO count + 100ASO variants")
    p.add_argument("--csv", required=True, help="Path to kd_two_tier_campaign_summary.csv")
    p.add_argument("--com", type=float, default=12, help="COM cutoff to plot (default: 12)")
    p.add_argument("--outdir", default="kd_split_plots", help="Output directory")
    p.add_argument("--log-right", action="store_true", help="Use log x-scale on the right panel")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "name" not in df.columns:
        raise ValueError("CSV must contain a 'name' column")

    com_cols = discover_com_cols(df)
    chosen = int(args.com) if float(args.com).is_integer() else float(args.com)
    if chosen not in com_cols:
        raise ValueError(f"Requested COM cutoff {chosen} not found. Available: {list(com_cols.keys())}")

    kd_col = com_cols[chosen]
    df["kd_mM"] = df[kd_col].map(finite_or_nan) * 1e3
    df = df[np.isfinite(df["kd_mM"])].copy()

    # ----------------------------
    # Left panel: ASO count series
    # ----------------------------
    count_df = df[df["name"].map(is_unmodified_baseline)].copy()
    count_df["aso_count"] = count_df["name"].map(extract_aso_count)
    count_df = count_df[np.isfinite(count_df["aso_count"])].copy()
    count_df = count_df.sort_values("aso_count")

    # ----------------------------
    # Right panel: 100ASO variants
    # include all 100ASO entries, including unmodified baseline
    # ----------------------------
    design_df = df[df["name"].astype(str).str.startswith("100ASO")].copy()
    design_df["label"] = design_df["name"].map(clean_design_label)
    design_df = design_df.sort_values("kd_mM", ascending=True)

    if count_df.empty:
        raise ValueError("No ASO count-series rows found (expected e.g. 25ASO_unmod, 50ASO_unmod, 100ASO_unmod, 200ASO_unmod)")
    if design_df.empty:
        raise ValueError("No 100ASO rows found for design-variant panel")

    # ----------------------------
    # Plot
    # ----------------------------
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.2),
        gridspec_kw={"width_ratios": [1.0, 1.5]}
    )

    # Left: ASO count
    x = count_df["aso_count"].to_numpy(dtype=int)
    y = count_df["kd_mM"].to_numpy(dtype=float)

    ax1.plot(x, y, marker="o", linewidth=2.2, markersize=7)
    ax1.set_title(f"A) Kd vs ASO number (COM ≤ {chosen} Å)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of ASOs")
    ax1.set_ylabel("Kd (mM)")
    ax1.set_xticks(x)
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    y_pad = 0.03 * max(y) if len(y) else 1.0
    for xi, yi in zip(x, y):
        ax1.text(xi, yi + y_pad, f"{yi:.1f}", ha="center", va="bottom", fontsize=9)

    # Right: 100ASO design variants
    labels = design_df["label"].tolist()
    vals = design_df["kd_mM"].to_numpy(dtype=float)

    bars = ax2.barh(labels, vals, edgecolor="black", linewidth=0.5, alpha=0.88)
    ax2.set_title(f"B) Kd across 100ASO design variants (COM ≤ {chosen} Å)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Kd (mM)")
    ax2.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    if args.log_right and np.all(vals > 0):
        ax2.set_xscale("log")

    xmax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    for bar, v in zip(bars, vals):
        ax2.text(v + 0.01 * xmax, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}", va="center", fontsize=9)

    plt.suptitle("Kd summary across ASO count and design perturbations", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_png = os.path.join(args.outdir, f"kd_split_panels_com_{chosen}.png")
    out_csv_left = os.path.join(args.outdir, f"kd_count_series_com_{chosen}.csv")
    out_csv_right = os.path.join(args.outdir, f"kd_100aso_variants_com_{chosen}.csv")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    count_df[["name", "aso_count", kd_col, "kd_mM"]].to_csv(out_csv_left, index=False)
    design_df[["name", "label", kd_col, "kd_mM"]].to_csv(out_csv_right, index=False)

    print("=" * 70)
    print("KD SPLIT-PANEL PLOT COMPLETE")
    print("=" * 70)
    print(f"Input CSV: {args.csv}")
    print(f"COM cutoff: {chosen} Å")
    print("\nSaved:")
    print(f"  {out_png}")
    print(f"  {out_csv_left}")
    print(f"  {out_csv_right}")


if __name__ == "__main__":
    main()