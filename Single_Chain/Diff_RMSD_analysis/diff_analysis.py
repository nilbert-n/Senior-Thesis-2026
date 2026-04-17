#!/usr/bin/env python3
"""
diff_simple.py
--------------
Compare enhanced sampling (Week 3) vs regular MD (Week 2) for the 5 configs:
1, 4, 5, 7, 8. Produces difference plots (enhanced - regular) for:
- Rg vs time
- RMSD vs time
- Potential energy vs time

Week 2 metrics filenames: metrics_config_<ID>_<PDB>.pdb1_<CHAIN>.dat.csv
Week 3 metrics filenames: metrics_config_<ID>_<PDB>_<CHAIN>.dat.csv
(PE files use same core but start with 'pe_' and have columns time_ns, Pe)

Usage example:
  python diff_simple.py \
    --week2 "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 2/outputs" \
    --week3 "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week3_2/outputs" \
    --outdir "./diff_outputs" --verbose
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Known mapping for the 5 systems ----
# ID -> (PDB, CHAIN) inferred from your Week 3 cores:
#   1 -> 2DER C
#   4 -> 4GXY A
#   5 -> 5ML7 A
#   7 -> 6G7Z A
#   8 -> 4U7U L
DEFAULT_CONFIG_MAP = {
    2: ("6CK4", "A"),
    3: ("4QLN", "A"),
    6: ("4OJI", "A"),
}

def parse_args():
    ap = argparse.ArgumentParser(description="Diff plots (enhanced - regular) for Rg, RMSD, PE.")
    ap.add_argument("--week2", required=True,
                    help="Week 2 outputs root (either .../outputs or .../outputs/metrics).")
    ap.add_argument("--week3", required=True,
                    help="Week 3 outputs_wk4 root (either .../outputs or .../outputs_wk4/metrics).")
    ap.add_argument("--outdir", default="./diff_outputs_wk4", help="Where to write plots & CSVs.")
    ap.add_argument("--configs", nargs="*", type=int,
                    help="Subset of config IDs (default: 2 3 6).")
    ap.add_argument("--time-tolerance-ns", type=float, default=0.001,
                    help="merge_asof tolerance in ns when pairing time points (default 0.001 ns).")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def ensure_metrics_dir(path_str: str) -> Path:
    """Accept .../outputs or .../outputs/metrics or outputs_wk4; return the metrics dir."""
    p = Path(path_str).resolve()
    if p.name == "metrics":
        return p
    cand = p / "metrics"
    return cand if cand.exists() else p  # if user already pointed directly at metrics

def load_csv_safe(path: Path, want_cols):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # Make sure needed columns exist and are numeric.
        for c in want_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {path.name}")
        # Convert known numeric columns safely
        for c in want_cols:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass
        return df
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}", file=sys.stderr)
        return None

def merge_nearest_by_time(df2, df3, time_col="time_ns", tol=0.001):
    """asof-merge Week2 (left) with Week3 (right) on nearest time within tolerance."""
    df2s = df2.sort_values(time_col).reset_index(drop=True)
    df3s = df3.sort_values(time_col).reset_index(drop=True)
    merged = pd.merge_asof(
        df2s, df3s,
        on=time_col,
        direction="nearest",
        tolerance=tol,
        suffixes=("_reg", "_enh"),
    )
    return merged

def plot_and_save(x, y, xlabel, ylabel, title, outpath):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def core_week2(config_id, pdb, chain):
    # metrics_config_<id>_<PDB>.pdb1_<CHAIN>.dat.csv
    return f"config_{config_id}_{pdb}.pdb1_{chain}.dat"

def core_week3(config_id, pdb, chain):
    # metrics_config_<id>_<PDB>_<CHAIN>.dat.csv
    return f"config_{config_id}_{pdb}_{chain}.dat"

def main():
    args = parse_args()
    metrics2 = ensure_metrics_dir(args.week2)
    metrics3 = ensure_metrics_dir(args.week3)
    outroot = Path(args.outdir).resolve()
    (outroot / "plots").mkdir(parents=True, exist_ok=True)
    (outroot / "diff_csv").mkdir(parents=True, exist_ok=True)

    config_ids = args.configs if args.configs else [2, 3, 6]

    any_done = False

    for cid in config_ids:
        if cid not in DEFAULT_CONFIG_MAP:
            print(f"[WARN] No mapping for config {cid}; skipping.")
            continue

        pdb, chain = DEFAULT_CONFIG_MAP[cid]
        core2 = core_week2(cid, pdb, chain)
        core3 = core_week3(cid, pdb, chain)

        # Files
        m2 = metrics2 / f"metrics_{core2}.csv"
        m3 = metrics3 / f"metrics_{core3}.csv"
        p2 = metrics2 / f"pe_{core2}.csv"
        p3 = metrics3 / f"pe_{core3}.csv"

        if args.verbose:
            print(f"\n[config {cid}] PDB={pdb} chain={chain}")
            print(f"  Week2 metrics: {m2}")
            print(f"  Week3 metrics: {m3}")
            print(f"  Week2 PE     : {p2}")
            print(f"  Week3 PE     : {p3}")

        dfm2 = load_csv_safe(m2, ["time_ns", "Rg", "RMSD"])
        dfm3 = load_csv_safe(m3, ["time_ns", "Rg", "RMSD"])

        if dfm2 is None or dfm3 is None:
            print(f"[WARN] Missing metrics for config {cid}; skipping metrics plots.")
        else:
            # Merge and compute diffs
            merged_m = merge_nearest_by_time(dfm2, dfm3, "time_ns", tol=args.time_tolerance_ns)
            # Drop rows without an enhanced match
            merged_m = merged_m.dropna(subset=["Rg_enh", "RMSD_enh"])

            merged_m["dRg"] = merged_m["Rg_enh"] - merged_m["Rg_reg"]
            merged_m["dRMSD"] = merged_m["RMSD_enh"] - merged_m["RMSD_reg"]

            # Save diff CSV
            csv_out = outroot / "diff_csv" / f"diff_metrics_{core3}.csv"
            merged_m[["time_ns", "Rg_reg", "Rg_enh", "dRg", "RMSD_reg", "RMSD_enh", "dRMSD"]].to_csv(csv_out, index=False)

            # Plots
            title_tag = f"{cid}_{pdb}_{chain}"
            plot_and_save(
                merged_m["time_ns"], merged_m["dRg"],
                "Time (ns)", "ΔRg (enh - reg, Å)", f"ΔRg vs time — {title_tag}",
                outroot / "plots" / f"diff_Rg_{title_tag}.png"
            )
            plot_and_save(
                merged_m["time_ns"], merged_m["dRMSD"],
                "Time (ns)", "ΔRMSD (enh - reg, Å)", f"ΔRMSD vs time — {title_tag}",
                outroot / "plots" / f"diff_RMSD_{title_tag}.png"
            )
            any_done = True

        # Potential Energy
        dfp2 = load_csv_safe(p2, ["time_ns", "Pe"])
        dfp3 = load_csv_safe(p3, ["time_ns", "Pe"])

        if dfp2 is None or dfp3 is None:
            print(f"[WARN] Missing PE for config {cid}; skipping PE plot.")
        else:
            merged_p = merge_nearest_by_time(dfp2, dfp3, "time_ns", tol=args.time_tolerance_ns)
            merged_p = merged_p.dropna(subset=["Pe_enh"])
            merged_p["dPe"] = merged_p["Pe_enh"] - merged_p["Pe_reg"]

            csv_out_pe = outroot / "diff_csv" / f"diff_pe_{core3}.csv"
            merged_p[["time_ns", "Pe_reg", "Pe_enh", "dPe"]].to_csv(csv_out_pe, index=False)

            title_tag = f"{cid}_{pdb}_{chain}"
            plot_and_save(
                merged_p["time_ns"], merged_p["dPe"],
                "Time (ns)", "ΔPE (enh - reg, energy units)", f"ΔPotential Energy vs time — {title_tag}",
                outroot / "plots" / f"diff_PE_{title_tag}.png"
            )
            any_done = True

    if not any_done:
        print("\nNo outputs were produced. Double-check paths and filenames.", file=sys.stderr)
    else:
        print(f"\nDone. Plots -> {outroot}/plots ; CSVs -> {outroot}/diff_csv")

if __name__ == "__main__":
    main()
