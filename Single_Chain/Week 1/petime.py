#!/usr/bin/env python3
"""
Plot Potential Energy vs Time from a LAMMPS run.

Usage examples:
  # From a raw LAMMPS log (assumes timestep = 10 fs)
  python plot_pe_time.py --log log.lammps --dt-fs 10 --color blue --lw 1.2 --out-stem pe_vs_time_298

  # From a CSV we already exported (has columns: step, pe, ke, temp, time_ps)
  python plot_pe_time.py --csv outputs_298_repro/pe_temp_timeseries_298.csv --color blue --lw 1.2
"""
import argparse, re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_fourcols(path):
    """Parse lines like: step pe ke temp"""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if re.match(r"^\s*\d+\s+[-+Ee0-9\.]+\s+[-+Ee0-9\.]+\s+[-+Ee0-9\.]+\s*$", line):
                s, pe, ke, temp = line.split()
                rows.append((int(s), float(pe), float(ke), float(temp)))
    df = pd.DataFrame(rows, columns=["step", "pe", "ke", "temp"])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="LAMMPS log file (raw).", default=None)
    ap.add_argument("--csv", help="CSV with columns including 'time_ps' & 'pe'.", default=None)
    ap.add_argument("--dt-fs", type=float, default=10.0, help="Timestep in femtoseconds (if reading raw log).")
    ap.add_argument("--out-stem", default="pe_vs_time_298", help="Output filename stem (no extension).")
    ap.add_argument("--color", default="blue", help="Line color (e.g., blue, black, #1f77b4).")
    ap.add_argument("--lw", type=float, default=1.2, help="Line width.")
    ap.add_argument("--title", default="PE vs Time (298 K)", help="Plot title.")
    ap.add_argument("--figsize", default="8,5", help="Figure size as 'W,H' in inches.")
    args = ap.parse_args()

    if not args.csv and not args.log:
        ap.error("Provide either --csv or --log.")

    if args.csv:
        df = pd.read_csv(args.csv)
        if "time_ps" not in df.columns:
            ap.error("CSV must contain a 'time_ps' column.")
    else:
        df = parse_log_fourcols(args.log)
        df["time_ps"] = df["step"] * (args.dt_fs / 1000.0)

    w, h = map(float, args.figsize.split(","))
    plt.figure(figsize=(w, h), dpi=200)
    plt.plot(df["time_ps"], df["pe"], color=args.color, linewidth=args.lw)
    plt.xlabel("Time (ps)", fontsize=14)
    plt.ylabel("Potential Energy", fontsize=14)
    plt.title(args.title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{args.out_stem}.png", dpi=600)
    plt.savefig(f"{args.out_stem}.pdf")   # vector version
    print(f"Wrote {args.out_stem}.png and {args.out_stem}.pdf")

if __name__ == "__main__":
    main()
