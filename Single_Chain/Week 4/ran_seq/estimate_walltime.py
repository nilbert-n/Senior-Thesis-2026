# estimate_walltime.py
import re, argparse, math, sys
ap = argparse.ArgumentParser()
ap.add_argument("--log", help="LAMMPS log from a short calibration run")
ap.add_argument("--steps", type=int, default=408_000_000, help="Total production steps (default: 408e6)")
ap.add_argument("--s-per-step", type=float, help="Override seconds per step (skip reading log)")
args = ap.parse_args()

def secs_to_hms(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{sec:06.3f}"

s_per_step = args.s_per_step
if s_per_step is None:
    if not args.log:
        print("Provide --log or --s-per-step", file=sys.stderr); sys.exit(1)
    sec = None; steps = None
    with open(args.log, "r", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("Loop time of"):
                # Example: "Loop time of 12.34 on 1 procs for 100000 steps ..."
                m = re.search(r"Loop time of\s+([0-9.eE+-]+).+for\s+([0-9]+)\s+steps", ln)
                if m:
                    sec = float(m.group(1)); steps = int(m.group(2))
    if not (sec and steps):
        print("Could not parse Loop time/steps from log.", file=sys.stderr); sys.exit(2)
    s_per_step = sec / steps

proj_sec = args.steps * s_per_step
print(f"Measured seconds/step : {s_per_step:.6g}")
print(f"Projected steps       : {args.steps:,}")
print(f"Projected walltime    : {proj_sec:.1f} s  (~{secs_to_hms(proj_sec)})")
print(f"Suggested #SBATCH -t  : {math.ceil(proj_sec/3600)+2}:00:00  (adds ~2h buffer)")
