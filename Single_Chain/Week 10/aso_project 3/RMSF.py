#!/usr/bin/env python3
"""
Standalone hairpin RMSF campaign analysis for multiple LAMMPS simulations.

What it does
------------
1. Matches each .dat file to its corresponding .lammpstrj trajectory.
2. Extracts the hairpin (assumed molecule 2) from each trajectory.
3. Aligns each frame on a chosen set of hairpin residues (default = all non-loop residues).
4. Computes per-residue RMSF for the hairpin.
5. Writes a summary CSV with full-hairpin and loop-region RMSF metrics.
6. Generates campaign plots:
   - full-hairpin RMSF overlay
   - loop-region RMSF overlay
   - sorted horizontal bar plot of loop mean RMSF (recommended thesis summary)
7. Optionally saves per-simulation plots.

Notes
-----
- This script intentionally forces single-threaded BLAS/LAPACK because many tiny
  SVD calls are often slower on clusters when MKL/OpenBLAS oversubscribe threads.
- For thesis use, the conventional summary metric is NOT minimum RMSF. Use either:
    * mean loop RMSF (default, recommended)
    * median loop RMSF (robust alternative)
    * max loop RMSF (useful as a secondary “peak flexibility” metric)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TYPE_TO_NUC = {1: "A", 2: "C", 3: "G", 4: "U"}


def parse_args():
    p = argparse.ArgumentParser(description="Multi-simulation hairpin RMSF analysis")
    p.add_argument("--dat-dir", default="configs", help="Directory containing .dat files")
    p.add_argument("--traj-dir", default="outputs", help="Directory containing .lammpstrj files")
    p.add_argument(
        "--dat-glob",
        default="*.dat",
        help="Glob pattern for dat files inside --dat-dir (default: *.dat)",
    )
    p.add_argument(
        "--loop",
        required=True,
        help="Loop/junction residues in 1-based indexing, e.g. 13-20 or 10-23",
    )
    p.add_argument(
        "--fit-res",
        default=None,
        help=(
            "Alignment residues in 1-based indexing, e.g. 1-12,21-32. "
            "If omitted, all non-loop hairpin residues are used."
        ),
    )
    p.add_argument(
        "--summary-metric",
        choices=["mean", "median", "max"],
        default="mean",
        help="Metric used for the campaign bar plot (default: mean)",
    )
    p.add_argument(
        "--per-sim-plots",
        action="store_true",
        help="Also save per-simulation full and loop RMSF plots",
    )
    p.add_argument(
        "--outdir",
        default="rmsf_campaign",
        help="Output directory for CSV and plots",
    )
    return p.parse_args()


def parse_range_spec(spec: str) -> list[int]:
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(chunk))
    # preserve order but deduplicate
    seen = set()
    ordered = []
    for x in out:
        if x not in seen:
            ordered.append(x)
            seen.add(x)
    return ordered


def parse_dat(dat_path):
    with open(dat_path) as f:
        lines = f.readlines()

    n_atoms = int(next(l for l in lines if l.strip().endswith("atoms")).split()[0])
    atom_start = next(i for i, l in enumerate(lines) if l.strip() == "Atoms") + 2

    atom_info = {}
    mol_atoms = defaultdict(list)
    xlo = xhi = ylo = yhi = zlo = zhi = None

    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            if parts[2:] == ["xlo", "xhi"]:
                xlo, xhi = float(parts[0]), float(parts[1])
            elif parts[2:] == ["ylo", "yhi"]:
                ylo, yhi = float(parts[0]), float(parts[1])
            elif parts[2:] == ["zlo", "zhi"]:
                zlo, zhi = float(parts[0]), float(parts[1])

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError(f"Could not parse box bounds from {dat_path}")

    for line in lines[atom_start:atom_start + n_atoms]:
        p = line.split()
        if len(p) < 7:
            continue
        aid, mid, atype = int(p[0]), int(p[1]), int(p[2])
        atom_info[aid] = (mid, atype)
        mol_atoms[mid].append(aid)

    for mid in mol_atoms:
        mol_atoms[mid].sort()

    box_lengths = np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=float)
    return atom_info, dict(mol_atoms), box_lengths


def resolve_trajectory(dat_file: Path, traj_dir: Path) -> Path | None:
    stem = dat_file.stem
    exact = traj_dir / f"{stem}.lammpstrj"
    if exact.is_file():
        return exact

    # relaxed matching if naming is slightly inconsistent
    candidates = sorted(traj_dir.glob(f"*{stem}*.lammpstrj"))
    if candidates:
        return candidates[0]

    # final fallback: strip obvious config_ prefix
    alt_stem = stem.removeprefix("config_")
    if alt_stem != stem:
        exact_alt = traj_dir / f"{alt_stem}.lammpstrj"
        if exact_alt.is_file():
            return exact_alt
        candidates = sorted(traj_dir.glob(f"*{alt_stem}*.lammpstrj"))
        if candidates:
            return candidates[0]

    return None


def discover_pairs(dat_dir: Path, traj_dir: Path, dat_glob: str):
    pairs = []
    missing = []
    for dat in sorted(dat_dir.glob(dat_glob)):
        traj = resolve_trajectory(dat, traj_dir)
        if traj is None:
            missing.append(dat.name)
        else:
            pairs.append((dat, traj))
    return pairs, missing


def parse_traj(traj_path, hairpin_aids):
    hairpin_set = set(hairpin_aids)
    frames = []
    timesteps = []

    print(f"Reading trajectory: {traj_path}")
    with open(traj_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            ts_line = f.readline()
            if not ts_line:
                break
            ts = int(ts_line.strip())

            number_hdr = f.readline().strip()
            if not number_hdr.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Malformed trajectory near timestep {ts}: missing NUMBER OF ATOMS header")

            n_atoms_line = f.readline()
            if not n_atoms_line:
                break
            n_atoms = int(n_atoms_line.strip())

            box_hdr = f.readline().strip()
            if not box_hdr.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Malformed trajectory near timestep {ts}: missing BOX BOUNDS header")

            # read 3 box lines
            b1 = f.readline()
            b2 = f.readline()
            b3 = f.readline()
            if not (b1 and b2 and b3):
                print(f"Warning: truncated box bounds at timestep {ts}; stopping parse.")
                break

            atoms_hdr = f.readline().strip()
            if not atoms_hdr.startswith("ITEM: ATOMS"):
                raise ValueError(f"Malformed trajectory near timestep {ts}: missing ATOMS header")

            cols = atoms_hdr.split()[2:]
            col_index = {c: i for i, c in enumerate(cols)}
            if "id" not in col_index:
                raise ValueError("Trajectory ATOMS header must include id column")

            coord_triplets = [("xu", "yu", "zu"), ("x", "y", "z"), ("xs", "ys", "zs")]
            chosen = None
            for cx, cy, cz in coord_triplets:
                if cx in col_index and cy in col_index and cz in col_index:
                    chosen = (col_index[cx], col_index[cy], col_index[cz])
                    break
            if chosen is None:
                raise ValueError("Could not find coordinates in ATOMS header")

            ix, iy, iz = chosen
            id_col = col_index["id"]

            coords = {}
            frame_ok = True

            for _ in range(n_atoms):
                pos = f.tell()
                atom_line = f.readline()
                if not atom_line:
                    print(f"Warning: truncated frame at timestep {ts}; stopping parse.")
                    frame_ok = False
                    break

                p = atom_line.split()
                if not p:
                    print(f"Warning: blank atom line at timestep {ts}; skipping frame.")
                    frame_ok = False
                    break

                # this is the key corruption guard
                if p[0] == "ITEM:":
                    print(f"Warning: early frame header encountered at timestep {ts}; skipping incomplete frame.")
                    f.seek(pos)
                    frame_ok = False
                    break

                try:
                    aid = int(p[id_col])
                except Exception:
                    print(f"Warning: malformed atom record at timestep {ts}; skipping frame.")
                    frame_ok = False
                    break

                try:
                    if aid in hairpin_set:
                        coords[aid] = np.array(
                            [float(p[ix]), float(p[iy]), float(p[iz])],
                            dtype=float
                        )
                except Exception:
                    print(f"Warning: malformed coordinate record at timestep {ts}; skipping frame.")
                    frame_ok = False
                    break

            if not frame_ok:
                continue

            missing = [aid for aid in hairpin_aids if aid not in coords]
            if missing:
                print(f"Warning: missing {len(missing)} hairpin atoms at timestep {ts}; skipping frame.")
                continue

            timesteps.append(ts)
            frames.append(np.array([coords[aid] for aid in hairpin_aids], dtype=np.float32))

            if len(timesteps) % 500 == 0:
                print(f"  ... {len(timesteps)} frames")

    if not frames:
        raise ValueError(f"No frames parsed from {traj_path}")

    print(f"Done: {len(timesteps)} frames total, {timesteps[0]} -> {timesteps[-1]} steps")
    return np.array(frames, dtype=np.float32), np.array(timesteps, dtype=int)
    
def kabsch_align_subset(X, ref, fit_idx):
    """
    Align full structure X onto ref using only fit_idx residues.
    X, ref: (n_res, 3)
    fit_idx: 0-based indices to use in rigid-body fit
    """
    X_fit = X[fit_idx]
    R_fit = ref[fit_idx]

    X_centroid = X_fit.mean(axis=0)
    R_centroid = R_fit.mean(axis=0)

    X0 = X - X_centroid
    R0 = ref - R_centroid

    H = X0[fit_idx].T @ R0[fit_idx]
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return X0 @ R + R_centroid


def compute_rmsf(traj, fit_idx):
    ref = traj[0].astype(float).copy()
    aligned = np.empty_like(traj, dtype=float)

    for i in range(traj.shape[0]):
        aligned[i] = kabsch_align_subset(traj[i].astype(float), ref, fit_idx)

    mean_pos = aligned.mean(axis=0)
    sq_disp = np.sum((aligned - mean_pos) ** 2, axis=2)
    rmsf = np.sqrt(np.mean(sq_disp, axis=0))
    return rmsf


def save_per_sim_plots(name, residues, loop_res, loop_idx, rmsf, outdir):
    sim_dir = outdir / "per_sim"
    sim_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(residues, rmsf, lw=2)
    plt.axvspan(loop_res.min() - 0.5, loop_res.max() + 0.5, alpha=0.18)
    plt.xlabel("Hairpin residue")
    plt.ylabel("RMSF (Å)")
    plt.title(f"Hairpin RMSF vs Residue — {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(sim_dir / f"{name}_hairpin_rmsf_full.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(loop_res, rmsf[loop_idx], marker="o", lw=2)
    plt.xlabel("Hairpin loop residue")
    plt.ylabel("RMSF (Å)")
    plt.title(f"Hairpin Loop RMSF vs Residue — {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(sim_dir / f"{name}_hairpin_rmsf_loop.png", dpi=300)
    plt.close()


def choose_summary_value(row, metric):
    key = {
        "mean": "loop_mean_rmsf_A",
        "median": "loop_median_rmsf_A",
        "max": "loop_max_rmsf_A",
    }[metric]
    return row[key]


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dat_dir = Path(args.dat_dir)
    traj_dir = Path(args.traj_dir)
    if not dat_dir.is_dir():
        raise SystemExit(f"--dat-dir not found: {dat_dir}")
    if not traj_dir.is_dir():
        raise SystemExit(f"--traj-dir not found: {traj_dir}")

    loop_res = np.array(parse_range_spec(args.loop), dtype=int)
    if len(loop_res) == 0:
        raise SystemExit("No loop residues parsed from --loop")

    pairs, missing = discover_pairs(dat_dir, traj_dir, args.dat_glob)
    print("=" * 70)
    print("HAIRPIN RMSF CAMPAIGN ANALYSIS")
    print("=" * 70)
    print(f"Matched simulations: {len(pairs)}")
    if missing:
        print("Missing trajectories for:")
        for name in missing:
            print(f"  - {name}")

    if not pairs:
        raise SystemExit("No dat/trajectory pairs found.")

    summary_rows = []
    full_curves = []
    loop_curves = []
    n_res_expected = None

    for idx, (dat_path, traj_path) in enumerate(pairs, start=1):
        name = dat_path.stem
        print("\n" + "-" * 70)
        print(f"[{idx}/{len(pairs)}] {name}")
        print("-" * 70)

        atom_info, mol_atoms, box_lengths = parse_dat(dat_path)
        if 2 not in mol_atoms:
            raise ValueError(f"{dat_path.name}: expected hairpin to be molecule 2")
        hairpin_aids = mol_atoms[2]
        n_res = len(hairpin_aids)
        residues = np.arange(1, n_res + 1)
        hairpin_seq = "".join(TYPE_TO_NUC.get(atom_info[aid][1], "?") for aid in hairpin_aids)

        if n_res_expected is None:
            n_res_expected = n_res
        elif n_res != n_res_expected:
            raise ValueError(
                f"{dat_path.name}: hairpin has {n_res} residues, but previous simulations had {n_res_expected}. "
                "Campaign overlay plotting expects the same hairpin length."
            )

        bad_loop = [r for r in loop_res if r < 1 or r > n_res]
        if bad_loop:
            raise ValueError(f"{dat_path.name}: loop residues out of range: {bad_loop}")
        loop_idx = np.array([r - 1 for r in loop_res], dtype=int)

        if args.fit_res:
            fit_res = np.array(parse_range_spec(args.fit_res), dtype=int)
        else:
            fit_res = np.array([r for r in residues if r not in set(loop_res)], dtype=int)

        bad_fit = [r for r in fit_res if r < 1 or r > n_res]
        if bad_fit:
            raise ValueError(f"{dat_path.name}: fit residues out of range: {bad_fit}")
        if len(fit_res) < 3:
            raise ValueError(f"{dat_path.name}: need at least 3 fit residues, got {len(fit_res)}")
        fit_idx = np.array([r - 1 for r in fit_res], dtype=int)

        print(f"Hairpin residues: {n_res}")
        print(f"Hairpin sequence: {hairpin_seq}")
        print(f"Loop residues:    {loop_res.tolist()}")
        print(f"Fit residues:     {fit_res.tolist()}")

        hp_traj, timesteps = parse_traj(traj_path, hairpin_aids)
        rmsf = compute_rmsf(hp_traj, fit_idx)

        full_mean = float(np.mean(rmsf))
        full_median = float(np.median(rmsf))
        full_max = float(np.max(rmsf))
        loop_vals = rmsf[loop_idx]
        loop_mean = float(np.mean(loop_vals))
        loop_median = float(np.median(loop_vals))
        loop_max = float(np.max(loop_vals))
        loop_min = float(np.min(loop_vals))
        duration_ns = float((timesteps[-1] - timesteps[0]) * 10.0 / 1e6)

        print(f"Full mean RMSF: {full_mean:.3f} Å")
        print(f"Loop mean RMSF: {loop_mean:.3f} Å")
        print(f"Loop max RMSF:  {loop_max:.3f} Å")

        row = {
            "name": name,
            "dat": str(dat_path),
            "traj": str(traj_path),
            "frames": int(len(timesteps)),
            "duration_ns": duration_ns,
            "hairpin_length": n_res,
            "hairpin_sequence": hairpin_seq,
            "loop_residues": ",".join(map(str, loop_res.tolist())),
            "fit_residues": ",".join(map(str, fit_res.tolist())),
            "full_mean_rmsf_A": full_mean,
            "full_median_rmsf_A": full_median,
            "full_max_rmsf_A": full_max,
            "loop_mean_rmsf_A": loop_mean,
            "loop_median_rmsf_A": loop_median,
            "loop_max_rmsf_A": loop_max,
            "loop_min_rmsf_A": loop_min,
        }
        summary_rows.append(row)
        full_curves.append((name, residues.copy(), rmsf.copy()))
        loop_curves.append((name, loop_res.copy(), loop_vals.copy()))

        if args.per_sim_plots:
            save_per_sim_plots(name, residues, loop_res, loop_idx, rmsf, outdir)

    # Write summary CSV
    csv_path = outdir / "rmsf_campaign_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary CSV saved -> {csv_path}")

    # Full overlay plot
    plt.figure(figsize=(10, 6))
    for name, residues, rmsf in full_curves:
        plt.plot(residues, rmsf, lw=2, label=name)
    plt.axvspan(loop_res.min() - 0.5, loop_res.max() + 0.5, alpha=0.12)
    plt.xlabel("Hairpin residue")
    plt.ylabel("RMSF (Å)")
    plt.title("Hairpin RMSF vs Residue — All Conditions")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    full_overlay = outdir / "rmsf_full_overlay.png"
    plt.savefig(full_overlay, dpi=300)
    plt.close()

    # Loop overlay plot
    plt.figure(figsize=(10, 6))
    for name, res_loop, vals in loop_curves:
        plt.plot(res_loop, vals, marker="o", lw=2, label=name)
    plt.xlabel("Hairpin loop residue")
    plt.ylabel("RMSF (Å)")
    plt.title("Hairpin Loop RMSF vs Residue — All Conditions")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    loop_overlay = outdir / "rmsf_loop_overlay.png"
    plt.savefig(loop_overlay, dpi=300)
    plt.close()

    # Recommended thesis summary bar plot
    sorted_rows = sorted(summary_rows, key=lambda r: choose_summary_value(r, args.summary_metric))
    labels = [r["name"] for r in sorted_rows]
    values = [choose_summary_value(r, args.summary_metric) for r in sorted_rows]

    plt.figure(figsize=(10, max(4.5, 0.45 * len(labels))))
    y = np.arange(len(labels))
    bars = plt.barh(y, values)
    plt.yticks(y, labels)
    plt.xlabel(f"Loop {args.summary_metric.capitalize()} RMSF (Å)")
    plt.title(f"Hairpin Loop {args.summary_metric.capitalize()} RMSF by Condition")
    plt.grid(axis="x", alpha=0.3)
    for yi, v in enumerate(values):
        plt.text(v, yi, f"  {v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    summary_bar = outdir / f"rmsf_loop_{args.summary_metric}_barplot.png"
    plt.savefig(summary_bar, dpi=300)
    plt.close()

    # Secondary dot plot for max loop RMSF (optional but useful)
    sorted_rows_max = sorted(summary_rows, key=lambda r: r["loop_max_rmsf_A"])
    labels_max = [r["name"] for r in sorted_rows_max]
    vals_max = [r["loop_max_rmsf_A"] for r in sorted_rows_max]
    plt.figure(figsize=(10, max(4.5, 0.45 * len(labels_max))))
    y = np.arange(len(labels_max))
    plt.plot(vals_max, y, "o")
    plt.yticks(y, labels_max)
    plt.xlabel("Loop Max RMSF (Å)")
    plt.title("Hairpin Loop Peak RMSF by Condition")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    peak_dot = outdir / "rmsf_loop_peak_dotplot.png"
    plt.savefig(peak_dot, dpi=300)
    plt.close()

    print("Plots saved:")
    print(f"  {full_overlay}")
    print(f"  {loop_overlay}")
    print(f"  {summary_bar}")
    print(f"  {peak_dot}")
    if args.per_sim_plots:
        print(f"  {outdir / 'per_sim'}/*")
    print("Done.")


if __name__ == "__main__":
    main()
