#!/usr/bin/env python3
"""
Campaign RMSF analysis for hairpin trajectories using a folded PDB reference.

Key idea:
- Use the PDB only as a folded alignment scaffold.
- Align each trajectory frame to the PDB on non-loop (typically stem) residues.
- Compute RMSF from the aligned trajectory relative to the mean aligned positions.

Typical usage:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python rmsf_campaign_pdbref.py \
      --dat-dir configs \
      --traj-dir outputs \
      --loop 13-20 \
      --ref-pdb 1YMO.pdb1 \
      --ref-chain A \
      --ref-start 15 \
      --ref-end 46 \
      --outdir rmsf_campaign_pdbref
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import glob
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------- Parsing helpers -----------------------------
TYPE_TO_NUC = {1: "A", 2: "C", 3: "G", 4: "U"}
PDB_RESNAME_TO_NUC = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "RA": "A", "RC": "C", "RG": "G", "RU": "U",
    "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U",
}


def parse_range_list(spec):
    """Parse strings like '13-20,25,27-29' into sorted 1-based residue indices."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(chunk))
    out = sorted(set(out))
    if not out:
        raise ValueError("Empty residue range specification")
    return out


def parse_dat(dat_path):
    with open(dat_path) as f:
        lines = f.readlines()

    n_atoms = int(next(l for l in lines if l.strip().endswith("atoms")).split()[0])
    start = next(i for i, l in enumerate(lines) if l.strip() == "Atoms") + 2

    atom_info = {}
    mol_atoms = defaultdict(list)

    for line in lines[start:start + n_atoms]:
        p = line.split()
        if len(p) < 7:
            continue
        aid, mid, atype = int(p[0]), int(p[1]), int(p[2])
        atom_info[aid] = (mid, atype)
        mol_atoms[mid].append(aid)

    for mid in mol_atoms:
        mol_atoms[mid].sort()

    if 2 not in mol_atoms:
        raise ValueError(f"{dat_path}: expected molecule 2 to be the hairpin")

    hairpin_aids = mol_atoms[2]
    hairpin_seq = "".join(TYPE_TO_NUC.get(atom_info[aid][1], "?") for aid in hairpin_aids)
    return hairpin_aids, hairpin_seq


def parse_traj(traj_path, hairpin_aids, start_frame=0, stop_frame=None):
    hairpin_set = set(hairpin_aids)
    frames = []
    timesteps = []

    print(f"Reading trajectory: {traj_path}")
    with open(traj_path) as f:
        frame_count = 0
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
                raise ValueError("Trajectory ATOMS header must include an id column")

            chosen = None
            for names in (("xu", "yu", "zu"), ("x", "y", "z"), ("xs", "ys", "zs")):
                if all(n in col_index for n in names):
                    chosen = tuple(col_index[n] for n in names)
                    break
            if chosen is None:
                raise ValueError("Could not find coordinate columns in trajectory")

            id_col = col_index["id"]
            ix, iy, iz = chosen
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
                if p[0] == "ITEM:":
                    print(f"Warning: early frame header encountered at timestep {ts}; skipping incomplete frame.")
                    f.seek(pos)
                    frame_ok = False
                    break
                try:
                    aid = int(p[id_col])
                    if aid in hairpin_set:
                        coords[aid] = np.array([float(p[ix]), float(p[iy]), float(p[iz])], dtype=np.float32)
                except Exception:
                    print(f"Warning: malformed atom record at timestep {ts}; skipping frame.")
                    frame_ok = False
                    break

            if not frame_ok:
                continue

            if any(aid not in coords for aid in hairpin_aids):
                print(f"Warning: missing hairpin atom(s) at timestep {ts}; skipping frame.")
                continue

            if frame_count >= start_frame and (stop_frame is None or frame_count < stop_frame):
                timesteps.append(ts)
                frames.append(np.array([coords[aid] for aid in hairpin_aids], dtype=np.float32))
                if len(frames) % 500 == 0:
                    print(f"  ... {len(frames)} kept frames")
            frame_count += 1
            if stop_frame is not None and frame_count >= stop_frame:
                break

    if not frames:
        raise ValueError(f"No usable frames parsed from {traj_path}")

    print(f"Done: {len(timesteps)} kept frames total, {timesteps[0]} -> {timesteps[-1]} steps")
    return np.array(frames, dtype=np.float32), np.array(timesteps, dtype=int)


def parse_pdb_reference(pdb_path, chain="A", start_res=15, end_res=46, atom_name="C3'"):
    atom_name_fallbacks = [
        atom_name, atom_name.replace("'", "*"),
        "C3'", "C3*", "P", "C4'", "C4*", "C1'", "C1*"
    ]
    models = defaultdict(lambda: {})
    current_model = 1

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "MODEL":
                try:
                    current_model = int(line[10:14].strip())
                except Exception:
                    current_model += 1
                continue
            if rec not in ("ATOM", "HETATM"):
                continue
            ch = line[21].strip()
            if ch != chain:
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            if resi < start_res or resi > end_res:
                continue
            resn = line[17:20].strip()
            atom = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            models[current_model].setdefault((resi, resn), {})[atom] = np.array([x, y, z], dtype=float)

    if not models:
        raise ValueError(f"No coordinates found in {pdb_path} for chain {chain} residues {start_res}-{end_res}")

    first_model = sorted(models)[0]
    res_records = models[first_model]
    coords = []
    seq = []
    resnums = list(range(start_res, end_res + 1))

    for resi in resnums:
        matches = [(key, atoms) for key, atoms in res_records.items() if key[0] == resi]
        if not matches:
            raise ValueError(f"Missing residue {resi} in PDB reference")
        (rr, resn), atoms = matches[0]
        nuc = PDB_RESNAME_TO_NUC.get(resn, resn)
        seq.append(nuc)
        chosen = None
        for name in atom_name_fallbacks:
            if name in atoms:
                chosen = atoms[name]
                break
        if chosen is None:
            raise ValueError(f"No suitable atom found for residue {resi} in PDB reference")
        coords.append(chosen)

    return np.array(coords, dtype=float), "".join(seq), resnums, first_model


# ----------------------------- RMSF math -----------------------------------
def kabsch_align_subset(X, ref, fit_idx):
    X = np.asarray(X, dtype=float)
    ref = np.asarray(ref, dtype=float)
    fit_idx = np.asarray(fit_idx, dtype=int)

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


def align_to_reference(traj, ref, fit_idx):
    aligned = np.empty_like(traj, dtype=float)
    for i in range(traj.shape[0]):
        aligned[i] = kabsch_align_subset(traj[i], ref, fit_idx)
    return aligned


def refine_mean_alignment(aligned, fit_idx, n_iter=2):
    cur = np.asarray(aligned, dtype=float)
    for _ in range(max(0, n_iter)):
        mean_struct = cur.mean(axis=0)
        new_cur = np.empty_like(cur)
        for i in range(cur.shape[0]):
            new_cur[i] = kabsch_align_subset(cur[i], mean_struct, fit_idx)
        cur = new_cur
    return cur


def compute_rmsf_from_aligned(aligned):
    mean_pos = aligned.mean(axis=0)
    sq_disp = np.sum((aligned - mean_pos) ** 2, axis=2)
    return np.sqrt(np.mean(sq_disp, axis=0)), mean_pos


# ----------------------------- Plotting ------------------------------------
def make_overlay_plot(resids, curves, ylabel, title, out_png, legend_loc="best"):
    plt.figure(figsize=(9, 5.4))
    for label, y in curves:
        plt.plot(resids, y, lw=2, label=label)
    plt.xlabel("Hairpin residue")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc=legend_loc, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def make_dotplot(names, values, title, xlabel, out_png):
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    y = np.arange(len(names))
    plt.figure(figsize=(9, max(5, 0.38 * len(names) + 1.5)))
    plt.plot(values, y, "o")
    plt.yticks(y, names)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def extract_aso_count(name):
    m = re.match(r"^(\d+)ASO(?:_|$)", str(name))
    return int(m.group(1)) if m else None


def is_unmodified_baseline(name):
    """
    Baseline count-series entries:
      25ASO_unmod
      50ASO_unmod
      100ASO_unmod
      200ASO_unmod
    Also accepts _unmodified or no suffix.
    """
    s = str(name)
    m = re.match(r"^(\d+)ASO(.*)$", s)
    if not m:
        return False
    suffix = m.group(2).strip("_").lower()
    return suffix in {"", "unmod", "unmodified"}


def clean_design_label(name):
    s = str(name)
    s = re.sub(r"^100ASO_?", "", s)

    if s.lower() in {"", "unmod", "unmodified"}:
        return "Unmodified"

    s = s.replace("_unmodified", "")
    s = s.replace("_unmod", "")
    s = s.replace("AAtoCC", "AA→CC")
    s = s.replace("GG_to_AA", "GG→AA")
    s = s.replace("_", " ").strip()

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


def make_split_summary_plot(rows, metric_key, metric_label, out_png):
    """
    Left panel: baseline ASO-count series (25/50/100/200 ASO)
    Right panel: 100ASO design variants
    """
    count_rows = [dict(r) for r in rows if is_unmodified_baseline(r["name"])]
    for r in count_rows:
        r["_aso_count"] = extract_aso_count(r["name"])
    count_rows = [r for r in count_rows if r["_aso_count"] is not None]
    count_rows = sorted(count_rows, key=lambda r: r["_aso_count"])

    variant_rows = [dict(r) for r in rows if str(r["name"]).startswith("100ASO")]
    for r in variant_rows:
        r["_label"] = clean_design_label(r["name"])
    variant_rows = sorted(variant_rows, key=lambda r: r[metric_key])

    if not count_rows:
        raise ValueError("No ASO count-series rows found for split RMSF plot")
    if not variant_rows:
        raise ValueError("No 100ASO variant rows found for split RMSF plot")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.2),
        gridspec_kw={"width_ratios": [1.0, 1.5]}
    )

    # Left panel
    x = np.array([r["_aso_count"] for r in count_rows], dtype=int)
    y = np.array([r[metric_key] for r in count_rows], dtype=float)

    ax1.plot(x, y, marker="o", linewidth=2.2, markersize=7)
    ax1.set_title(f"A) {metric_label} vs ASO number", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of ASOs")
    ax1.set_ylabel(f"{metric_label} (Å)")
    ax1.set_xticks(x)
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    y_pad = 0.03 * max(y) if len(y) else 1.0
    for xi, yi in zip(x, y):
        ax1.text(xi, yi + y_pad, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    # Right panel
    labels = [r["_label"] for r in variant_rows]
    vals = np.array([r[metric_key] for r in variant_rows], dtype=float)

    bars = ax2.barh(labels, vals, edgecolor="black", linewidth=0.5, alpha=0.88)
    ax2.set_title(f"B) {metric_label} across 100ASO design variants", fontsize=13, fontweight="bold")
    ax2.set_xlabel(f"{metric_label} (Å)")
    ax2.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    xmax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    for bar, v in zip(bars, vals):
        ax2.text(v + 0.01 * xmax, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)

    plt.suptitle(
        f"{metric_label} summary across ASO count and design perturbations",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------- Campaign logic ------------------------------
def discover_pairs(dat_dir, traj_dir):
    pairs = []
    for dat_path in sorted(glob.glob(os.path.join(dat_dir, "*.dat"))):
        stem = os.path.splitext(os.path.basename(dat_path))[0]
        traj_path = os.path.join(traj_dir, stem + ".lammpstrj")
        if os.path.isfile(traj_path):
            pairs.append((stem, dat_path, traj_path))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Campaign RMSF analysis using a folded PDB reference")
    ap.add_argument("--dat-dir", required=True, help="Directory containing *.dat files")
    ap.add_argument("--traj-dir", required=True, help="Directory containing *.lammpstrj files")
    ap.add_argument("--loop", required=True, help="1-based loop/core residues, e.g. 13-20 or 10-23")
    ap.add_argument("--fit-res", default=None,
                    help="Optional 1-based residues to use for alignment. Default: complement of --loop")
    ap.add_argument("--ref-pdb", required=True, help="Folded reference PDB file")
    ap.add_argument("--ref-chain", default="A", help="Reference PDB chain (default: A)")
    ap.add_argument("--ref-start", type=int, default=15, help="Start residue in PDB chain (default: 15)")
    ap.add_argument("--ref-end", type=int, default=46, help="End residue in PDB chain (default: 46)")
    ap.add_argument("--ref-atom", default="C3'", help="Primary atom name to extract per residue (default: C3')")
    ap.add_argument("--start-frame", type=int, default=0, help="Discard frames before this 0-based frame index")
    ap.add_argument("--stop-frame", type=int, default=None, help="Optional exclusive stop frame")
    ap.add_argument("--refine-iters", type=int, default=2,
                    help="Extra align-to-mean refinement iterations after PDB alignment (default: 2)")
    ap.add_argument("--summary-metric", choices=["mean", "median", "max"], default="mean",
                    help="Metric for the main split-panel plot (default: mean)")
    ap.add_argument("--outdir", default="rmsf_campaign_pdbref", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    loop_res = parse_range_list(args.loop)
    n_res = args.ref_end - args.ref_start + 1
    if any(r < 1 or r > n_res for r in loop_res):
        raise ValueError(f"Loop residues must lie within 1..{n_res}")

    if args.fit_res is None:
        fit_res = [r for r in range(1, n_res + 1) if r not in set(loop_res)]
    else:
        fit_res = parse_range_list(args.fit_res)

    fit_idx = np.array([r - 1 for r in fit_res], dtype=int)
    loop_idx = np.array([r - 1 for r in loop_res], dtype=int)
    all_res = np.arange(1, n_res + 1)

    ref_coords, ref_seq, ref_resnums, model_id = parse_pdb_reference(
        args.ref_pdb,
        chain=args.ref_chain,
        start_res=args.ref_start,
        end_res=args.ref_end,
        atom_name=args.ref_atom,
    )

    pairs = discover_pairs(args.dat_dir, args.traj_dir)
    if not pairs:
        raise SystemExit("No matching .dat/.lammpstrj pairs found")

    print("=" * 70)
    print("HAIRPIN RMSF CAMPAIGN ANALYSIS (PDB-REFERENCED)")
    print("=" * 70)
    print(f"Matched simulations: {len(pairs)}")
    print(f"Reference PDB:      {args.ref_pdb}")
    print(f"Reference segment:  chain {args.ref_chain} residues {args.ref_start}-{args.ref_end} (model {model_id})")
    print(f"Reference sequence: {ref_seq}")
    print(f"Loop residues:      {loop_res}")
    print(f"Fit residues:       {fit_res}")
    if args.start_frame:
        print(f"Start frame:        {args.start_frame}")
    if args.stop_frame is not None:
        print(f"Stop frame:         {args.stop_frame}")

    rows = []
    full_curves = []
    loop_curves = []
    failures = []

    for idx, (name, dat_path, traj_path) in enumerate(pairs, start=1):
        print("\n" + "-" * 70)
        print(f"[{idx}/{len(pairs)}] {name}")
        print("-" * 70)
        try:
            hairpin_aids, hairpin_seq = parse_dat(dat_path)
            print(f"Hairpin residues: {len(hairpin_aids)}")
            print(f"Hairpin sequence: {hairpin_seq}")
            if len(hairpin_aids) != len(ref_coords):
                raise ValueError(
                    f"Hairpin length {len(hairpin_aids)} does not match reference length {len(ref_coords)}"
                )
            if "?" not in hairpin_seq and hairpin_seq != ref_seq:
                print("Warning: hairpin sequence does not exactly match reference sequence")

            hp_traj, timesteps = parse_traj(
                traj_path, hairpin_aids, start_frame=args.start_frame, stop_frame=args.stop_frame
            )

            aligned0 = align_to_reference(hp_traj, ref_coords, fit_idx)
            aligned = refine_mean_alignment(aligned0, fit_idx, n_iter=args.refine_iters)
            rmsf, mean_struct = compute_rmsf_from_aligned(aligned)

            loop_vals = rmsf[loop_idx]
            full_mean = float(rmsf.mean())
            loop_mean = float(loop_vals.mean())
            loop_median = float(np.median(loop_vals))
            loop_max = float(loop_vals.max())
            loop_min = float(loop_vals.min())

            print(f"Full mean RMSF: {full_mean:.3f} Å")
            print(f"Loop mean RMSF: {loop_mean:.3f} Å")
            print(f"Loop max RMSF:  {loop_max:.3f} Å")

            rows.append({
                "name": name,
                "dat": dat_path,
                "traj": traj_path,
                "n_frames": int(len(timesteps)),
                "start_timestep": int(timesteps[0]),
                "end_timestep": int(timesteps[-1]),
                "hairpin_seq": hairpin_seq,
                "ref_seq": ref_seq,
                "loop_residues": ",".join(map(str, loop_res)),
                "fit_residues": ",".join(map(str, fit_res)),
                "full_mean_rmsf_A": full_mean,
                "loop_mean_rmsf_A": loop_mean,
                "loop_median_rmsf_A": loop_median,
                "loop_max_rmsf_A": loop_max,
                "loop_min_rmsf_A": loop_min,
            })
            full_curves.append((name, rmsf.copy()))
            loop_curves.append((name, loop_vals.copy()))

        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"ERROR: {exc}")
            continue

    if not rows:
        raise SystemExit("No successful simulations to summarize")

    csv_path = os.path.join(args.outdir, "rmsf_campaign_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        fail_path = os.path.join(args.outdir, "rmsf_campaign_failures.txt")
        with open(fail_path, "w") as f:
            for name, msg in failures:
                f.write(f"{name}: {msg}\n")
    else:
        fail_path = None

    make_overlay_plot(
        all_res,
        full_curves,
        ylabel="RMSF (Å)",
        title="Hairpin RMSF vs residue (PDB-referenced alignment)",
        out_png=os.path.join(args.outdir, "rmsf_full_overlay.png"),
        legend_loc="best",
    )
    make_overlay_plot(
        loop_res,
        loop_curves,
        ylabel="Loop-core RMSF (Å)",
        title="Loop-core RMSF vs residue (PDB-referenced alignment)",
        out_png=os.path.join(args.outdir, "rmsf_loop_overlay.png"),
        legend_loc="best",
    )

    if args.summary_metric == "mean":
        metric_key = "loop_mean_rmsf_A"
        metric_label = "Mean loop-core RMSF"
        split_fname = "rmsf_loop_mean_split_panels.png"
    elif args.summary_metric == "median":
        metric_key = "loop_median_rmsf_A"
        metric_label = "Median loop-core RMSF"
        split_fname = "rmsf_loop_median_split_panels.png"
    else:
        metric_key = "loop_max_rmsf_A"
        metric_label = "Peak loop-core RMSF"
        split_fname = "rmsf_loop_max_split_panels.png"

    make_split_summary_plot(
        rows,
        metric_key=metric_key,
        metric_label=metric_label,
        out_png=os.path.join(args.outdir, split_fname),
    )

    make_dotplot(
        [row["name"] for row in rows],
        [row["loop_max_rmsf_A"] for row in rows],
        "Peak loop-core RMSF across ASO conditions",
        "Peak loop-core RMSF (Å)",
        os.path.join(args.outdir, "rmsf_loop_peak_dotplot.png"),
    )

    readme_path = os.path.join(args.outdir, "README_results.txt")
    with open(readme_path, "w") as f:
        f.write("Hairpin RMSF campaign analysis (PDB-referenced)\n")
        f.write(f"Reference PDB: {args.ref_pdb}\n")
        f.write(f"Reference segment: chain {args.ref_chain} residues {args.ref_start}-{args.ref_end} (model {model_id})\n")
        f.write(f"Reference sequence: {ref_seq}\n")
        f.write(f"Loop residues: {loop_res}\n")
        f.write(f"Fit residues: {fit_res}\n")
        f.write(f"Start frame: {args.start_frame}\n")
        f.write(f"Stop frame: {args.stop_frame}\n")
        f.write(f"Refine iterations: {args.refine_iters}\n")
        f.write("\nRecommended thesis plot:\n")
        f.write("  Use the split-panel loop-core RMSF figure as the main comparison figure.\n")
        f.write("  Left panel: ASO-count series. Right panel: 100ASO design variants.\n")
        f.write("  Use the loop RMSF vs residue overlay as support/appendix.\n")
        if failures:
            f.write("\nFailed simulations:\n")
            for name, msg in failures:
                f.write(f"  {name}: {msg}\n")

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {os.path.join(args.outdir, 'rmsf_full_overlay.png')}")
    print(f"  {os.path.join(args.outdir, 'rmsf_loop_overlay.png')}")
    print(f"  {os.path.join(args.outdir, split_fname)}")
    print(f"  {os.path.join(args.outdir, 'rmsf_loop_peak_dotplot.png')}")
    print(f"  {readme_path}")
    if fail_path:
        print(f"  {fail_path}")
    print("Done.")


if __name__ == "__main__":
    main()