#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, math
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# PDB: extract 1 coord per residue
# ----------------------------
def pdb_c3_coords(pdb_path: str, slice_str: str | None = None, allowed=("C3'", "C3*")) -> np.ndarray:
    """
    Extract exactly one coordinate per residue using ONLY C3' (or C3*).
    Residues are taken in the order they appear in the file.

    slice_str format: "start:end" where start/end are 1-indexed and end is inclusive.
    Example: "1:32"
    """
    residues = []
    seen = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom = line[12:16].strip()
            if atom not in allowed:
                continue

            chain = line[21].strip()
            resseq = int(line[22:26])
            icode = line[26].strip()
            key = (chain, resseq, icode)

            if key in seen:
                continue
            seen.add(key)

            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            residues.append(np.array([x, y, z], dtype=float))

    if not residues:
        raise ValueError(f"No C3' (or C3*) atoms found in PDB: {pdb_path}")

    coords = np.vstack(residues)

    if slice_str:
        a, b = slice_str.split(":")
        start = int(a) - 1
        end = int(b)          # inclusive -> python exclusive
        coords = coords[start:end]

    return coords




# ----------------------------
# LAMMPS .lammpstrj parser
# ----------------------------
def parse_lammpstrj(path: str, poscols=("xu","yu","zu")):
    """
    Yields: (timestep:int, box_len: (3,), atoms: (N, ncols), cols: dict)
    """
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep = int(f.readline().strip())

            f.readline()  # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline().strip())

            f.readline()  # ITEM: BOX BOUNDS ...
            bounds = []
            for _ in range(3):
                lo, hi = f.readline().split()[:2]
                bounds.append((float(lo), float(hi)))
            box_lo = np.array([b[0] for b in bounds], dtype=float)
            box_hi = np.array([b[1] for b in bounds], dtype=float)
            box_len = box_hi - box_lo

            header = f.readline().strip().split()
            # ITEM: ATOMS ...
            colnames = header[2:]
            cols = {name:i for i,name in enumerate(colnames)}

            data = np.zeros((natoms, len(colnames)), dtype=float)
            for i in range(natoms):
                row = f.readline().split()
                data[i,:] = [float(x) for x in row[:len(colnames)]]

            # make sure required columns exist
            for needed in ("id","mol", *poscols):
                if needed not in cols:
                    raise ValueError(f"Missing column '{needed}' in dump. Have: {list(cols.keys())}")

            yield timestep, box_len, data, cols


def pbc_delta(dx: np.ndarray, box_len: np.ndarray) -> np.ndarray:
    return dx - box_len * np.round(dx / box_len)


def unwrap_positions(X: np.ndarray, box_len: np.ndarray) -> np.ndarray:
    """
    Minimal unwrapping across frames by removing PBC jumps.
    X: (T,N,3)
    """
    U = X.copy()
    for t in range(1, U.shape[0]):
        d = U[t] - U[t-1]
        d = d - box_len * np.round(d / box_len)
        U[t] = U[t-1] + d
    return U


# ----------------------------
# Alignment / RMSD / RMSF / Rg
# ----------------------------
def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Rotation R such that P@R best matches Q (both centered).
    """
    C = P.T @ Q
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    return V @ D @ Wt


def rmsd_vs_ref(X: np.ndarray, ref: np.ndarray, align=True) -> np.ndarray:
    """
    X: (T,N,3), ref: (N,3)
    """
    if X.shape[1] != ref.shape[0]:
        raise ValueError(f"RMSD mismatch: traj N={X.shape[1]} vs ref N={ref.shape[0]}")
    T, N, _ = X.shape
    out = np.zeros(T, dtype=float)

    ref0 = ref - ref.mean(axis=0)
    for t in range(T):
        P = X[t] - X[t].mean(axis=0)
        if align:
            R = kabsch(P, ref0)
            P = P @ R
        diff = P - ref0
        out[t] = math.sqrt((diff*diff).sum() / N)
    return out


def align_trajectory(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Align each frame of X to ref and return aligned coords in the original reference frame.
    """
    if X.shape[1] != ref.shape[0]:
        raise ValueError(f"Align mismatch: traj N={X.shape[1]} vs ref N={ref.shape[0]}")
    T = X.shape[0]
    ref_center = ref.mean(axis=0)
    ref0 = ref - ref_center
    out = np.zeros_like(X)

    for t in range(T):
        P = X[t] - X[t].mean(axis=0)
        R = kabsch(P, ref0)
        out[t] = (P @ R) + ref_center
    return out


def rmsf(X_aligned: np.ndarray) -> np.ndarray:
    """
    X_aligned: (T,N,3) after alignment
    Returns RMSF per bead: (N,)
    """
    mean = X_aligned.mean(axis=0, keepdims=True)
    d = X_aligned - mean
    return np.sqrt((d*d).mean(axis=(0,2)))


def radius_of_gyration(X: np.ndarray) -> np.ndarray:
    """
    X: (T,N,3)
    """
    com = X.mean(axis=1, keepdims=True)
    d = X - com
    return np.sqrt((d*d).sum(axis=(1,2)) / X.shape[1])


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="LAMMPS .lammpstrj")
    ap.add_argument("--outdir", required=True, help="Output directory (must be writable)")
    ap.add_argument("--mol_aso", type=int, default=1, help="mol id for ASO")
    ap.add_argument("--mol_hairpin", type=int, default=2, help="mol id for hairpin")
    ap.add_argument("--len_aso", type=int, default=10, help="expected ASO beads")
    ap.add_argument("--len_hairpin", type=int, default=42, help="expected hairpin beads")

    ap.add_argument("--pdb_complex", required=True, help="Reference PDB for complex (ASO+hairpin)")
    ap.add_argument("--pdb_hairpin", required=True, help="Reference PDB for hairpin")
    ap.add_argument("--pdb_aso", required=True, help="Reference PDB for ASO")

    # NEW: optional PDB residue slicing (1-indexed, inclusive)
    ap.add_argument("--pdb_hairpin_slice", default=None,
                    help="Slice residues from hairpin PDB by 1-indexed order start:end (inclusive). Example: 1:32")
    ap.add_argument("--pdb_aso_slice", default=None,
                    help="Slice residues from ASO PDB by 1-indexed order start:end (inclusive). Example: 1:10")
    ap.add_argument("--pdb_complex_slice", default=None,
                    help="Slice residues from complex PDB by 1-indexed order start:end (inclusive).")

    ap.add_argument("--poscols", default="xu,yu,zu", help="position columns in dump (default xu,yu,zu)")
    ap.add_argument("--burnin_frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dt_ps", type=float, default=None,
                    help="Timestep size in ps (if timestep in dump is step count). "
                         "If omitted, x-axis uses raw timestep.")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)


    poscols = tuple(s.strip() for s in args.poscols.split(","))

    # ---- Load references (1 coord per residue)
    ref_complex = pdb_c3_coords(args.pdb_complex, slice_str=args.pdb_complex_slice)
    ref_hairpin = pdb_c3_coords(args.pdb_hairpin, slice_str=args.pdb_hairpin_slice)
    ref_aso     = pdb_c3_coords(args.pdb_aso,     slice_str=args.pdb_aso_slice)

    # ---- Read trajectory frames, sorting by atom id each frame
    timesteps = []
    X_frames = []
    box_lens = []
    mol_ids = None
    ids0 = None

    for frame_i, (ts, box_len, data, cols) in enumerate(parse_lammpstrj(args.traj, poscols=poscols)):
        if frame_i < args.burnin_frames:
            continue
        if args.stride > 1 and ((frame_i - args.burnin_frames) % args.stride != 0):
            continue

        atom_id = data[:, cols["id"]].astype(int)
        order = np.argsort(atom_id)
        data = data[order]
        atom_id = atom_id[order]

        x = data[:, cols[poscols[0]]]
        y = data[:, cols[poscols[1]]]
        z = data[:, cols[poscols[2]]]
        P = np.stack([x,y,z], axis=1)

        if mol_ids is None:
            mol_ids = data[:, cols["mol"]].astype(int)
            ids0 = atom_id
        else:
            # sanity: id ordering should match after sorting
            if not np.array_equal(ids0, atom_id):
                raise RuntimeError("Atom IDs changed across frames after sorting — mapping is not stable.")

        timesteps.append(ts)
        X_frames.append(P)
        box_lens.append(box_len)

    if not X_frames:
        raise RuntimeError("No frames loaded. Check burnin/stride/path.")

    timesteps = np.array(timesteps, dtype=int)
    X = np.stack(X_frames, axis=0)  # (T,N,3)
    box_len = np.mean(np.stack(box_lens, axis=0), axis=0)

    # unwrap for stable Rg / RMSF
    X = unwrap_positions(X, box_len)

    # split molecules
    idx_aso = np.where(mol_ids == args.mol_aso)[0]
    idx_hp  = np.where(mol_ids == args.mol_hairpin)[0]

    if len(idx_aso) != args.len_aso:
        print(f"[WARN] ASO beads in traj = {len(idx_aso)} (expected {args.len_aso})")
    if len(idx_hp) != args.len_hairpin:
        print(f"[WARN] Hairpin beads in traj = {len(idx_hp)} (expected {args.len_hairpin})")

    X_aso = X[:, idx_aso, :]
    X_hp  = X[:, idx_hp,  :]
    X_cpx = X

    # ---- Reference length checks
    # These must match your bead counts (1 bead per residue)
    def check_len(name, trajN, refN, pdbpath):
        if trajN != refN:
            raise ValueError(
                f"{name} length mismatch: traj has {trajN} beads, but PDB-derived reference has {refN} residues.\n"
                f"Fix: choose correct PDB, or adjust --pdb_atom_priority, or ensure PDB contains exactly that molecule."
            )

    check_len("Hairpin", X_hp.shape[1], ref_hairpin.shape[0], args.pdb_hairpin)
    check_len("ASO",     X_aso.shape[1], ref_aso.shape[0],     args.pdb_aso)
    check_len("Complex", X_cpx.shape[1], ref_complex.shape[0], args.pdb_complex)

    # ---- Time axis
    if args.dt_ps is None:
        time_x = timesteps.astype(float)
        time_label = "Timestep"
    else:
        # assume timesteps are MD steps
        time_ns = (timesteps - timesteps[0]) * args.dt_ps / 1000.0
        time_x = time_ns
        time_label = "Time (ns)"

    # ---- Compute metrics (aligned-to-PDB RMSD; aligned RMSF; unaligned Rg)
    rmsd_complex = rmsd_vs_ref(X_cpx, ref_complex, align=True)
    rmsd_hairpin = rmsd_vs_ref(X_hp,  ref_hairpin, align=True)
    rmsd_aso     = rmsd_vs_ref(X_aso, ref_aso,     align=True)

    rg_complex = radius_of_gyration(X_cpx)

    X_hp_aligned = align_trajectory(X_hp, ref_hairpin)
    rmsf_hp = rmsf(X_hp_aligned)

    # ---- Plot: paper-like multi-panel layout
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs[0,0].plot(time_x, rmsd_complex)
    axs[0,0].set_title("RMSD of Complex")
    axs[0,0].set_xlabel(time_label)
    axs[0,0].set_ylabel("RMSD (Å)")

    axs[0,1].plot(time_x, rmsd_hairpin)
    axs[0,1].set_title("RMSD of Hairpin")
    axs[0,1].set_xlabel(time_label)
    axs[0,1].set_ylabel("RMSD (Å)")

    axs[0,2].plot(np.arange(1, len(rmsf_hp)+1), rmsf_hp)
    axs[0,2].set_title("RMSF (Hairpin)")
    axs[0,2].set_xlabel("Residue (number)")
    axs[0,2].set_ylabel("RMSF (Å)")

    axs[1,0].plot(time_x, rmsd_aso)
    axs[1,0].set_title("RMSD of ASO")
    axs[1,0].set_xlabel(time_label)
    axs[1,0].set_ylabel("RMSD (Å)")

    axs[1,1].plot(time_x, rg_complex)
    axs[1,1].set_title("Radius of Gyration of Complex")
    axs[1,1].set_xlabel(time_label)
    axs[1,1].set_ylabel("Rg (Å)")

    axs[1,2].axis("off")  # empty like the paper figure (they put a schematic here)

    plt.tight_layout()
    out_png = os.path.join(args.outdir, "paper_style_metrics.png")
    plt.savefig(out_png, dpi=300)
    plt.close()

    # also save individual plots for convenience
    def save1(x, y, title, ylab, fname):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(time_label if x is not None else "")
        plt.ylabel(ylab)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=300)
        plt.close()

    save1(time_x, rmsd_complex, "RMSD of Complex", "RMSD (Å)", "rmsd_complex.png")
    save1(time_x, rmsd_hairpin, "RMSD of Hairpin", "RMSD (Å)", "rmsd_hairpin.png")
    save1(time_x, rmsd_aso, "RMSD of ASO", "RMSD (Å)", "rmsd_aso.png")
    save1(time_x, rg_complex, "Radius of Gyration of Complex", "Rg (Å)", "rg_complex.png")

    plt.figure()
    plt.plot(np.arange(1, len(rmsf_hp)+1), rmsf_hp)
    plt.title("RMSF (Hairpin)")
    plt.xlabel("Residue (number)")
    plt.ylabel("RMSF (Å)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "rmsf_hairpin.png"), dpi=300)
    plt.close()

    print("Wrote:")
    print(" ", out_png)
    print(" ", os.path.join(args.outdir, "rmsd_complex.png"))
    print(" ", os.path.join(args.outdir, "rmsd_hairpin.png"))
    print(" ", os.path.join(args.outdir, "rmsd_aso.png"))
    print(" ", os.path.join(args.outdir, "rg_complex.png"))
    print(" ", os.path.join(args.outdir, "rmsf_hairpin.png"))


if __name__ == "__main__":
    main()
