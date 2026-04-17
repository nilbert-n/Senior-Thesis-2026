#!/usr/bin/env python3
"""
analyze_secondary.py

From a LAMMPS data file (config_*.dat) and trajectory (config_*.dat.lammpstrj),
infer a simulation-based secondary structure via distance-based base-pair
occupancies and output:

- <out_prefix>_secondary.txt : sequence + dot-bracket
- <out_prefix>_pairs.csv     : i, j, occupancy
- <out_prefix>_arc.png       : arc diagram of base pairs
- <out_prefix>_ovito.data    : LAMMPS data file incl. Bonds for OVITO

Assumptions:
- Data file first line looks like:
    "LAMMPS data file for single RNA: UGGUAA..."
- There are N atoms and N nucleotides (1 atom per nucleotide).
- Trajectory has frames like:
    ITEM: TIMESTEP
    ...
    ITEM: ATOMS id mol type q xu yu zu
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


# ---------------------- I/O HELPERS ---------------------- #

def read_sequence_and_natoms(data_path):
    """
    Read sequence (from first line after ':') and number of atoms from LAMMPS data file.
    """
    seq = None
    n_atoms = None
    with open(data_path, "r") as f:
        first = f.readline().strip()
        if ":" in first:
            seq = first.split(":", 1)[1].strip()
        # Find the '<int> atoms' line
        for line in f:
            line = line.strip()
            if line.endswith("atoms"):
                n_atoms = int(line.split()[0])
                break

    if seq is None:
        raise ValueError(f"Could not find sequence in first line of {data_path}")
    if n_atoms is None:
        raise ValueError(f"Could not find 'atoms' line in {data_path}")

    if len(seq) != n_atoms:
        print(f"[WARN] Sequence length ({len(seq)}) != number of atoms ({n_atoms}). "
              f"Proceeding assuming 1 atom per nucleotide, indexed by id.")

    return seq, n_atoms


def iter_frames(traj_path):
    """
    Generator over frames in a LAMMPS text trajectory.

    Yields:
        timestep (int), coords (N,3) numpy array ordered by atom id (1..N -> 0..N-1)
    """
    with open(traj_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                return  # EOF

            # Synchronize to "ITEM: TIMESTEP"
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            # TIMESTEP value
            ts_line = f.readline()
            if not ts_line:
                return
            timestep = int(ts_line.strip())

            # NUMBER OF ATOMS
            line = f.readline()  # ITEM: NUMBER OF ATOMS
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Unexpected format: expected 'ITEM: NUMBER OF ATOMS'")
            n_atoms_line = f.readline()
            if not n_atoms_line:
                return
            n_atoms = int(n_atoms_line.strip())

            # BOX BOUNDS (3 lines)
            line = f.readline()  # ITEM: BOX BOUNDS ...
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected format: expected 'ITEM: BOX BOUNDS'")
            bounds = [f.readline() for _ in range(3)]
            if any(b == "" for b in bounds):
                return

            # ATOMS header
            header = f.readline()
            if not header.startswith("ITEM: ATOMS"):
                raise ValueError("Unexpected format: expected 'ITEM: ATOMS' line")

            # Parse which columns contain id, xu, yu, zu
            cols = header.strip().split()[2:]  # skip "ITEM:", "ATOMS"
            try:
                idx_id = cols.index("id")
                idx_xu = cols.index("xu")
                idx_yu = cols.index("yu")
                idx_zu = cols.index("zu")
            except ValueError as e:
                raise ValueError(f"Expected columns id, xu, yu, zu in ATOMS header, got: {cols}") from e

            coords = np.empty((n_atoms, 3), dtype=float)

            for _ in range(n_atoms):
                parts = f.readline().split()
                if not parts:
                    raise ValueError("Unexpected end of file while reading ATOMS section")
                atom_id = int(parts[idx_id])
                x = float(parts[idx_xu])
                y = float(parts[idx_yu])
                z = float(parts[idx_zu])
                coords[atom_id - 1] = (x, y, z)

            yield timestep, coords


# ---------------------- ANALYSIS CORE ---------------------- #

def compute_pair_occupancies(traj_path, N, index_sep=4,
                             target_dist=13.8, dist_tol=3.0,
                             max_frames=None):
    """
    Compute occupancies ONLY for pairs that are exactly `index_sep`
    beads apart along the chain, and whose distance is ~target_dist.

    A pair (i, j) with j = i + index_sep is counted as "paired" in a
    frame if |d_ij - target_dist| <= dist_tol.

    Returns:
        occ: (N, N) array of occupancies in [0,1]
        total_frames: number of frames actually processed
    """
    pair_counts = np.zeros((N, N), dtype=float)
    total_frames = 0

    for frame_idx, (ts, coords) in enumerate(iter_frames(traj_path)):
        if max_frames is not None and frame_idx >= max_frames:
            break

        total_frames += 1

        # only i and i+index_sep
        for i in range(N - index_sep):
            j = i + index_sep
            dij = np.linalg.norm(coords[i] - coords[j])
            if abs(dij - target_dist) <= dist_tol:
                pair_counts[i, j] += 1.0
                pair_counts[j, i] += 1.0

    if total_frames == 0:
        raise ValueError("No frames read from trajectory; check traj file/path.")

    occ = pair_counts / total_frames
    return occ, total_frames



def occupancy_to_partners(occ, threshold=0.5):
    """
    Convert occupancy matrix to a set of consensus partners.

    Greedy algorithm:
        - collect all (i,j) with occ >= threshold
        - sort by occupancy descending
        - assign pairs so that each residue is in at most one pair
    """
    N = occ.shape[0]
    partners = [-1] * N  # -1 = unpaired

    candidates = []
    for i in range(N):
        for j in range(i + 1, N):
            if occ[i, j] >= threshold:
                candidates.append((occ[i, j], i, j))

    candidates.sort(reverse=True)  # highest occupancy first

    for o, i, j in candidates:
        if partners[i] == -1 and partners[j] == -1:
            partners[i] = j
            partners[j] = i

    return partners


def partners_to_dotbracket(partners):
    """
    Convert partners list (0-based) to dot-bracket string.
    """
    N = len(partners)
    db = []
    for i in range(N):
        if partners[i] == -1:
            db.append(".")
        elif partners[i] > i:
            db.append("(")
        else:
            db.append(")")
    return "".join(db)


# ---------------------- PLOTTING ---------------------- #

def plot_arc_structure(partners, occ, out_png, title="Secondary structure"):
    """
    Make an arc diagram of secondary structure and save to PNG.
    """
    N = len(partners)
    x = np.arange(1, N + 1)
    y = np.zeros_like(x)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.scatter(x, y, s=10)

    for i in range(N):
        j = partners[i]
        if j <= i:
            continue
        o = occ[i, j]
        if o <= 0.0:
            continue

        xs = np.linspace(i + 1, j + 1, 100)
        center = 0.5 * ((i + 1) + (j + 1))
        radius = 0.5 * ((j + 1) - (i + 1))
        # semicircle in +y
        ys_sq = radius ** 2 - (xs - center) ** 2
        ys_sq = np.clip(ys_sq, 0.0, None)
        ys = np.sqrt(ys_sq)
        if ys.max() > 0:
            ys /= ys.max()
        ys *= 0.5

        ax.plot(xs, ys, alpha=float(o), linewidth=2)

    ax.set_title(title)
    ax.set_ylim(-0.1, 0.8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------- OVITO DATA WRITER ---------------------- #

def write_ovito_data_with_bonds(data_path, out_path, partners):
    """
    Create a LAMMPS data file with Bonds section for OVITO.

    - Reuses box, Masses, Atoms from the original data file.
    - Adds a Bonds section where each consensus base pair is bond type 1.
    """
    with open(data_path, "r") as f:
        lines = f.readlines()

    # Grab x/y/z box lines
    xline = yline = zline = None
    for line in lines:
        if "xlo" in line and "xhi" in line:
            xline = line
        elif "ylo" in line and "yhi" in line:
            yline = line
        elif "zlo" in line and "zhi" in line:
            zline = line
    if xline is None or yline is None or zline is None:
        raise ValueError("Could not find xlo/xhi, ylo/yhi, zlo/zhi lines in data file.")

    # Find Masses and Atoms sections
    masses_start = None
    atoms_start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Masses"):
            masses_start = idx
        if line.strip().startswith("Atoms"):
            atoms_start = idx
            break

    if masses_start is None or atoms_start is None:
        raise ValueError("Could not locate 'Masses' or 'Atoms' section in data file.")

    # Masses: from two lines after 'Masses' line up to just before 'Atoms'
    masses_lines = []
    for line in lines[masses_start + 2:atoms_start]:
        if line.strip():
            masses_lines.append(line.rstrip("\n"))

    # Atoms: from two lines after 'Atoms' line to end (ignore empty trailing lines)
    atoms_lines = []
    for line in lines[atoms_start + 2:]:
        if line.strip():
            atoms_lines.append(line.rstrip("\n"))

    # Build bonds list from partners
    bonds = []
    for i, j in enumerate(partners):
        if j > i and j != -1:
            bonds.append((i + 1, j + 1))  # convert to 1-based

    n_atoms = len(atoms_lines)
    n_bonds = len(bonds)

    with open(out_path, "w") as f:
        f.write(f"LAMMPS data file with base-pair bonds derived from {data_path}\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_bonds} bonds\n")
        f.write("0 angles\n0 dihedrals\n0 impropers\n\n")

        f.write(xline)
        f.write(yline)
        f.write(zline)
        if not xline.endswith("\n"):
            f.write("\n")
        f.write("\nMasses\n\n")
        for ml in masses_lines:
            f.write(ml + "\n")

        f.write("\nAtoms\n\n")
        for al in atoms_lines:
            f.write(al + "\n")

        f.write("\nBonds\n\n")
        bond_id = 1
        for (i_atom, j_atom) in bonds:
            # bond_type = 1 for all base pairs
            f.write(f"{bond_id} 1 {i_atom} {j_atom}\n")
            bond_id += 1

    print(f"[INFO] Wrote OVITO-ready data file with bonds to {out_path}")


# ---------------------- MAIN ---------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Infer secondary structure from LAMMPS trajectory via contact occupancies."
    )
    ap.add_argument("--data", required=True, help="LAMMPS data file (config_*.dat)")
    ap.add_argument("--traj", required=True, help="LAMMPS trajectory file (.lammpstrj)")
    ap.add_argument("--out-prefix", required=True, help="Prefix for output files")
    ap.add_argument("--index-sep", type=int, default=4,
                    help="Index separation |i-j| to consider (default: 4)")
    ap.add_argument("--target-dist", type=float, default=13.8,
                    help="Target distance between paired beads (default: 13.8)")
    ap.add_argument("--dist-tol", type=float, default=3.0,
                    help="Distance tolerance around target-dist (default: 3.0)")

    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Occupancy threshold for pairing (default: 0.5)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Optionally limit number of frames (default: all)")
    args = ap.parse_args()

    seq, N = read_sequence_and_natoms(args.data)
    print(f"[INFO] Sequence length: {len(seq)}, N atoms: {N}")
    print(f"[INFO] Sequence: {seq}")

    print(f"[INFO] Computing pair occupancies from {args.traj} ...")
    occ, n_frames = compute_pair_occupancies(
        args.traj,
        N,
        index_sep=args.index_sep,
        target_dist=args.target_dist,
        dist_tol=args.dist_tol,
        max_frames=args.max_frames,
    )
    print(f"[INFO] Processed {n_frames} frames.")

    partners = occupancy_to_partners(occ, threshold=args.threshold)
    dotbracket = partners_to_dotbracket(partners)

    # Save sequence + dot-bracket
    sec_path = f"{args.out_prefix}_secondary.txt"
    with open(sec_path, "w") as f:
        f.write(f"Sequence : {seq}\n")
        f.write(f"Structure: {dotbracket}\n")
    print(f"[INFO] Wrote secondary structure to {sec_path}")
    print(f"         {seq}")
    print(f"         {dotbracket}")

    # Save pairs + occupancies
    pairs_path = f"{args.out_prefix}_pairs.csv"
    with open(pairs_path, "w") as f:
        f.write("i,j,occupancy\n")
        N_local = len(partners)
        for i in range(N_local):
            j = partners[i]
            if j > i:
                f.write(f"{i+1},{j+1},{occ[i,j]:.4f}\n")
    print(f"[INFO] Wrote pair occupancies to {pairs_path}")

    # Plot arc diagram
    png_path = f"{args.out_prefix}_arc.png"
    plot_arc_structure(partners, occ, png_path,
                       title=f"{args.out_prefix} secondary structure")
    print(f"[INFO] Wrote arc diagram to {png_path}")

    # OVITO-ready LAMMPS data with Bonds
    ovito_path = f"{args.out_prefix}_ovito.data"
    write_ovito_data_with_bonds(args.data, ovito_path, partners)


if __name__ == "__main__":
    main()
