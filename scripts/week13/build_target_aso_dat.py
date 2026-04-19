#!/usr/bin/env python3
"""
Build a combined RNA-target + ASO LAMMPS data file for Week 13 binding simulations.

Molecule convention (matches Week 9/10 generate_all_configs.py):
  mol 1      = ASO (primary, placed at RNA centroid + x-offset)
  mol 2      = RNA target (recentered to origin)
  mol 3..N   = free ASOs scattered in box (only if --n-aso > 1)

Force field: same coarse-grained 4-type scheme used throughout the thesis.
  A = type 1 (329.20 Da)   C = type 2 (305.20 Da)
  G = type 3 (345.20 Da)   U = type 4 (306.20 Da)

Unmodified ASOs reuse the same 4 atom types — no new pair_coeff lines needed.
Box default ±80 Å matches the Week 9/10 binding run standard.

Usage examples:

  # Single ASO pilot for 7LYJ (placeholder sequence):
  python3 build_target_aso_dat.py \\
    --rna-dat  ../../Single_Chain/Week\\ 13/runs/7LYJ/inputs/7LYJ.dat \\
    --aso-seq  ACGUACGUACGU \\
    --out      ../../Single_Chain/Week\\ 13/runs/7LYJ/inputs/7LYJ_aso_combined.dat

  # 100-copy binding run for 7LYJ (1 docked + 99 free ASOs):
  python3 build_target_aso_dat.py \\
    --rna-dat  ../../Single_Chain/Week\\ 13/runs/7LYJ/inputs/7LYJ.dat \\
    --aso-seq  ACGUACGUACGU \\
    --n-aso    100 \\
    --out      ../../Single_Chain/Week\\ 13/runs/7LYJ/inputs/7LYJ_aso_100_combined.dat

  # Custom box and offset:
  python3 build_target_aso_dat.py \\
    --rna-dat  ../../Single_Chain/Week\\ 13/runs/1ANR/inputs/1ANR.dat \\
    --aso-seq  GGGCUGUUU \\
    --box-half 80.0 \\
    --offset   50.0 \\
    --out      ../../Single_Chain/Week\\ 13/runs/1ANR/inputs/1ANR_aso_combined.dat
"""

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — identical to Week 9/10 generate_all_configs.py
# ---------------------------------------------------------------------------

TYPE_MAP = {"A": 1, "C": 2, "G": 3, "U": 4}
MASS_MAP = {1: 329.20, 2: 305.20, 3: 345.20, 4: 306.20}
BOND_EQ  = 5.9    # Å — harmonic bond equilibrium distance (bond_coeff 1 7.5 5.9)

# ---------------------------------------------------------------------------
# RNA .dat parser
# ---------------------------------------------------------------------------

def parse_rna_dat(path: Path):
    """
    Parse a prepared RNA LAMMPS data file (from prepare_target pipeline).
    Returns:
      atoms  : list of [atom_id, mol_id, type, charge, x, y, z]  (strings)
      bonds  : list of [bond_id, type, i, j]
      angles : list of [angle_id, type, i, j, k]
    """
    atoms, bonds, angles = [], [], []
    section = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s == "Atoms":
                section = "atoms"; continue
            if s == "Bonds":
                section = "bonds"; continue
            if s == "Angles":
                section = "angles"; continue
            if s in ("Masses", "Velocities"):
                section = s; continue
            # skip any header / box / count lines
            if any(kw in s for kw in ["xlo", "ylo", "zlo", "atom type", "bond type",
                                       "angle type", "LAMMPS", "atoms", "bonds", "angles",
                                       "Masses"]):
                section = None if s.endswith(("atoms","bonds","angles")) else section
                continue

            parts = s.split()
            if section == "atoms"  and len(parts) >= 7:
                atoms.append(parts[:7])
            elif section == "bonds"  and len(parts) >= 4:
                bonds.append(parts[:4])
            elif section == "angles" and len(parts) >= 5:
                angles.append(parts[:5])

    return atoms, bonds, angles


def centroid(atoms):
    n = len(atoms)
    cx = sum(float(a[4]) for a in atoms) / n
    cy = sum(float(a[5]) for a in atoms) / n
    cz = sum(float(a[6]) for a in atoms) / n
    return cx, cy, cz


def translate_atoms(atoms, dx, dy, dz):
    """Return new atom list with coordinates shifted."""
    out = []
    for a in atoms:
        out.append([a[0], a[1], a[2], a[3],
                    f"{float(a[4])+dx:.6f}",
                    f"{float(a[5])+dy:.6f}",
                    f"{float(a[6])+dz:.6f}"])
    return out


# ---------------------------------------------------------------------------
# ASO coordinate builder
# ---------------------------------------------------------------------------

def build_aso_coords(sequence: str, start: tuple, step: float = BOND_EQ):
    """
    Build a straight-chain ASO along the +x axis.
    start  : (x0, y0, z0) for bead 1
    step   : Å between successive beads (bond equilibrium length)
    Returns list of (type_int, x, y, z).
    """
    seq = sequence.upper()
    coords = []
    x0, y0, z0 = start
    for i, base in enumerate(seq):
        t = TYPE_MAP.get(base)
        if t is None:
            raise ValueError(f"Unknown base '{base}' in ASO sequence '{sequence}'. Use A/C/G/U only.")
        coords.append((t, x0 + i * step, y0, z0))
    return coords


# ---------------------------------------------------------------------------
# Free-ASO scatter (mirrors Week 9/10 scatter_positions)
# ---------------------------------------------------------------------------

def scatter_positions(n, box_half, exclusion_radius=40.0, min_sep=15.0, rng=None):
    """
    Place n centres randomly in the box, avoiding the central exclusion zone
    and maintaining min_sep between any two centres.
    """
    if rng is None:
        rng = random
    positions = []
    attempts = 0
    max_attempts = n * 2000
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        px = rng.uniform(-box_half + 10, box_half - 10)
        py = rng.uniform(-box_half + 10, box_half - 10)
        pz = rng.uniform(-box_half + 10, box_half - 10)
        if math.sqrt(px**2 + py**2 + pz**2) < exclusion_radius:
            continue
        if any(math.sqrt((px-ex)**2+(py-ey)**2+(pz-ez)**2) < min_sep
               for ex, ey, ez in positions):
            continue
        positions.append((px, py, pz))

    if len(positions) < n:
        print(f"  WARNING: only placed {len(positions)}/{n} free ASOs — box may be too small")
    return positions


# ---------------------------------------------------------------------------
# Bond / angle generators
# ---------------------------------------------------------------------------

def bonds_for_chain(first_atom_id: int, n_beads: int, bond_id_start: int):
    bonds = []
    for i in range(n_beads - 1):
        bonds.append(f"{bond_id_start+i} 1 {first_atom_id+i} {first_atom_id+i+1}")
    return bonds


def angles_for_chain(first_atom_id: int, n_beads: int, angle_id_start: int):
    angles = []
    for i in range(n_beads - 2):
        angles.append(f"{angle_id_start+i} 1 {first_atom_id+i} {first_atom_id+i+1} {first_atom_id+i+2}")
    return angles


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_combined_dat(rna_dat: Path, aso_seq: str, out_path: Path,
                       n_aso: int = 1, box_half: float = 80.0,
                       offset: float = 50.0, seed: int = 42):
    """
    Assemble the combined RNA+ASO LAMMPS data file.

    Molecule numbering:
      mol 1     = primary ASO (placed at RNA centroid + offset along +x)
      mol 2     = RNA target  (recentered to origin)
      mol 3..N  = (n_aso - 1) free ASOs scattered in box
    """
    rng = random.Random(seed)

    # -- Parse RNA --
    rna_atoms, rna_bonds, rna_angles = parse_rna_dat(rna_dat)
    if not rna_atoms:
        raise ValueError(f"No atoms parsed from {rna_dat}. Check file format.")

    n_rna = len(rna_atoms)
    n_aso_beads = len(aso_seq)
    n_free = max(0, n_aso - 1)

    # -- Center RNA at origin --
    cx, cy, cz = centroid(rna_atoms)
    rna_atoms = translate_atoms(rna_atoms, -cx, -cy, -cz)

    # -- Build primary ASO coords: start at (offset, 0, 0), chain along +x --
    aso_start = (offset, 0.0, 0.0)
    aso_primary = build_aso_coords(aso_seq, aso_start)

    # -- Scatter free ASOs --
    free_centres = scatter_positions(n_free, box_half, rng=rng)
    actual_free = len(free_centres)

    # -- Atom totals --
    total_atoms  = n_aso_beads + n_rna + actual_free * n_aso_beads
    total_bonds  = (n_aso_beads-1) + (n_rna-1) + actual_free*(n_aso_beads-1)
    total_angles = max(0,n_aso_beads-2) + max(0,n_rna-2) + actual_free*max(0,n_aso_beads-2)

    out_lines_atoms  = []
    out_lines_bonds  = []
    out_lines_angles = []

    atom_id  = 0
    bond_id  = 0
    angle_id = 0

    # ── mol 1: primary ASO ──
    aso1_first = atom_id + 1
    for atype, x, y, z in aso_primary:
        atom_id += 1
        out_lines_atoms.append(f"{atom_id} 1 {atype} 0.000000 {x:.6f} {y:.6f} {z:.6f}")

    for b in bonds_for_chain(aso1_first, n_aso_beads, bond_id+1):
        bond_id += 1
        out_lines_bonds.append(b)
    for a in angles_for_chain(aso1_first, n_aso_beads, angle_id+1):
        angle_id += 1
        out_lines_angles.append(a)

    # ── mol 2: RNA target ──
    rna_first = atom_id + 1
    for ra in rna_atoms:
        atom_id += 1
        out_lines_atoms.append(
            f"{atom_id} 2 {ra[2]} {ra[3]} {ra[4]} {ra[5]} {ra[6]}"
        )
    for b in bonds_for_chain(rna_first, n_rna, bond_id+1):
        bond_id += 1
        out_lines_bonds.append(b)
    for a in angles_for_chain(rna_first, n_rna, angle_id+1):
        angle_id += 1
        out_lines_angles.append(a)

    # ── mol 3..N: free ASOs ──
    # ASO centre of mass for translation
    aso_cx = sum(c[1] for c in aso_primary) / n_aso_beads
    aso_cy = sum(c[2] for c in aso_primary) / n_aso_beads
    aso_cz = sum(c[3] for c in aso_primary) / n_aso_beads

    for mol_idx, (px, py, pz) in enumerate(free_centres, start=3):
        first = atom_id + 1
        for atype, tx, ty, tz in aso_primary:
            atom_id += 1
            nx = tx - aso_cx + px
            ny = ty - aso_cy + py
            nz = tz - aso_cz + pz
            out_lines_atoms.append(f"{atom_id} {mol_idx} {atype} 0.000000 {nx:.6f} {ny:.6f} {nz:.6f}")
        for b in bonds_for_chain(first, n_aso_beads, bond_id+1):
            bond_id += 1
            out_lines_bonds.append(b)
        for a in angles_for_chain(first, n_aso_beads, angle_id+1):
            angle_id += 1
            out_lines_angles.append(a)

    # ── Write .dat ──
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = (f"Week 13 combined RNA+ASO | target={rna_dat.stem} | "
             f"aso={aso_seq} | n_aso={n_aso} | box=±{box_half}")
    with open(out_path, "w") as fh:
        fh.write(f"LAMMPS data file — {title}\n\n")
        fh.write(f"{atom_id} atoms\n{bond_id} bonds\n{angle_id} angles\n\n")
        fh.write("4 atom types\n1 bond types\n1 angle types\n\n")
        fh.write(f"{-box_half:.6f}  {box_half:.6f}  xlo xhi\n")
        fh.write(f"{-box_half:.6f}  {box_half:.6f}  ylo yhi\n")
        fh.write(f"{-box_half:.6f}  {box_half:.6f}  zlo zhi\n\n")
        fh.write("Masses\n\n")
        for t, m in MASS_MAP.items():
            fh.write(f"{t} {m:.2f}\n")
        fh.write("\nAtoms\n\n")
        fh.write("\n".join(out_lines_atoms) + "\n")
        fh.write("\nBonds\n\n")
        fh.write("\n".join(out_lines_bonds) + "\n")
        fh.write("\nAngles\n\n")
        fh.write("\n".join(out_lines_angles) + "\n")

    # ── Write summary ──
    summary_path = out_path.with_suffix(".summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(f"build_target_aso_dat.py — output summary\n")
        fh.write(f"Generated : {datetime.now().isoformat()}\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"RNA target  : {rna_dat}  ({n_rna} beads)\n")
        fh.write(f"ASO seq     : {aso_seq}  ({n_aso_beads} beads)\n")
        fh.write(f"n_aso total : {n_aso}  (1 docked + {actual_free} free)\n")
        fh.write(f"Box         : ±{box_half} Å\n")
        fh.write(f"Offset      : +{offset} Å along x from RNA centroid\n")
        fh.write(f"Seed        : {seed}\n\n")
        fh.write(f"Molecule layout:\n")
        fh.write(f"  mol 1   ASO (primary, atoms 1–{n_aso_beads})\n")
        fh.write(f"  mol 2   RNA target (atoms {n_aso_beads+1}–{n_aso_beads+n_rna})\n")
        if actual_free:
            fh.write(f"  mol 3–{actual_free+2}  free ASOs ({actual_free} copies)\n")
        fh.write(f"\nTotals: {atom_id} atoms, {bond_id} bonds, {angle_id} angles\n")
        fh.write(f"Output: {out_path}\n")

    print(f"[OK] {out_path.name}  "
          f"({atom_id} atoms | mol1=ASO {n_aso_beads}nt | mol2=RNA {n_rna}nt | "
          f"{actual_free} free ASOs)")
    print(f"     Summary: {summary_path.name}")

    return {"atoms": atom_id, "bonds": bond_id, "angles": angle_id,
            "n_free_placed": actual_free}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build a combined RNA+ASO LAMMPS .dat for Week 13 binding simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--rna-dat",  required=True,
                    help="Prepared RNA .dat from prepare_target pipeline")
    ap.add_argument("--aso-seq",  required=True,
                    help="ASO sequence 5'→3', single-letter ACGU (e.g. ACGUACGUACGU)")
    ap.add_argument("--out",      required=True,
                    help="Output combined .dat path")
    ap.add_argument("--n-aso",   type=int,   default=1,
                    help="Total ASO count: 1=single pilot, 100=full binding run (default 1)")
    ap.add_argument("--box-half", type=float, default=80.0,
                    help="Box half-length in Å (default 80.0, Week 9/10 standard)")
    ap.add_argument("--offset",   type=float, default=50.0,
                    help="Å offset from RNA centroid for initial ASO placement (default 50.0)")
    ap.add_argument("--seed",     type=int,   default=42,
                    help="Random seed for free-ASO scatter (default 42)")
    args = ap.parse_args()

    # Validate sequence
    bad = set(args.aso_seq.upper()) - set("ACGU")
    if bad:
        ap.error(f"ASO sequence contains invalid characters: {bad}. Use A/C/G/U only.")

    if len(args.aso_seq) < 2:
        ap.error("ASO sequence must be at least 2 nucleotides.")

    build_combined_dat(
        rna_dat  = Path(args.rna_dat),
        aso_seq  = args.aso_seq.upper(),
        out_path = Path(args.out),
        n_aso    = args.n_aso,
        box_half = args.box_half,
        offset   = args.offset,
        seed     = args.seed,
    )


if __name__ == "__main__":
    main()
