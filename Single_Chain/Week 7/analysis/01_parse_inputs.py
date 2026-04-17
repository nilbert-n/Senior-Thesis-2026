"""
Script 1 – Parse all input files and build reference structures from PDB.
Outputs:  parsed_data.npz   (reference coords + trajectory + thermo)

Usage:
    python 01_parse_inputs.py

Inputs expected in same directory (edit paths at top if different):
    100ASO.dat                          LAMMPS data file
    100ASO_dat_NVE.lammpstrj            LAMMPS dump trajectory
    thermo_100ASO_dat_NVE_averaged.dat  thermodynamic log
    1YMO.pdb1                           NMR reference structure (MODEL 1)
"""

import numpy as np
from collections import defaultdict

# ── File paths (edit here if needed) ─────────────────────────────────────────
DAT_FILE    = "100ASO.dat"
TRAJ_FILE   = "100ASO.dat_NVE.lammpstrj"
THERMO_FILE = "thermo_100ASO.dat_NVE_averaged.dat"
PDB_FILE    = "1YMO.pdb1"
OUT_FILE    = "parsed_data.npz"

# ── Atom-type → nucleotide name  (from masses in .dat) ───────────────────────
#   type 1 = A (329.20), 2 = C (305.20), 3 = G (345.20), 4 = U (306.20)
TYPE_TO_NUC = {1: "A", 2: "C", 3: "G", 4: "U"}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Parse LAMMPS .dat  →  topology
# ═════════════════════════════════════════════════════════════════════════════
def parse_dat(fname):
    with open(fname) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "Atoms") + 2
    n_atoms = int(next(l for l in lines if "atoms" in l and "atom" in l.lower()).split()[0])

    atom_info = {}   # atom_id → (mol_id, atom_type, np.array xyz)
    for line in lines[start : start + n_atoms]:
        p = line.split()
        if len(p) < 7:
            continue
        aid, mid, atype = int(p[0]), int(p[1]), int(p[2])
        xyz = np.array([float(p[4]), float(p[5]), float(p[6])])
        atom_info[aid] = (mid, atype, xyz)

    mol_atoms = defaultdict(list)
    for aid, (mid, _, _) in atom_info.items():
        mol_atoms[mid].append(aid)
    for mid in mol_atoms:
        mol_atoms[mid].sort()

    return atom_info, dict(mol_atoms)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Parse trajectory
# ═════════════════════════════════════════════════════════════════════════════
def parse_traj(fname):
    """Returns list of (timestep, {atom_id: np.array([x,y,z])})."""
    frames = []
    with open(fname) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if "TIMESTEP" in lines[i]:
            ts = int(lines[i + 1])
            n  = int(lines[i + 3])
            coords = {}
            for j in range(9, 9 + n):
                p = lines[i + j].split()
                coords[int(p[0])] = np.array([float(p[4]), float(p[5]), float(p[6])])
            frames.append((ts, coords))
            i += 9 + n
        else:
            i += 1
    return frames


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Parse thermodynamics file
# ═════════════════════════════════════════════════════════════════════════════
def parse_thermo(fname):
    data = {"timestep": [], "PE": [], "KE": [], "T": [], "E_total": []}
    with open(fname) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) == 5:
                data["timestep"].append(int(p[0]))
                data["PE"].append(float(p[1]))
                data["KE"].append(float(p[2]))
                data["T"].append(float(p[3]))
                data["E_total"].append(float(p[4]))
    return {k: np.array(v) for k, v in data.items()}


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Parse PDB → reference C3' coordinates (MODEL 1 only)
# ═════════════════════════════════════════════════════════════════════════════
def parse_pdb(fname):
    pdb_c3    = {}   # resid → xyz
    pdb_names = {}   # resid → nucleotide name (G/C/A/U)
    in_m1 = False
    with open(fname) as f:
        for line in f:
            if line.startswith("MODEL        1"):
                in_m1 = True
            elif line.startswith("ENDMDL") and in_m1:
                break
            if in_m1 and line.startswith("ATOM") and line[12:16].strip() == "C3'":
                resid = int(line[22:26])
                pdb_c3[resid]    = np.array([float(line[30:38]),
                                             float(line[38:46]),
                                             float(line[46:54])])
                pdb_names[resid] = line[17:20].strip()
    return pdb_c3, pdb_names


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Build per-molecule coordinate arrays from trajectory
# ═════════════════════════════════════════════════════════════════════════════
def mol_coords_over_time(frames, atom_list):
    """→ (n_frames, n_atoms, 3)"""
    arr = np.zeros((len(frames), len(atom_list), 3))
    for fi, (_, coords) in enumerate(frames):
        for ai, aid in enumerate(atom_list):
            arr[fi, ai] = coords[aid]
    return arr


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Parsing .dat …")
    atom_info, mol_atoms = parse_dat(DAT_FILE)

    # Molecule assignments:
    #   mol 1  → docked ASO (10 beads = PDB res 1-10)
    #   mol 2  → hairpin     (32 beads = PDB res 15-46)
    #   mol 3-101 → 99 free ASOs (10 beads each)
    docked_aso_aids = mol_atoms[1]
    hairpin_aids    = mol_atoms[2]
    free_aso_mols   = list(range(3, 102))

    print("Parsing trajectory …")
    frames = parse_traj(TRAJ_FILE)
    print(f"  {len(frames)} frames,  ts {frames[0][0]} → {frames[-1][0]}")

    print("Parsing thermodynamics …")
    thermo = parse_thermo(THERMO_FILE)

    print("Parsing PDB …")
    pdb_c3, pdb_names = parse_pdb(PDB_FILE)

    # PDB reference vectors (already verified 0.000 Å mismatch)
    pdb_aso_ref = np.array([pdb_c3[r] for r in range(1, 11)])       # (10, 3)
    pdb_hp_ref  = np.array([pdb_c3[r] for r in range(15, 47)])      # (32, 3)

    # Sequences
    aso_seq = [TYPE_TO_NUC[atom_info[a][1]] for a in docked_aso_aids]
    hp_seq  = [pdb_names[r] for r in range(15, 47)]

    # Coarse-grained residue labels
    #   ASO:     paper numbers 1-10  (3'→5': U10…U7,G6,U5,C4,G3,G2,G1)
    #   Hairpin: paper numbers 11-42 → map to PDB 15-46
    aso_labels = [f"{i+1}({aso_seq[i]})" for i in range(10)]
    hp_labels  = [f"{i+11}({hp_seq[i]})" for i in range(32)]

    print("Extracting coordinate arrays …")
    aso_traj = mol_coords_over_time(frames, docked_aso_aids)   # (F,10,3)
    hp_traj  = mol_coords_over_time(frames, hairpin_aids)      # (F,32,3)

    free_aso_trajs = []
    for mid in free_aso_mols:
        aids = mol_atoms[mid]
        free_aso_trajs.append(mol_coords_over_time(frames, aids))
    free_aso_trajs = np.array(free_aso_trajs)   # (99, F, 10, 3)

    timesteps = np.array([f[0] for f in frames])

    print(f"Saving → {OUT_FILE}")
    np.savez_compressed(
        OUT_FILE,
        timesteps       = timesteps,
        aso_traj        = aso_traj,
        hp_traj         = hp_traj,
        free_aso_trajs  = free_aso_trajs,
        pdb_aso_ref     = pdb_aso_ref,
        pdb_hp_ref      = pdb_hp_ref,
        aso_seq         = np.array(aso_seq),
        hp_seq          = np.array(hp_seq),
        aso_labels      = np.array(aso_labels),
        hp_labels       = np.array(hp_labels),
        thermo_ts       = thermo["timestep"],
        thermo_PE       = thermo["PE"],
        thermo_KE       = thermo["KE"],
        thermo_T        = thermo["T"],
        thermo_E        = thermo["E_total"],
    )
    print("Done.")
    print(f"  Docked ASO sequence : {' '.join(aso_seq)}")
    print(f"  Hairpin sequence    : {' '.join(hp_seq)}")
