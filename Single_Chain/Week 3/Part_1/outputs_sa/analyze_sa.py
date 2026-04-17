#!/usr/bin/env python3
import argparse, gzip, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- masses by type (your CG model) ----
TYPE_MASS = {1:329.20, 2:305.20, 3:345.20, 4:306.20}

# Recognize nucleic acid residues and map to A/C/G/U
RES2BASE = {
    'A':'A','ADE':'A','RA':'A','DA':'A',
    'C':'C','CYT':'C','RC':'C','DC':'C',
    'G':'G','GUA':'G','RG':'G','DG':'G',
    'U':'U','URA':'U','RU':'U','DT':'U','T':'U'
}
BASE2TYPE = {'A':1,'C':2,'G':3,'U':4}

def open_text(path):
    p = Path(path)
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="ignore")
    return open(p, "rt", encoding="utf-8", errors="ignore")

# ----------------- dump parsing -----------------
def parse_lammpstrj(dump_path):
    """Yield (step:int, ids:np.ndarray, types:np.ndarray, xyz:np.ndarray[N,3]) for each frame."""
    frames = []
    with open_text(dump_path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline().strip())
            if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Unexpected dump format (NUMBER OF ATOMS).")
            n = int(fh.readline().strip())

            # BOX BOUNDS (3 lines)
            if not fh.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected dump format (BOX BOUNDS).")
            for _ in range(3):
                fh.readline()

            # ATOMS header
            header = fh.readline().strip()
            if not header.startswith("ITEM: ATOMS"):
                raise ValueError("Unexpected dump format (ATOMS header).")
            cols = header.split()[2:]
            want = {c:i for i,c in enumerate(cols)}
            for k in ["id","type","xu","yu","zu"]:
                if k not in want:
                    raise ValueError(f"Dump missing column {k}; got columns: {cols}")

            rows = [fh.readline().split() for _ in range(n)]
            ids = np.array([int(r[want["id"]]) for r in rows])
            types = np.array([int(r[want["type"]]) for r in rows])
            xyz = np.stack([[float(r[want["xu"]]), float(r[want["yu"]]), float(r[want["zu"]])] for r in rows])

            order = np.argsort(ids)
            frames.append((step, ids[order], types[order], xyz[order]))
    return frames

# ----------------- math helpers -----------------
def kabsch(P, Q):
    """Return 3x3 rotation aligning P->Q (both centered Nx3)."""
    C = P.T @ Q
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    return V @ D @ Wt

def rmsd_aligned(P, Q, masses=None):
    """RMSD after optimal superposition of P onto Q. P,Q: Nx3. masses optional (N,)."""
    if masses is None:
        masses = np.ones(len(P))
    M = masses[:, None]
    Pc = P - (np.sum(M * P, axis=0) / np.sum(masses))
    Qc = Q - (np.sum(M * Q, axis=0) / np.sum(masses))
    R = kabsch(Pc, Qc)
    Pr = Pc @ R.T
    diff2 = ((Pr - Qc) ** 2).sum(axis=1)
    return float(np.sqrt(np.sum(masses * diff2) / np.sum(masses)))

def radius_of_gyration(X, masses=None):
    if masses is None:
        masses = np.ones(len(X))
    Mtot = np.sum(masses)
    rcm = np.sum(masses[:, None] * X, axis=0) / Mtot
    rg2 = np.sum(masses * np.sum((X - rcm) ** 2, axis=1)) / Mtot
    return float(np.sqrt(rg2))

def types_to_masses(types):
    return np.array([TYPE_MASS.get(int(t), 320.0) for t in types], dtype=float)

def steps_to_time_ns(steps, timestep_fs=10.0):
    return np.array(steps, dtype=float) * (timestep_fs / 1e6)

# ----------------- thermo/log parsing -----------------
def parse_thermo_avg(avg_path):
    """
    Parse 'fix ave/time' output: first numeric column is step,
    then we expect v_vpe, v_vke, v_vT (by your setup). Returns DataFrame(step, pe).
    """
    rows = []
    with open_text(avg_path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if not parts:
                continue
            if not re.match(r"^-?\d+$", parts[0]):
                continue
            step = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            if len(vals) >= 1:
                rows.append((step, vals[0]))
    if not rows:
        return pd.DataFrame(columns=["step","pe"])
    return pd.DataFrame(rows, columns=["step","pe"]).drop_duplicates("step").sort_values("step")

def parse_log_pe(log_path):
    """Parse PE from LAMMPS log sections with headers like 'Step Temp E_pair ...' or 'Step pe ke temp'."""
    steps, pes = [], []
    headers = None
    in_table = False
    with open_text(log_path) as fh:
        for ln in fh:
            s = ln.strip()
            # Header line starts with Step
            if s.startswith("Step"):
                headers = s.split()
                in_table = True
                continue
            if in_table:
                if not s or s.startswith("Loop time"):
                    in_table = False
                    headers = None
                    continue
                parts = s.split()
                if headers and len(parts) >= len(headers):
                    d = dict(zip(headers, parts))
                    # try common PE header keys
                    pe_val = None
                    for key in ("pe", "E_pair", "Epair", "PotEng"):
                        if key in d:
                            try:
                                pe_val = float(d[key])
                                break
                            except ValueError:
                                pass
                    if pe_val is None:
                        continue
                    steps.append(int(d.get("Step", parts[0])))
                    pes.append(pe_val)
    if not steps:
        return pd.DataFrame(columns=["step","pe"])
    return pd.DataFrame({"step":steps, "pe":pes}).drop_duplicates("step").sort_values("step")

# ----------------- PDB → reference -----------------
def parse_pdb_reference(pdb_path, chain='C', atom_priority=None):
    """
    Returns (ref_xyz Nx3, ref_base_letters list[str]) for selected chain.
    Picks one coordinate per nucleotide residue using atom_priority order,
    else residue centroid as fallback.
    """
    if atom_priority is None:
        atom_priority = ["C3'", "C3*", "P", "C4'", "C1'"]

    residues = {}  # {(resseq,icode): {'resn':..., 'atoms':{atom_name:(x,y,z)}}}
    with open_text(pdb_path) as fh:
        for ln in fh:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            resn = ln[17:20].strip().upper()
            ch = (ln[21] or ' ').strip()
            if ch != chain:
                continue
            if resn not in RES2BASE:
                continue
            resseq = ln[22:26].strip()
            icode = ln[26].strip()
            try:
                resseq_key = int(resseq)
            except Exception:
                resseq_key = resseq
            key = (resseq_key, icode)

            atom_name = ln[12:16].strip().upper().replace('*', "'")
            try:
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except Exception:
                continue

            if key not in residues:
                residues[key] = {'resn': resn, 'atoms': {}}
            residues[key]['atoms'][atom_name] = (x, y, z)

    if not residues:
        raise SystemExit(f"No nucleotide residues found in chain {chain} of {pdb_path}")

    keys_sorted = sorted(residues.keys(), key=lambda k: (k[0] if isinstance(k[0], int) else -1, str(k[1])))
    ref_xyz, ref_bases = [], []
    for key in keys_sorted:
        entry = residues[key]
        atoms = entry['atoms']
        pos = None
        for a in atom_priority:
            a2 = a.replace('*', "'").upper()
            if a2 in atoms:
                pos = atoms[a2]
                break
        if pos is None:
            arr = np.array(list(atoms.values()), dtype=float)
            pos = tuple(arr.mean(axis=0))
        ref_xyz.append(pos)
        ref_bases.append(RES2BASE.get(entry['resn'], 'U'))
    return np.array(ref_xyz, dtype=float), ref_bases

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Analyze SA: RMSD/Rg from dump, PE from thermo/log, RMSD vs PDB reference.")
    ap.add_argument("--dump", required=True, help="outputs/<tag>_SA.lammpstrj[.gz]")
    ap.add_argument("--thermo_avg", help="outputs/thermo_<tag>_SA_averaged.dat (optional)")
    ap.add_argument("--log", help="outputs/log.<tag>.sa.lammps (optional)")
    ap.add_argument("--timestep_fs", type=float, default=10.0)
    ap.add_argument("--outdir", default="metrics_sa")
    ap.add_argument("--ref_pdb", help="Reference PDB/PDB1 to compute RMSD against")
    ap.add_argument("--ref_chain", default="C", help="Chain ID in reference PDB (default: C)")
    ap.add_argument("--ref_atom", default="C3'", help="Preferred atom name for per-residue reference (e.g., C3', P).")
    ap.add_argument("--truncate_mismatch", action="store_true",
                    help="If lengths differ, trim to min length instead of error.")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # ---- parse dump ----
    frames = parse_lammpstrj(args.dump)
    if not frames:
        raise SystemExit(f"No frames found in {args.dump}")
    ref_dump_step, ref_dump_ids, ref_dump_types, ref_xyz0 = frames[0]
    dump_masses_full = types_to_masses(ref_dump_types)

    # ---- PDB reference (optional) ----
    pdb_ref_xyz = None
    N_use = len(ref_dump_types)
    if args.ref_pdb:
        atom_priority = [args.ref_atom.replace('*', "'")]
        for a in ["C3'", "P", "C4'", "C1'"]:
            if a not in atom_priority:
                atom_priority.append(a)
        pdb_ref_xyz, _ = parse_pdb_reference(args.ref_pdb, chain=args.ref_chain,
                                             atom_priority=atom_priority)

        n_dump = len(ref_dump_types)
        n_pdb = len(pdb_ref_xyz)
        if n_dump != n_pdb:
            msg = f"[WARN] Length mismatch: dump has {n_dump} beads; PDB chain {args.ref_chain} has {n_pdb} residues."
            if args.truncate_mismatch:
                N_use = min(n_dump, n_pdb)
                print(msg + f" Truncating to first {N_use}.", file=sys.stderr)
                pdb_ref_xyz = pdb_ref_xyz[:N_use]
                dump_masses = dump_masses_full[:N_use]
            else:
                raise SystemExit(msg + " (use --truncate_mismatch to auto-trim).")
        else:
            dump_masses = dump_masses_full
    else:
        dump_masses = dump_masses_full

    # ---- per-frame metrics ----
    records = []
    for step, ids, types, xyz in frames:
        if not np.array_equal(ids, ref_dump_ids):
            raise SystemExit("Atom IDs change between frames; cannot compute RMSD reliably.")
        coords = xyz[:N_use]
        masses = dump_masses[:N_use]

        rg = radius_of_gyration(coords, masses)

        if pdb_ref_xyz is not None:
            target = pdb_ref_xyz[:N_use]
        else:
            target = ref_xyz0[:N_use]
        rmsd = rmsd_aligned(coords, target, masses)

        records.append((step, rg, rmsd))

    df_geom = pd.DataFrame(records, columns=["step","Rg","RMSD"]).sort_values("step")
    df_geom["time_ns"] = steps_to_time_ns(df_geom["step"].values, args.timestep_fs)

    # ---- PE: prefer thermo_avg, else log ----
    if args.thermo_avg and Path(args.thermo_avg).exists():
        df_pe = parse_thermo_avg(args.thermo_avg)
    elif args.log and Path(args.log).exists():
        df_pe = parse_log_pe(args.log)
    else:
        df_pe = pd.DataFrame(columns=["step","pe"])
        print("No thermo source provided; skipping PE.", file=sys.stderr)

    if not df_pe.empty:
        df_pe["time_ns"] = steps_to_time_ns(df_pe["step"].values, args.timestep_fs)

    # ---- file tag for outputs ----
    tag = Path(args.dump).name
    if tag.endswith(".gz"):
        tag = tag[:-3]
    if tag.endswith(".lammpstrj"):
        tag = tag[:-11]

    # ---- write CSVs ----
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df_geom[["step","time_ns","Rg","RMSD"]].to_csv(Path(args.outdir, f"geom_{tag}.csv"), index=False)
    if not df_pe.empty:
        df_pe[["step","time_ns","pe"]].to_csv(Path(args.outdir, f"pe_{tag}.csv"), index=False)

    # ---- plots ----
    plt.figure()
    plt.plot(df_geom["time_ns"], df_geom["RMSD"])
    plt.xlabel("Time (ns)"); plt.ylabel("RMSD (Å)"); plt.title("RMSD vs time")
    plt.tight_layout(); plt.savefig(Path(args.outdir, f"RMSD_{tag}.png"), dpi=200)

    plt.figure()
    plt.plot(df_geom["time_ns"], df_geom["Rg"])
    plt.xlabel("Time (ns)"); plt.ylabel("Rg (Å)"); plt.title("Radius of gyration vs time")
    plt.tight_layout(); plt.savefig(Path(args.outdir, f"Rg_{tag}.png"), dpi=200)

    if not df_pe.empty:
        plt.figure()
        plt.plot(df_pe["time_ns"], df_pe["pe"])
        plt.xlabel("Time (ns)"); plt.ylabel("Potential energy"); plt.title("Potential energy vs time")
        plt.tight_layout(); plt.savefig(Path(args.outdir, f"PE_{tag}.png"), dpi=200)

    print("Done.",
          f"\n  geom CSV: {Path(args.outdir, f'geom_{tag}.csv')}"
          f"\n  pe CSV:   {Path(args.outdir, f'pe_{tag}.csv') if not df_pe.empty else 'n/a'}"
          f"\n  plots:    {Path(args.outdir, f'RMSD_{tag}.png')}, {Path(args.outdir, f'Rg_{tag}.png')}"
          f"{', ' + str(Path(args.outdir, f'PE_{tag}.png')) if not df_pe.empty else ''}")

if __name__ == "__main__":
    main()
