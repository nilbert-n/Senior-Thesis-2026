#!/usr/bin/env python3
"""
analyze_strict.py
Compute Rg vs time, RMSD (to PDB C3'), and Potential Energy vs time.

"Tight" filename rules (no directory rglob scanning):
- Trajectory: config_*.dat.lammpstrj  [default glob]
- Log:        log.config_<core>.lammps (same or parent dir of traj)
- Thermo:     thermo_config_<core>_averaged.dat (same or parent dir)
- PDB:        <stem>.pdb1 (preferred), else <stem>.pdb (optionally .gz)

<core> = "config_<stem>_<CHAIN>.dat", e.g., config_1_2DER_C.dat
<stem> = the middle part, e.g., 1_2DER

Examples:
  traj:  config_1_2DER_C.dat.lammpstrj
  log:   log.config_1_2DER_C.dat.lammps
  pdb:   1_2DER.pdb1
  thermo:thermo_config_1_2DER_C.dat_averaged.dat
"""

import re, gzip, sys, glob
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Args ----------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Rg/RMSD/PE vs time with strict filename matching.")
    ap.add_argument("--pdb", help="Explicit PDB path (reference for ALL trajectories).")
    ap.add_argument("--pdb-dir", default=".", help="Directory to look for PDBs (default: '.').")
    ap.add_argument("--chain", help="Preferred chain ID (e.g., C). If missing, auto-pick by C3' count.")
    ap.add_argument("--traj", help="One trajectory (.lammpstrj or .lammpstrj.gz if --allow-gz).")
    ap.add_argument("--traj-glob", default="config_*.dat.lammpstrj", help="Strict glob for trajectories.")
    ap.add_argument("--allow-gz", action="store_true", help="Also accept .gz for traj/log/PDB.")
    ap.add_argument("--log", help="Explicit log path (.lammps or .gz).")
    ap.add_argument("--timestep-fs", type=float, default=None, help="Override timestep fs (else read from log; fallback --default-timestep-fs).")
    ap.add_argument("--default-timestep-fs", type=float, default=10.0, help="Fallback timestep fs if not in log (default 10).")
    ap.add_argument("--outdir", default=".", help="Output root directory (default '.').")
    ap.add_argument("--no-plots", action="store_true", help="Skip plotting (write CSVs only).")
    ap.add_argument("--plot-downsample", type=int, default=0, help="If >0, downsample plotted points to this many.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

# ---------------- Helpers ----------------
def core_from_traj_name(name: str) -> str:
    # "config_1_2DER_C.dat.lammpstrj" -> "config_1_2DER_C.dat"
    base = Path(name).name
    m = re.match(r"(config_.+?\.dat)(?:\..*)?$", base)
    return m.group(1) if m else base

def parse_chain_from_core(core: str) -> Optional[str]:
    m = re.match(r"config_.+_([A-Za-z0-9])\.dat$", core)
    return m.group(1) if m else None

def pdb_stem_from_core(core: str) -> Optional[str]:
    m = re.match(r"config_(.+)_[A-Za-z0-9]\.dat$", core)
    return m.group(1) if m else None

def open_text_auto(path: Path) -> List[str]:
    s = str(path).lower()
    if s.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            return fh.readlines()
    return path.read_text(errors="ignore").splitlines()

def open_stream_auto(path: Path):
    s = str(path).lower()
    if s.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")

# ---------------- Strict discovery ----------------
def find_pdb_for_core(pdb_dir: Path, core: str, allow_gz: bool, verbose=False) -> Optional[Path]:
    stem = pdb_stem_from_core(core)
    if not stem:
        return None
    candidates = [pdb_dir / f"{stem}.pdb1", pdb_dir / f"{stem}.pdb"]
    if allow_gz:
        candidates += [pdb_dir / f"{stem}.pdb1.gz", pdb_dir / f"{stem}.pdb.gz"]
    for c in candidates:
        if c.exists():
            if verbose: print(f"[PDB] {c}")
            return c
    if verbose: print(f"[PDB] not found for stem={stem} in {pdb_dir}")
    return None

def find_log_near(traj: Path, core: str, allow_gz: bool, verbose=False) -> Optional[Path]:
    # search strictly in traj dir and its parent
    dirs = [traj.parent, traj.parent.parent]
    names = [f"log.{core}.lammps"]
    if allow_gz: names.append(f"log.{core}.lammps.gz")
    for d in dirs:
        for n in names:
            p = d / n
            if p.exists():
                if verbose: print(f"[LOG] {p}")
                return p
    if verbose: print(f"[LOG] not found for {core} near {traj.parent}")
    return None

def find_thermo_near(traj: Path, core: str, verbose=False) -> Optional[Path]:
    # exact name only (your convention)
    names = [f"thermo_config_{core.split('config_')[-1]}_averaged.dat"]
    dirs = [traj.parent, traj.parent.parent]
    for d in dirs:
        for n in names:
            p = d / n
            if p.exists():
                if verbose: print(f"[THERMO] {p}")
                return p
    return None

# ---------------- PDB: C3' ----------------
def c3_counts_by_chain(pdb_path: Path) -> Dict[str,int]:
    counts = {}
    for ln in open_text_auto(pdb_path):
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")): continue
        atn = ln[12:16].strip()
        if atn not in ("C3'", "C3*") and atn.upper() not in ("C3'", "C3*"): continue
        ch = (ln[21].strip() or "_")
        counts[ch] = counts.get(ch, 0) + 1
    return counts

def c3_coords_sorted(pdb_path: Path, chain: str) -> np.ndarray:
    rows = []
    for ln in open_text_auto(pdb_path):
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")): continue
        ch = (ln[21].strip() or "_")
        if ch != chain: continue
        atn = ln[12:16].strip()
        if atn not in ("C3'", "C3*") and atn.upper() not in ("C3'", "C3*"): continue
        resseq = ln[22:26].strip(); icode = ln[26].strip()
        try:
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        except ValueError:
            continue
        try:
            rnum = int(resseq)
        except Exception:
            rnum = 10**9
        rows.append((rnum, icode, x, y, z))
    rows.sort(key=lambda t: (t[0], t[1]))
    return np.array([[r[2], r[3], r[4]] for r in rows], float)

# ---------------- Log / Thermo ----------------
def parse_log_timestep_and_pe(log_path: Optional[Path], default_fs: float, verbose=False):
    ts = default_fs; steps = np.array([]); pe = np.array([])
    if not log_path or not log_path.exists():
        if verbose: print(f"[LOG] missing: {log_path}")
        return ts, steps, pe
    lines = open_text_auto(log_path)
    # timestep
    for ln in lines:
        m = re.search(r"^\s*timestep\s+([0-9.+-Ee]+)", ln, flags=re.I)
        if m:
            try:
                ts = float(m.group(1))
                if verbose: print(f"[LOG] timestep = {ts} fs")
                break
            except ValueError:
                pass
    # thermo blocks
    out_steps, out_pe = [], []
    i = 0
    while i < len(lines):
        if re.search(r"^\s*Step\b", lines[i]):
            header = lines[i].split()
            lo = [h.lower() for h in header]
            idx_step = lo.index("step") if "step" in lo else None
            idx_pe = None
            for name in ("pe", "poteng", "e_pot", "epot"):
                if name in lo:
                    idx_pe = lo.index(name); break
            i += 1
            if idx_step is None:
                continue
            while i < len(lines):
                parts = lines[i].split()
                if len(parts) < len(header):
                    break
                try:
                    s = float(parts[idx_step])
                except ValueError:
                    break
                pe_val = float(parts[idx_pe]) if (idx_pe is not None and parts[idx_pe].lower() not in ("nan",)) else float("nan")
                out_steps.append(int(s)); out_pe.append(pe_val)
                i += 1
            continue
        i += 1
    if verbose: print(f"[LOG] PE points: {len(out_steps)}")
    return ts, np.array(out_steps, float), np.array(out_pe, float)

def parse_thermo_pe(thermo_path: Path, verbose=False):
    try:
        df = pd.read_csv(thermo_path, delim_whitespace=True, comment="#", engine="python")
    except Exception:
        df = pd.read_csv(thermo_path, comment="#")
    cols_lower = {c.lower(): c for c in df.columns}
    step_col = cols_lower.get("step")
    pe_col = None
    for key in ("pe","poteng","e_pot","epot"):
        if key in cols_lower:
            pe_col = cols_lower[key]; break
    if step_col is None or pe_col is None:
        if verbose: print(f"[THERMO] missing Step/PE in {thermo_path}")
        return np.array([]), np.array([])
    steps = pd.to_numeric(df[step_col], errors="coerce").to_numpy()
    pes   = pd.to_numeric(df[pe_col],   errors="coerce").to_numpy()
    m = np.isfinite(steps) & np.isfinite(pes)
    return steps[m], pes[m]

# ---------------- Trajectory ----------------
def iter_frames(path: Path):
    with open_stream_auto(path) as fh:
        while True:
            line = fh.readline()
            if not line: break
            if not line.startswith("ITEM: TIMESTEP"): continue
            step_line = fh.readline()
            if not step_line: break
            try: step = int(step_line.strip())
            except Exception: continue
            # NUMBER OF ATOMS
            lbl = fh.readline()
            if not lbl or not lbl.strip().startswith("ITEM: NUMBER OF ATOMS"): continue
            try: n_atoms = int(fh.readline().strip())
            except Exception: continue
            # BOX BOUNDS (skip 3)
            bb = fh.readline()
            if not bb or not bb.strip().startswith("ITEM: BOX BOUNDS"): continue
            _ = fh.readline(); _ = fh.readline(); _ = fh.readline()
            # ATOMS header + rows
            atoms_hdr = fh.readline()
            if not atoms_hdr: break
            atoms_hdr = atoms_hdr.strip()
            if not atoms_hdr.startswith("ITEM: ATOMS"): continue
            cols = atoms_hdr.split()[2:]
            if not cols: continue
            rows = []
            ok = True
            for _ in range(n_atoms):
                atom_line = fh.readline()
                if not atom_line:
                    ok = False; break
                rows.append(atom_line.split())
            if not ok or len(rows) != n_atoms: continue
            df = pd.DataFrame(rows, columns=cols)
            # Safe numeric conversion without FutureWarning:
            for c in df.columns:
                try:
                    # Try float first (covers ints too), else leave as-is
                    df[c] = df[c].astype(float)
                except Exception:
                    # stay string/object for mixed columns
                    pass
            yield {"timestep": step, "atoms": df}

def atoms_to_xyz(df: pd.DataFrame) -> np.ndarray:
    for triplet in (("xu","yu","zu"), ("x","y","z")):
        if all(c in df.columns for c in triplet):
            arr = df.loc[:, triplet].astype(float).to_numpy()
            if "id" in df.columns:
                order = np.argsort(df["id"].to_numpy())
                arr = arr[order]
            return arr
    raise ValueError("No coordinate columns in dump (need xu/yu/zu or x/y/z).")

# ---------------- Geometry ----------------
def radius_of_gyration(xyz: np.ndarray) -> float:
    cm = xyz.mean(axis=0)
    return float(np.sqrt(((xyz - cm)**2).sum(axis=1).mean()))

def kabsch(P: np.ndarray, Q: np.ndarray):
    Pc = P - P.mean(axis=0); Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    if (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0:
        V[:, -1] = -V[:, -1]
    U = V @ Wt
    return Pc @ U, Qc

def rmsd_kabsch(P: np.ndarray, Q: np.ndarray) -> float:
    Pf, Qc = kabsch(P, Q)
    d = Pf - Qc
    return float(np.sqrt((d*d).sum()/P.shape[0]))

# ---------------- Main ----------------
def main():
    args = parse_args()
    outroot = Path(args.outdir).resolve()
    plots = outroot / "plots"; plots.mkdir(exist_ok=True, parents=True)
    metrics = outroot / "metrics"; metrics.mkdir(exist_ok=True, parents=True)

    # Gather trajectories (strict glob, no rglob)
    trajs = []
    if args.traj:
        p = Path(args.traj)
        if p.exists():
            trajs.append(p.resolve())
    else:
        trajs.extend(Path(".").glob(args.traj_glob))
        if args.allow_gz:
            trajs.extend(Path(".").glob(args.traj_glob + ".gz"))
    trajs = sorted({t.resolve() for t in trajs if t.exists()})
    if not trajs:
        print(f"No trajectories with pattern: {args.traj or args.traj_glob}")
        sys.exit(1)

    pdb_dir = Path(args.pdb_dir).resolve()
    fs_to_ns = 1e-6
    summary = []

    for traj in trajs:
        core = core_from_traj_name(traj.name)
        preferred = args.chain if args.chain else parse_chain_from_core(core)

        # PDB
        if args.pdb:
            pdb_path = Path(args.pdb).resolve()
        else:
            pdb_path = find_pdb_for_core(pdb_dir, core, args.allow_gz, verbose=args.verbose)

        # Log / Thermo
        log_path = Path(args.log).resolve() if args.log else find_log_near(traj, core, args.allow_gz, verbose=args.verbose)
        ts_default = args.default_timestep_fs if args.timestep_fs is None else args.timestep_fs
        ts_fs, steps, pe = parse_log_timestep_and_pe(log_path, ts_default, verbose=args.verbose)
        if args.timestep_fs is not None:
            ts_fs = args.timestep_fs

        if (steps.size == 0 or not np.any(np.isfinite(pe))) and not args.log:
            th = find_thermo_near(traj, core, verbose=args.verbose)
            if th and th.exists():
                th_steps, th_pe = parse_thermo_pe(th, verbose=args.verbose)
                if th_steps.size:
                    steps, pe = th_steps, th_pe

        # Reference build
        ref = None; used_chain = preferred; c3map = {}
        if pdb_path and pdb_path.exists():
            c3map = c3_counts_by_chain(pdb_path)
            if used_chain and c3map.get(used_chain, 0) > 0:
                ref = c3_coords_sorted(pdb_path, used_chain)
            elif c3map:
                used_chain = max(c3map, key=c3map.get)
                ref = c3_coords_sorted(pdb_path, used_chain)
        else:
            if args.verbose: print(f"[WARN] PDB not found for {core}; RMSD will be NaN.")

        if args.verbose:
            print(f"[DISCOVER] traj={traj}")
            print(f"[DISCOVER] core={core}")
            print(f"[DISCOVER] pdb={pdb_path if (pdb_path and pdb_path.exists()) else 'None'}")
            print(f"[DISCOVER] log={log_path if (log_path and log_path.exists()) else 'None'}")
            print(f"[REF] preferred={preferred} used={used_chain} C3'={(0 if ref is None else ref.shape[0])} counts={c3map}")

        times_ns, rgs, rmsds = [], [], []
        Nref = ref.shape[0] if ref is not None else None
        first_N = None; good_frames = 0

        for fr in iter_frames(traj):
            try:
                xyz = atoms_to_xyz(fr["atoms"])
            except Exception:
                continue
            t_ns = fr["timestep"] * ts_fs * fs_to_ns
            times_ns.append(t_ns)
            rgs.append(radius_of_gyration(xyz))
            if first_N is None: first_N = xyz.shape[0]
            if ref is not None and Nref and Nref > 0:
                mlen = min(Nref, xyz.shape[0])
                if mlen > 0:
                    rmsds.append(rmsd_kabsch(xyz[:mlen,:], ref[:mlen,:]))
                    good_frames += 1
                else:
                    rmsds.append(np.nan)
            else:
                rmsds.append(np.nan)

        # CSVs
        dfm = pd.DataFrame({"time_ns": times_ns, "Rg": rgs, "RMSD": rmsds})
        dfm.to_csv(metrics / f"metrics_{core}.csv", index=False)

        if steps.size:
            tns = steps * ts_fs * fs_to_ns
            pd.DataFrame({"time_ns": tns, "Pe": pe}).to_csv(metrics / f"pe_{core}.csv", index=False)

        # Plots (optional & downsampling to keep savefig snappy)
        if not args.no_plots:
            def maybe_downsample(x, y):
                if args.plot_downsample and len(x) > args.plot_downsample:
                    idx = np.linspace(0, len(x)-1, args.plot_downsample).astype(int)
                    return np.asarray(x)[idx], np.asarray(y)[idx]
                return x, y

            x, y = maybe_downsample(times_ns, rgs)
            plt.figure(); plt.plot(x, y, linewidth=0.8)
            plt.xlabel("Time (ns)"); plt.ylabel("Radius of gyration (Rg)")
            plt.title(f"Rg vs time — {core}")
            plt.savefig(plots / f"Rg_{core}.png", dpi=150, bbox_inches="tight"); plt.close()

            x, y = maybe_downsample(times_ns, rmsds)
            plt.figure(); plt.plot(x, y, linewidth=0.8)
            plt.xlabel("Time (ns)"); plt.ylabel("RMSD to PDB C3' (Å)")
            plt.title(f"RMSD vs time — {core} (chain {used_chain})")
            plt.savefig(plots / f"RMSD_{core}.png", dpi=150, bbox_inches="tight"); plt.close()

            if steps.size:
                x, y = maybe_downsample(tns, pe)
                plt.figure(); plt.plot(x, y, linewidth=0.8)
                plt.xlabel("Time (ns)"); plt.ylabel("Potential energy")
                plt.title(f"PE vs time — {core}")
                plt.savefig(plots / f"PE_{core}.png", dpi=150, bbox_inches="tight"); plt.close()

        # Debug file
        with (metrics / f"rmsd_debug_{core}.txt").open("w") as dbg:
            dbg.write(f"traj={traj.name}\n")
            dbg.write(f"core={core}\n")
            dbg.write(f"preferred_chain={preferred}\n")
            dbg.write(f"used_chain={used_chain}\n")
            dbg.write(f"pdb_path={str(pdb_path) if pdb_path else 'None'}\n")
            dbg.write(f"log_path={str(log_path) if log_path else 'None'}\n")
            dbg.write(f"timestep_fs={ts_fs}\n")
            dbg.write(f"traj_first_N={first_N}\n")
            dbg.write(f"ref_C3prime_N={(0 if ref is None else Nref)}\n")
            dbg.write(f"frames_with_numeric_RMSD={good_frames}\n")

        summary.append((traj.name, f"chain={used_chain}", f"C3'={(0 if ref is None else Nref)}", f"frames={len(times_ns)}", f"finite_RMSD={good_frames}"))

    print("\nSummary:")
    for row in summary:
        print("  " + " | ".join(row))
    print(f"\nOutputs: {outroot}/plots  and  {outroot}/metrics")

if __name__ == "__main__":
    main()
