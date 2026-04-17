#!/usr/bin/env python3
"""
analyze_all_ns.py
-----------------
Compute Rg vs time, RMSD vs time (to **C3'** atoms in a reference PDB chain),
and Potential Energy vs time (from LAMMPS logs).

CSV time axis is always in **nanoseconds**.
Plot axis unit is configurable: **ns** or **µs**.

Run from your outputs folder:
  cd "/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 2/outputs"
  python analyze_all_ns.py --pdb-dir "."

Outputs in the chosen --outdir (default: current dir):
  plots/Rg_<config>.png
  plots/RMSD_<config>.png        (RMSD to PDB C3′, with Kabsch alignment)
  plots/PE_<config>.png          (if log found)
  metrics/metrics_<config>.csv   (time_ns, Rg, RMSD)  [full timeline]
  metrics/pe_<config>.csv        (time_ns, Pe)        [full timeline]
  metrics/rmsd_debug_<config>.txt (small diagnostics)

Notes:
- Finds PDBs in --pdb-dir, current ".", and parent "..".
- Accepts C3' and legacy C3* atom names.
- If your requested chain has no C3′, auto-picks the chain with the most C3′ atoms.
- Parses PE from thermo headers: pe, Pe, PotEng, E_pot, Epot (case-insensitive).
- Prefers xu/yu/zu; falls back to x/y/z.
"""

import re, gzip, sys, math, glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Utilities ----------

def open_text_auto(path: Path) -> List[str]:
    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            return fh.readlines()
    return path.read_text(errors="ignore").splitlines()

def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Rg/RMSD(to C3')/PE vs time; CSV in ns, plot unit configurable.")
    ap.add_argument("--pdb", help="Explicit PDB path (reference for ALL trajectories).")
    ap.add_argument("--pdb-dir", default=".", help="Directory to look for PDBs (default: current '.').")
    ap.add_argument("--chain", help="Preferred chain ID (e.g., C). If missing or empty, script will auto-pick the best chain.")
    ap.add_argument("--traj", help="One trajectory (.lammpstrj).")
    ap.add_argument("--traj-glob", default="config_*.dat_*_nvt.lammpstrj", help="Glob for many trajectories (default matches your files).")
    ap.add_argument("--log", help="Explicit log path (.lammps). If omitted, script searches near traj.")
    ap.add_argument("--timestep-fs", type=float, default=None, help="Override timestep fs (else read from log; fallback 10).")
    ap.add_argument("--default-timestep-fs", type=float, default=10.0, help="Fallback timestep fs if not in log (default 10).")
    ap.add_argument("--outdir", default=".", help="Output root directory (default current '.').")
    ap.add_argument("--time-unit", choices=["ns","us"], default="ns", help="Plot axis unit (ns or µs). CSVs remain in ns.")
    ap.add_argument("--xmax", type=float, help="Clamp x-axis to this value in the chosen --time-unit (e.g., 500 for 500 ns, 1.0 for 1 µs).")
    ap.add_argument("--clip", type=float, help="Optionally discard data AFTER this value for plotting only, in the chosen --time-unit.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def core_from_traj_name(name: str) -> str:
    # "config_1_2DER.pdb1_C.dat_298_nvt.lammpstrj" -> "config_1_2DER.pdb1_C.dat"
    m = re.match(r"(config_.+?\.dat)(?:_.*)?$", name)
    return m.group(1) if m else name

def parse_chain_from_core(core: str) -> Optional[str]:
    m = re.match(r"config_.+_(\w)\.dat$", core)
    return m.group(1) if m else None

def pdb_stem_from_core(core: str) -> Optional[str]:
    # "config_<stem>_<CHAIN>.dat" -> "<stem>"
    m = re.match(r"config_(.+)_(\w)\.dat$", core)
    return m.group(1) if m else None

def find_pdb_for_core(search_dirs: List[Path], core: str, verbose=False) -> Optional[Path]:
    stem = pdb_stem_from_core(core)
    if not stem:
        return None
    exts = ["", ".pdb", ".pdb1", ".pdb.gz", ".pdb1.gz", ".gz"]
    for d in search_dirs:
        for ext in exts:
            cand = d / f"{stem}{ext}"
            if cand.exists():
                if verbose: print(f"[PDB] matched: {cand}")
                return cand
    # Fallback: prefix match (e.g., "1_2DER")
    first = stem.split("_")[0]
    for d in search_dirs:
        for cand in d.glob(f"{first}_*.pdb*"):
            if verbose: print(f"[PDB] fallback matched: {cand}")
            return cand
    if verbose: print(f"[PDB] no match for stem={stem} in {search_dirs}")
    return None

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
        try: x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        except ValueError: continue
        try: rnum = int(resseq)
        except Exception: rnum = float('inf')
        rows.append((rnum, icode, x, y, z))
    rows.sort(key=lambda t: (t[0], t[1]))
    return np.array([[r[2], r[3], r[4]] for r in rows], float)

def parse_log_timestep_and_pe(log_path: Optional[Path], default_fs: float, verbose=False):
    ts = default_fs; steps = np.array([]); pe = np.array([])
    if not log_path or not log_path.exists():
        if verbose: print(f"[LOG] missing: {log_path}")
        return ts, steps, pe
    # timestep
    for ln in open_text_auto(log_path):
        m = re.search(r"^\s*timestep\s+([0-9.+-Ee]+)", ln, flags=re.I)
        if m:
            try:
                ts = float(m.group(1))
                if verbose: print(f"[LOG] timestep = {ts} fs")
                break
            except ValueError:
                pass
    # thermo blocks
    lines = open_text_auto(log_path)
    out_steps, out_pe = [], []
    i=0
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
            if idx_step is None: continue
            while i < len(lines):
                parts = lines[i].split()
                if len(parts) < len(header): break
                try: s = float(parts[idx_step])
                except ValueError: break
                pe_val = float(parts[idx_pe]) if idx_pe is not None else float("nan")
                out_steps.append(int(s)); out_pe.append(pe_val)
                i += 1
            continue
        i += 1
    if verbose: print(f"[LOG] PE points: {len(out_steps)}")
    return ts, np.array(out_steps, float), np.array(out_pe, float)

def iter_frames(path: Path):
    with open(path, "r") as fh:
        while True:
            line = fh.readline()
            if not line: break
            if not line.startswith("ITEM: TIMESTEP"): continue
            try:
                step = int(fh.readline().strip())
                # NUMBER OF ATOMS
                if not fh.readline().strip().startswith("ITEM: NUMBER OF ATOMS"):
                    continue
                n_atoms = int(fh.readline().strip())
                # BOX BOUNDS
                if not fh.readline().strip().startswith("ITEM: BOX BOUNDS"):
                    continue
                _ = fh.readline(); _ = fh.readline(); _ = fh.readline()
                # ATOMS ...
                atoms_hdr = fh.readline().strip()
                if not atoms_hdr.startswith("ITEM: ATOMS"):
                    continue
                cols = atoms_hdr.split()[2:]
                if not cols: continue
                rows = []
                for _ in range(n_atoms):
                    atom_line = fh.readline()
                    if not atom_line:
                        rows = []
                        break
                    rows.append(atom_line.split())
                if len(rows) != n_atoms:
                    continue
                df = pd.DataFrame(rows, columns=cols)
                for c in df.columns:
                    try: df[c] = pd.to_numeric(df[c])
                    except Exception: pass
                yield {"timestep": step, "atoms": df}
            except Exception:
                continue

def atoms_to_xyz(df: pd.DataFrame) -> np.ndarray:
    for triplet in (("xu","yu","zu"), ("x","y","z")):
        if all(c in df.columns for c in triplet):
            arr = df.loc[:, triplet].astype(float).to_numpy()
            if "id" in df.columns:
                order = np.argsort(df["id"].to_numpy())
                arr = arr[order]
            return arr
    raise ValueError("No coordinate columns in dump (need xu/yu/zu or x/y/z).")

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

# ---------- Main ----------

def main():
    args = parse_args()
    outroot = Path(args.outdir).resolve()
    plots = outroot / "plots"; plots.mkdir(exist_ok=True, parents=True)
    metrics = outroot / "metrics"; metrics.mkdir(exist_ok=True, parents=True)

    # Plot unit helpers
    unit_label = "ns" if args.time_unit == "ns" else "µs"
    def to_plot_units(arr_ns):
        arr_ns = np.asarray(arr_ns, float)
        return arr_ns if args.time_unit == "ns" else arr_ns / 1000.0
    clip_ns = None if args.clip is None else (args.clip if args.time_unit == "ns" else args.clip * 1000.0)

    # Gather trajectories
    trajs = []
    if args.traj: trajs.append(Path(args.traj))
    if args.traj_glob: trajs.extend(Path(p) for p in glob.glob(args.traj_glob))
    trajs = sorted({t.resolve() for t in trajs if t.exists()})
    if not trajs:
        print("No trajectories. Use --traj or --traj-glob."); sys.exit(1)

    # Helper: find log near traj
    def find_log(core: str, tdir: Path) -> Optional[Path]:
        candidates = [
            tdir / f"log.{core}.lammps",
            tdir.parent / f"log.{core}.lammps",
            tdir / "outputs" / f"log.{core}.lammps",
            tdir.parent / "outputs" / f"log.{core}.lammps",
            # generic fallbacks
            tdir / "log.lammps",
            tdir / "outputs" / "log.lammps",
            tdir.parent / "log.lammps",
            tdir.parent / "outputs" / "log.lammps",
        ]
        for c in candidates:
            if c.exists(): return c
        return None

    # PDB search dirs: user-provided, plus current and parent as fallbacks
    pdb_search_dirs = []
    if args.pdb_dir: pdb_search_dirs.append(Path(args.pdb_dir))
    pdb_search_dirs.extend([Path("."), Path("..")])
    pdb_search_dirs = [d.resolve() for d in pdb_search_dirs]

    fs_to_ns = 1e-6
    summary = []

    for traj in trajs:
        core = core_from_traj_name(traj.name)
        preferred = args.chain if args.chain else parse_chain_from_core(core)
        # PDB
        if args.pdb:
            pdb_path = Path(args.pdb).resolve()
        else:
            pdb_path = find_pdb_for_core(pdb_search_dirs, core, verbose=args.verbose)
        # Log
        log_path = Path(args.log).resolve() if args.log else find_log(core, traj.parent)
        # Timestep
        ts_default = args.default_timestep_fs if args.timestep_fs is None else args.timestep_fs
        ts_fs, steps, pe = parse_log_timestep_and_pe(log_path, ts_default, verbose=args.verbose)
        if args.timestep_fs is not None:
            ts_fs = args.timestep_fs

        # Build reference (auto-pick chain if needed)
        ref = None; used_chain = preferred; c3map = {}
        if pdb_path and pdb_path.exists():
            c3map = c3_counts_by_chain(pdb_path)
            if used_chain and c3map.get(used_chain, 0) > 0:
                ref = c3_coords_sorted(pdb_path, used_chain)
            else:
                if c3map:
                    used_chain = max(c3map, key=c3map.get)
                    ref = c3_coords_sorted(pdb_path, used_chain)
        else:
            if args.verbose: print(f"[WARN] PDB not found for {core}; RMSD will be NaN.")

        if args.verbose:
            print(f"[REF] core={core} preferred={preferred} used={used_chain} C3'={(0 if ref is None else ref.shape[0])} counts={c3map}")

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
            if ref is not None:
                mlen = min(Nref, xyz.shape[0])
                if mlen > 0:
                    rmsds.append(rmsd_kabsch(xyz[:mlen,:], ref[:mlen,:]))
                    good_frames += 1
                else:
                    rmsds.append(np.nan)
            else:
                rmsds.append(np.nan)

        # --- Prepare plot arrays (optional clipping) ---
        times_ns_plot = np.asarray(times_ns)
        rgs_plot = np.asarray(rgs)
        rmsds_plot = np.asarray(rmsds)
        if clip_ns is not None:
            mask = times_ns_plot <= clip_ns
            times_ns_plot = times_ns_plot[mask]
            rgs_plot = rgs_plot[mask]
            rmsds_plot = rmsds_plot[mask]

        # Debug file
        dbg = (metrics / f"rmsd_debug_{core}.txt").open("w")
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
        dbg.write(f"C3_counts_by_chain={c3map}\n")
        dbg.close()

        # CSVs (always full timeline in ns)
        dfm = pd.DataFrame({"time_ns": times_ns, "Rg": rgs, "RMSD": rmsds})
        dfm.to_csv(metrics / f"metrics_{core}.csv", index=False)

        # PE CSV (full) and plot arrays (maybe clipped)
        have_pe = False
        if steps.size:
            tns_full = steps * ts_fs * fs_to_ns
            pe_full = pe
            pd.DataFrame({"time_ns": tns_full, "Pe": pe_full}).to_csv(metrics / f"pe_{core}.csv", index=False)
            tns_plot = tns_full
            pe_plot = pe_full
            if clip_ns is not None:
                mask_pe = tns_plot <= clip_ns
                tns_plot = tns_plot[mask_pe]
                pe_plot = pe_plot[mask_pe]
            have_pe = (tns_plot.size > 0)

        # Plots (use plot arrays; axis unit per --time-unit; optional xmax)
        plt.figure()
        plt.plot(to_plot_units(times_ns_plot), rgs_plot)
        plt.xlabel(f"Time ({unit_label})")
        plt.ylabel("Radius of gyration (Rg)")
        plt.title(f"Rg vs time — {core}")
        if args.xmax is not None: plt.xlim(0, args.xmax)
        plt.savefig(plots / f"Rg_{core}.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(to_plot_units(times_ns_plot), rmsds_plot)
        plt.xlabel(f"Time ({unit_label})")
        plt.ylabel("RMSD to PDB C3' (Å)")
        plt.title(f"RMSD vs time — {core} (chain {used_chain})")
        if args.xmax is not None: plt.xlim(0, args.xmax)
        plt.savefig(plots / f"RMSD_{core}.png", dpi=150, bbox_inches="tight")
        plt.close()

        if have_pe:
            plt.figure()
            plt.plot(to_plot_units(tns_plot), pe_plot)
            plt.xlabel(f"Time ({unit_label})")
            plt.ylabel("Potential energy (kcal/mol)")
            plt.title(f"PE vs time — {core}")
            if args.xmax is not None: plt.xlim(0, args.xmax)
            plt.savefig(plots / f"PE_{core}.png", dpi=150, bbox_inches="tight")
            plt.close()

        summary.append((traj.name, f"chain={used_chain}", f"C3'={(0 if ref is None else Nref)}", f"frames={len(times_ns)}", f"finite_RMSD={good_frames}"))

    # Print a compact summary
    print("\nSummary:")
    for row in summary:
        print("  " + " | ".join(row))

if __name__ == "__main__":
    main()
