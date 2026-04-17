#!/usr/bin/env python3
"""
OVITO helper: report timestep for MIN/MAX RMSD to a reference structure.

By default, the reference is frame 0 of the trajectory.
Optionally, you can provide a C3-only PDB as the reference.

Examples
--------
# 1) Min RMSD in first 8160 frames, reference = frame 0 (default)
python ovito_rmsd.py --traj config_1_2DER_C.dat.lammpstrj --limit-frames 8160 --stat min

# 2) Max RMSD in first 8160 frames (most different from frame 0)
python ovito_rmsd.py --traj config_1_2DER_C.dat.lammpstrj --limit-frames 8160 --stat max

# 3) Min RMSD to a C3-only PDB (NuFold prediction) over first 8160 frames
python ovito_rmsd.py --traj config_1_2DER_C.dat.lammpstrj \
    --ref-pdb seq1_rank_1_C3only.pdb \
    --limit-frames 8160 \
    --stat min

# 4) Just the timestep + RMSD (to reference) for a specific frame index
python ovito_rmsd.py --traj config_1_2DER_C.dat.lammpstrj --exact-frame 8160

# 5) Min & max RMSD within ±100 frames around frame 8160
python ovito_rmsd.py --traj config_1_2DER_C.dat.lammpstrj \
    --exact-frame 8160 --window 100 --stat both
"""

import argparse, gzip
from pathlib import Path
import numpy as np


# ---------- IO helpers ----------

def open_stream(path: Path):
    s = str(path).lower()
    if s.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def parse_xyz_from_frame(fh):
    """
    File handle must be positioned just AFTER reading the 'ITEM: TIMESTEP' line.
    Returns (xyz_array, ok_flag). Advances fh through the frame block.

    xyz_array has shape (N, 3) and is sorted by atom id.
    """
    line = fh.readline()
    if not line or not line.startswith("ITEM: NUMBER OF ATOMS"):
        return None, False
    try:
        n_atoms = int(fh.readline().strip())
    except Exception:
        return None, False

    line = fh.readline()
    if not line or not line.startswith("ITEM: BOX BOUNDS"):
        return None, False
    # skip 3 box lines
    _ = fh.readline(); _ = fh.readline(); _ = fh.readline()

    hdr = fh.readline()
    if not hdr or not hdr.startswith("ITEM: ATOMS"):
        return None, False
    cols = hdr.strip().split()[2:]  # after 'ITEM: ATOMS'

    def idx(name):
        return cols.index(name) if name in cols else None

    # prefer unwrapped xu,yu,zu if present
    xi, yi, zi = idx("xu"), idx("yu"), idx("zu")
    if xi is None or yi is None or zi is None:
        xi, yi, zi = idx("x"), idx("y"), idx("z")
        if xi is None or yi is None or zi is None:
            # consume lines but bail
            for _ in range(n_atoms):
                if not fh.readline():
                    break
            return None, False

    id_i = idx("id")

    rows = []
    for row_i in range(n_atoms):
        line = fh.readline()
        if not line:
            return None, False
        parts = line.split()
        try:
            aid = int(parts[id_i]) if id_i is not None else row_i
            x = float(parts[xi])
            y = float(parts[yi])
            z = float(parts[zi])
        except Exception:
            return None, False
        rows.append((aid, x, y, z))

    # sort by id so frames are consistently ordered
    rows.sort(key=lambda t: t[0])
    xyz = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
    return xyz, True


def load_pdb_xyz(pdb_path: Path):
    """
    Load ATOM/HETATM coordinates from a PDB file.
    Returns an array of shape (N, 3) sorted by atom serial.
    """
    rows = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    serial = int(line[6:11])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except Exception:
                    continue
                rows.append((serial, x, y, z))
    if not rows:
        raise ValueError(f"No ATOM/HETATM records found in PDB: {pdb_path}")
    rows.sort(key=lambda t: t[0])
    xyz = np.array([[x, y, z] for _, x, y, z in rows], dtype=float)
    return xyz


# ---------- RMSD / Kabsch ----------

def kabsch_rmsd(P, Q):
    """
    RMSD after optimal superposition of P onto Q (same length).

    P, Q: arrays of shape (N, 3)
    """
    # center both
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    # covariance and SVD
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    # handle improper rotation
    if np.linalg.det(V) * np.linalg.det(Wt) < 0.0:
        V[:, -1] *= -1.0
    U = V @ Wt

    # rotate P onto Q
    Pf = Pc @ U
    diff = Pf - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def scan_frames(traj_path: Path,
                limit_frames: int | None,
                exact_frame: int | None,
                window: int | None):
    """
    Stream the trajectory, returning:
      ref_xyz: coordinates of frame 0 (used only if no ref-pdb is given)
      hits: list of tuples (frame_idx, timestep, xyz)
    """
    hits = []
    ref_xyz = None
    fmin = None
    fmax = None
    if exact_frame is not None and window is not None and window > 0:
        fmin = max(0, exact_frame - window)
        fmax = exact_frame + window

    with open_stream(traj_path) as fh:
        frame_idx = -1
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            ts_line = fh.readline()
            if not ts_line:
                break
            try:
                timestep = int(ts_line.strip())
            except Exception:
                continue

            xyz, ok = parse_xyz_from_frame(fh)
            if not ok:
                continue

            frame_idx += 1
            if ref_xyz is None:
                ref_xyz = xyz.copy()

            # gating by limit / window / exact-frame
            if exact_frame is not None:
                if window is None:
                    if frame_idx == exact_frame:
                        hits.append((frame_idx, timestep, xyz))
                        break
                else:
                    if fmin is not None and frame_idx < fmin:
                        continue
                    if fmax is not None and frame_idx > fmax:
                        break
                    hits.append((frame_idx, timestep, xyz))
            else:
                if limit_frames is not None and frame_idx >= limit_frames:
                    break
                hits.append((frame_idx, timestep, xyz))

    return ref_xyz, hits


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Report timesteps for MIN/MAX RMSD to a reference (frame 0 or PDB)."
    )
    ap.add_argument("--traj", required=True,
                    help="LAMMPS dump (.lammpstrj or .lammpstrj.gz)")
    ap.add_argument("--limit-frames", type=int, default=8160,
                    help="Search only first N frames (default 8160). "
                         "Use <=0 for all frames. Ignored for --exact-frame without --window.")
    ap.add_argument("--include-frame0", action="store_true",
                    help="Include frame 0 in the search (default: exclude).")
    ap.add_argument("--exact-frame", type=int,
                    help="If set, report only this frame's timestep (+RMSD to reference).")
    ap.add_argument("--window", type=int,
                    help="With --exact-frame, search within ±window of that frame.")
    ap.add_argument("--stat", choices=["min", "max", "both"], default="both",
                    help="Which statistic over the search range to report (default: both).")
    ap.add_argument("--ref-pdb",
                    help="Use this PDB (e.g. C3-only prediction) as RMSD reference "
                         "instead of frame 0 of the trajectory.")
    args = ap.parse_args()

    traj = Path(args.traj)
    if not traj.exists():
        raise SystemExit(f"Trajectory not found: {traj}")

    limit = None if (args.limit_frames is not None and args.limit_frames <= 0) else args.limit_frames

    # Parse trajectory frames
    ref_xyz, hits = scan_frames(traj, limit, args.exact_frame, args.window)
    if ref_xyz is None or not hits:
        raise SystemExit("No frames parsed. Check dump format / window range.")

    # If a reference PDB is provided, override the default reference
    if args.ref_pdb:
        ref_xyz = load_pdb_xyz(Path(args.ref_pdb))

    # Mode: exact frame only (no window) — just report that frame's RMSD
    if args.exact_frame is not None and args.window is None:
        f_idx, ts, xyz = hits[0]
        m = min(len(ref_xyz), len(xyz))
        rmsd = kabsch_rmsd(xyz[:m], ref_xyz[:m]) if m > 0 else float("nan")
        print("== Exact frame ==")
        print(f"Frame index : {f_idx}")
        print(f"Timestep    : {ts}")
        print(f"RMSD to reference (Å): {rmsd:.6f}")
        return

    best_min = None
    best_max = None

    for f_idx, ts, xyz in hits:
        if (not args.include_frame0) and (args.ref_pdb is None) and f_idx == 0:
            # When using frame 0 as reference, it's often excluded from the search
            continue
        m = min(len(ref_xyz), len(xyz))
        if m <= 0:
            continue
        val = kabsch_rmsd(xyz[:m], ref_xyz[:m])

        if (best_min is None) or (val < best_min["rmsd"]):
            best_min = {"frame_index": f_idx, "timestep": ts, "rmsd": val}
        if (best_max is None) or (val > best_max["rmsd"]):
            best_max = {"frame_index": f_idx, "timestep": ts, "rmsd": val}

    # Report MIN
    if args.stat in ("min", "both"):
        if best_min is None:
            print("No frames for MIN RMSD (maybe only reference frame in range).")
        else:
            if args.ref_pdb:
                hdr = (f"== Minimum RMSD to PDB within ±{args.window} of frame {args.exact_frame} =="
                       if (args.exact_frame is not None and args.window is not None)
                       else f"== Minimum RMSD to PDB (searched {len(hits)} frames) ==")
            else:
                hdr = (f"== Minimum RMSD to frame 0 within ±{args.window} of frame {args.exact_frame} =="
                       if (args.exact_frame is not None and args.window is not None)
                       else f"== Minimum RMSD to frame 0 (searched {len(hits)} frames"
                            f"{'; excl. frame 0' if not args.include_frame0 else ''}) ==")
            print(hdr)
            print(f"Frame index : {best_min['frame_index']}")
            print(f"Timestep    : {best_min['timestep']}")
            print(f"RMSD (Å)    : {best_min['rmsd']:.6f}")

    # Report MAX
    if args.stat in ("max", "both"):
        if best_max is None:
            print("No frames for MAX RMSD (maybe only reference frame in range).")
        else:
            if args.ref_pdb:
                hdr = (f"== Maximum RMSD to PDB within ±{args.window} of frame {args.exact_frame} =="
                       if (args.exact_frame is not None and args.window is not None)
                       else f"== Maximum RMSD to PDB (searched {len(hits)} frames) ==")
            else:
                hdr = (f"== Maximum RMSD to frame 0 within ±{args.window} of frame {args.exact_frame} =="
                       if (args.exact_frame is not None and args.window is not None)
                       else f"== Maximum RMSD to frame 0 (searched {len(hits)} frames"
                            f"{'; excl. frame 0' if not args.include_frame0 else ''}) ==")
            print(hdr)
            print(f"Frame index : {best_max['frame_index']}")
            print(f"Timestep    : {best_max['timestep']}")
            print(f"RMSD (Å)    : {best_max['rmsd']:.6f}")


if __name__ == "__main__":
    main()

