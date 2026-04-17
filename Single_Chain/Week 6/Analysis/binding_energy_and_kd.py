#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
import numpy as np

V0_A3 = 1660.0  # Å^3 per molecule at 1 M
R_KCAL = 0.00198720425864083  # kcal/mol/K


def read_thermo_averaged(path: str):
    """
    Reads file like:
    # TimeStep v_vpe v_vke v_vT
    100000 190.245 353.543 2892.84
    ...
    Returns:
      step (M,), vpe (M,), vke (M,), vT (M,)
    """
    steps = []
    vpe = []
    vke = []
    vT = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            steps.append(int(float(parts[0])))
            vpe.append(float(parts[1]))
            vke.append(float(parts[2]))
            vT.append(float(parts[3]))
    if not steps:
        raise RuntimeError(f"No data read from thermo file: {path}")
    return (np.array(steps, dtype=int),
            np.array(vpe, dtype=float),
            np.array(vke, dtype=float),
            np.array(vT, dtype=float))


def parse_lammpstrj(path: str, poscols=("xu", "yu", "zu")):
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            ts = int(f.readline().strip())

            f.readline()  # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline().strip())

            f.readline()  # ITEM: BOX BOUNDS
            bounds = []
            for _ in range(3):
                lo, hi = f.readline().split()[:2]
                bounds.append((float(lo), float(hi)))
            box_lo = np.array([b[0] for b in bounds], float)
            box_hi = np.array([b[1] for b in bounds], float)
            box_len = box_hi - box_lo

            header = f.readline().strip().split()
            colnames = header[2:]
            cols = {name: i for i, name in enumerate(colnames)}

            for needed in ("id", "mol", *poscols):
                if needed not in cols:
                    raise ValueError(f"Missing column '{needed}'. Have {list(cols.keys())}")

            data = np.zeros((natoms, len(colnames)), float)
            for i in range(natoms):
                row = f.readline().split()
                data[i, :] = [float(x) for x in row[:len(colnames)]]

            # stable mapping by sorting id
            atom_id = data[:, cols["id"]].astype(int)
            order = np.argsort(atom_id)
            data = data[order]

            yield ts, box_len, data, cols


def pbc_delta(dx: np.ndarray, box_len: np.ndarray) -> np.ndarray:
    return dx - box_len * np.round(dx / box_len)


def com(pos: np.ndarray) -> np.ndarray:
    return pos.mean(axis=0)


def nearest_index(sorted_steps: np.ndarray, target: int) -> int:
    # sorted_steps is ascending
    i = int(np.searchsorted(sorted_steps, target))
    if i <= 0:
        return 0
    if i >= len(sorted_steps):
        return len(sorted_steps) - 1
    # choose closer of i-1 and i
    if abs(sorted_steps[i] - target) < abs(sorted_steps[i - 1] - target):
        return i
    return i - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--thermo", required=True, help="thermo averaged file (Step v_vpe v_vke v_vT)")
    ap.add_argument("--mol_aso", type=int, default=1)
    ap.add_argument("--mol_hairpin", type=int, default=2)
    ap.add_argument("--poscols", default="xu,yu,zu")
    ap.add_argument("--burnin_frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--r_cut", type=float, default=15.0, help="Bound if COM distance <= r_cut (Å)")
    ap.add_argument("--T", type=float, default=300.0, help="Temperature for ΔG° from Kd (K)")
    ap.add_argument("--match_tol", type=int, default=0,
                    help="If >0, require |traj_step-thermo_step|<=match_tol, else drop that point. "
                         "If 0, always take nearest thermo point.")

    args = ap.parse_args()
    poscols = tuple(s.strip() for s in args.poscols.split(","))

    # --- load thermo series
    th_step, th_vpe, th_vke, th_vT = read_thermo_averaged(args.thermo)
    order = np.argsort(th_step)
    th_step = th_step[order]
    th_vpe = th_vpe[order]
    th_vke = th_vke[order]
    th_vT = th_vT[order]

    # --- walk trajectory and compute r(t), V_box(t)
    traj_steps = []
    r_list = []
    V_list = []
    keep_idxA = None
    keep_idxB = None

    for fi, (ts, box_len, data, cols) in enumerate(parse_lammpstrj(args.traj, poscols=poscols)):
        if fi < args.burnin_frames:
            continue
        if args.stride > 1 and ((fi - args.burnin_frames) % args.stride != 0):
            continue

        mol = data[:, cols["mol"]].astype(int)
        if keep_idxA is None:
            keep_idxA = np.where(mol == args.mol_aso)[0]
            keep_idxB = np.where(mol == args.mol_hairpin)[0]
            if keep_idxA.size == 0 or keep_idxB.size == 0:
                raise ValueError("Could not find ASO/hairpin mol IDs in trajectory.")

        pos = np.stack([
            data[:, cols[poscols[0]]],
            data[:, cols[poscols[1]]],
            data[:, cols[poscols[2]]]
        ], axis=1)

        comA = com(pos[keep_idxA])
        comB = com(pos[keep_idxB])
        dvec = pbc_delta(comB - comA, box_len)
        r = float(np.linalg.norm(dvec))

        traj_steps.append(ts)
        r_list.append(r)
        V_list.append(float(np.prod(box_len)))

    traj_steps = np.array(traj_steps, dtype=int)
    r_arr = np.array(r_list, dtype=float)
    V_box = float(np.mean(V_list))

    if traj_steps.size == 0:
        raise RuntimeError("No trajectory frames processed.")

    # --- map each traj step -> thermo vpe (nearest or within tol)
    vpe_at = np.empty_like(r_arr)
    used = np.ones_like(r_arr, dtype=bool)

    for i, ts in enumerate(traj_steps):
        j = nearest_index(th_step, ts)
        if args.match_tol > 0 and abs(int(th_step[j]) - int(ts)) > args.match_tol:
            used[i] = False
            continue
        vpe_at[i] = th_vpe[j]

    if not used.any():
        raise RuntimeError("No matched points between traj and thermo. Increase --match_tol or check timesteps.")

    r_use = r_arr[used]
    vpe_use = vpe_at[used]

    # --- bound/unbound classification
    bound = (r_use <= args.r_cut)
    P_bound = float(bound.mean())
    P_unbound = max(1.0 - P_bound, 1e-15)

    # Two-state Ka/Kd (same formula we used before)
    Ka = (P_bound / (P_unbound ** 2)) * (V_box / V0_A3)
    Kd = 1.0 / Ka
    dG = -R_KCAL * args.T * math.log(max(Ka, 1e-300))

    # Energy conditional averages
    if bound.any():
        vpe_b = float(vpe_use[bound].mean())
    else:
        vpe_b = float("nan")
    if (~bound).any():
        vpe_u = float(vpe_use[~bound].mean())
    else:
        vpe_u = float("nan")

    dE = vpe_b - vpe_u

    print("=== Binding thermodynamics from traj + thermo ===")
    print(f"Frames used (after burnin/stride/match): {len(r_use)}")
    print(f"Mean box volume V_box: {V_box:.6e} Å^3")
    print(f"r_cut: {args.r_cut:.2f} Å")
    print(f"P_bound: {P_bound:.6f}")
    print(f"Ka: {Ka:.6e} M^-1")
    print(f"Kd: {Kd:.6e} M   ({Kd*1e9:.3f} nM)")
    print(f"ΔG° (from Kd): {dG:.3f} kcal/mol  (T={args.T:.1f} K)")

    print("\n=== Energy conditional on bound/unbound (thermo v_vpe) ===")
    print(f"<v_vpe>_bound:   {vpe_b:.6f}")
    print(f"<v_vpe>_unbound: {vpe_u:.6f}")
    print(f"ΔE_bind ≈ <v_vpe>_bound - <v_vpe>_unbound: {dE:.6f}")
    print("\nNotes:")
    print("- v_vpe is total potential energy (unless you computed group/group separately).")
    print("- ΔE_bind is an enthalpy-like estimate; compare to ΔG° which includes entropy.")


if __name__ == "__main__":
    main()
