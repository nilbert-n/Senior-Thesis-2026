
#!/usr/bin/env python3
"""
Analyze a 298 K LAMMPS run (2:40 job) and produce PE/T + RMSD/Rg plots.
Usage:
  python analyze_298_run.py --log log.lammps --traj 26_298_nvt.lammpstrj --out outputs_298_2m40
"""
import re, os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def parse_log_fourcols(path):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if re.match(r"^\s*\d+\s+[-+Ee0-9\.]+\s+[-+Ee0-9\.]+\s+[-+Ee0-9\.]+\s*$", line):
                s, pe, ke, temp = line.split()
                rows.append([int(s), float(pe), float(ke), float(temp)])
    return pd.DataFrame(rows, columns=["step","pe","ke","temp"])

def read_traj_positions(path):
    times=[]; coords=[]; types=[]
    with open(path,"r") as f:
        while True:
            line=f.readline()
            if not line: break
            if not line.startswith("ITEM: TIMESTEP"): continue
            step=int(f.readline().strip())
            assert f.readline().startswith("ITEM: NUMBER OF ATOMS")
            nat=int(f.readline().strip())
            assert f.readline().startswith("ITEM: BOX BOUNDS")
            for _ in range(3): f.readline()
            header=f.readline().split()[2:]
            id_i=header.index("id"); typ_i=header.index("type")
            x_i=header.index("xu"); y_i=header.index("yu"); z_i=header.index("zu")
            arr=np.zeros((nat,3),float); tarr=np.zeros(nat,int)
            for _ in range(nat):
                parts=f.readline().split()
                idx=int(parts[id_i])-1
                tarr[idx]=int(parts[typ_i])
                arr[idx,0]=float(parts[x_i]); arr[idx,1]=float(parts[y_i]); arr[idx,2]=float(parts[z_i])
            times.append(step); coords.append(arr); types.append(tarr)
    return np.array(times), np.array(coords), np.array(types)

def read_pdb_coords(path):
    """Read atomic coordinates from a PDB file."""
    coords = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append([x, y, z])
    return np.array(coords, dtype=float)

def kabsch(P, Q):
    C = P.T @ Q
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1,1,d])
    U = V @ D @ Wt
    return U

def compute_rmsd_rg(times, coords, exclude_ids=None, dt_fs=10.0, ref_coords=None):
    nat = coords.shape[1]
    mask = np.ones(nat, dtype=bool)
    if exclude_ids:
        for eid in exclude_ids:
            if 1 <= eid <= nat:
                mask[eid-1] = False
    
    # Apply mask to coordinates
    X = coords[:, mask, :]
    
    # Use provided reference or first frame
    if ref_coords is not None:
        # Apply same mask to reference coordinates
        if len(ref_coords) == nat:
            ref = ref_coords[mask]
        else:
            # Assume reference already has same atoms as trajectory
            ref = ref_coords
    else:
        ref = X[0]
    
    ref_c = ref - ref.mean(axis=0, keepdims=True)
    
    rmsd=[]; rg=[]
    for frame in X:
        Y = frame - frame.mean(axis=0, keepdims=True)
        U = kabsch(Y, ref_c)
        diff = (Y @ U) - ref_c
        rmsd.append(np.sqrt((diff**2).mean()))
        rg.append(np.sqrt((Y**2).mean()))
    t_ps = times * dt_fs / 1000.0
    return pd.DataFrame({"step": times, "time_ps": t_ps, "RMSD": rmsd, "Rg": rg})

def estimate_equilibration(time_ps, series, window=200, epsilon=1e-3, sustain=100):
    if len(series) < window*2:
        window = max(10, len(series)//10)
    slopes=[]
    for i in range(len(series)-window):
        xt = time_ps[i:i+window]; yt = series[i:i+window]
        x0 = xt - xt.mean()
        denom = (x0**2).sum()
        slopes.append(0.0 if denom<=1e-12 else (x0*((yt-yt.mean()))).sum()/denom)
    slopes = np.array(slopes)
    eq_idx=None
    for i in range(max(0,len(slopes)-sustain)):
        if np.all(np.abs(slopes[i:i+sustain]) < epsilon):
            eq_idx=i; break
    return float(time_ps[eq_idx]) if eq_idx is not None else float(time_ps[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="log.lammps")
    ap.add_argument("--traj", default="26_298_nvt.lammpstrj")
    ap.add_argument("--out", default="outputs_298_2m40")
    ap.add_argument("--dt_fs", type=float, default=10.0)
    ap.add_argument("--ref", default="3NJ6.pdb1", help="Reference PDB file for RMSD calculation")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # log → PE/T
    df_log = parse_log_fourcols(args.log)
    df_log.to_csv(os.path.join(args.out,"pe_temp_timeseries_298.csv"), index=False)
    if not df_log.empty:
        tps = df_log["step"]*args.dt_fs/1000.0
        plt.figure(); plt.plot(tps, df_log["pe"]); plt.xlabel("Time (ps)"); plt.ylabel("Potential Energy"); plt.title("PE vs Time (298 K)"); plt.tight_layout()
        plt.savefig(os.path.join(args.out,"plot_pe_vs_time_298.png"), dpi=300); plt.close()
        plt.figure(); plt.plot(tps, df_log["temp"]); plt.xlabel("Time (ps)"); plt.ylabel("Temperature (K)"); plt.title("Temperature vs Time (298 K)"); plt.tight_layout()
        plt.savefig(os.path.join(args.out,"plot_temp_vs_time_298.png"), dpi=300); plt.close()

    # traj → RMSD/Rg
    times, coords, types = read_traj_positions(args.traj)
    
    # Load reference structure
    ref_coords = read_pdb_coords(args.ref)
    print(f"Loaded reference structure from {args.ref} with {len(ref_coords)} atoms")
    print(f"Trajectory has {coords.shape[1]} atoms per frame")

    # full
    df_full = compute_rmsd_rg(times, coords, exclude_ids=None, dt_fs=args.dt_fs, ref_coords=ref_coords)
    df_full.to_csv(os.path.join(args.out,"rmsd_rg_timeseries_298.csv"), index=False)
    t = df_full["time_ps"].values
    plt.figure(); plt.plot(t, df_full["RMSD"]); plt.xlabel("Time (ps)"); plt.ylabel("RMSD (Å)"); plt.title("RMSD vs Time (298 K)"); plt.tight_layout()
    plt.savefig(os.path.join(args.out,"plot_rmsd_vs_time_298.png"), dpi=300); plt.close()
    plt.figure(); plt.plot(t, df_full["Rg"]); plt.xlabel("Time (ps)"); plt.ylabel("Rg (Å)"); plt.title("Radius of Gyration vs Time (298 K)"); plt.tight_layout()
    plt.savefig(os.path.join(args.out,"plot_rg_vs_time_298.png"), dpi=300); plt.close()
    eq_full = estimate_equilibration(df_full["time_ps"].values, df_full["RMSD"].values)
    open(os.path.join(args.out,"rmsd_equilibration_298.txt"),"w").write(f"{eq_full:.2f}")

    # exclude middle AAAA (ids 12..15)
    df_ex = compute_rmsd_rg(times, coords, exclude_ids=list(range(12,16)), dt_fs=args.dt_fs, ref_coords=ref_coords)
    df_ex.to_csv(os.path.join(args.out,"rmsd_rg_timeseries_298_excl4A.csv"), index=False)
    t2 = df_ex["time_ps"].values
    plt.figure(); plt.plot(t2, df_ex["RMSD"]); plt.xlabel("Time (ps)"); plt.ylabel("RMSD (Å)"); plt.title("RMSD vs Time (298 K, excl. AAAA)"); plt.tight_layout()
    plt.savefig(os.path.join(args.out,"plot_rmsd_vs_time_298_excl4A.png"), dpi=300); plt.close()
    plt.figure(); plt.plot(t2, df_ex["Rg"]); plt.xlabel("Time (ps)"); plt.ylabel("Rg (Å)"); plt.title("Rg vs Time (298 K, excl. AAAA)"); plt.tight_layout()
    plt.savefig(os.path.join(args.out,"plot_rg_vs_time_298_excl4A.png"), dpi=300); plt.close()
    eq_ex = estimate_equilibration(df_ex["time_ps"].values, df_ex["RMSD"].values)
    open(os.path.join(args.out,"rmsd_equilibration_298_excl4A.txt"),"w").write(f"{eq_ex:.2f}")

if __name__ == "__main__":
    main()
