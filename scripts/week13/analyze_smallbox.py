#!/usr/bin/env python3
"""
Analyse the 3 small-box 7LYJ runs (n=5, 20, 50 in ±100 Å box)
and produce the corrected concentration-scan plot.

Overlays:
  * large-box series (box scaled as 80·N^(1/3), diffusion-limited)
  * small-box series (fixed ±100 Å, concentration scales with N)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 13")
OUT  = ROOT / "analysis"
TRAJ_DIR = ROOT / "runs/7LYJ/outputs"

CUTOFF = 15.0
FRAME_STRIDE = 5


def iter_frames(traj_path, stride=1):
    with open(traj_path) as f:
        frame_idx = 0
        while True:
            line = f.readline()
            if not line:
                return
            step = int(f.readline())
            f.readline()
            n_atoms = int(f.readline())
            f.readline()
            bx = f.readline().split()
            f.readline(); f.readline()
            box_lx = float(bx[1]) - float(bx[0])
            f.readline()
            atoms = np.empty((n_atoms, 6), dtype=float)
            for i in range(n_atoms):
                parts = f.readline().split()
                atoms[i, 0] = int(parts[0])
                atoms[i, 1] = int(parts[1])
                atoms[i, 2] = int(parts[2])
                atoms[i, 3] = float(parts[4])
                atoms[i, 4] = float(parts[5])
                atoms[i, 5] = float(parts[6])
            if frame_idx % stride == 0:
                yield step, box_lx, atoms
            frame_idx += 1


def analyze(traj_path, n_aso, n_rna):
    steps, bound = [], []
    for step, box_lx, atoms in iter_frames(traj_path, FRAME_STRIDE):
        order = np.argsort(atoms[:, 0])
        atoms = atoms[order]
        mol_ids = atoms[:, 1].astype(int)
        coords  = atoms[:, 3:6]
        rna_xyz = coords[mol_ids == 2]
        if len(rna_xyz) != n_rna:
            continue
        aso_mols = np.unique(mol_ids[mol_ids != 2])
        n_bound = 0
        for m in aso_mols:
            aso_xyz = coords[mol_ids == m]
            dxyz = aso_xyz[:, None, :] - rna_xyz[None, :, :]
            dxyz -= box_lx * np.round(dxyz / box_lx)
            dist = np.linalg.norm(dxyz, axis=2)
            if (dist < CUTOFF).any():
                n_bound += 1
        bound.append(n_bound / max(1, len(aso_mols)))
        steps.append(step)
    return np.array(steps), np.array(bound)


def main():
    # --- analyse small-box runs ---
    smallbox_rows = []
    traces = {}
    for n in [5, 20, 50]:
        tag  = f"7LYJ_aso_n{n}_smallbox"
        path = TRAJ_DIR / f"{tag}.lammpstrj"
        print(f"parsing {tag} ...", flush=True)
        steps, bf = analyze(path, n_aso=n, n_rna=66)
        # discard first 20% as transient
        stable = bf[len(bf)//5:]
        smallbox_rows.append({
            "n_aso":             n,
            "box_half":          100,
            "mean_bound_frac":   float(stable.mean()),
            "std_bound_frac":    float(stable.std()),
            "n_bound_mean":      float(stable.mean()) * n,
            "bound_frac_final":  float(bf[-1]),
            "n_frames":          len(bf),
        })
        traces[n] = (steps, bf)
        print(f"  n={n:>2}  bound_frac = {stable.mean():.3f} ± {stable.std():.3f}  "
              f"→ {stable.mean()*n:.1f} / {n} bound")

    sb_df = pd.DataFrame(smallbox_rows)
    sb_df.to_csv(OUT / "smallbox_binding_summary.csv", index=False)
    print(f"\n→ {OUT/'smallbox_binding_summary.csv'}")

    # --- load large-box results from previous analysis ---
    lb_df = pd.read_csv(OUT / "binding_summary.csv")
    lb_7lyj = lb_df[(lb_df.target == "7LYJ") & (lb_df.n_aso.isin([5, 10, 20, 50, 100, 200, 300, 400, 500]))] \
                .sort_values("n_aso")

    # --- main concentration scan plot (overlay) ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axs[0]
    ax.plot(lb_7lyj.n_aso, lb_7lyj.bound_frac_mean, "o-",
            color="tab:gray", label="large box (±80·N^(1/3) Å)  — diffusion-limited",
            mfc="white")
    ax.errorbar(sb_df.n_aso, sb_df.mean_bound_frac, yerr=sb_df.std_bound_frac,
                fmt="o-", color="tab:red", capsize=4, lw=2,
                label="small box (±100 Å)  — equilibrated")
    ax.set_xscale("log")
    ax.set_xlabel("n_aso  (ASO copies in box)")
    ax.set_ylabel("mean bound fraction")
    ax.set_title("7LYJ bound fraction vs [ASO]  (cutoff 15 Å)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    ax = axs[1]
    ax.plot(lb_7lyj.n_aso, lb_7lyj.n_bound_mean, "s-",
            color="tab:gray", label="large box", mfc="white")
    ax.plot(sb_df.n_aso, sb_df.n_bound_mean, "s-",
            color="tab:red", lw=2, label="small box")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_aso")
    ax.set_ylabel("mean # ASOs bound")
    ax.set_title("Absolute bound count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("7LYJ concentration scan — small-box re-run corrects the "
                 "diffusion-limited artefact", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "7lyj_concentration_scan_corrected.png", dpi=130)
    plt.close(fig)
    print(f"→ {OUT/'7lyj_concentration_scan_corrected.png'}")

    # --- time traces in a separate figure ---
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {5: "tab:blue", 20: "tab:orange", 50: "tab:red"}
    for n, (steps, bf) in traces.items():
        ax.plot(steps / 1e6, bf, lw=0.9, color=colors[n], label=f"n_aso = {n}")
    ax.set_xlabel("step (×10⁶)")
    ax.set_ylabel("bound fraction")
    ax.set_title("Small-box (±100 Å) bound-fraction traces  (cutoff 15 Å)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "7lyj_smallbox_traces.png", dpi=130)
    plt.close(fig)
    print(f"→ {OUT/'7lyj_smallbox_traces.png'}")

    # --- print summary table ---
    print("\n=== Small-box series ===")
    print(sb_df.to_string(index=False))
    print("\n=== Large-box (for reference) ===")
    print(lb_7lyj[["n_aso", "bound_frac_mean", "n_bound_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
