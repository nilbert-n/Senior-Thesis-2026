#!/usr/bin/env python3
"""
Week 13 ASO-binding analysis.

Per run:
  - thermo stability (PE, KE, T, Rg vs step)
  - bound-fraction vs time  (ASO mol with any bead within CUTOFF of any RNA bead)
  - per-RNA-residue contact frequency

Summary:
  - cross-target comparison at n=50
  - 7LYJ concentration scan
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 13")
RUNS = ROOT / "runs"
OUT  = ROOT / "analysis"
(OUT / "thermo_plots").mkdir(parents=True, exist_ok=True)
(OUT / "binding_plots").mkdir(parents=True, exist_ok=True)

CUTOFF = 15.0          # Å — "bound" threshold
FRAME_STRIDE = 5        # analyse every 5th dump frame
ASO_BEADS_PER_MOL = 10  # 10-mer ASO

# -------- registry of all runs ----------------------------------------
RUN_REGISTRY = []
for n in [5, 10, 20, 50, 100, 200, 300, 400, 500]:
    RUN_REGISTRY.append({
        "target": "7LYJ",
        "n_aso": n,
        "tag":   f"7LYJ_aso_n{n}",
        "n_rna": 66,
        "desc":  f"7LYJ n={n}",
    })
RUN_REGISTRY.append({
    "target": "7LYJ", "n_aso": 1, "tag": "7LYJ_aso_bind", "n_rna": 66,
    "desc": "7LYJ n=1",
})
for t, nrna in [("1ANR", 29), ("1E95", 36), ("1P5O", 77),
                ("1RNK", 34), ("1XJR", 47)]:
    RUN_REGISTRY.append({
        "target": t, "n_aso": 50, "tag": f"{t}_aso_n50", "n_rna": nrna,
        "desc": f"{t} n=50",
    })


def run_paths(r):
    base = RUNS / r["target"] / "outputs"
    return {
        "traj":   base / f"{r['tag']}.lammpstrj",
        "thermo": base / f"thermo_{r['tag']}_production.dat",
    }


# -------- thermo parsing ----------------------------------------------
def parse_thermo(path):
    """Return DataFrame with columns: step, pe, ke, T, E, Rg."""
    df = pd.read_csv(path, sep=r"\s+", comment="#",
                     names=["step", "pe", "ke", "T", "E", "Rg"])
    return df


def plot_thermo(df, r, outpath):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axs[0, 0].plot(df["step"] / 1e6, df["pe"], lw=0.7)
    axs[0, 0].set_ylabel("PE (kcal/mol)")
    axs[0, 1].plot(df["step"] / 1e6, df["T"],  lw=0.7, color="tab:red")
    axs[0, 1].set_ylabel("T (K)")
    axs[0, 1].axhline(300, color="k", ls=":", lw=0.5)
    axs[1, 0].plot(df["step"] / 1e6, df["E"],  lw=0.7, color="tab:green")
    axs[1, 0].set_ylabel("E_total")
    axs[1, 0].set_xlabel("step (×10⁶)")
    axs[1, 1].plot(df["step"] / 1e6, df["Rg"], lw=0.7, color="tab:purple")
    axs[1, 1].set_ylabel("R_g all (Å)")
    axs[1, 1].set_xlabel("step (×10⁶)")
    fig.suptitle(f"Thermo — {r['desc']}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)


# -------- trajectory parsing ------------------------------------------
def iter_frames(traj_path, stride=1):
    """Yield (step, box_lx_half, atoms_np) per sampled frame.

    atoms_np columns: id, mol, type, xu, yu, zu
    """
    with open(traj_path) as f:
        frame_idx = 0
        while True:
            line = f.readline()
            if not line:
                return
            # ITEM: TIMESTEP
            step = int(f.readline())
            f.readline()                 # NUMBER OF ATOMS
            n_atoms = int(f.readline())
            f.readline()                 # BOX BOUNDS
            bx = f.readline().split()
            by = f.readline().split()
            bz = f.readline().split()
            box_lx = float(bx[1]) - float(bx[0])
            f.readline()                 # ATOMS header
            atoms = np.empty((n_atoms, 6), dtype=float)
            for i in range(n_atoms):
                parts = f.readline().split()
                # id mol type q xu yu zu
                atoms[i, 0] = int(parts[0])
                atoms[i, 1] = int(parts[1])
                atoms[i, 2] = int(parts[2])
                atoms[i, 3] = float(parts[4])
                atoms[i, 4] = float(parts[5])
                atoms[i, 5] = float(parts[6])
            if frame_idx % stride == 0:
                yield step, box_lx, atoms
            frame_idx += 1


def analyze_trajectory(r):
    """Return dict with bound_frac_trace, mean_bound_frac, per_res_contact."""
    paths = run_paths(r)
    n_rna = r["n_rna"]
    n_aso = r["n_aso"]

    # Sort by atom id, then mol 2 = RNA, other mols = ASO
    steps = []
    bound_frac = []
    per_res_contacts = np.zeros(n_rna, dtype=np.int64)
    n_frames_used = 0

    for step, box_lx, atoms in iter_frames(paths["traj"], FRAME_STRIDE):
        # Sort by id
        order = np.argsort(atoms[:, 0])
        atoms = atoms[order]
        mol_ids = atoms[:, 1].astype(int)
        coords  = atoms[:, 3:6]

        rna_mask = mol_ids == 2
        rna_xyz  = coords[rna_mask]         # (n_rna, 3)

        # RNA residue order along chain: atoms already sorted by id,
        # RNA atoms 11..(10+n_rna) for primary ASO at 1..10, RNA next
        # The first ASO occupies ids 1..10, RNA 11..10+n_rna
        # residue index i corresponds to rna_xyz[i].

        if len(rna_xyz) != n_rna:
            # safety check
            continue

        # For each ASO molecule (mol id != 2), gather its 10 beads
        aso_mol_ids = np.unique(mol_ids[~rna_mask])
        n_bound = 0
        for m in aso_mol_ids:
            aso_xyz = coords[mol_ids == m]     # (10, 3)
            # Minimum image distances to each RNA bead
            dxyz = aso_xyz[:, None, :] - rna_xyz[None, :, :]
            # Apply periodic image (unwrapped coords in cubic box)
            dxyz -= box_lx * np.round(dxyz / box_lx)
            dist = np.linalg.norm(dxyz, axis=2)  # (10, n_rna)
            in_contact = dist < CUTOFF
            if in_contact.any():
                n_bound += 1
                # mark residues that are in contact with ANY bead of this ASO
                per_res_contacts += in_contact.any(axis=0).astype(int)

        bound_frac.append(n_bound / max(1, len(aso_mol_ids)))
        steps.append(step)
        n_frames_used += 1

    bound_frac = np.array(bound_frac)
    steps      = np.array(steps)
    return {
        "steps": steps,
        "bound_frac": bound_frac,
        "mean_bound_frac": float(bound_frac.mean()) if len(bound_frac) else float("nan"),
        "per_res_contact_freq": per_res_contacts / max(1, n_frames_used) / max(1, n_aso),
        "n_frames": n_frames_used,
    }


# -------- main pipeline ------------------------------------------------
def main():
    thermo_rows = []
    binding_rows = []
    all_bind_traces = {}
    all_res_profiles = {}

    print("=== Thermo pass ===")
    for r in RUN_REGISTRY:
        paths = run_paths(r)
        if not paths["thermo"].exists():
            print(f"  skip {r['desc']}: no thermo")
            continue
        df = parse_thermo(paths["thermo"])
        # Use last 80% as "equilibrium"
        tail = df.iloc[len(df) // 5:]
        thermo_rows.append({
            "target":  r["target"],
            "n_aso":   r["n_aso"],
            "tag":     r["tag"],
            "n_steps": int(df.step.iloc[-1]),
            "T_mean":  float(tail["T"].mean()),
            "T_std":   float(tail["T"].std()),
            "PE_mean": float(tail["pe"].mean()),
            "Rg_mean": float(tail["Rg"].mean()),
            "E_drift": float((df["E"].iloc[-1] - df["E"].iloc[0])
                             / max(1, df.step.iloc[-1]) * 1e6),
        })
        plot_thermo(df, r, OUT / "thermo_plots" / f"{r['tag']}.png")
        print(f"  {r['desc']:<20}  T={tail['T'].mean():.1f}  Rg={tail['Rg'].mean():.1f}  frames={len(df)}")

    thermo_df = pd.DataFrame(thermo_rows)
    thermo_df.to_csv(OUT / "thermo_summary.csv", index=False)
    print(f"  → {OUT/'thermo_summary.csv'}")

    print("\n=== Trajectory binding pass ===")
    for r in RUN_REGISTRY:
        paths = run_paths(r)
        if not paths["traj"].exists():
            print(f"  skip {r['desc']}: no traj")
            continue
        print(f"  {r['desc']:<20} parsing...", flush=True)
        res = analyze_trajectory(r)
        binding_rows.append({
            "target":     r["target"],
            "n_aso":      r["n_aso"],
            "tag":        r["tag"],
            "n_frames":   res["n_frames"],
            "bound_frac_mean":  res["mean_bound_frac"],
            "bound_frac_last":  float(res["bound_frac"][-1]) if len(res["bound_frac"]) else float("nan"),
            "n_bound_mean":     res["mean_bound_frac"] * r["n_aso"],
        })
        all_bind_traces[r["tag"]] = (res["steps"], res["bound_frac"])
        all_res_profiles[r["tag"]] = res["per_res_contact_freq"]
        print(f"    frames={res['n_frames']}  bound_frac_mean={res['mean_bound_frac']:.3f}  "
              f"n_bound_mean={res['mean_bound_frac']*r['n_aso']:.1f}/{r['n_aso']}")

        # Per-run plot: bound fraction over time
        if len(res["bound_frac"]):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(res["steps"] / 1e6, res["bound_frac"], lw=0.8)
            ax.set_xlabel("step (×10⁶)")
            ax.set_ylabel("bound fraction")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"Bound-fraction trace — {r['desc']}  (cutoff {CUTOFF} Å)")
            fig.tight_layout()
            fig.savefig(OUT / "binding_plots" / f"{r['tag']}_trace.png", dpi=110)
            plt.close(fig)

    binding_df = pd.DataFrame(binding_rows)
    binding_df.to_csv(OUT / "binding_summary.csv", index=False)
    print(f"\n  → {OUT/'binding_summary.csv'}")

    # ---------- 7LYJ concentration scan --------------------------------
    scan = binding_df[binding_df.target == "7LYJ"].sort_values("n_aso")
    if len(scan):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(scan.n_aso, scan.bound_frac_mean, "o-", color="tab:blue")
        ax1.set_xscale("log")
        ax1.set_xlabel("n_aso (copies in box, log scale)")
        ax1.set_ylabel("mean bound fraction")
        ax1.set_title("7LYJ concentration scan")
        ax1.grid(True, which="both", alpha=0.3)

        ax2.plot(scan.n_aso, scan.n_bound_mean, "s-", color="tab:orange")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("n_aso")
        ax2.set_ylabel("mean # bound ASOs")
        ax2.set_title("Absolute binding count vs n_aso")
        ax2.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "7lyj_concentration_scan.png", dpi=120)
        plt.close(fig)
        print(f"  → 7lyj_concentration_scan.png")

    # ---------- cross-target comparison at n=50 ------------------------
    at50 = binding_df[binding_df.n_aso == 50].sort_values("target")
    if len(at50):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(at50.target, at50.bound_frac_mean, color="teal")
        ax.set_ylabel("mean bound fraction")
        ax.set_title(f"Cross-target binding comparison at n_aso = 50  (cutoff {CUTOFF} Å)")
        for i, (x, v, nb) in enumerate(zip(at50.target, at50.bound_frac_mean,
                                           at50.n_bound_mean)):
            ax.text(i, v + 0.01, f"{v:.2f}\n({nb:.0f}/50)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(at50.bound_frac_mean) * 1.3 + 0.05)
        fig.tight_layout()
        fig.savefig(OUT / "cross_target_n50.png", dpi=120)
        plt.close(fig)
        print(f"  → cross_target_n50.png")

    # ---------- per-target binding-site profiles ----------------------
    # one panel per target (use n=50 run)
    profile_targets = [("1ANR", "1ANR_aso_n50"), ("1E95", "1E95_aso_n50"),
                       ("1P5O", "1P5O_aso_n50"), ("1RNK", "1RNK_aso_n50"),
                       ("1XJR", "1XJR_aso_n50"), ("7LYJ", "7LYJ_aso_n50")]
    fig, axs = plt.subplots(3, 2, figsize=(12, 9))
    for ax, (t, tag) in zip(axs.flat, profile_targets):
        if tag in all_res_profiles:
            prof = all_res_profiles[tag]
            ax.bar(np.arange(1, len(prof) + 1), prof, color="tab:blue")
            ax.set_title(f"{t}  (mean contact-freq per residue, n_aso=50)")
            ax.set_xlabel("residue")
            ax.set_ylabel("freq")
    fig.tight_layout()
    fig.savefig(OUT / "binding_site_profiles_n50.png", dpi=120)
    plt.close(fig)
    print(f"  → binding_site_profiles_n50.png")

    # save residue profiles to CSV
    rows = []
    for tag, prof in all_res_profiles.items():
        for i, v in enumerate(prof, 1):
            rows.append({"tag": tag, "residue": i, "contact_freq": float(v)})
    pd.DataFrame(rows).to_csv(OUT / "residue_contact_freq.csv", index=False)

    print("\n=== Done ===")
    print(f"Artifacts in {OUT}")


if __name__ == "__main__":
    main()
