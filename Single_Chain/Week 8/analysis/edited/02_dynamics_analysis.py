"""
Script 2 – Structural dynamics: RMSD (vs PDB), RMSF, Radius of Gyration, Thermodynamics.

Reads:   parsed_data.npz  (produced by 01_parse_inputs.py)
Outputs: dynamics_results.npz
         fig_dynamics.png

Usage:
    python 02_dynamics_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress

IN_FILE  = "parsed_data.npz"
OUT_NPZ  = "dynamics_results.npz"
OUT_FIG  = "fig_dynamics.png"

# ─────────────────────────────────────────────────────────────────────────────
# Utility: Kabsch rotation (align P onto Q, both zero-centred on entry)
# ─────────────────────────────────────────────────────────────────────────────
def kabsch_rotate(P, Q):
    """Return P rotated onto Q (both already mean-centred). No in-place mod."""
    H   = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d   = np.sign(np.linalg.det(Vt.T @ U.T))
    R   = Vt.T @ np.diag([1, 1, d]) @ U.T
    return P @ R.T


def align_traj_to_ref(traj, ref):
    """
    Align each frame of traj (F, N, 3) onto ref (N, 3) via Kabsch.
    Returns aligned trajectory and per-frame RMSD.
    """
    ref_c   = ref - ref.mean(axis=0)
    aligned = np.empty_like(traj)
    rmsds   = np.empty(len(traj))
    for f, frame in enumerate(traj):
        P   = frame - frame.mean(axis=0)
        Pa  = kabsch_rotate(P, ref_c)
        aligned[f]  = Pa
        diff        = Pa - ref_c
        rmsds[f]    = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return aligned, rmsds


def rmsf(traj_aligned):
    """Per-atom RMSF from aligned trajectory (F, N, 3) → (N,)"""
    mean = traj_aligned.mean(axis=0)
    return np.sqrt(np.mean(np.sum((traj_aligned - mean)**2, axis=2), axis=0))


def radius_of_gyration(traj):
    """Per-frame Rg from trajectory (F, N, 3) → (F,)"""
    rg = np.empty(len(traj))
    for f, frame in enumerate(traj):
        com  = frame.mean(axis=0)
        rg[f] = np.sqrt(np.mean(np.sum((frame - com)**2, axis=1)))
    return rg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    d = np.load(IN_FILE, allow_pickle=True)

    ts          = d["timesteps"].astype(float) * 1e-5   # millions of steps
    aso_traj    = d["aso_traj"]        # (F, 10, 3)
    hp_traj     = d["hp_traj"]         # (F, 32, 3)
    pdb_aso_ref = d["pdb_aso_ref"]     # (10, 3)  PDB MODEL 1
    pdb_hp_ref  = d["pdb_hp_ref"]      # (32, 3)
    free_trajs  = d["free_aso_trajs"]  # (99, F, 10, 3)
    aso_labels  = d["aso_labels"]
    hp_labels   = d["hp_labels"]
    thermo_ts   = d["thermo_ts"].astype(float) * 1e-5
    PE, KE, T_thermo, E_tot = d["thermo_PE"], d["thermo_KE"], d["thermo_T"], d["thermo_E"]

    # ── Align docked complex ────────────────────────────────────────────────
    # Use hairpin as the alignment target (fixed biological scaffold)
    aso_aligned, aso_rmsd = align_traj_to_ref(aso_traj,   pdb_aso_ref)
    hp_aligned,  hp_rmsd  = align_traj_to_ref(hp_traj,    pdb_hp_ref)
    complex_rmsd = (aso_rmsd + hp_rmsd) / 2

    # ── RMSF docked ─────────────────────────────────────────────────────────
    aso_rmsf_vals = rmsf(aso_aligned)
    hp_rmsf_vals  = rmsf(hp_aligned)

    # ── RMSF free ASOs (internal fluctuations within each ASO) ─────────────
    free_rmsf_all = np.zeros((99, 10))
    for mi in range(99):
        fa = free_trajs[mi]   # (F, 10, 3)
        fa_c = fa - fa.mean(axis=1, keepdims=True)   # center at COM each frame
        free_rmsf_all[mi] = rmsf(fa_c)

    free_rmsf_mean = free_rmsf_all.mean(axis=0)
    free_rmsf_std  = free_rmsf_all.std(axis=0)
    z_score        = (aso_rmsf_vals - free_rmsf_mean) / (free_rmsf_std + 1e-9)

    # ── Radius of Gyration ───────────────────────────────────────────────────
    complex_all = np.concatenate([aso_traj, hp_traj], axis=1)
    rog_complex  = radius_of_gyration(complex_all)
    rog_hp       = radius_of_gyration(hp_traj)

    # ── Thermodynamics statistics ────────────────────────────────────────────
    T_mean, T_std = T_thermo.mean(), T_thermo.std()
    E_drift_slope, _, _, _, _ = linregress(thermo_ts, E_tot)

    print(f"Hairpin RMSD vs PDB:    {hp_rmsd.mean():.2f} ± {hp_rmsd.std():.2f} Å")
    print(f"Docked ASO RMSD vs PDB: {aso_rmsd.mean():.2f} ± {aso_rmsd.std():.2f} Å")
    print(f"Temperature:            {T_mean:.1f} ± {T_std:.1f} K")
    print(f"Energy drift slope:     {E_drift_slope:.4f} kcal/(mol·10⁶ steps)")
    print(f"Z-scores (ASO beads):   {np.round(z_score, 2)}")

    np.savez_compressed(
        OUT_NPZ,
        ts=ts, aso_rmsd=aso_rmsd, hp_rmsd=hp_rmsd, complex_rmsd=complex_rmsd,
        aso_rmsf=aso_rmsf_vals, hp_rmsf=hp_rmsf_vals,
        free_rmsf_mean=free_rmsf_mean, free_rmsf_std=free_rmsf_std,
        z_score=z_score, rog_complex=rog_complex, rog_hp=rog_hp,
        thermo_ts=thermo_ts, PE=PE, KE=KE, T=T_thermo, E_tot=E_tot,
    )
    print(f"Saved → {OUT_NPZ}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE – 2×3 dashboard
    # ════════════════════════════════════════════════════════════════════════
    sm = lambda x, w=5: uniform_filter1d(x.astype(float), w)

    fig = plt.figure(figsize=(17, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, hspace=0.48, wspace=0.38)

    # ── A: RMSD vs PDB reference ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ts, sm(hp_rmsd),    color="#2196F3", lw=1.8, label="Hairpin")
    ax.plot(ts, sm(aso_rmsd),   color="#E91E63", lw=1.8, label="Docked ASO")
    ax.plot(ts, sm(complex_rmsd), color="#4CAF50", lw=1.4, ls="--", label="Mean complex")
    ax.set_xlabel("Time (ns)", fontsize=10)
    ax.set_ylabel("RMSD vs PDB (Å)", fontsize=10)
    ax.set_title("A  –  RMSD vs NMR Reference (PDB 1YMO)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.5); ax.set_facecolor("#fafafa")

    # ── B: Radius of Gyration ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ts, sm(rog_complex), color="#9C27B0", lw=2, label="Complex")
    ax.plot(ts, sm(rog_hp),      color="#FF9800", lw=2, label="Hairpin only")
    ax.set_xlabel("Time (ns)", fontsize=10)
    ax.set_ylabel("Radius of Gyration (Å)", fontsize=10)
    ax.set_title("B  –  Radius of Gyration", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.5); ax.set_facecolor("#fafafa")

    # ── C: Temperature ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(thermo_ts, T_thermo, color="#607D8B", lw=1.2, alpha=0.7)
    ax.axhline(T_mean, color="k", ls="--", lw=1.5, label=f"Mean = {T_mean:.1f} K")
    ax.axhspan(T_mean - 2*T_std, T_mean + 2*T_std, alpha=0.15, color="#607D8B")
    ax.set_xlabel("Time (ns)", fontsize=10)
    ax.set_ylabel("Temperature (K)", fontsize=10)
    ax.set_title(f"C  –  Temperature  (σ = {T_std:.1f} K)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.5); ax.set_facecolor("#fafafa")

    # ── D: RMSF docked vs free ──────────────────────────────────────────────
    res_idx = np.arange(1, 11)
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(res_idx, free_rmsf_mean - 2*free_rmsf_std,
                             free_rmsf_mean + 2*free_rmsf_std,
                    alpha=0.15, color="gray", label="Free ASO ±2σ")
    ax.fill_between(res_idx, free_rmsf_mean - free_rmsf_std,
                             free_rmsf_mean + free_rmsf_std,
                    alpha=0.30, color="gray", label="Free ASO ±1σ")
    ax.plot(res_idx, free_rmsf_mean, "k--", lw=1.5, label="Free ASO mean")
    ax.plot(res_idx, aso_rmsf_vals, "o-", color="#E91E63", lw=2, ms=7, label="Docked ASO")
    ax.set_xticks(res_idx)
    ax.set_xticklabels(aso_labels, fontsize=7, rotation=25, ha="right")
    ax.set_xlabel("ASO Residue", fontsize=10)
    ax.set_ylabel("RMSF (Å)", fontsize=10)
    ax.set_title("D  –  RMSF: Docked vs 99 Free ASOs", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.5); ax.set_facecolor("#fafafa")

    # ── E: Z-score significance ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    colors_z = ["#E91E63" if z > 0 else "#2196F3" for z in z_score]
    bars = ax.bar(res_idx, z_score, color=colors_z, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline( 2, color="k", ls="--", lw=1.2, alpha=0.6, label="|z| = 2")
    ax.axhline(-2, color="k", ls="--", lw=1.2, alpha=0.6)
    ax.axhline( 0, color="k", lw=0.8)
    for bar, z in zip(bars, z_score):
        offset = 0.3 if z >= 0 else -0.6
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f"{z:.1f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(res_idx)
    ax.set_xticklabels(aso_labels, fontsize=7, rotation=25, ha="right")
    ax.set_xlabel("ASO Residue", fontsize=10)
    ax.set_ylabel("Z-score (RMSF)", fontsize=10)
    ax.set_title("E  –  Thermal Fluctuation Significance\n(Docked vs 99 Free ASOs)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.5, axis="y"); ax.set_facecolor("#fafafa")
    ax.text(0.02, 0.97, "Pink = more flexible\nBlue = more constrained",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # ── F: Energy components ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(thermo_ts, PE,     color="#F44336", lw=1.3, alpha=0.8, label="PE")
    ax.plot(thermo_ts, KE,     color="#2196F3", lw=1.3, alpha=0.8, label="KE")
    ax.plot(thermo_ts, E_tot,  color="#4CAF50", lw=2,   label=f"E_total  (drift {E_drift_slope:+.3f})")
    ax.set_xlabel("Time (ns)", fontsize=10)
    ax.set_ylabel("Energy (kcal/mol)", fontsize=10)
    ax.set_title("F  –  Energy Components (NVE Conservation)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.5); ax.set_facecolor("#fafafa")

    plt.suptitle("ASO–RNA Hairpin Dynamics  |  Reference: PDB 1YMO (NMR MODEL 1)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {OUT_FIG}")
