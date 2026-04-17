"""
Script 3 – Close-contact (CLOSS) analysis.

Computes:
  • Per-frame minimum distance between every ASO bead and every hairpin bead
  • Time-averaged contact probability matrix (contact map)
  • Contact frequency per hairpin residue (binding footprint)
  • Triple-base-pair (TBP) contact detection: the paper's key U5·A31/A21 and G6·C20/A32 pairs
    mapped onto the coarse-grained model
  • Contact persistence: how long each contact is maintained (autocorrelation)
  • Free-energy of contact  ΔG = -kT ln(p / (1-p))

Reads:   parsed_data.npz
Outputs: contact_results.npz
         fig_contacts.png

Usage:
    python 03_contact_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter1d

IN_FILE      = "parsed_data.npz"
OUT_NPZ      = "contact_results.npz"
OUT_FIG      = "fig_contacts.png"

# Contact thresholds (Å)  – coarse-grained, so looser than atomistic
CONTACT_CUTOFF = 16.56   # 1.2x thermal fluctuation buffer
CLOSE_CUTOFF   = 13.80   # Optimal base/rna equilibrium distance

# Temperature for ΔG (must match simulation)
kT_kcal = 0.592   # kB·T at 298 K in kcal/mol

# ── Paper residue numbering convention ───────────────────────────────────────
#  ASO beads    : 1-10  (LAMMPS mol1)
#  Hairpin beads: 11-42 (paper) = LAMMPS mol2 beads 1-32  = PDB residues 15-46
#
#  Key TBP contacts from paper Fig 1B / Results:
#  U5 ↔ A21 (hairpin loop)  → ASO bead 5, HP bead 7  (PDB res 21 = HP bead 7)
#  U5 ↔ A31 (hairpin loop)  → ASO bead 5, HP bead 17 (PDB res 31 = HP bead 17)
#  G6 ↔ C20 (hairpin loop)  → ASO bead 6, HP bead 6  (PDB res 20 = HP bead 6)
#  G6 ↔ A32 (hairpin loop)  → ASO bead 6, HP bead 18 (PDB res 32 = HP bead 18)
#  U10 anchoring stem        → ASO bead 10, HP bead ~22 (A36)
#
#  Note: HP bead index = PDB_resid - 14  (since HP starts at PDB res 15)

TBP_CONTACTS = [
    ("U5·A21", 5-1, 21-15,  "#E91E63"),   # ASO idx 4, HP idx 6
    ("U5·A31", 5-1, 31-15,  "#E91E63"),   # ASO idx 4, HP idx 16
    ("G6·C20", 6-1, 20-15,  "#2196F3"),   # ASO idx 5, HP idx 5
    ("G6·A32", 6-1, 32-15,  "#2196F3"),   # ASO idx 5, HP idx 17
    ("U10·A36",10-1, 36-15, "#FF9800"),   # ASO idx 9, HP idx 21 (stem anchor)
]

# ─────────────────────────────────────────────────────────────────────────────
def contact_map_and_mindist(aso_traj, hp_traj, cutoff):
    """
    Returns:
        contact_prob  (10, 32)  – fraction of frames in contact
        min_dist      (F, 10, 32) – per-frame distances (full matrix)
        mean_mindist  (10, 32)  – time-averaged distances
    """
    F = len(aso_traj)
    n_aso, n_hp = aso_traj.shape[1], hp_traj.shape[1]
    contact_sum = np.zeros((n_aso, n_hp))
    dist_sum    = np.zeros((n_aso, n_hp))
    # store full dist per frame – only if memory allows (113 × 10 × 32 = trivial)
    all_dists = np.zeros((F, n_aso, n_hp))
    for f in range(F):
        D = cdist(aso_traj[f], hp_traj[f])
        all_dists[f]  = D
        contact_sum  += (D < cutoff).astype(float)
        dist_sum     += D
    return contact_sum / F, all_dists, dist_sum / F


def contact_autocorr(contact_ts):
    """Autocorrelation of a binary contact time series (normalised)."""
    x  = contact_ts.astype(float) - contact_ts.mean()
    if x.std() < 1e-9:
        return np.ones(len(x))
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    return ac / ac[0]


def delta_G(prob, kT=kT_kcal):
    """ΔG = -kT ln(p / (1-p));  returns NaN where prob = 0 or 1."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return -kT * np.log(p / (1 - p))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    d         = np.load(IN_FILE, allow_pickle=True)
    aso_traj  = d["aso_traj"]       # (F,10,3)
    hp_traj   = d["hp_traj"]        # (F,32,3)
    ts        = d["timesteps"].astype(float) * 1e-5
    aso_labels = d["aso_labels"]
    hp_labels  = d["hp_labels"]

    F, n_aso, n_hp = len(ts), 10, 32

    print("Computing contact maps …")
    prob10, all_dists10, mean_dist10 = contact_map_and_mindist(aso_traj, hp_traj, CONTACT_CUTOFF)
    prob8,  all_dists8,  mean_dist8  = contact_map_and_mindist(aso_traj, hp_traj, CLOSE_CUTOFF)

    # ── ΔG surface ──────────────────────────────────────────────────────────
    dG_map = delta_G(prob10)

    # ── Binding footprint: per hairpin bead, max contact prob across ASO ────
    hp_footprint    = prob10.max(axis=0)   # (32,)
    aso_footprint   = prob10.max(axis=1)   # (10,)

    # ── TBP contact time series ──────────────────────────────────────────────
    tbp_ts    = {}
    tbp_ac    = {}
    tbp_prob  = {}
    tbp_dist  = {}
    for name, ai, hi, col in TBP_CONTACTS:
        ts_bin          = (all_dists10[:, ai, hi] < CONTACT_CUTOFF).astype(float)
        tbp_ts[name]    = ts_bin
        tbp_ac[name]    = contact_autocorr(ts_bin)
        tbp_prob[name]  = ts_bin.mean()
        tbp_dist[name]  = all_dists10[:, ai, hi]
        print(f"  {name}:  prob = {tbp_prob[name]:.3f}   mean dist = {tbp_dist[name].mean():.1f} Å")

    # ── Contact persistence (autocorr decay length) ──────────────────────────
    persist = {}
    for name, ac in tbp_ac.items():
        # find lag where AC drops below 1/e
        below = np.where(ac < 1/np.e)[0]
        persist[name] = below[0] if len(below) else len(ac)

    np.savez_compressed(
        OUT_NPZ,
        prob10=prob10, prob8=prob8, mean_dist10=mean_dist10,
        dG_map=dG_map, hp_footprint=hp_footprint, aso_footprint=aso_footprint,
        tbp_probs=np.array([tbp_prob[n] for n, *_ in TBP_CONTACTS]),
        tbp_dists=np.array([tbp_dist[n] for n, *_ in TBP_CONTACTS]),
        tbp_names=np.array([n for n, *_ in TBP_CONTACTS]),
    )
    print(f"Saved → {OUT_NPZ}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE – 2×3 contact dashboard
    # ════════════════════════════════════════════════════════════════════════
    cmap_contact = LinearSegmentedColormap.from_list(
        "contact", ["#f5f5f5", "#FFF176", "#FF9800", "#E91E63", "#880E4F"])
    cmap_dg      = "RdYlGn_r"

    fig = plt.figure(figsize=(17, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.38)

    hp_tick_idx   = list(range(0, 32, 3))
    hp_tick_label = [hp_labels[i] for i in hp_tick_idx]
    aso_tick_idx  = list(range(10))
    aso_tick_label= list(aso_labels)

    # ── A: Contact probability map (10 Å) ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(prob10, aspect="auto", origin="lower",
                   cmap=cmap_contact, vmin=0, vmax=max(prob10.max(), 0.05))
    plt.colorbar(im, ax=ax, label="Contact probability", shrink=0.85, pad=0.02)
    ax.set_xlabel("Hairpin residue", fontsize=9)
    ax.set_ylabel("Docked ASO residue", fontsize=9)
    ax.set_title("A  –  Contact Map  (cutoff 10 Å)", fontsize=10, fontweight="bold")
    ax.set_yticks(aso_tick_idx); ax.set_yticklabels(aso_tick_label, fontsize=7)
    ax.set_xticks(hp_tick_idx);  ax.set_xticklabels(hp_tick_label, fontsize=7, rotation=40, ha="right")
    # Mark TBP contacts
    for name, ai, hi, col in TBP_CONTACTS:
        ax.plot(hi, ai, "s", ms=7, mfc="none", mec=col, mew=1.8)
    ax.text(0.02, 0.97, "□ = TBP contacts", transform=ax.transAxes,
            fontsize=7, va="top", bbox=dict(fc="white", alpha=0.7, boxstyle="round"))

    # ── B: ΔG map ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    dg_plot = np.clip(dG_map, -3, 3)
    im = ax.imshow(dg_plot, aspect="auto", origin="lower", cmap=cmap_dg, vmin=-3, vmax=3)
    plt.colorbar(im, ax=ax, label="ΔG (kcal/mol)", shrink=0.85, pad=0.02)
    ax.set_xlabel("Hairpin residue", fontsize=9)
    ax.set_ylabel("Docked ASO residue", fontsize=9)
    ax.set_title("B  –  Contact Free Energy ΔG = −kT ln(p/(1−p))", fontsize=10, fontweight="bold")
    ax.set_yticks(aso_tick_idx); ax.set_yticklabels(aso_tick_label, fontsize=7)
    ax.set_xticks(hp_tick_idx);  ax.set_xticklabels(hp_tick_label, fontsize=7, rotation=40, ha="right")

    # ── C: Mean distance map ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(mean_dist10, aspect="auto", origin="lower",
                   cmap="viridis_r", vmin=0, vmax=40)
    plt.colorbar(im, ax=ax, label="Mean distance (Å)", shrink=0.85, pad=0.02)
    ax.set_xlabel("Hairpin residue", fontsize=9)
    ax.set_ylabel("Docked ASO residue", fontsize=9)
    ax.set_title("C  –  Mean Pairwise Distance", fontsize=10, fontweight="bold")
    ax.set_yticks(aso_tick_idx); ax.set_yticklabels(aso_tick_label, fontsize=7)
    ax.set_xticks(hp_tick_idx);  ax.set_xticklabels(hp_tick_label, fontsize=7, rotation=40, ha="right")
    ax.contour(mean_dist10, levels=[CONTACT_CUTOFF], colors="white", linewidths=1.2)

    # ── D: Hairpin binding footprint ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    bar_colors = plt.cm.YlOrRd(hp_footprint / max(hp_footprint.max(), 0.01))
    ax.bar(range(32), hp_footprint, color=bar_colors, edgecolor="none")
    ax.set_xticks(hp_tick_idx); ax.set_xticklabels(hp_tick_label, fontsize=7, rotation=40, ha="right")
    ax.set_xlabel("Hairpin residue", fontsize=9)
    ax.set_ylabel("Max contact probability", fontsize=9)
    ax.set_title("D  –  Hairpin Binding Footprint", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, lw=0.5, axis="y"); ax.set_facecolor("#fafafa")
    # Mark loop region (paper: hairpin loop ~residues 20-32 = HP beads 6-18)
    ax.axvspan(5.5, 18.5, alpha=0.08, color="#E91E63")
    ax.text(12, hp_footprint.max()*0.95, "loop", ha="center", fontsize=8, color="#E91E63")

    # ── E: TBP contact distances over time ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    sm5 = lambda x: uniform_filter1d(x.astype(float), 5)
    for name, ai, hi, col in TBP_CONTACTS:
        ax.plot(ts, sm5(tbp_dist[name]), color=col, lw=1.5, label=f"{name} ({tbp_prob[name]:.2f})", alpha=0.85)
    ax.axhline(CONTACT_CUTOFF, color="k", ls="--", lw=1, label=f"{CONTACT_CUTOFF} Å cutoff")
    ax.set_xlabel("Time (ns)", fontsize=9)
    ax.set_ylabel("Distance (Å)", fontsize=9)
    ax.set_title("E  –  Key TBP Contact Distances vs Time\n(label: contact probability)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.3, lw=0.5); ax.set_facecolor("#fafafa")

    # ── F: TBP contact probability bar chart ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    names  = [n for n, *_ in TBP_CONTACTS]
    probs  = [tbp_prob[n] for n in names]
    cols   = [c for _, __, ___, c in TBP_CONTACTS]
    bars   = ax.bar(range(len(names)), probs, color=cols, edgecolor="white", lw=0.7)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Contact probability", fontsize=9)
    ax.set_ylim(0, min(1.0, max(probs)*1.5 + 0.05))
    ax.set_title("F  –  Triple-BP Contact Probabilities", fontsize=10, fontweight="bold")
    for bar, p, pers in zip(bars, probs, [persist[n] for n in names]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{p:.3f}\n(τ={pers})", ha="center", fontsize=8)
    ax.grid(alpha=0.3, lw=0.5, axis="y"); ax.set_facecolor("#fafafa")

    plt.suptitle("Close-Contact (CLOSS) Analysis  |  ASO–RNA Hairpin Complex",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {OUT_FIG}")
