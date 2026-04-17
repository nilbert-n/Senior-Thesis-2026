"""
Script 4 – Free-ASO binding probability analysis.

For each of the 99 free ASOs, computes:
  • Fraction of frames in contact with hairpin (binding probability)
  • First-passage time to reach hairpin
  • Distance to hairpin centre-of-mass over time
  • Comparison of free ASO RMSF distribution

Reads:   parsed_data.npz
Outputs: binding_results.npz
         fig_binding.png

Usage:
    python 04_binding_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter1d

IN_FILE = "parsed_data.npz"
OUT_NPZ = "binding_results.npz"
OUT_FIG = "fig_binding.png"

BIND_CUTOFF  = 12.0   # Å  – free ASO "contacts" hairpin
CLOSE_CUTOFF =  8.0   # Å  – "close encounter"


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    d          = np.load(IN_FILE, allow_pickle=True)
    ts         = d["timesteps"].astype(float) / 1e6
    hp_traj    = d["hp_traj"]         # (F, 32, 3)
    free_trajs = d["free_aso_trajs"]  # (99, F, 10, 3)
    aso_labels = d["aso_labels"]

    F, n_free = len(ts), 99

    # ── Hairpin COM per frame ─────────────────────────────────────────────────
    hp_com = hp_traj.mean(axis=1)   # (F, 3)

    # ── Per-free-ASO metrics ──────────────────────────────────────────────────
    bind_prob   = np.zeros(n_free)    # fraction of frames "bound"
    close_prob  = np.zeros(n_free)    # fraction at < 8 Å
    dist_to_hp  = np.zeros((n_free, F))  # COM-to-COM distance
    first_passage = np.full(n_free, np.nan)  # first frame bound

    for mi in range(n_free):
        fa = free_trajs[mi]   # (F, 10, 3)
        for f in range(F):
            D_min = cdist(fa[f], hp_traj[f]).min()
            dist_to_hp[mi, f] = D_min
            if D_min < BIND_CUTOFF:
                bind_prob[mi] += 1
                if np.isnan(first_passage[mi]):
                    first_passage[mi] = ts[f]
            if D_min < CLOSE_CUTOFF:
                close_prob[mi] += 1
        bind_prob[mi]  /= F
        close_prob[mi] /= F

    n_ever_bound  = int((bind_prob  > 0).sum())
    n_ever_close  = int((close_prob > 0).sum())

    print(f"Bind (< {BIND_CUTOFF} Å):  {n_ever_bound}/{n_free} ASOs  "
          f"(mean prob = {bind_prob.mean()*100:.4f}%)")
    print(f"Close (< {CLOSE_CUTOFF} Å): {n_ever_close}/{n_free} ASOs  "
          f"(mean prob = {close_prob.mean()*100:.4f}%)")
    print(f"Mean first-passage time: {np.nanmean(first_passage):.2f}  ×10⁶ steps")

    # ── Distance distribution ────────────────────────────────────────────────
    dist_flat = dist_to_hp.flatten()

    # ── Identify the "best binders" (top 5 by prob) ──────────────────────────
    top5_idx = np.argsort(bind_prob)[-5:][::-1]

    np.savez_compressed(
        OUT_NPZ,
        bind_prob=bind_prob, close_prob=close_prob,
        dist_to_hp=dist_to_hp, first_passage=first_passage,
        ts=ts
    )
    print(f"Saved → {OUT_NPZ}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE
    # ════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, hspace=0.48, wspace=0.40)

    # ── A: Binding probability bar (all 99) ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    sorted_idx = np.argsort(bind_prob)[::-1]
    bp_sorted  = bind_prob[sorted_idx] * 100
    cp_sorted  = close_prob[sorted_idx] * 100
    bar_col = ["#FF7043" if b > 0 else "#B0BEC5" for b in bp_sorted]
    ax.bar(range(n_free), bp_sorted, color=bar_col, width=1.0, edgecolor="none", label="Bound (<12Å)")
    ax.plot(range(n_free), cp_sorted, "^", ms=3, color="#9C27B0", alpha=0.7, label="Close (<8Å)")
    ax.set_xlabel("Free ASO (sorted by bind prob)", fontsize=9)
    ax.set_ylabel("Binding probability (%)", fontsize=9)
    ax.set_title(f"A  –  Free ASO Binding Probability\n({n_ever_bound}/{n_free} ever contacted hairpin)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.5, axis="y"); ax.set_facecolor("#fafafa")
    ax.legend(handles=[
        Patch(color="#FF7043", label=f"Contacted (<{BIND_CUTOFF}Å)"),
        Patch(color="#B0BEC5", label="Never bound"),
    ], fontsize=8)

    # ── B: Pie chart ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    sizes  = [n_free - n_ever_bound, n_ever_bound]
    cols   = ["#B0BEC5", "#FF7043"]
    labels = [f"Never bound\n({n_free - n_ever_bound})", f"Contacted\n({n_ever_bound})"]
    wedges, _, autotexts = ax.pie(sizes, colors=cols, labels=labels,
                                  autopct="%1.1f%%", startangle=90,
                                  wedgeprops={"edgecolor": "white", "lw": 2})
    ax.set_title(f"B  –  Binding Events\n(cutoff {BIND_CUTOFF} Å)", fontsize=10, fontweight="bold")

    # ── C: Distance distribution histogram ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 150, 60)
    ax.hist(dist_flat, bins=bins, color="#607D8B", edgecolor="none", density=True, alpha=0.7, label="All frames/ASOs")
    ax.axvline(BIND_CUTOFF,  color="#FF7043", ls="--", lw=1.5, label=f"Bind cutoff ({BIND_CUTOFF} Å)")
    ax.axvline(CLOSE_CUTOFF, color="#9C27B0", ls="--", lw=1.5, label=f"Close cutoff ({CLOSE_CUTOFF} Å)")
    ax.set_xlabel("Min distance ASO → Hairpin (Å)", fontsize=9)
    ax.set_ylabel("Probability density", fontsize=9)
    ax.set_title("C  –  Min-Distance Distribution\n(all 99 free ASOs, all frames)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.5); ax.set_facecolor("#fafafa")

    # ── D: Best binders – distance vs time ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    palette = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for rank, mi in enumerate(top5_idx):
        prob = bind_prob[mi] * 100
        ax.plot(ts, dist_to_hp[mi], color=palette[rank], lw=1.2, alpha=0.85,
                label=f"ASO #{mi+3}  p={prob:.2f}%")
    ax.axhline(BIND_CUTOFF, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Timestep (×10⁶)", fontsize=9)
    ax.set_ylabel("Min distance to hairpin (Å)", fontsize=9)
    ax.set_title("D  –  Top-5 Binders: Distance vs Time", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.5); ax.set_facecolor("#fafafa")

    # ── E: First-passage time distribution ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    fpt_valid = first_passage[~np.isnan(first_passage)]
    if len(fpt_valid) > 0:
        ax.hist(fpt_valid, bins=max(5, len(fpt_valid)//2 + 1),
                color="#FF9800", edgecolor="white", alpha=0.85)
        ax.axvline(fpt_valid.mean(), color="k", ls="--", lw=1.5,
                   label=f"Mean = {fpt_valid.mean():.2f}")
        ax.set_xlabel("First-passage time (×10⁶ steps)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No binding events observed", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_title(f"E  –  First-Passage Time to Hairpin\n({len(fpt_valid)} binding ASOs)", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, lw=0.5); ax.set_facecolor("#fafafa")

    # ── F: Radial heat map  (distance vs ASO index vs time) ──────────────────
    ax = fig.add_subplot(gs[1, 2])
    # Show distance heatmap for all 99 ASOs at all frames
    sorted_dist = dist_to_hp[sorted_idx]  # sorted by bind prob
    im = ax.imshow(sorted_dist, aspect="auto", origin="lower",
                   cmap="RdYlGn_r", vmin=0, vmax=80,
                   extent=[ts[0], ts[-1], 0, n_free])
    plt.colorbar(im, ax=ax, label="Min dist ASO→Hairpin (Å)", shrink=0.85)
    ax.axhline(n_ever_bound + 0.5, color="white", ls="--", lw=1.5,
               label=f"Bound / unbound ({n_ever_bound}/{n_free})")
    ax.set_xlabel("Timestep (×10⁶)", fontsize=9)
    ax.set_ylabel("Free ASO rank (sorted by prob)", fontsize=9)
    ax.set_title("F  –  Distance Heatmap: All 99 Free ASOs", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")

    plt.suptitle("Free-ASO Binding Probability to RNA Hairpin",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {OUT_FIG}")
