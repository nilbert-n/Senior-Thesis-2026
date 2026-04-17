"""
ASO–RNA Hairpin Binding Analysis
=================================
Parses a LAMMPS trajectory directly and produces a 4-panel figure:
  A  Docked ASO–Hairpin min distance over time
  B  Per-bead contact probability (docked ASO)
  C  Free ASO bound-frame distribution
  D  Binding free energy ΔG per ASO bead

Usage
-----
  python aso_binding_analysis.py

Edit the FILE PATHS and PARAMETERS section below if your files are
named differently or live in another directory.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PATHS  –  edit these
# ═══════════════════════════════════════════════════════════════════════════════
DAT_FILE  = "100ASO.dat"                   # LAMMPS data file
TRAJ_FILE = "100ASO.dat_NVE.lammpstrj"     # LAMMPS dump trajectory (all frames)
OUT_PNG   = "aso_binding_analysis.png"     # output figure

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
CUTOFF   = 16.56  # Å  contact cutoff (appropriate for CG bead model)
kT       = 0.592  # kcal/mol at 298 K  (change to 0.547 for 274 K if desired)
SMOOTH   = 50     # frames for uniform_filter1d smoothing in panel A

# Molecule IDs in the LAMMPS data file:
MOL_DOCKED_ASO = 1          # docked ASO
MOL_HAIRPIN    = 2          # RNA hairpin
MOL_FREE_FIRST = 3          # first free ASO
MOL_FREE_LAST  = 101        # last  free ASO  (99 free ASOs total)

# Atom-type → nucleotide (from masses in .dat:
#   type 1=A 329.20, type 2=C 305.20, type 3=G 345.20, type 4=U 306.20)
TYPE_TO_NUC = {1: "A", 2: "C", 3: "G", 4: "U"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PARSE .dat  →  topology
# ═══════════════════════════════════════════════════════════════════════════════
def parse_dat(fname):
    print(f"  Reading topology from {fname} …")
    with open(fname) as f:
        lines = f.readlines()

    n_atoms = int(next(l for l in lines if "atoms" in l and l.split()[1] == "atoms").split()[0])
    start   = next(i for i, l in enumerate(lines) if l.strip() == "Atoms") + 2

    atom_info = {}          # atom_id → (mol_id, atom_type)
    mol_atoms = defaultdict(list)
    for line in lines[start : start + n_atoms]:
        p = line.split()
        if len(p) < 5:
            continue
        aid, mid, atype = int(p[0]), int(p[1]), int(p[2])
        atom_info[aid] = (mid, atype)
        mol_atoms[mid].append(aid)

    for mid in mol_atoms:
        mol_atoms[mid].sort()

    print(f"    {n_atoms} atoms, {len(mol_atoms)} molecules")
    return atom_info, dict(mol_atoms)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PARSE TRAJECTORY  →  per-molecule coordinate arrays
#     Returns:
#       aso_traj     (F, 10,  3)
#       hp_traj      (F, 32,  3)
#       free_trajs   (99, F, 10, 3)
#       timesteps    (F,)
# ═══════════════════════════════════════════════════════════════════════════════
def parse_traj(fname, mol_atoms):
    docked_aids = mol_atoms[MOL_DOCKED_ASO]
    hp_aids     = mol_atoms[MOL_HAIRPIN]
    free_mids   = list(range(MOL_FREE_FIRST, MOL_FREE_LAST + 1))
    free_aids   = [mol_atoms[m] for m in free_mids]
    n_free      = len(free_mids)
    n_aso       = len(docked_aids)
    n_hp        = len(hp_aids)

    print(f"  Reading trajectory from {fname} …")
    file_size = os.path.getsize(fname) / 1e6
    print(f"    File size: {file_size:.1f} MB  (this may take a while for large runs)")

    aso_frames  = []
    hp_frames   = []
    free_frames = [[] for _ in range(n_free)]
    timesteps   = []

    with open(fname) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "TIMESTEP" not in line:
                continue

            ts   = int(f.readline())
            f.readline()                        # ITEM: NUMBER OF ATOMS
            n    = int(f.readline())
            f.readline()                        # ITEM: BOX BOUNDS
            f.readline(); f.readline(); f.readline()
            f.readline()                        # ITEM: ATOMS ...

            coords = {}
            for _ in range(n):
                p = f.readline().split()
                # dump format: id mol type xs ys zs x y z   (wrapped + unwrapped)
                # fall back gracefully if fewer columns
                aid = int(p[0])
                coords[aid] = np.array([float(p[4]), float(p[5]), float(p[6])])

            timesteps.append(ts)
            aso_frames.append( np.array([coords[a] for a in docked_aids]) )
            hp_frames.append(  np.array([coords[a] for a in hp_aids])     )
            for i, aids in enumerate(free_aids):
                free_frames[i].append( np.array([coords[a] for a in aids]) )

            if len(timesteps) % 500 == 0:
                print(f"    … {len(timesteps)} frames read (ts={ts})")

    n_frames = len(timesteps)
    print(f"    Done: {n_frames} frames, ts {timesteps[0]} → {timesteps[-1]}")

    aso_traj  = np.array(aso_frames,  dtype=np.float32)   # (F, 10,  3)
    hp_traj   = np.array(hp_frames,   dtype=np.float32)   # (F, 32,  3)
    free_trajs = np.array([np.array(ff, dtype=np.float32)
                            for ff in free_frames])        # (99, F, 10, 3)
    timesteps  = np.array(timesteps)

    return aso_traj, hp_traj, free_trajs, timesteps


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BINDING CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def calc_binding(aso_traj, hp_traj, free_trajs):
    n_frames = aso_traj.shape[0]
    n_beads  = aso_traj.shape[1]
    n_free   = free_trajs.shape[0]

    print("  Computing docked ASO–hairpin distances …")
    min_dists = np.array([
        cdist(aso_traj[f].astype(float), hp_traj[f].astype(float)).min()
        for f in range(n_frames)
    ])
    bound_frames = min_dists < CUTOFF
    print(f"    Docked ASO bound: {bound_frames.sum()}/{n_frames} frames "
          f"({100*bound_frames.mean():.1f}%)")

    print("  Computing per-bead contact probability …")
    per_bead_prob = np.zeros(n_beads)
    for i in range(n_beads):
        per_bead_prob[i] = np.mean([
            cdist(aso_traj[f, i:i+1].astype(float),
                  hp_traj[f].astype(float)).min() < CUTOFF
            for f in range(n_frames)
        ])

    print("  Computing free ASO binding …")
    free_bound_counts = np.zeros(n_free, dtype=int)
    for a in range(n_free):
        cnt = sum(
            cdist(free_trajs[a, f].astype(float),
                  hp_traj[f].astype(float)).min() < CUTOFF
            for f in range(n_frames)
        )
        free_bound_counts[a] = cnt
        if (a + 1) % 20 == 0:
            print(f"    … {a+1}/{n_free} free ASOs processed")

    n_ever = (free_bound_counts > 0).sum()
    print(f"    Free ASOs that ever bind: {n_ever}/{n_free}")
    print(f"    Bound-frame counts (binders): {free_bound_counts[free_bound_counts>0].tolist()}")

    return min_dists, bound_frames, per_bead_prob, free_bound_counts


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
def make_figure(timesteps, min_dists, bound_frames,
                per_bead_prob, free_bound_counts, aso_seq, out_png):

    n_frames = len(timesteps)
    frames_ax = np.arange(n_frames)
    kT_local  = kT

    def dG(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -kT_local * np.log(p / (1 - p))

    delta_g = dG(per_bead_prob)

    # ── colours / style ──────────────────────────────────────────────────────
    PINK   = '#f4a0c0'
    BLUE   = '#7ab8f5'
    GOLD   = '#f5c842'
    GREEN  = '#5dd479'
    BG     = '#0f1117'
    PANEL  = '#181c27'
    GRID   = '#2a2f3f'
    TEXT   = '#e8eaf0'
    SUBTEXT= '#9aa0b8'

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.42, wspace=0.38,
                            left=0.08, right=0.96, top=0.91, bottom=0.09)

    bead_idx = np.arange(1, len(aso_seq) + 1)
    labels   = [f"{i}\n({s})" for i, s in enumerate(aso_seq, 1)]

    # ── Panel A: distance vs frame ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, 'A  |  Docked ASO–Hairpin Distance')

    colors_pts = np.where(bound_frames, PINK, '#4a5060')
    ax_a.scatter(frames_ax, min_dists, c=colors_pts, s=4, zorder=3, alpha=0.6)
    ax_a.axhline(CUTOFF, color=GOLD, lw=1.3, ls='--',
                 label=f'Cutoff {CUTOFF} Å', zorder=4)
    ax_a.fill_between(frames_ax, 0, CUTOFF, where=bound_frames,
                      alpha=0.10, color=PINK)
    smooth = uniform_filter1d(min_dists, size=SMOOTH)
    ax_a.plot(frames_ax, smooth, color=PINK, lw=2, zorder=5, label='Smoothed')

    pct = 100 * bound_frames.mean()
    ax_a.text(0.97, 0.95,
              f"{bound_frames.sum()}/{n_frames} frames\nbound ({pct:.1f}%)",
              transform=ax_a.transAxes, ha='right', va='top',
              color=PINK, fontsize=9, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#2a1a24',
                        edgecolor=PINK, alpha=0.85))

    ax_a.set_xlabel('Frame', fontsize=9)
    ax_a.set_ylabel('Min Distance (Å)', fontsize=9)
    ax_a.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax_a.set_ylim(0, max(min_dists.max() * 1.2, CUTOFF * 1.5))

    # ── Panel B: per-bead contact prob ────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, 'B  |  Per-Bead Contact Probability')

    bar_colors = [PINK if p >= 0.5 else BLUE for p in per_bead_prob]
    bars = ax_b.bar(bead_idx, per_bead_prob, color=bar_colors,
                    edgecolor='none', width=0.7, zorder=3)
    ax_b.axhline(0.5, color=GOLD, lw=1.2, ls='--', alpha=0.8, label='p = 0.5')
    for bar, p in zip(bars, per_bead_prob):
        ax_b.text(bar.get_x() + bar.get_width() / 2, p + 0.02,
                  f'{p:.2f}', ha='center', va='bottom',
                  color=TEXT, fontsize=7.5, fontweight='bold')

    ax_b.set_xticks(bead_idx)
    ax_b.set_xticklabels(labels, fontsize=8)
    ax_b.set_xlabel('ASO Bead (nucleotide)', fontsize=9)
    ax_b.set_ylabel('P(contact < 12 Å)', fontsize=9)
    ax_b.set_ylim(0, 1.12)
    ax_b.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel C: free ASO bound-frame distribution ────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    style_ax(ax_c, 'C  |  Free ASO Bound-Frame Distribution')

    max_count = max(free_bound_counts.max(), 1)
    bins = np.arange(0, max_count + 2) - 0.5

    # plot non-binders
    n0 = (free_bound_counts == 0).sum()
    ax_c.bar(0, n0, width=1, color=SUBTEXT, edgecolor=BG, lw=0.8, zorder=3)
    # plot binders in green
    for v in range(1, max_count + 1):
        cnt = (free_bound_counts == v).sum()
        if cnt > 0:
            ax_c.bar(v, cnt, width=1, color=GREEN, edgecolor=BG, lw=0.8, zorder=4)

    n_bind  = (free_bound_counts > 0).sum()
    n_never = (free_bound_counts == 0).sum()
    ax_c.text(0.97, 0.95,
              f"Never bind: {n_never}/{len(free_bound_counts)}\n"
              f"Ever bind:   {n_bind}/{len(free_bound_counts)}",
              transform=ax_c.transAxes, ha='right', va='top',
              color=TEXT, fontsize=9,
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#1a2a1f',
                        edgecolor=GREEN, alpha=0.85))

    ax_c.set_xlabel(f'Frames Bound (out of {n_frames})', fontsize=9)
    ax_c.set_ylabel('Count of free ASOs', fontsize=9)
    ax_c.set_xticks(range(max_count + 1))

    # ── Panel D: ΔG per bead ──────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    style_ax(ax_d, 'D  |  Binding Free Energy per Bead')

    bar_c2 = [PINK if dg < 0 else '#4a5060' for dg in delta_g]
    ax_d.bar(bead_idx, delta_g, color=bar_c2, edgecolor='none', width=0.7, zorder=3)
    ax_d.axhline(0, color=TEXT, lw=0.8, alpha=0.5)

    for i, dg in zip(bead_idx, delta_g):
        va     = 'top'    if dg < 0 else 'bottom'
        offset = -0.03    if dg < 0 else 0.03
        ax_d.text(i, dg + offset, f'{dg:.2f}',
                  ha='center', va=va, color=TEXT, fontsize=7.5, fontweight='bold')

    ax_d.set_xticks(bead_idx)
    ax_d.set_xticklabels(labels, fontsize=8)
    ax_d.set_xlabel('ASO Bead (nucleotide)', fontsize=9)
    ax_d.set_ylabel('ΔG (kcal/mol)', fontsize=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f'ASO–RNA Hairpin Binding Analysis  |  CG-MD Simulation ({n_frames} frames)',
        color=TEXT, fontsize=13, fontweight='bold', y=0.97)

    plt.savefig(out_png, dpi=180, facecolor=BG, bbox_inches='tight')
    print(f"\n  Figure saved → {out_png}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    for f in [DAT_FILE, TRAJ_FILE]:
        if not os.path.isfile(f):
            sys.exit(f"ERROR: cannot find {f}\n"
                     "  Edit the FILE PATHS section at the top of the script.")

    print("=== Step 1: Topology ===")
    atom_info, mol_atoms = parse_dat(DAT_FILE)

    aso_seq = [TYPE_TO_NUC[atom_info[a][1]] for a in mol_atoms[MOL_DOCKED_ASO]]
    print(f"  Docked ASO sequence: {' '.join(aso_seq)}")

    print("\n=== Step 2: Trajectory ===")
    aso_traj, hp_traj, free_trajs, timesteps = parse_traj(TRAJ_FILE, mol_atoms)

    print("\n=== Step 3: Binding analysis ===")
    min_dists, bound_frames, per_bead_prob, free_bound_counts = \
        calc_binding(aso_traj, hp_traj, free_trajs)

    print("\n=== Step 4: Figure ===")
    make_figure(timesteps, min_dists, bound_frames,
                per_bead_prob, free_bound_counts, aso_seq, OUT_PNG)

    print("\nAll done.")