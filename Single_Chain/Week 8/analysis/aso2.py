"""
ASO–RNA Hairpin Binding Analysis  (with trajectory centering)
==============================================================
Step 1 – Centers the LAMMPS dump in-place (xu/yu/zu unwrapped coords)
Step 2 – Parses topology from .dat
Step 3 – Reads centered trajectory and builds coordinate arrays
Step 4 – Computes binding metrics for docked + free ASOs
Step 5 – Saves 4-panel figure

Usage
-----
  python aso_binding_analysis.py

Edit FILE PATHS and PARAMETERS below.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ═══════════════════════════════════════════════════════════════════════════════
DAT_FILE      = "100ASO.dat"
TRAJ_FILE     = "100ASO.dat_NVE.lammpstrj"
CENTERED_TRAJ = "100ASO_centered.lammpstrj"   # written by Step 1
OUT_PNG       = "aso_binding_analysis.png"

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
CUTOFF  = 12.0   # Å  contact cutoff for CG bead model
kT      = 0.592  # kcal/mol (298 K); use 0.547 for 274 K
SMOOTH  = 50     # frames for smoothing in Panel A

# Molecule IDs  (from 100ASO.dat)
MOL_DOCKED_ASO = 1
MOL_HAIRPIN    = 2
MOL_FREE_FIRST = 3
MOL_FREE_LAST  = 101        # 99 free ASOs

TYPE_TO_NUC = {1: "A", 2: "C", 3: "G", 4: "U"}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – CENTER TRAJECTORY
#   Subtracts the per-frame centroid from all atom coordinates.
#   Handles xu/yu/zu (unwrapped) or x/y/z columns automatically.
#   Writes CENTERED_TRAJ; skips if file already exists.
# ═══════════════════════════════════════════════════════════════════════════════
def center_dump(in_path, out_path, hairpin_mol_id=MOL_HAIRPIN):
    """
    Center each frame on the hairpin (mol 2) centroid.
    This pins the RNA at the origin so all inter-molecular distances are meaningful,
    regardless of how far the system has drifted under unwrapped (xu/yu/zu) coords.
    """
    if os.path.isfile(out_path):
        print(f"  Centered trajectory already exists ({out_path}), skipping.")
        return

    print(f"  Centering {in_path} -> {out_path} (reference: mol {hairpin_mol_id}) ...")
    n_frames = 0
    col_mol = col_cx = col_cy = col_cz = None

    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        while True:
            line = fin.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                fout.write(line)
                continue

            fout.write(line)
            ts = fin.readline()
            if not ts:
                break
            fout.write(ts)

            fout.write(fin.readline())          # ITEM: NUMBER OF ATOMS
            n = int(fin.readline())
            fout.write(f"{n}\n")

            fout.write(fin.readline())          # ITEM: BOX BOUNDS
            for _ in range(3):
                fout.write(fin.readline())

            atoms_hdr = fin.readline()
            if not atoms_hdr.startswith("ITEM: ATOMS"):
                raise RuntimeError("Expected 'ITEM: ATOMS' header line.")
            fout.write(atoms_hdr)
            cols = atoms_hdr.strip().split()[2:]

            # Detect columns once
            if col_cx is None:
                col_mol = cols.index("mol") if "mol" in cols else None
                if {"xu", "yu", "zu"}.issubset(cols):
                    col_cx, col_cy, col_cz = cols.index("xu"), cols.index("yu"), cols.index("zu")
                elif {"x", "y", "z"}.issubset(cols):
                    col_cx, col_cy, col_cz = cols.index("x"), cols.index("y"), cols.index("z")
                else:
                    for _ in range(n): fout.write(fin.readline())
                    continue
                print(f"    Columns: mol={col_mol}, coords={col_cx},{col_cy},{col_cz}")

            rows = [fin.readline().rstrip("\n").split() for _ in range(n)]

            # Centroid of hairpin beads only
            if col_mol is not None:
                hp_coords = np.array(
                    [[float(r[col_cx]), float(r[col_cy]), float(r[col_cz])]
                     for r in rows if int(r[col_mol]) == hairpin_mol_id],
                    dtype=float)
            else:
                # fallback: use all atoms (old behaviour)
                hp_coords = np.array(
                    [[float(r[col_cx]), float(r[col_cy]), float(r[col_cz])]
                     for r in rows], dtype=float)

            cen = hp_coords.mean(axis=0)

            for r in rows:
                r[col_cx] = f"{float(r[col_cx]) - cen[0]:.6f}"
                r[col_cy] = f"{float(r[col_cy]) - cen[1]:.6f}"
                r[col_cz] = f"{float(r[col_cz]) - cen[2]:.6f}"
                fout.write(" ".join(r) + "\n")

            n_frames += 1
            if n_frames % 500 == 0:
                print(f"    ... {n_frames} frames centred")

    print(f"  Done: {n_frames} frames centred.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – PARSE .dat  ->  topology
# ═══════════════════════════════════════════════════════════════════════════════
def parse_dat(fname):
    print(f"  Reading topology from {fname} ...")
    with open(fname) as f:
        lines = f.readlines()

    n_atoms = int(next(l for l in lines
                       if "atoms" in l and l.split()[1] == "atoms").split()[0])
    start   = next(i for i, l in enumerate(lines) if l.strip() == "Atoms") + 2

    atom_info = {}
    mol_atoms = defaultdict(list)
    for line in lines[start : start + n_atoms]:
        p = line.split()
        if len(p) < 5:
            continue
        aid, mid, atype = int(p[0]), int(p[1]), int(p[2])
        atom_info[aid]  = (mid, atype)
        mol_atoms[mid].append(aid)

    for mid in mol_atoms:
        mol_atoms[mid].sort()

    print(f"    {n_atoms} atoms, {len(mol_atoms)} molecules")
    return atom_info, dict(mol_atoms)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – PARSE CENTERED TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════════
def parse_traj(fname, mol_atoms):
    docked_aids = mol_atoms[MOL_DOCKED_ASO]
    hp_aids     = mol_atoms[MOL_HAIRPIN]
    free_mids   = list(range(MOL_FREE_FIRST, MOL_FREE_LAST + 1))
    free_aids   = [mol_atoms[m] for m in free_mids]
    n_free      = len(free_mids)

    print(f"  Reading trajectory from {fname} ...")
    print(f"    File size: {os.path.getsize(fname)/1e6:.1f} MB")

    aso_frames  = []
    hp_frames   = []
    free_frames = [[] for _ in range(n_free)]
    timesteps   = []
    col_cx = col_cy = col_cz = None

    with open(fname) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" not in line:
                continue

            ts = int(f.readline())
            f.readline()                        # ITEM: NUMBER OF ATOMS
            n  = int(f.readline())
            f.readline()                        # ITEM: BOX BOUNDS
            f.readline(); f.readline(); f.readline()

            atoms_hdr = f.readline()
            cols = atoms_hdr.strip().split()[2:]

            if col_cx is None:
                if {"xu", "yu", "zu"}.issubset(cols):
                    col_cx, col_cy, col_cz = (cols.index("xu"),
                                              cols.index("yu"),
                                              cols.index("zu"))
                elif {"x", "y", "z"}.issubset(cols):
                    col_cx, col_cy, col_cz = (cols.index("x"),
                                              cols.index("y"),
                                              cols.index("z"))
                else:
                    sys.exit("ERROR: cannot find coordinate columns in dump.")
                print(f"    Coordinate columns: indices {col_cx},{col_cy},{col_cz} "
                      f"({cols[col_cx]},{cols[col_cy]},{cols[col_cz]})")

            coords = {}
            for _ in range(n):
                p   = f.readline().split()
                aid = int(p[0])
                coords[aid] = np.array([float(p[col_cx]),
                                        float(p[col_cy]),
                                        float(p[col_cz])], dtype=np.float32)

            timesteps.append(ts)
            aso_frames.append( np.array([coords[a] for a in docked_aids]) )
            hp_frames.append(  np.array([coords[a] for a in hp_aids])     )
            for i, aids in enumerate(free_aids):
                free_frames[i].append( np.array([coords[a] for a in aids]) )

            if len(timesteps) % 500 == 0:
                print(f"    ... {len(timesteps)} frames read (ts={ts})")

    n_frames = len(timesteps)
    print(f"    Done: {n_frames} frames, ts {timesteps[0]} -> {timesteps[-1]}")

    aso_traj   = np.array(aso_frames,  dtype=np.float32)
    hp_traj    = np.array(hp_frames,   dtype=np.float32)
    free_trajs = np.array([np.array(ff, dtype=np.float32)
                            for ff in free_frames])

    d0 = cdist(aso_traj[0].astype(float), hp_traj[0].astype(float)).min()
    print(f"    Sanity check - frame 0 docked ASO-HP min dist: {d0:.2f} A")

    return aso_traj, hp_traj, free_trajs, np.array(timesteps)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – BINDING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def calc_binding(aso_traj, hp_traj, free_trajs):
    n_frames = aso_traj.shape[0]
    n_beads  = aso_traj.shape[1]
    n_free   = free_trajs.shape[0]

    def md(a, b):
        return cdist(a.astype(float), b.astype(float)).min()

    print("  Computing docked ASO-hairpin distances ...")
    min_dists    = np.array([md(aso_traj[f], hp_traj[f]) for f in range(n_frames)])
    bound_frames = min_dists < CUTOFF
    print(f"    Docked ASO bound: {bound_frames.sum()}/{n_frames} frames "
          f"({100*bound_frames.mean():.1f}%)")
    print(f"    Distance range: {min_dists.min():.1f} - {min_dists.max():.1f} A  "
          f"(mean {min_dists.mean():.1f} +/- {min_dists.std():.1f} A)")

    print("  Computing per-bead contact probability ...")
    per_bead_prob = np.zeros(n_beads)
    for i in range(n_beads):
        per_bead_prob[i] = np.mean([
            md(aso_traj[f, i:i+1], hp_traj[f]) < CUTOFF
            for f in range(n_frames)
        ])

    print("  Computing free ASO binding ...")
    free_bound_counts = np.zeros(n_free, dtype=int)
    for a in range(n_free):
        cnt = sum(md(free_trajs[a, f], hp_traj[f]) < CUTOFF
                  for f in range(n_frames))
        free_bound_counts[a] = cnt
        if (a + 1) % 20 == 0:
            print(f"    ... {a+1}/{n_free} free ASOs processed")

    n_ever = (free_bound_counts > 0).sum()
    print(f"    Free ASOs that ever bind: {n_ever}/{n_free}")
    if n_ever:
        print(f"    Bound-frame counts (binders): "
              f"{sorted(free_bound_counts[free_bound_counts>0].tolist(), reverse=True)}")

    return min_dists, bound_frames, per_bead_prob, free_bound_counts


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
def make_figure(timesteps, min_dists, bound_frames,
                per_bead_prob, free_bound_counts, aso_seq, out_png):

    n_frames = len(timesteps)

    def dG(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -kT * np.log(p / (1 - p))

    delta_g = dG(per_bead_prob)

    PINK    = '#f4a0c0'
    BLUE    = '#7ab8f5'
    GOLD    = '#f5c842'
    GREEN   = '#5dd479'
    BG      = '#0f1117'
    PANEL   = '#181c27'
    GRID    = '#2a2f3f'
    TEXT    = '#e8eaf0'
    SUBTEXT = '#9aa0b8'

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

    bead_idx  = np.arange(1, len(aso_seq) + 1)
    labels    = [f"{i}\n({s})" for i, s in enumerate(aso_seq, 1)]
    frames_ax = np.arange(n_frames)

    # ── Panel A ───────────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, 'A  |  Docked ASO-Hairpin Distance')

    colors_pts = np.where(bound_frames, PINK, '#4a5060')
    ax_a.scatter(frames_ax, min_dists, c=colors_pts, s=3, zorder=3, alpha=0.5)
    ax_a.axhline(CUTOFF, color=GOLD, lw=1.3, ls='--',
                 label=f'Cutoff {CUTOFF} A', zorder=4)
    ax_a.fill_between(frames_ax, 0, CUTOFF,
                      where=bound_frames, alpha=0.15, color=PINK)
    smooth_n = min(SMOOTH, max(n_frames // 10, 1))
    smooth   = uniform_filter1d(min_dists, size=smooth_n)
    ax_a.plot(frames_ax, smooth, color=PINK, lw=2, zorder=5, label='Smoothed')

    pct = 100 * bound_frames.mean()
    ax_a.text(0.97, 0.95,
              f"{bound_frames.sum()}/{n_frames} frames\nbound ({pct:.1f}%)",
              transform=ax_a.transAxes, ha='right', va='top',
              color=PINK, fontsize=9, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#2a1a24',
                        edgecolor=PINK, alpha=0.85))
    ax_a.set_xlabel('Frame', fontsize=9)
    ax_a.set_ylabel('Min Distance (A)', fontsize=9)
    ax_a.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax_a.set_ylim(0, min(min_dists.max() * 1.15, 200))

    # ── Panel B ───────────────────────────────────────────────────────────────
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
    ax_b.set_ylabel('P(contact < 12 A)', fontsize=9)
    ax_b.set_ylim(0, 1.12)
    ax_b.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel C ───────────────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    style_ax(ax_c, 'C  |  Free ASO Bound-Frame Distribution')

    max_count = max(free_bound_counts.max(), 1)
    ax_c.bar(0, (free_bound_counts == 0).sum(),
             width=1, color=SUBTEXT, edgecolor=BG, lw=0.8, zorder=3)
    for v in range(1, max_count + 1):
        cnt = (free_bound_counts == v).sum()
        if cnt > 0:
            ax_c.bar(v, cnt, width=1, color=GREEN,
                     edgecolor=BG, lw=0.8, zorder=4)

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

    # ── Panel D ───────────────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    style_ax(ax_d, 'D  |  Binding Free Energy per Bead')

    bar_c2 = [PINK if dg < 0 else '#4a5060' for dg in delta_g]
    ax_d.bar(bead_idx, delta_g, color=bar_c2, edgecolor='none', width=0.7, zorder=3)
    ax_d.axhline(0, color=TEXT, lw=0.8, alpha=0.5)
    for i, dg in zip(bead_idx, delta_g):
        va     = 'top'  if dg < 0 else 'bottom'
        offset = -0.03  if dg < 0 else  0.03
        ax_d.text(i, dg + offset, f'{dg:.2f}',
                  ha='center', va=va, color=TEXT, fontsize=7.5, fontweight='bold')
    ax_d.set_xticks(bead_idx)
    ax_d.set_xticklabels(labels, fontsize=8)
    ax_d.set_xlabel('ASO Bead (nucleotide)', fontsize=9)
    ax_d.set_ylabel('dG (kcal/mol)', fontsize=9)

    fig.suptitle(
        f'ASO-RNA Hairpin Binding Analysis  |  CG-MD Simulation ({n_frames} frames)',
        color=TEXT, fontsize=13, fontweight='bold', y=0.97)

    plt.savefig(out_png, dpi=180, facecolor=BG, bbox_inches='tight')
    print(f"\n  Figure saved -> {out_png}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    for f in [DAT_FILE, TRAJ_FILE]:
        if not os.path.isfile(f):
            sys.exit(f"ERROR: cannot find {f!r} - check FILE PATHS at top of script.")

    print("=== Step 1: Center trajectory ===")
    center_dump(TRAJ_FILE, CENTERED_TRAJ)

    print("\n=== Step 2: Topology ===")
    atom_info, mol_atoms = parse_dat(DAT_FILE)
    aso_seq = [TYPE_TO_NUC[atom_info[a][1]] for a in mol_atoms[MOL_DOCKED_ASO]]
    print(f"  Docked ASO sequence: {' '.join(aso_seq)}")

    print("\n=== Step 3: Trajectory ===")
    aso_traj, hp_traj, free_trajs, timesteps = parse_traj(CENTERED_TRAJ, mol_atoms)

    print("\n=== Step 4: Binding analysis ===")
    min_dists, bound_frames, per_bead_prob, free_bound_counts = \
        calc_binding(aso_traj, hp_traj, free_trajs)

    print("\n=== Step 5: Figure ===")
    make_figure(timesteps, min_dists, bound_frames,
                per_bead_prob, free_bound_counts, aso_seq, OUT_PNG)

    print("\nAll done.")