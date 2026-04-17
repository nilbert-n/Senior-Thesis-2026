#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
ASO–RNA Hairpin: Unified 6-Panel Analysis
═══════════════════════════════════════════════════════════════════════════════
Produces one 6-panel figure per simulation:

  A  Docked ASO–Hairpin min distance vs time
  B  Free ASO bound-frame distribution
  C  Binding free energy ΔG per ASO bead
  D  RMSD vs time (ASO + Hairpin vs initial structure)
  E  Radius of Gyration vs time
  F  Potential Energy vs time

Auto-detects:
  - Number of ASOs (25/50/100/200)
  - ASO length (7/10/12/14 beads)
  - Hairpin variant (always 32 beads, mol 2)
  - Handles trajectories that ended early (SLURM truncation)

Usage:
    python run_all_analysis.py

    OR for a single simulation:
    python run_all_analysis.py --dat configs/100ASO_unmod.dat \
        --traj outputs/100ASO_unmod.lammpstrj \
        --thermo outputs/thermo_100ASO_unmod_production.dat

Directory layout expected (from aso_project/):
    configs/*.dat
    outputs/*.lammpstrj
    outputs/thermo_*_production.dat   (preferred) or thermo_*_averaged.dat
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress

# ── Parameters ───────────────────────────────────────────────────────────────
CUTOFF   = 16.56   # Å – contact cutoff for CG bead model
kT       = 0.592   # kcal/mol at 298 K
SMOOTH   = 30      # frames for smoothing
DT_FS    = 10      # timestep in fs
DUMP_STRIDE = 50000  # dump every N steps

TYPE_TO_NUC = {1: "A", 2: "C", 3: "G", 4: "U"}

# ── Light theme colours ──────────────────────────────────────────────────────
PINK    = '#E91E63'
BLUE    = '#2196F3'
GOLD    = '#FF9800'
GREEN   = '#4CAF50'
PURPLE  = '#9C27B0'
ORANGE  = '#FF5722'
BG      = '#FFFFFF'
PANEL   = '#FAFAFA'
GRID    = '#E0E0E0'
TEXT    = '#333333'
SUBTEXT = '#666666'


# ═════════════════════════════════════════════════════════════════════════════
# PARSING
# ═════════════════════════════════════════════════════════════════════════════
def parse_dat(fname):
    """Parse LAMMPS data file → atom_info, mol_atoms (auto-detect topology)."""
    with open(fname) as f:
        lines = f.readlines()

    n_atoms = int(next(l for l in lines if l.strip().endswith("atoms")).split()[0])
    start = next(i for i, l in enumerate(lines) if l.strip() == "Atoms") + 2

    atom_info = {}
    mol_atoms = defaultdict(list)
    for line in lines[start:start + n_atoms]:
        p = line.split()
        if len(p) < 7:
            continue
        aid, mid, atype = int(p[0]), int(p[1]), int(p[2])
        xyz = np.array([float(p[4]), float(p[5]), float(p[6])])
        atom_info[aid] = (mid, atype, xyz)
        mol_atoms[mid].append(aid)

    for mid in mol_atoms:
        mol_atoms[mid].sort()

    # Auto-detect topology
    n_mols = len(mol_atoms)
    aso_beads = len(mol_atoms[1])
    hp_beads = len(mol_atoms[2])
    n_free = n_mols - 2
    total_aso = n_free + 1  # docked + free

    print(f"  Topology: {total_aso} ASOs ({aso_beads}-mer), "
          f"1 hairpin ({hp_beads} beads), {n_free} free ASOs")

    return atom_info, dict(mol_atoms), aso_beads, hp_beads, n_free


def parse_traj(fname, mol_atoms, n_free):
    """Parse LAMMPS trajectory → per-molecule coordinate arrays."""
    docked_aids = mol_atoms[1]
    hp_aids = mol_atoms[2]
    free_mids = list(range(3, 3 + n_free))
    free_aids = [mol_atoms[m] for m in free_mids]

    n_aso = len(docked_aids)
    n_hp = len(hp_aids)

    aso_frames = []
    hp_frames = []
    free_frames = [[] for _ in range(n_free)]
    timesteps = []

    file_size_mb = os.path.getsize(fname) / 1e6
    print(f"  Reading {fname} ({file_size_mb:.1f} MB) ...")

    with open(fname) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "TIMESTEP" not in line:
                continue

            ts = int(f.readline())
            f.readline()  # NUMBER OF ATOMS
            n = int(f.readline())
            f.readline()  # BOX BOUNDS
            f.readline(); f.readline(); f.readline()
            f.readline()  # ATOMS header

            coords = {}
            for _ in range(n):
                p = f.readline().split()
                aid = int(p[0])
                coords[aid] = np.array([float(p[4]), float(p[5]), float(p[6])])

            timesteps.append(ts)
            aso_frames.append(np.array([coords[a] for a in docked_aids]))
            hp_frames.append(np.array([coords[a] for a in hp_aids]))
            for i, aids in enumerate(free_aids):
                free_frames[i].append(np.array([coords[a] for a in aids]))

            if len(timesteps) % 200 == 0:
                print(f"    ... {len(timesteps)} frames (ts={ts})")

    n_frames = len(timesteps)
    print(f"    Done: {n_frames} frames, ts {timesteps[0]} → {timesteps[-1]}")

    aso_traj = np.array(aso_frames, dtype=np.float32)
    hp_traj = np.array(hp_frames, dtype=np.float32)

    if n_free > 0:
        free_trajs = np.array([np.array(ff, dtype=np.float32) for ff in free_frames])
    else:
        free_trajs = np.empty((0, n_frames, n_aso, 3), dtype=np.float32)

    return aso_traj, hp_traj, free_trajs, np.array(timesteps)


def parse_thermo(fname):
    """Parse thermo file (handles both _production and _averaged formats)."""
    data = {"timestep": [], "PE": [], "KE": [], "T": [], "E_total": []}
    extra_col = False

    with open(fname) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) == 5:
                data["timestep"].append(float(p[0]))
                data["PE"].append(float(p[1]))
                data["KE"].append(float(p[2]))
                data["T"].append(float(p[3]))
                data["E_total"].append(float(p[4]))
            elif len(p) == 6:
                # Some thermo files have Rg as 6th column
                data["timestep"].append(float(p[0]))
                data["PE"].append(float(p[1]))
                data["KE"].append(float(p[2]))
                data["T"].append(float(p[3]))
                data["E_total"].append(float(p[4]))
                extra_col = True

    if not data["timestep"]:
        print(f"  WARNING: No data in {fname}")
        return None

    return {k: np.array(v) for k, v in data.items()}


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
def kabsch_rotate(P, Q):
    """Rotate P onto Q (both mean-centred)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return P @ R.T


def calc_rmsd_traj(traj, ref):
    """RMSD of each frame vs reference after Kabsch alignment."""
    ref_c = ref - ref.mean(axis=0)
    rmsds = np.empty(len(traj))
    for f, frame in enumerate(traj):
        P = frame - frame.mean(axis=0)
        Pa = kabsch_rotate(P.astype(np.float64), ref_c.astype(np.float64))
        diff = Pa - ref_c
        rmsds[f] = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsds


def radius_of_gyration(traj):
    """Per-frame Rg."""
    rg = np.empty(len(traj))
    for f, frame in enumerate(traj):
        com = frame.mean(axis=0)
        rg[f] = np.sqrt(np.mean(np.sum((frame - com)**2, axis=1)))
    return rg


def calc_binding(aso_traj, hp_traj, free_trajs, cutoff):
    """Compute docked distance, per-bead prob, free ASO bound counts."""
    n_frames = len(aso_traj)
    n_beads = aso_traj.shape[1]
    n_free = len(free_trajs)

    # Docked ASO min distance to hairpin per frame
    min_dists = np.array([
        cdist(aso_traj[f].astype(float), hp_traj[f].astype(float)).min()
        for f in range(n_frames)
    ])
    bound_frames = min_dists < cutoff

    # Per-bead contact probability
    per_bead_prob = np.zeros(n_beads)
    for i in range(n_beads):
        per_bead_prob[i] = np.mean([
            cdist(aso_traj[f, i:i+1].astype(float),
                  hp_traj[f].astype(float)).min() < cutoff
            for f in range(n_frames)
        ])

    # Free ASO binding
    free_bound_counts = np.zeros(n_free, dtype=int)
    for a in range(n_free):
        for f in range(n_frames):
            if cdist(free_trajs[a, f].astype(float),
                     hp_traj[f].astype(float)).min() < cutoff:
                free_bound_counts[a] += 1
        if (a + 1) % 50 == 0:
            print(f"    ... {a+1}/{n_free} free ASOs processed")

    return min_dists, bound_frames, per_bead_prob, free_bound_counts


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═════════════════════════════════════════════════════════════════════════════
def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=8)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5)


def make_figure(sim_name, timesteps, min_dists, bound_frames,
                per_bead_prob, free_bound_counts, aso_seq,
                aso_rmsd, hp_rmsd, rog_complex, rog_hp,
                thermo, out_png):

    n_frames = len(timesteps)
    # Convert timesteps to ns
    time_ns = (timesteps - timesteps[0]) * DT_FS / 1e6  # steps * 10fs / 1e6 = ns

    def dG(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -kT * np.log(p / (1 - p))

    delta_g = dG(per_bead_prob)

    n_beads = len(aso_seq)
    bead_idx = np.arange(1, n_beads + 1)
    labels = [f"{i}\n({s})" for i, s in enumerate(aso_seq, 1)]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.35,
                           left=0.06, right=0.97, top=0.90, bottom=0.07)

    sm = lambda x, w=SMOOTH: uniform_filter1d(x.astype(float), min(w, max(1, len(x)//5)))

    # ── A: Docked ASO–Hairpin Distance ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, 'A  |  Docked ASO–Hairpin Distance')

    colors_pts = np.where(bound_frames, PINK, '#CCCCCC')
    ax.scatter(time_ns, min_dists, c=colors_pts, s=3, zorder=3, alpha=0.5)
    ax.axhline(CUTOFF, color=GOLD, lw=1.3, ls='--', label=f'Cutoff {CUTOFF} Å')
    ax.fill_between(time_ns, 0, CUTOFF, where=bound_frames, alpha=0.08, color=PINK)
    if len(min_dists) > 3:
        ax.plot(time_ns, sm(min_dists), color=PINK, lw=2, zorder=5, label='Smoothed')

    pct = 100 * bound_frames.mean()
    ax.text(0.97, 0.95,
            f"{bound_frames.sum()}/{n_frames} frames\nbound ({pct:.1f}%)",
            transform=ax.transAxes, ha='right', va='top',
            color=PINK, fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDE8EF',
                      edgecolor=PINK, alpha=0.85))
    ax.set_xlabel('Time (ns)', fontsize=9)
    ax.set_ylabel('Min Distance (Å)', fontsize=9)
    ax.legend(fontsize=7, facecolor='white', edgecolor=GRID, labelcolor=TEXT)
    ax.set_ylim(0, max(min_dists.max() * 1.15, CUTOFF * 1.5))

    # ── B: Free ASO Bound-Frame Distribution ─────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, 'B  |  Free ASO Bound-Frame Distribution')

    n_free = len(free_bound_counts)
    if n_free > 0:
        max_count = max(free_bound_counts.max(), 1)
        n0 = (free_bound_counts == 0).sum()
        ax.bar(0, n0, width=1, color=SUBTEXT, edgecolor=BG, lw=0.8, zorder=3)
        for v in range(1, max_count + 1):
            cnt = (free_bound_counts == v).sum()
            if cnt > 0:
                ax.bar(v, cnt, width=1, color=GREEN, edgecolor=BG, lw=0.8, zorder=4)

        n_bind = (free_bound_counts > 0).sum()
        ax.text(0.97, 0.95,
                f"Never bind: {n0}/{n_free}\nEver bind: {n_bind}/{n_free}",
                transform=ax.transAxes, ha='right', va='top',
                color=TEXT, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                          edgecolor=GREEN, alpha=0.85))
        ax.set_xlabel(f'Frames Bound (of {n_frames})', fontsize=9)
        ax.set_ylabel('Count of free ASOs', fontsize=9)
        max_tick = min(max_count, 20)
        ax.set_xticks(range(0, max_tick + 1, max(1, max_tick // 10)))
    else:
        ax.text(0.5, 0.5, "No free ASOs\nin this simulation",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color=SUBTEXT)

    # ── C: Binding Free Energy per Bead ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, 'C  |  Binding Free Energy per Bead')

    bar_c = [PINK if dg < 0 else '#BDBDBD' for dg in delta_g]
    ax.bar(bead_idx, delta_g, color=bar_c, edgecolor='none', width=0.65, zorder=3)
    ax.axhline(0, color=TEXT, lw=0.8, alpha=0.5)
    for i, dg in zip(bead_idx, delta_g):
        va = 'top' if dg < 0 else 'bottom'
        offset = -0.04 if dg < 0 else 0.04
        ax.text(i, dg + offset, f'{dg:.2f}',
                ha='center', va=va, color=TEXT, fontsize=6.5, fontweight='bold')
    ax.set_xticks(bead_idx)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel('ASO Bead (nucleotide)', fontsize=9)
    ax.set_ylabel('ΔG (kcal/mol)', fontsize=9)

    # ── D: RMSD vs Time ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, 'D  |  RMSD vs Initial Structure')

    if len(aso_rmsd) > 3:
        ax.plot(time_ns, sm(hp_rmsd), color=BLUE, lw=1.8, label='Hairpin')
        ax.plot(time_ns, sm(aso_rmsd), color=PINK, lw=1.8, label='Docked ASO')
        mean_complex = (aso_rmsd + hp_rmsd) / 2
        ax.plot(time_ns, sm(mean_complex), color=GREEN, lw=1.2, ls='--', label='Mean')
    ax.set_xlabel('Time (ns)', fontsize=9)
    ax.set_ylabel('RMSD (Å)', fontsize=9)
    ax.legend(fontsize=7, facecolor='white', edgecolor=GRID, labelcolor=TEXT)

    # stats box
    ax.text(0.97, 0.05,
            f"HP: {hp_rmsd.mean():.1f}±{hp_rmsd.std():.1f} Å\n"
            f"ASO: {aso_rmsd.mean():.1f}±{aso_rmsd.std():.1f} Å",
            transform=ax.transAxes, ha='right', va='bottom',
            color=SUBTEXT, fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=GRID, alpha=0.85))

    # ── E: Radius of Gyration ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, 'E  |  Radius of Gyration')

    if len(rog_complex) > 3:
        ax.plot(time_ns, sm(rog_complex), color=PURPLE, lw=2, label='Complex')
        ax.plot(time_ns, sm(rog_hp), color=ORANGE, lw=2, label='Hairpin only')
    ax.set_xlabel('Time (ns)', fontsize=9)
    ax.set_ylabel('Rg (Å)', fontsize=9)
    ax.legend(fontsize=7, facecolor='white', edgecolor=GRID, labelcolor=TEXT)

    ax.text(0.97, 0.05,
            f"Complex: {rog_complex.mean():.1f}±{rog_complex.std():.1f} Å\n"
            f"Hairpin: {rog_hp.mean():.1f}±{rog_hp.std():.1f} Å",
            transform=ax.transAxes, ha='right', va='bottom',
            color=SUBTEXT, fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=GRID, alpha=0.85))

    # ── F: Potential Energy vs Time ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, 'F  |  Potential Energy vs Time')

    if thermo is not None:
        t_thermo = (thermo["timestep"] - thermo["timestep"][0]) * DT_FS / 1e6
        PE = thermo["PE"]
        if len(PE) > 3:
            ax.plot(t_thermo, PE, color='#607D8B', lw=0.8, alpha=0.5)
            ax.plot(t_thermo, sm(PE, min(100, len(PE)//3)),
                    color=GOLD, lw=2, label='Smoothed PE')
            slope, _, _, _, _ = linregress(t_thermo, PE)
            ax.text(0.97, 0.95,
                    f"Mean: {PE.mean():.1f} kcal/mol\n"
                    f"Drift: {slope:.3f} kcal/(mol·ns)",
                    transform=ax.transAxes, ha='right', va='top',
                    color=TEXT, fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=GRID, alpha=0.85))
        ax.set_xlabel('Time (ns)', fontsize=9)
        ax.set_ylabel('PE (kcal/mol)', fontsize=9)
        ax.legend(fontsize=7, facecolor='white', edgecolor=GRID, labelcolor=TEXT)
    else:
        ax.text(0.5, 0.5, "No thermo data", ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color=SUBTEXT)

    # ── Supertitle ───────────────────────────────────────────────────────
    fig.suptitle(
        f'{sim_name}  |  {n_frames} frames  '
        f'({time_ns[-1]:.0f} ns)  |  {n_beads}-mer ASO',
        color=TEXT, fontsize=13, fontweight='bold', y=0.97)

    plt.savefig(out_png, dpi=180, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved → {out_png}")


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE SIMULATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def run_single(dat_file, traj_file, thermo_file, out_dir="analysis"):
    """Run full 6-panel analysis on one simulation."""

    sim_name = os.path.splitext(os.path.basename(traj_file))[0].replace(".lammpstrj", "")
    print(f"\n{'='*70}")
    print(f"  ANALYZING: {sim_name}")
    print(f"{'='*70}")

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"fig_{sim_name}.png")

    # ── Parse topology ───────────────────────────────────────────────────
    print("  [1/5] Parsing topology ...")
    atom_info, mol_atoms, aso_beads, hp_beads, n_free = parse_dat(dat_file)
    aso_seq = [TYPE_TO_NUC[atom_info[a][1]] for a in mol_atoms[1]]

    # ── Parse trajectory ─────────────────────────────────────────────────
    print("  [2/5] Parsing trajectory ...")
    aso_traj, hp_traj, free_trajs, timesteps = parse_traj(traj_file, mol_atoms, n_free)
    n_frames = len(timesteps)

    if n_frames < 2:
        print(f"  SKIP: Only {n_frames} frames — not enough data")
        return

    # ── Parse thermo ─────────────────────────────────────────────────────
    print("  [3/5] Parsing thermo ...")
    thermo = None
    if thermo_file and os.path.isfile(thermo_file):
        thermo = parse_thermo(thermo_file)
        if thermo:
            print(f"    {len(thermo['timestep'])} thermo records")
    else:
        print(f"    No thermo file found ({thermo_file})")

    # ── Dynamics: RMSD & Rg ──────────────────────────────────────────────
    print("  [4/5] Computing RMSD & Rg ...")

    # Use frame 0 as reference (since we don't have the PDB for all variants)
    aso_ref = aso_traj[0].astype(np.float64)
    hp_ref = hp_traj[0].astype(np.float64)

    aso_rmsd = calc_rmsd_traj(aso_traj, aso_ref)
    hp_rmsd = calc_rmsd_traj(hp_traj, hp_ref)

    complex_all = np.concatenate([aso_traj, hp_traj], axis=1)
    rog_complex = radius_of_gyration(complex_all)
    rog_hp = radius_of_gyration(hp_traj)

    # ── Binding analysis ─────────────────────────────────────────────────
    print("  [5/5] Computing binding ...")
    min_dists, bound_frames, per_bead_prob, free_bound_counts = \
        calc_binding(aso_traj, hp_traj, free_trajs, CUTOFF)

    n_ever = (free_bound_counts > 0).sum() if len(free_bound_counts) > 0 else 0
    pct_bound = 100 * bound_frames.mean()
    print(f"    Docked bound: {bound_frames.sum()}/{n_frames} ({pct_bound:.1f}%)")
    print(f"    Free binders: {n_ever}/{n_free}")

    # ── Figure ───────────────────────────────────────────────────────────
    print("  Generating figure ...")
    make_figure(sim_name, timesteps, min_dists, bound_frames,
                per_bead_prob, free_bound_counts, aso_seq,
                aso_rmsd, hp_rmsd, rog_complex, rog_hp,
                thermo, out_png)

    # ── Save summary stats ───────────────────────────────────────────────
    summary_file = os.path.join(out_dir, f"summary_{sim_name}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Simulation: {sim_name}\n")
        f.write(f"Frames: {n_frames}\n")
        f.write(f"Duration: {(timesteps[-1] - timesteps[0]) * DT_FS / 1e6:.1f} ns\n")
        f.write(f"ASO length: {aso_beads} beads\n")
        f.write(f"ASO sequence: {' '.join(aso_seq)}\n")
        f.write(f"Free ASOs: {n_free}\n")
        f.write(f"Docked bound fraction: {pct_bound:.2f}%\n")
        f.write(f"Free ASOs ever bound: {n_ever}/{n_free}\n")
        f.write(f"HP RMSD: {hp_rmsd.mean():.2f} ± {hp_rmsd.std():.2f} Å\n")
        f.write(f"ASO RMSD: {aso_rmsd.mean():.2f} ± {aso_rmsd.std():.2f} Å\n")
        f.write(f"Complex Rg: {rog_complex.mean():.2f} ± {rog_complex.std():.2f} Å\n")
        f.write(f"Hairpin Rg: {rog_hp.mean():.2f} ± {rog_hp.std():.2f} Å\n")
        if thermo:
            f.write(f"Mean PE: {thermo['PE'].mean():.2f} kcal/mol\n")
            f.write(f"Mean T: {thermo['T'].mean():.2f} K\n")
        f.write(f"\nPer-bead contact prob: {np.array2string(per_bead_prob, precision=3)}\n")
        f.write(f"Per-bead ΔG: {np.array2string(-kT * np.log(np.clip(per_bead_prob, 1e-6, 1-1e-6) / (1 - np.clip(per_bead_prob, 1e-6, 1-1e-6))), precision=2)} kcal/mol\n")
        if len(free_bound_counts) > 0:
            f.write(f"Free ASO bound counts: {free_bound_counts[free_bound_counts > 0].tolist()}\n")
    print(f"  Summary → {summary_file}")

    return {
        "name": sim_name,
        "n_frames": n_frames,
        "duration_ns": (timesteps[-1] - timesteps[0]) * DT_FS / 1e6,
        "aso_beads": aso_beads,
        "docked_bound_pct": pct_bound,
        "free_binders": n_ever,
        "n_free": n_free,
        "hp_rmsd_mean": hp_rmsd.mean(),
        "aso_rmsd_mean": aso_rmsd.mean(),
        "rg_complex_mean": rog_complex.mean(),
        "rg_hp_mean": rog_hp.mean(),
        "mean_pe": thermo['PE'].mean() if thermo else np.nan,
    }


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-DISCOVERY OF ALL SIMULATIONS
# ═════════════════════════════════════════════════════════════════════════════
def discover_simulations(base_dir="."):
    """Find all matching (dat, traj, thermo) triplets."""
    config_dir = os.path.join(base_dir, "configs")
    output_dir = os.path.join(base_dir, "outputs")

    sims = []

    # Find all trajectory files
    traj_files = sorted(glob.glob(os.path.join(output_dir, "*.lammpstrj")))

    for traj_path in traj_files:
        traj_name = os.path.basename(traj_path).replace(".lammpstrj", "")

        # Match to config file
        dat_path = os.path.join(config_dir, traj_name + ".dat")
        if not os.path.isfile(dat_path):
            # Try without path prefix
            print(f"  WARNING: No config for {traj_name}, trying alternate names...")
            candidates = glob.glob(os.path.join(config_dir, f"*{traj_name}*.dat"))
            if candidates:
                dat_path = candidates[0]
            else:
                print(f"  SKIP: No config found for {traj_name}")
                continue

        # Match to thermo file (prefer _production, fall back to _averaged)
        thermo_path = os.path.join(output_dir, f"thermo_{traj_name}_production.dat")
        if not os.path.isfile(thermo_path):
            thermo_path = os.path.join(output_dir, f"thermo_{traj_name}_averaged.dat")
        if not os.path.isfile(thermo_path):
            thermo_path = None

        sims.append({
            "dat": dat_path,
            "traj": traj_path,
            "thermo": thermo_path,
            "name": traj_name,
        })

    return sims


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASO-Hairpin 6-Panel Analysis")
    parser.add_argument("--dat", help="Single .dat config file")
    parser.add_argument("--traj", help="Single .lammpstrj trajectory file")
    parser.add_argument("--thermo", help="Single thermo file")
    parser.add_argument("--base", default=".", help="Base directory for auto-discovery")
    parser.add_argument("--outdir", default="analysis", help="Output directory for figures")
    args = parser.parse_args()

    if args.dat and args.traj:
        # Single simulation mode
        run_single(args.dat, args.traj, args.thermo, args.outdir)
    else:
        # Auto-discovery mode
        print("="*70)
        print("  ASO-Hairpin Campaign Analysis — Auto-Discovery Mode")
        print("="*70)

        sims = discover_simulations(args.base)
        print(f"\nFound {len(sims)} simulations:\n")
        for s in sims:
            thermo_status = "✓" if s["thermo"] else "✗"
            print(f"  {s['name']:45s}  thermo: {thermo_status}")

        if not sims:
            print("\nNo simulations found! Check directory structure.")
            print("Expected: configs/*.dat  outputs/*.lammpstrj  outputs/thermo_*_production.dat")
            sys.exit(1)

        all_results = []
        for s in sims:
            try:
                result = run_single(s["dat"], s["traj"], s["thermo"], args.outdir)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"  ERROR on {s['name']}: {e}")
                import traceback
                traceback.print_exc()

        # ── Summary table ────────────────────────────────────────────────
        if all_results:
            print(f"\n{'='*90}")
            print(f"  CAMPAIGN SUMMARY — {len(all_results)} simulations analyzed")
            print(f"{'='*90}")
            print(f"{'Simulation':<42s} {'Frames':>6s} {'ns':>6s} {'Bound%':>7s} "
                  f"{'Free':>5s} {'HP RMSD':>8s} {'Rg':>7s} {'PE':>10s}")
            print("-"*90)
            for r in all_results:
                print(f"{r['name']:<42s} {r['n_frames']:>6d} {r['duration_ns']:>6.0f} "
                      f"{r['docked_bound_pct']:>6.1f}% "
                      f"{r['free_binders']:>3d}/{r['n_free']:<2d}"
                      f" {r['hp_rmsd_mean']:>7.2f} {r['rg_complex_mean']:>6.1f}"
                      f" {r['mean_pe']:>10.1f}")

            summary_csv = os.path.join(args.outdir, "campaign_summary.csv")
            with open(summary_csv, 'w') as f:
                f.write("simulation,frames,duration_ns,aso_beads,docked_bound_pct,"
                        "free_binders,n_free,hp_rmsd,aso_rmsd,rg_complex,rg_hp,mean_pe\n")
                for r in all_results:
                    f.write(f"{r['name']},{r['n_frames']},{r['duration_ns']:.1f},"
                            f"{r['aso_beads']},{r['docked_bound_pct']:.2f},"
                            f"{r['free_binders']},{r['n_free']},"
                            f"{r['hp_rmsd_mean']:.3f},{r['aso_rmsd_mean']:.3f},"
                            f"{r['rg_complex_mean']:.3f},{r['rg_hp_mean']:.3f},"
                            f"{r['mean_pe']:.3f}\n")
            print(f"\nSummary CSV → {summary_csv}")

        print("\nAll done.")
        