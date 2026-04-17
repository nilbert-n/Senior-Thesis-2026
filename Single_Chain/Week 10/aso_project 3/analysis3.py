#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Two-Tier Kd Analysis for Multi-ASO Systems
═══════════════════════════════════════════════════════════════════════════════

Single simulation mode:
    python kd_two_tier.py --dat configs/25ASO_unmod.dat \
        --traj outputs/25ASO_unmod.lammpstrj

Auto-discovery mode:
    python kd_two_tier.py --base . --outdir kd_analysis

Directory layout expected (from aso_project/):
    configs/*.dat
    outputs/*.lammpstrj

Tier A — Encounter / Pocket Occupancy:
  Default rule for 10-mers: ≥ 8/10 ASO beads within 16.8 Å of any hairpin bead.
  For non-10-mer ASOs discovered in batch mode, this scales to ceil(0.8 * ASO length)
  unless explicitly overridden.

Tier B — Successful Bound State:
  Default rule: all ASO beads within 16.8 Å
  + ASO–hairpin COM distance ≤ COM_CUTOFF
  + ≥ MIN_CONSEC consecutive frames (dwell filter)

Multi-ligand Kd formula (N_aso ligands, 1 hairpin):
  Kd = (1 - f)(N_aso - f) / (Na · V · f)

where f = fraction of frames with at least one successfully bound ASO.

IMPORTANT GEOMETRY NOTE:
  This version uses minimum-image periodic distances for intermolecular geometry.
  That matters because the trajectory uses xu/yu/zu (unwrapped coordinates), and
  raw Euclidean subtraction can badly overestimate distances across box boundaries.
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# ── Parameters ───────────────────────────────────────────────────────────
DEFAULT_BEAD_CUTOFF = 16.8
DEFAULT_COM_CUTOFFS = [10, 12, 15, 20]
DEFAULT_MIN_CONSEC = 1
DEFAULT_ENCOUNTER_BEADS_10MER = 8
DEFAULT_BINDING_BEADS_10MER = 10

Na = 6.022e23
T = 300.0
DT_FS = 10
TYPE_TO_NUC = {1: "A", 2: "C", 3: "G", 4: "U"}


# ═════════════════════════════════════════════════════════════════════════════
# PARSING
# ═════════════════════════════════════════════════════════════════════════════
def parse_dat(fname):
    with open(fname) as f:
        lines = f.readlines()

    n_atoms = int(next(l for l in lines if l.strip().endswith("atoms")).split()[0])
    start = next(i for i, l in enumerate(lines) if l.strip() == "Atoms") + 2

    atom_info = {}
    mol_atoms = defaultdict(list)

    xlo = xhi = ylo = yhi = zlo = zhi = None

    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            if parts[2:] == ["xlo", "xhi"]:
                xlo, xhi = float(parts[0]), float(parts[1])
            elif parts[2:] == ["ylo", "yhi"]:
                ylo, yhi = float(parts[0]), float(parts[1])
            elif parts[2:] == ["zlo", "zhi"]:
                zlo, zhi = float(parts[0]), float(parts[1])

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError(f"Could not parse box bounds from {fname}")

    for line in lines[start:start + n_atoms]:
        p = line.split()
        if len(p) < 7:
            continue
        aid, mid, atype = int(p[0]), int(p[1]), int(p[2])
        atom_info[aid] = (mid, atype)
        mol_atoms[mid].append(aid)

    for mid in mol_atoms:
        mol_atoms[mid].sort()

    box_lengths = np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=float)
    box_bounds = {
        "xlo": xlo, "xhi": xhi,
        "ylo": ylo, "yhi": yhi,
        "zlo": zlo, "zhi": zhi,
    }

    return atom_info, dict(mol_atoms), box_lengths, box_bounds


def parse_traj(fname, mol_atoms, n_aso_total):
    hp_aids = mol_atoms[2]
    all_aso_mids = [1] + list(range(3, 3 + n_aso_total - 1))  # docked + free
    all_aso_aids = [mol_atoms[m] for m in all_aso_mids]

    hp_frames, aso_frames_all = [], [[] for _ in range(len(all_aso_mids))]
    timesteps = []

    print(f"  Reading trajectory: {fname}")
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
            f.readline()
            f.readline()
            f.readline()
            f.readline()  # ATOMS header

            coords = {}
            for _ in range(n):
                p = f.readline().split()
                coords[int(p[0])] = np.array([float(p[4]), float(p[5]), float(p[6])], dtype=float)

            timesteps.append(ts)
            hp_frames.append(np.array([coords[a] for a in hp_aids], dtype=float))
            for i, aids in enumerate(all_aso_aids):
                aso_frames_all[i].append(np.array([coords[a] for a in aids], dtype=float))

            if len(timesteps) % 500 == 0:
                print(f"    ... {len(timesteps)} frames")

    if not timesteps:
        raise ValueError(f"No frames parsed from trajectory: {fname}")

    print(f"    Done: {len(timesteps)} frames total, {timesteps[0]} → {timesteps[-1]} steps")

    hp_traj = np.array(hp_frames, dtype=np.float32)
    aso_trajs = np.array([np.array(af, dtype=np.float32) for af in aso_frames_all])
    return hp_traj, aso_trajs, np.array(timesteps), all_aso_mids


# ═════════════════════════════════════════════════════════════════════════════
# PBC GEOMETRY HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def minimum_image_displacements(diff, box_lengths):
    """
    Apply minimum-image convention to displacement vectors.

    diff can be shape (..., 3)
    box_lengths should be shape (3,)
    """
    box = np.asarray(box_lengths, dtype=float)
    return diff - box * np.round(diff / box)


def pairwise_pbc_distances(A, B, box_lengths):
    """
    Pairwise minimum-image distances between point sets A and B.

    A: shape (nA, 3)
    B: shape (nB, 3)
    returns: shape (nA, nB)
    """
    diff = A[:, None, :] - B[None, :, :]
    diff = minimum_image_displacements(diff, box_lengths)
    return np.linalg.norm(diff, axis=2)


def com_distance_pbc(A, B, box_lengths):
    """
    Minimum-image distance between COM(A) and COM(B).
    A and B are arrays of shape (n, 3).
    """
    com_A = A.mean(axis=0)
    com_B = B.mean(axis=0)
    disp = com_A - com_B
    disp = minimum_image_displacements(disp, box_lengths)
    return np.linalg.norm(disp)


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def scaled_encounter_beads(aso_beads, override=None):
    if override is not None:
        return int(override)
    if aso_beads == DEFAULT_BINDING_BEADS_10MER:
        return DEFAULT_ENCOUNTER_BEADS_10MER
    return max(1, math.ceil(0.8 * aso_beads))


def scaled_binding_beads(aso_beads, override=None):
    if override is not None:
        return int(override)
    if aso_beads == DEFAULT_BINDING_BEADS_10MER:
        return DEFAULT_BINDING_BEADS_10MER
    return aso_beads


def apply_dwell_filter(bound_mask, min_consec):
    """Only keep stretches of ≥ min_consec consecutive True values."""
    filtered = np.zeros_like(bound_mask, dtype=bool)
    count, start = 0, 0
    for i in range(len(bound_mask)):
        if bound_mask[i]:
            if count == 0:
                start = i
            count += 1
        else:
            if count >= min_consec:
                filtered[start:start + count] = True
            count = 0
    if count >= min_consec:
        filtered[start:start + count] = True
    return filtered


def compute_kd_multi(f_bound, n_aso, box_lengths):
    """
    Multi-ligand Kd: Kd = (1-f)(N-f) / (Na·V·f)
    f = fraction of frames with at least one successfully bound ASO
    N = total number of ASO ligands
    """
    if f_bound < 1e-10:
        return float("inf")
    lx, ly, lz = box_lengths
    v_liters = lx * ly * lz * 1e-27
    kd = (1 - f_bound) * (n_aso - f_bound) / (Na * v_liters * f_bound)
    return kd


def discover_simulations(base_dir="."):
    """Find matching (dat, traj) pairs using the same directory layout as run_all_analysis.py."""
    config_dir = os.path.join(base_dir, "configs")
    output_dir = os.path.join(base_dir, "outputs")
    sims = []

    traj_files = sorted(glob.glob(os.path.join(output_dir, "*.lammpstrj")))
    for traj_path in traj_files:
        traj_name = os.path.basename(traj_path).replace(".lammpstrj", "")
        dat_path = os.path.join(config_dir, traj_name + ".dat")

        if not os.path.isfile(dat_path):
            candidates = glob.glob(os.path.join(config_dir, f"*{traj_name}*.dat"))
            if candidates:
                dat_path = candidates[0]
            else:
                print(f"  SKIP: No config found for {traj_name}")
                continue

        sims.append({
            "name": traj_name,
            "dat": dat_path,
            "traj": traj_path,
        })

    return sims


# ═════════════════════════════════════════════════════════════════════════════
# CORE TWO-TIER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def analyze_two_tier(
    dat_file,
    traj_file,
    out_dir,
    bead_cutoff=DEFAULT_BEAD_CUTOFF,
    com_cutoffs=None,
    min_consec=DEFAULT_MIN_CONSEC,
    encounter_beads_override=None,
    binding_beads_override=None,
):
    com_cutoffs = list(com_cutoffs or DEFAULT_COM_CUTOFFS)
    sim_name = os.path.splitext(os.path.basename(traj_file))[0].replace(".lammpstrj", "")

    print("=" * 70)
    print(f"  TWO-TIER Kd ANALYSIS: {sim_name}")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    out_fig = os.path.join(out_dir, f"Kd_two_tier_{sim_name}.png")
    out_summary = os.path.join(out_dir, f"Kd_two_tier_{sim_name}.txt")

    print("\n[1/4] Parsing topology...")
    atom_info, mol_atoms, box_lengths, box_bounds = parse_dat(dat_file)
    n_mols = len(mol_atoms)

    if 1 not in mol_atoms or 2 not in mol_atoms:
        raise ValueError("Expected molecule 1 = ASO and molecule 2 = hairpin.")

    aso_beads = len(mol_atoms[1])
    n_free = n_mols - 2
    n_aso_total = n_free + 1  # docked + free
    aso_seq = [TYPE_TO_NUC.get(atom_info[a][1], str(atom_info[a][1])) for a in mol_atoms[1]]

    lx, ly, lz = box_lengths
    v_liters = lx * ly * lz * 1e-27
    conc_per_mol = 1 / (Na * v_liters)

    encounter_beads = scaled_encounter_beads(aso_beads, encounter_beads_override)
    binding_beads = scaled_binding_beads(aso_beads, binding_beads_override)

    print(f"  {n_aso_total} ASOs ({aso_beads}-mer), 1 hairpin")
    print(f"  Box lengths: Lx={lx:.1f} Å, Ly={ly:.1f} Å, Lz={lz:.1f} Å")
    print(f"  Volume: {v_liters:.2e} L")
    print(f"  1 molecule = {conc_per_mol * 1e3:.2f} mM")
    print(f"  Tier A threshold: ≥{encounter_beads}/{aso_beads} beads within {bead_cutoff} Å")
    print(f"  Tier B threshold: {binding_beads}/{aso_beads} beads within {bead_cutoff} Å")

    print("\n[2/4] Parsing trajectory...")
    hp_traj, aso_trajs, timesteps, aso_mids = parse_traj(traj_file, mol_atoms, n_aso_total)
    n_frames = len(timesteps)
    if n_frames < 2:
        raise ValueError(f"Only {n_frames} frame(s) found; need at least 2.")
    time_ns = (timesteps - timesteps[0]) * DT_FS / 1e6

    print("\n[3/4] Computing two-tier binding analysis with PBC distances...")
    n_aso = len(aso_mids)
    beads_in_range = np.zeros((n_aso, n_frames), dtype=int)
    com_distances = np.zeros((n_aso, n_frames), dtype=float)

    print("  Computing per-ASO, per-frame distances...")
    for ai in range(n_aso):
        for f in range(n_frames):
            aso_xyz = aso_trajs[ai, f].astype(float)
            hp_xyz = hp_traj[f].astype(float)

            dmat = pairwise_pbc_distances(aso_xyz, hp_xyz, box_lengths)
            beads_in_range[ai, f] = (dmat.min(axis=1) < bead_cutoff).sum()

            com_distances[ai, f] = com_distance_pbc(aso_xyz, hp_xyz, box_lengths)

        if (ai + 1) % 5 == 0 or ai == n_aso - 1:
            print(f"    {ai + 1}/{n_aso} ASOs done")

    # ── Tier A ──────────────────────────────────────────────────────────
    print("\n  ── TIER A: Encounter Occupancy ──")
    encounter_mask = beads_in_range >= encounter_beads
    n_in_pocket_per_frame = encounter_mask.sum(axis=0)
    any_in_pocket = n_in_pocket_per_frame > 0

    encounter_frac = any_in_pocket.mean()
    mean_occupancy = n_in_pocket_per_frame.mean()
    max_occupancy = int(n_in_pocket_per_frame.max())

    print(f"    Frames with any ASO in pocket: {any_in_pocket.sum()}/{n_frames} ({encounter_frac * 100:.1f}%)")
    print(f"    Mean ASOs in pocket/frame: {mean_occupancy:.2f}")
    print(f"    Max simultaneous: {max_occupancy}")

    # ── Tier B ──────────────────────────────────────────────────────────
    print("\n  ── TIER B: Successful Binding ──")
    tier_b_results = {}

    for com_cut in com_cutoffs:
        bound_raw = (beads_in_range >= binding_beads) & (com_distances <= com_cut)
        any_bound_per_frame = np.zeros(n_frames, dtype=bool)
        per_aso_bound_frac = np.zeros(n_aso, dtype=float)
        per_aso_events = np.zeros(n_aso, dtype=int)
        per_aso_dwell_times = []

        for ai in range(n_aso):
            filtered = apply_dwell_filter(bound_raw[ai], min_consec)
            any_bound_per_frame |= filtered
            per_aso_bound_frac[ai] = filtered.mean()

            in_event = False
            dwell = 0
            events = 0
            dwells = []
            for f in range(n_frames):
                if filtered[f]:
                    dwell += 1
                    if not in_event:
                        in_event = True
                        events += 1
                else:
                    if in_event:
                        dwells.append(dwell)
                        dwell = 0
                        in_event = False
            if in_event:
                dwells.append(dwell)

            per_aso_events[ai] = events
            per_aso_dwell_times.append(dwells)

        f_site = any_bound_per_frame.mean()
        kd = compute_kd_multi(f_site, n_aso_total, box_lengths)
        n_binders = int((per_aso_bound_frac > 0).sum())
        all_dwells = [d for dwells in per_aso_dwell_times for d in dwells]
        mean_dwell = float(np.mean(all_dwells)) if all_dwells else 0.0

        tier_b_results[com_cut] = {
            "f_site": f_site,
            "Kd": kd,
            "n_binders": n_binders,
            "n_events": int(per_aso_events.sum()),
            "mean_dwell": mean_dwell,
            "all_dwells": all_dwells,
            "per_aso_bound_frac": per_aso_bound_frac,
            "any_bound_per_frame": any_bound_per_frame,
        }

        print(f"\n    COM ≤ {com_cut} Å + {binding_beads}/{aso_beads} beads + dwell ≥ {min_consec}:")
        print(f"      f_site = {f_site:.4f} ({any_bound_per_frame.sum()}/{n_frames} frames)")
        print(f"      Binders: {n_binders}/{n_aso}")
        print(f"      Events: {per_aso_events.sum()}, mean dwell: {mean_dwell:.1f} frames")
        print(f"      Kd = {kd:.4f} M = {kd * 1e3:.1f} mM = {kd * 1e6:.0f} µM")

    com_vals = sorted(tier_b_results.keys())

    
    print("\n[4/4] Generating figure...")
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    gs = gridspec.GridSpec(
        2, 2,
        hspace=0.38, wspace=0.28,
        left=0.06, right=0.97, top=0.90, bottom=0.07
    )

    main_com = 12 if 12 in tier_b_results else sorted(tier_b_results.keys())[len(tier_b_results) // 2]
    res = tier_b_results[main_com]

    # ── A: Encounter over time — smoothed line curve ──────────────────────
    ax = fig.add_subplot(gs[0, 0])

    # Rolling-average window: ~2% of total frames, minimum 5
    _win = max(5, n_frames // 50)
    _kernel = np.ones(_win) / _win
    smoothed_occupancy = np.convolve(
        n_in_pocket_per_frame.astype(float), _kernel, mode="same"
    )

    ax.plot(time_ns, smoothed_occupancy,
            color="#1565C0", lw=2.2, label="occupancy")
    ax.axhline(
        mean_occupancy,
        color="#E91E63",
        ls="--",
        lw=1.5,
        label=f"Mean = {mean_occupancy:.2f} ASOs"
    )
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("ASOs in Pocket", fontsize=11)
    ax.set_title("Encounter", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    ax.text(
        0.02, 0.95,
        f"{encounter_frac * 100:.1f}% occupied",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", edgecolor="#1565C0")
    )

    # B: Binding success over time
    ax = fig.add_subplot(gs[0, 1])
    bound_mask_plot = res["any_bound_per_frame"].astype(float)
    ax.fill_between(
        time_ns, 0, bound_mask_plot,
        color="#4CAF50", alpha=0.5, step="mid",
        label=f"Site occupied (f = {res['f_site']:.4f})"
    )
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("Bound State (0/1)", fontsize=11)
    ax.set_ylim(-0.1, 1.3)
    ax.set_title("Binding Success", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.95,
        f"Kd = {res['Kd'] * 1e3:.1f} mM\n"
        f"{res['n_events']} events\n"
        f"mean dwell = {res['mean_dwell']:.0f} frames",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#4CAF50")
    )

    # C: Dwell times
    ax = fig.add_subplot(gs[1, 0])
    all_dwells = res["all_dwells"]
    if all_dwells:
        dwell_ns = np.array(all_dwells) * (timesteps[1] - timesteps[0]) * DT_FS / 1e6
        ax.hist(
            dwell_ns,
            bins=max(5, len(dwell_ns) // 3 + 1),
            color="#9C27B0",
            edgecolor="white",
            alpha=0.8
        )
        ax.axvline(
            dwell_ns.mean(),
            color="k",
            ls="--",
            lw=1.5,
            label=f"Mean = {dwell_ns.mean():.1f} ns"
        )
        ax.set_xlabel("Dwell Time (ns)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)
    else:
        ax.text(
            0.5, 0.5,
            "No binding events",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=12, color="gray"
        )
    ax.set_title("Dwell Times", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

    # Summary panel
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    summary = f"""ANALYSIS SUMMARY
{'─' * 44}

Simulation: {sim_name}
System:     {n_aso_total} ASOs + 1 Hairpin
Sequence:   {' '.join(aso_seq)}
Box:        {lx:.0f} × {ly:.0f} × {lz:.0f} Å³,  V = {v_liters:.2e} L
T:          {T:.0f} K
Frames:     {n_frames}
Duration:   {time_ns[-1]:.1f} ns

TIER A — ENCOUNTER
  Definition: ≥{encounter_beads}/{aso_beads} beads within {bead_cutoff} Å
  Occupied:   {encounter_frac * 100:.1f}% of frames
  Mean ASOs:  {mean_occupancy:.2f}/frame
  Max ASOs:   {max_occupancy}

TIER B — SUCCESSFUL BINDING  (COM ≤ {main_com} Å)
  Definition: {binding_beads}/{aso_beads} beads + COM ≤ {main_com} Å
              + dwell ≥ {min_consec} consecutive frames
  f:     {res['f_site']:.4f}

  Formula: Kd = (1−f)(N−f) / (Na·V·f)
           N = {n_aso_total} ASOs

  Kd = {res['Kd']:.4f}
  Binding Count = {res['n_events']}
  Mean dwell = {res['mean_dwell']:.1f} frames
"""
    ax.text(
        0.02, 0.98, summary,
        transform=ax.transAxes,
        fontsize=8.8,
        fontfamily="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#E0E0E0")
    )

    fig.suptitle(
        f"Two-Tier Kd Analysis  |  {sim_name}  |  {n_aso_total} ASOs + 1 Hairpin",
        fontsize=13,
        fontweight="bold"
    )
    plt.savefig(out_fig, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nFigure saved → {out_fig}")

    with open(out_summary, "w") as f:
        f.write(f"Simulation: {sim_name}\n")
        f.write(f"DAT: {dat_file}\n")
        f.write(f"TRAJ: {traj_file}\n")
        f.write(f"Frames: {n_frames}\n")
        f.write(f"Duration_ns: {time_ns[-1]:.4f}\n")
        f.write(f"ASO_length: {aso_beads}\n")
        f.write(f"ASO_sequence: {' '.join(aso_seq)}\n")
        f.write(f"Total_ASOs: {n_aso_total}\n")
        f.write(f"Box_lengths_A: {lx:.6f}, {ly:.6f}, {lz:.6f}\n")
        f.write(f"Volume_L: {v_liters:.8g}\n")
        f.write(f"Encounter_threshold_beads: {encounter_beads}\n")
        f.write(f"Binding_threshold_beads: {binding_beads}\n")
        f.write(f"Encounter_fraction: {encounter_frac:.6f}\n")
        f.write(f"Mean_occupancy: {mean_occupancy:.6f}\n")
        f.write(f"Max_occupancy: {max_occupancy}\n")
        for com_cut in com_vals:
            row = tier_b_results[com_cut]
            f.write(
                f"COM_cutoff_{com_cut}_A: f_site={row['f_site']:.6f}, "
                f"Kd_M={row['Kd']:.8g}, "
                f"n_binders={row['n_binders']}, n_events={row['n_events']}, "
                f"mean_dwell_frames={row['mean_dwell']:.6f}\n"
            )
    print(f"Summary saved → {out_summary}")
    print("Done!\n")

    result = {
        "name": sim_name,
        "dat": dat_file,
        "traj": traj_file,
        "frames": n_frames,
        "duration_ns": float(time_ns[-1]),
        "aso_beads": aso_beads,
        "n_aso_total": n_aso_total,
        "box_lx_A": float(lx),
        "box_ly_A": float(ly),
        "box_lz_A": float(lz),
        "volume_L": float(v_liters),
        "encounter_beads": encounter_beads,
        "binding_beads": binding_beads,
        "encounter_frac": float(encounter_frac),
        "mean_occupancy": float(mean_occupancy),
        "max_occupancy": max_occupancy,
        "figure": out_fig,
        "summary": out_summary,
    }
    for com_cut in com_vals:
        row = tier_b_results[com_cut]
        result[f"f_site_com_{com_cut}"] = float(row["f_site"])
        result[f"kd_molar_com_{com_cut}"] = float(row["Kd"])
        result[f"n_events_com_{com_cut}"] = int(row["n_events"])
        result[f"n_binders_com_{com_cut}"] = int(row["n_binders"])
        result[f"mean_dwell_frames_com_{com_cut}"] = float(row["mean_dwell"])
    return result


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Two-tier Kd analysis for ASO-hairpin simulations")
    parser.add_argument("--dat", help="Single .dat config file")
    parser.add_argument("--traj", help="Single .lammpstrj trajectory file")
    parser.add_argument("--base", default=".", help="Base directory for auto-discovery mode")
    parser.add_argument("--outdir", default="kd_analysis", help="Output directory")
    parser.add_argument("--bead-cutoff", type=float, default=DEFAULT_BEAD_CUTOFF,
                        help="Distance cutoff in Å for bead-pocket contact")
    parser.add_argument("--com-cutoffs", type=float, nargs="+", default=DEFAULT_COM_CUTOFFS,
                        help="One or more COM cutoff values in Å")
    parser.add_argument("--min-consec", type=int, default=DEFAULT_MIN_CONSEC,
                        help="Minimum consecutive frames for dwell filter")
    parser.add_argument("--encounter-beads", type=int, default=None,
                        help="Override Tier A required bead count")
    parser.add_argument("--binding-beads", type=int, default=None,
                        help="Override Tier B required bead count")
    return parser.parse_args()


def write_campaign_summary(outdir, results, com_cutoffs):
    if not results:
        return None

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "kd_two_tier_campaign_summary.csv")

    fieldnames = [
        "name", "dat", "traj", "frames", "duration_ns", "aso_beads", "n_aso_total",
        "box_lx_A", "box_ly_A", "box_lz_A", "volume_L",
        "encounter_beads", "binding_beads", "encounter_frac", "mean_occupancy",
        "max_occupancy", "figure", "summary"
    ]
    for com_cut in sorted(com_cutoffs):
        fieldnames.extend([
            f"f_site_com_{com_cut}",
            f"kd_molar_com_{com_cut}",
            f"n_events_com_{com_cut}",
            f"n_binders_com_{com_cut}",
            f"mean_dwell_frames_com_{com_cut}",
        ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return csv_path


def main():
    args = parse_args()

    if args.dat and args.traj:
        analyze_two_tier(
            dat_file=args.dat,
            traj_file=args.traj,
            out_dir=args.outdir,
            bead_cutoff=args.bead_cutoff,
            com_cutoffs=args.com_cutoffs,
            min_consec=args.min_consec,
            encounter_beads_override=args.encounter_beads,
            binding_beads_override=args.binding_beads,
        )
        return

    print("=" * 70)
    print("  TWO-TIER Kd ANALYSIS — Auto-Discovery Mode")
    print("=" * 70)
    sims = discover_simulations(args.base)
    print(f"\nFound {len(sims)} simulations.\n")
    for sim in sims:
        print(f"  {sim['name']}")

    if not sims:
        print("\nNo simulations found. Expected: configs/*.dat and outputs/*.lammpstrj")
        raise SystemExit(1)

    results = []
    for sim in sims:
        try:
            result = analyze_two_tier(
                dat_file=sim["dat"],
                traj_file=sim["traj"],
                out_dir=args.outdir,
                bead_cutoff=args.bead_cutoff,
                com_cutoffs=args.com_cutoffs,
                min_consec=args.min_consec,
                encounter_beads_override=args.encounter_beads,
                binding_beads_override=args.binding_beads,
            )
            results.append(result)
        except Exception as exc:
            print(f"  ERROR on {sim['name']}: {exc}")
            import traceback
            traceback.print_exc()

    csv_path = write_campaign_summary(args.outdir, results, args.com_cutoffs)
    if csv_path:
        print("=" * 70)
        print(f"Campaign summary saved → {csv_path}")
        print("=" * 70)


if __name__ == "__main__":
    main()