"""
Fabricated representative data for local previews.

The real ``.npz`` / ``.csv`` inputs live on Princeton's ``/scratch/gpfs``
filesystem and are not available on every machine. When a figure script
cannot find its data file, it imports helpers from this module to
generate statistically plausible surrogate data so the layout/style
changes can be reviewed without cluster access.

**Do not use this module to produce figures for the thesis writeup** —
always re-render on the cluster against the real data files. Every call
here is deterministic (``np.random.seed`` is set) so preview runs are
reproducible and side-effect-free.
"""
from __future__ import annotations
import numpy as np


SEED = 20260417
_RNG = np.random.default_rng(SEED)


def _rng():
    """Return a fresh RNG so multiple previews don't interfere."""
    return np.random.default_rng(SEED)


# ── Dynamics: RMSD / Rg / RMSF surrogate data (figs 1–4) ─────────────────────
def make_dynamics_results(n_frames: int = 3640) -> dict:
    """
    Emulate ``Week 8/analysis/dynamics_results.npz`` with reasonable
    magnitudes: hairpin RMSD ~ 28 Å, docked ASO RMSD ~ 4 Å, hairpin Rg
    ~ 29 Å, complex Rg ~ 30 Å (physically sensible — the real data
    currently shows an unwrapped-coords artefact at ~2300 Å; see
    dynamics fix in ``Single_Chain/Week 8/analysis``).

    ``ts`` is emitted in units of "millions of timesteps", matching the
    convention established in ``02_dynamics_analysis.py``
    (``ts = d["timesteps"] / 1e6``).  Fig scripts then multiply by 0.01
    to convert to µs (10 fs timestep, 100 Msteps = 1 µs).
    """
    rng = _rng()
    # 3.64 µs run at 10 fs/step sampled in 3640 frames → 364 Msteps total
    ts  = np.linspace(0, 364.0, n_frames)

    # Hairpin: strong conformational drift, ~28 Å with occasional basin hops.
    hp_rmsd = 28.0 + 2.0 * np.sin(np.linspace(0, 6 * np.pi, n_frames))
    hp_rmsd += rng.normal(0, 2.0, n_frames)
    # Sprinkle a few deep "refolding" dips
    for dip in rng.choice(n_frames, size=6, replace=False):
        w = rng.integers(20, 80)
        lo = max(0, dip - w); hi = min(n_frames, dip + w)
        hp_rmsd[lo:hi] -= 15 * np.exp(-((np.arange(lo, hi) - dip) / w) ** 2)

    # Docked ASO: small, tightly-bound fluctuations ~ 3.5 Å.
    aso_rmsd = 3.5 + 0.4 * np.sin(np.linspace(0, 12 * np.pi, n_frames))
    aso_rmsd += rng.normal(0, 0.35, n_frames)
    aso_rmsd = np.clip(aso_rmsd, 1.0, None)

    complex_rmsd = 0.5 * (hp_rmsd + aso_rmsd)

    # Rg — physically plausible after the nearest-image fix.
    rog_hp      = 29.0 + rng.normal(0, 0.25, n_frames)
    rog_complex = 30.2 + rng.normal(0, 0.35, n_frames) + \
                  0.3 * np.sin(np.linspace(0, 3 * np.pi, n_frames))

    # Per-bead RMSF for the docked ASO (10 beads).
    aso_rmsf = np.array([4.6, 2.4, 2.6, 2.7, 2.7, 2.6, 2.7, 2.5, 2.6, 4.6])

    # Per-bead internal RMSF for the free-ASO ensemble, AFTER per-ASO
    # Kabsch alignment onto each ASO's mean structure (the fix — previously
    # these values were 6–21 Å because rigid-body tumbling was not removed).
    # Internal bond fluctuations for a 10-mer CG ssRNA sit around 2–6 Å
    # with terminal beads a bit higher.
    free_rmsf_mean = np.array([5.0, 3.2, 2.9, 2.8, 2.7, 2.8, 2.9, 3.1, 3.4, 5.2])
    free_rmsf_std  = np.array([0.6, 0.45, 0.4, 0.35, 0.35, 0.4, 0.4, 0.45, 0.5, 0.7])
    z_score = (aso_rmsf - free_rmsf_mean) / free_rmsf_std

    return dict(
        ts=ts, hp_rmsd=hp_rmsd, aso_rmsd=aso_rmsd, complex_rmsd=complex_rmsd,
        rog_hp=rog_hp, rog_complex=rog_complex,
        aso_rmsf=aso_rmsf,
        free_rmsf_mean=free_rmsf_mean, free_rmsf_std=free_rmsf_std,
        z_score=z_score,
    )


# ── Parsed data: labels for axes (figs 3, 4, 5) ──────────────────────────────
def make_parsed_data() -> dict:
    aso_labels = np.array(
        ["1(G)", "2(G)", "3(G)", "4(C)", "5(U)",
         "6(G)", "7(U)", "8(U)", "9(U)", "10(U)"]
    )
    hp_seq = "GCGCGCAGAUUGGAUUGCGCUGCUAGCGCGCA"
    hp_labels = np.array([f"{i+11}({hp_seq[i]})" for i in range(32)])
    return dict(aso_labels=aso_labels, hp_labels=hp_labels)


# ── Contact results (figs 5, 6) ──────────────────────────────────────────────
def make_contact_results() -> dict:
    """
    Contact-probability map shaped like a V (ASO folded back onto itself
    against the loop) with realistic low magnitudes (~10⁻⁴) for the
    Week 8 high-loading NVE run.
    """
    rng = _rng()
    n_aso, n_hp = 10, 32

    ai = np.arange(n_aso)[:, None]
    hi = np.arange(n_hp)[None, :]
    # Two V-branches converging on hairpin bead 22
    d1 = np.abs(ai - (hi - 14))
    d2 = np.abs(ai - (30 - hi))
    prob10 = 8e-4 * np.exp(-np.minimum(d1, d2) ** 2 / 3.0)
    prob10 += rng.uniform(0, 1e-5, prob10.shape)

    tbp_names = np.array(["U5·A21", "U5·A31", "G6·C20", "G6·A32", "U10·A36"])
    tbp_probs = np.array([4.12e-4, 2.75e-4, 4.12e-4, 2.75e-4, 0.0])

    return dict(prob10=prob10, tbp_names=tbp_names, tbp_probs=tbp_probs)


# ── Binding results (fig 7) ──────────────────────────────────────────────────
def make_binding_results(n_frames: int = 3640, n_free: int = 99) -> dict:
    rng = _rng()
    ts = np.linspace(0, 364.0, n_frames)   # millions of timesteps

    # Only 11 of 99 ASOs ever contact the hairpin — mirror the real run.
    bind_prob = np.zeros(n_free)
    ever_bound = rng.choice(n_free, 11, replace=False)
    bind_prob[ever_bound] = rng.uniform(1e-4, 3.5e-4, size=11)

    # Distance-to-hairpin trace: mostly far (~150 Å), occasional dips.
    dist = 150 + 40 * rng.standard_normal((n_free, n_frames))
    dist = np.clip(dist, 10, 220)
    for mi in ever_bound:
        n_events = rng.integers(1, 4)
        for _ in range(n_events):
            centre = rng.integers(100, n_frames - 100)
            w = rng.integers(30, 120)
            lo, hi = max(0, centre - w), min(n_frames, centre + w)
            dist[mi, lo:hi] = np.minimum(
                dist[mi, lo:hi],
                15 + rng.uniform(0, 5, hi - lo)
            )

    return dict(bind_prob=bind_prob, dist_to_hp=dist, ts=ts)


# ── Kd and loop-RMSF campaign (figs 8, 9) ────────────────────────────────────
def make_kd_summary() -> list[dict]:
    # Rows mirror the real CSV header used by the fig8 script.
    return [
        dict(name="25ASO_unmod",                      label="",
             kd_main_mM=135,  kd_std_across_cutoffs_mM=110),
        dict(name="100ASO_truncated_unmodified",      label="",
             kd_main_mM=243,  kd_std_across_cutoffs_mM=230),
        dict(name="50ASO_unmod",                      label="",
             kd_main_mM=290,  kd_std_across_cutoffs_mM=250),
        dict(name="100ASO_unmodified_AAtoCC",         label="",
             kd_main_mM=322,  kd_std_across_cutoffs_mM=300),
        dict(name="200ASO_unmod",                     label="",
             kd_main_mM=350,  kd_std_across_cutoffs_mM=230),
        dict(name="100ASO_unmodified_loop_GG_to_AA",  label="",
             kd_main_mM=355,  kd_std_across_cutoffs_mM=210),
        dict(name="100ASO_mismatch_G6A_unmodified",   label="",
             kd_main_mM=535,  kd_std_across_cutoffs_mM=390),
        dict(name="100ASO_all_purine_unmodified",     label="",
             kd_main_mM=559,  kd_std_across_cutoffs_mM=390),
        dict(name="100ASO_scrambled_unmodified",      label="",
             kd_main_mM=741,  kd_std_across_cutoffs_mM=570),
        dict(name="100ASO_mismatch_U5C_unmodified",   label="",
             kd_main_mM=853,  kd_std_across_cutoffs_mM=610),
        dict(name="100ASO_extended_12mer_unmodified", label="",
             kd_main_mM=1219, kd_std_across_cutoffs_mM=980),
        dict(name="100ASO_extended_14mer_unmodified", label="",
             kd_main_mM=1828, kd_std_across_cutoffs_mM=1100),
    ]


def make_loop_rmsf_summary() -> list[dict]:
    return [
        dict(name="100ASO_unmod",                       loop_mean_rmsf_A=31.5, loop_max_rmsf_A=33.2),
        dict(name="100ASO_scrambled_unmodified",        loop_mean_rmsf_A=30.4, loop_max_rmsf_A=32.1),
        dict(name="25ASO_unmod",                        loop_mean_rmsf_A=29.0, loop_max_rmsf_A=30.5),
        dict(name="100ASO_unmodified_AAtoCC",           loop_mean_rmsf_A=28.7, loop_max_rmsf_A=30.2),
        dict(name="100ASO_mismatch_U5C_unmodified",     loop_mean_rmsf_A=28.1, loop_max_rmsf_A=29.7),
        dict(name="50ASO_unmod",                        loop_mean_rmsf_A=26.9, loop_max_rmsf_A=28.2),
        dict(name="100ASO_mismatch_G6A_unmodified",     loop_mean_rmsf_A=26.4, loop_max_rmsf_A=27.9),
        dict(name="100ASO_truncated_unmodified",        loop_mean_rmsf_A=26.2, loop_max_rmsf_A=27.6),
        dict(name="100ASO_all_purine_unmodified",       loop_mean_rmsf_A=24.7, loop_max_rmsf_A=26.1),
        dict(name="200ASO_unmod",                       loop_mean_rmsf_A=24.6, loop_max_rmsf_A=25.9),
        dict(name="100ASO_extended_12mer_unmodified",   loop_mean_rmsf_A=24.0, loop_max_rmsf_A=25.4),
        dict(name="100ASO_extended_14mer_unmodified",   loop_mean_rmsf_A=23.7, loop_max_rmsf_A=25.1),
        dict(name="100ASO_unmodified_loop_GG_to_AA",    loop_mean_rmsf_A=23.2, loop_max_rmsf_A=24.6),
    ]


# ── Safe loader: real data if present, fabricated data otherwise ─────────────
def load_or_fake(real_path: str, faker):
    """
    Load ``real_path`` with :func:`numpy.load` if the file exists,
    otherwise call ``faker()`` and emit a warning banner to stderr so
    nobody ships a preview-data figure into the thesis by accident.
    """
    import os, sys
    if os.path.isfile(real_path):
        return np.load(real_path, allow_pickle=True)
    print(f"  [PREVIEW] {real_path} not found — using fabricated data.",
          file=sys.stderr)
    return faker()
