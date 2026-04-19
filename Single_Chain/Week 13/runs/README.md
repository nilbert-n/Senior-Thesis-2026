# Week 13 Runs — Simulation Scaffolding

Scaffolding generated from `_generate_scaffolding.py`.
Do not edit that script's output files directly — edit the generator
and re-run if structural changes are needed across all targets.

## Per-target folders

Each folder contains:

```
<target>/
├── inputs/
│   ├── <target>.dat                  CG data file (copied from outputs/<target>/)
│   ├── NVT_<target>_target.in        Target-only LAMMPS input  ← EDIT before submit
│   └── NVT_<target>_aso_bind.in      ASO-binding TEMPLATE      ← NOT ready to run
├── outputs/                          Empty — populated at runtime by LAMMPS
├── analysis/                         Empty — add analysis scripts here after run
├── slurm/
│   ├── run_<target>_target.sh        SLURM job for target-only run  ← EDIT seeds
│   └── run_<target>_aso_bind.sh      SLURM job for ASO run          ← NOT ready
└── notes/
    └── <target>_notes.txt            Caveats, TODOs, file inventory
```

## Status

| Target | .dat | target .in | aso .in | target .sh | aso .sh | Ready? |
|--------|------|-----------|---------|-----------|---------|--------|
| 1ANR | ✓ | ✓ edit seeds | template | ✓ | template | after seeds |
| 1E95 | ✓ | ✓ edit seeds | template | ✓ | template | after seeds |
| 1P5O | ✓ | ✓ edit seeds | template | ✓ | template | after seeds |
| 1RNK | ✓ | ✓ edit seeds | template | ✓ | template | after seeds |
| 1XJR | ✓ | ✓ edit seeds | template | ✓ | template | after seeds |
| 7LYJ | ✓ | ✓ edit seeds | template | ✓ | template | after seeds |

## Immediate manual TODOs (required before first submission)

For each target, open `inputs/NVT_<target>_target.in` and:
1. Replace `seed1` / `seed2` values with fresh random integers
2. Confirm `T_target` (default 300 K)
3. Optionally add temperature variants (copy .in + .sh, change tag and T)

Then submit:
```bash
cd /scratch/gpfs/JERELLE/nilbert/Single_Chain/Week\ 13/runs/<target>
sbatch slurm/run_<target>_target.sh
```

## Future ASO-binding (not yet actionable)

Each `NVT_<target>_aso_bind.in` is clearly labelled NOT READY.
The remaining manual work before any ASO run:
- Design ASO sequence(s) for the target
- Build combined RNA+ASO coarse-grained data file using:
  `scripts/week13/build_target_aso_dat.py`

  Combined .dat convention (matches Week 9/10 generate_all_configs.py):
  - mol 1 = ASO  (N beads, same 4 atom types A=1 C=2 G=3 U=4)
  - mol 2 = RNA target  (from prepared .dat)
  - mol 3..N = free ASOs if --n-aso > 1

- No new pair_coeff lines needed for unmodified ASO (same 4 types)
- Box convention: ±80 Å (Week 9/10 standard), not the ±500.6 Å target-only default
- Only then remove the NOT READY warning and submit
