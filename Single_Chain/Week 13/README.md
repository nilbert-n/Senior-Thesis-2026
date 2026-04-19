# Week 13 — Multi-Target RNA Simulation Preparation

## Overview

Week 13 extends the thesis simulation pipeline from the single 1YMO target
(Weeks 8–11) to a panel of 6 diverse RNA structures, with the goal of running
target-only equilibrations followed by ASO-binding simulations.

## Active Targets

| Target | Description | nt | Seq (5'→3') |
|--------|-------------|-----|-------------|
| 1ANR | HIV-1 TAR RNA | 29 | GGCAGAUCUGAGCCUGGGAGCUCUCUGCC |
| 1E95 | SRV-1 frameshifting pseudoknot | 36 | GCGGCCAGCUCCAGGCCGCCAAACAAUAUGGAGCAC |
| 1P5O | HCV IRES domain II | 77 | GGCUGUGAGG...GCCUCCAGCC |
| 1RNK | -1 frameshifting pseudoknot | 34 | GGCGCAGUGGGCUAGCGCCACUCAAAAGGCCCAU |
| 1XJR | SARS conserved RNA element | 47 | GGAGUUCACCGAGG...UGAAUU |
| 7LYJ | SARS-CoV-2 frameshifting pseudoknot | 66 | CGGUGUAAGUGCAG...AGGGCU |

**2MNC** (miR-21 pre-element, 29 nt) is excluded for now (`skip: true` in manifest).

## Directory Structure

```
Week 13/
├── inputs/                      Raw PDB/pdb1 files for all targets
├── outputs/<target>/            Prepared files from prepare_target pipeline
│   ├── <target>_rna_full.pdb
│   ├── <target>_c3only.pdb
│   ├── <target>.dat             Coarse-grained LAMMPS data (C3' beads)
│   ├── <target>_sequence.txt
│   ├── <target>_metadata.json
│   └── <target>_validation.txt
├── runs/<target>/               Simulation scaffolding (this week's work)
│   ├── inputs/                  .dat + LAMMPS .in files
│   ├── outputs/                 Created at simulation runtime
│   ├── analysis/                Post-processing scripts go here
│   ├── slurm/                   Job submission scripts
│   └── notes/                   Per-target notes and TODOs
├── validation/                  (reserved for cross-target validation)
├── logs/                        (reserved for batch pipeline logs)
├── outputs/batch_summary.json   Summary of last prepare_target batch run
└── README.md                    This file
```

## Preparation Pipeline

Managed by `scripts/week13/`:
- `manifest_targets.yaml` — target list with chain/resrange overrides
- `prepare_target.py` — single-target parser + validator + writer
- `batch_prepare_targets.py` — batch runner (reads manifest)

Re-run preparation at any time:
```bash
cd /scratch/gpfs/JERELLE/nilbert/scripts/week13
python3 batch_prepare_targets.py
```

## Next Steps

1. Review each `runs/<target>/notes/<target>_notes.txt`
2. Edit seeds and temperature in each `NVT_<target>_target.in`
3. Submit target-only jobs: `sbatch slurm/run_<target>_target.sh`
4. After target-only runs stabilise, build combined RNA+ASO data files:
   ```bash
   python3 scripts/week13/build_target_aso_dat.py \
     --rna-dat runs/<target>/inputs/<target>.dat \
     --aso-seq <SEQUENCE> \
     --out runs/<target>/inputs/<target>_aso_combined.dat
   ```
   Combined .dat: mol 1 = ASO, mol 2 = RNA, mol 3..N = free ASOs (if --n-aso > 1)
5. Fill in ASO-binding templates (`NVT_<target>_aso_bind.in`)
