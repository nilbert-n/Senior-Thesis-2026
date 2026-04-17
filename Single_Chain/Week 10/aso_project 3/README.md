# ASO-Hairpin Simulation Campaign
## Replicating & Extending Hörberg et al. (2024)

---

## Quick Start

```bash
# 1. Copy this entire folder to your cluster
scp -r aso_project/ yourcluster:~/

# 2. Submit all 18 jobs at once
cd aso_project
bash launch_all.sh

# 3. Or submit individual jobs
sbatch slurm/run_100ASO_300K.sh
```

> **Screening mode (Option A):** Each job runs **1 µs production + 50 ns equilibration**
> with a 4-hour walltime. Expected wall time per job: **~1.5–2 hours**.
> All 18 jobs submitted in parallel finish within **~2 hours total**.
> After reviewing results, re-run the most interesting 4–5 variants at
> full 4 µs by changing `run_ns=1000` → `run_ns=4000` in
> `generate_all_configs.py` and regenerating.

---

## SLURM Optimization (IMPORTANT)

Your original setup used **32 MPI tasks** but only achieved **34.35% efficiency**. The reason:

- Your system has only **~1000 coarse-grained beads** (at 100 ASOs)
- With 32 MPI ranks, each rank handles ~31 atoms — the **MPI communication overhead dominates** the actual computation
- The domain decomposition creates 32 spatial domains with very few particles each

**Recommended task counts:**

| System Size | Atoms | Optimal ntasks | Expected Speedup |
|-------------|-------|---------------|-----------------|
| 25 ASOs     | 282   | **4**         | ~3.5x vs serial |
| 50 ASOs     | 532   | **4**         | ~3.5x           |
| 100 ASOs    | 1032  | **4**         | ~3.8x           |
| 200 ASOs    | 2032  | **8**         | ~6x             |

For coarse-grained models, MPI scaling breaks down rapidly beyond ~250 atoms/rank. All SLURM scripts in this package are already optimized accordingly.

---

## File Structure

```
aso_project/
├── generate_all_configs.py    # Master generator (re-run to regenerate)
├── launch_all.sh              # Submit all 18 jobs
├── README.md
│
├── configs/                   # LAMMPS data files (.dat)
│   ├── 25ASO_unmod.dat
│   ├── 50ASO_unmod.dat
│   ├── 100ASO_unmod.dat
│   ├── 200ASO_unmod.dat
│   ├── 100ASO_truncated_unmodified.dat
│   ├── 100ASO_extended_12mer_unmodified.dat
│   ├── 100ASO_extended_14mer_unmodified.dat
│   ├── 100ASO_mismatch_G6A_unmodified.dat
│   ├── 100ASO_mismatch_U5C_unmodified.dat
│   ├── 100ASO_all_purine_unmodified.dat
│   ├── 100ASO_scrambled_unmodified.dat
│   ├── 100ASO_unmodified_AAtoCC.dat
│   └── 100ASO_unmodified_loop_GG_to_AA.dat
│
├── inputs/                    # LAMMPS input scripts (.in)
│   ├── NVT_25ASO_300K.in     ... (18 input files)
│
└── slurm/                     # SLURM submission scripts (.sh)
    ├── run_25ASO_300K.sh      ... (18 SLURM scripts)
```

---

## Simulation Campaign Overview

### Part 1: Concentration Sweep (4 simulations)

All at 300K with unmodified 10-mer ASO (GGGCUGUUUU) and unmodified hairpin.

| # ASOs | Box Size (Å) | Atoms | Effective [ASO] | Purpose |
|--------|-------------|-------|-----------------|---------|
| 25     | 160³        | 282   | ~10 µM          | Dilute limit |
| 50     | 160³        | 532   | ~20 µM          | Low concentration |
| 100    | 160³        | 1032  | ~40 µM          | Original (baseline) |
| 200    | 220³        | 2032  | ~30 µM          | High density |

### Part 2: Temperature Sweep (5 simulations)

All with 100 ASOs and unmodified sequences. NVT ensemble with Langevin thermostat.

| Temperature | Purpose |
|------------|---------|
| 250K       | Below physiological — tighter binding expected |
| 275K       | Intermediate |
| 300K       | Physiological (baseline) |
| 325K       | Above physiological — increased dissociation |
| 350K       | High temperature — near melting |

### Part 3: ASO Design Variants (9 simulations)

All at 300K with 100 ASOs. Tests how ASO sequence/length affects binding.

| Variant | ASO Sequence (5'→3') | Hairpin | Rationale |
|---------|---------------------|---------|-----------|
| Truncated 7mer | GGGCUGU | Unmodified | Paper's truncated variant (lost A34-A36 contacts) |
| Extended 12mer | GGGCUGUUUUAA | Unmodified | More triple bp contacts with stem |
| Extended 14mer | GGGCUGUUUUAAGU | Unmodified | Even longer — test flexibility penalty |
| G6A mismatch | GGGCUAUUUU | Unmodified | Disrupts G6-C20-A32 triple bp |
| U5C mismatch | GGGCCGUUUU | Unmodified | Disrupts U5-A21-A31 triple bp |
| Purine-rich | GGGAGGAAAA | Unmodified | All-purine alternative design |
| Scrambled | UGUGUCGUGU | Unmodified | Same composition, random order |
| AAtoCC mutant | GGGCUGUUUU | A31C,A32C | Paper's AAtoCC variant (130-fold Kd increase) |
| G22A loop mutant | GGGCUGUUUU | G22A | Tests loop accessibility changes |

---

## Key Changes from Your Original NVE_4us.in

1. **NVT instead of NVE**: Langevin thermostat maintains target temperature — essential for temperature sweep and more realistic sampling
2. **1 µs screening runs**: 50 ns equilibration + 1 µs production (~2h per job vs ~12h). Extend to 4 µs for promising variants.
3. **Radius of gyration tracking**: `gyration(all)` computed and logged automatically
4. **Separate production averages**: Equilibration data excluded from production stats
5. **Optimized MPI**: 4 tasks (not 32) for ~4x better efficiency
6. **4-hour walltime**: Generous buffer; most jobs finish in ~1.5–2h

---

## Recommended Other Molecules to Simulate

Based on the literature, these RNA structures are excellent candidates for the same coarse-grained ASO-binding simulation framework:

### Tier 1: Directly relevant (same methodology, high-impact)

1. **SARS-CoV-2 Frameshifting Pseudoknot (PDB: 7LYJ)**
   - 1.3 Å X-ray structure, three coaxially stacked helices with base triples
   - Li et al. (the same group that inspired your paper) designed ASOs targeting this element
   - ASO binding disrupts the pseudoknot and inhibits −1 ribosomal frameshifting
   - Directly comparable to 1YMO since both involve triple bp interactions

2. **SARS-CoV-2 Attenuator Hairpin (PDB: 7LYJ region / modeled from 8VCI)**
   - Contains a UU internal loop targetable by small molecules (Kd = 11 nM for compound C5)
   - ASO gapmers targeting this region inhibit viral replication
   - Good test case for ASO binding to internal loops vs. terminal loops

3. **SARS-CoV-2 s2m Element (3' UTR stem-loop II-like motif)**
   - Highly conserved across coronaviruses and astroviruses
   - Lulla et al. showed ASOs targeting exposed loops trigger RNase H cleavage
   - LNA gapmer ASOs showed dose-dependent inhibition of viral replication
   - No high-res PDB available — would need homology modeling from SARS-CoV s2m (PDB: 1XJR)

4. **MMTV Frameshifting Pseudoknot (PDB: 1RNK)**
   - Classic H-type pseudoknot, well-characterized thermodynamically
   - Studied in PNAS coarse-grained folding simulations alongside 1YMO
   - Different loop topology than 1YMO — good for testing generalizability

### Tier 2: Broader RNA targets with ASO therapeutic relevance

5. **SRV-1 Pseudoknot (PDB: 1E95)**
   - Simian retrovirus frameshifting element
   - Folds cooperatively (different mechanism than 1YMO)
   - Coarse-grained parameters already validated in published work

6. **HIV-1 TAR Element (PDB: 1ANR)**
   - Small RNA hairpin (29 nt) with a trinucleotide bulge
   - Major drug target — extensive ASO and small molecule data available
   - Well-suited for your coarse-grained model size

7. **Hepatitis C IRES Domain II (PDB: 1P5M)**
   - Internal ribosome entry site hairpin
   - ASO therapeutics (e.g., miravirsen concept) target this region
   - Larger structure — would test scalability of your approach

8. **Pre-miR-21 Hairpin**
   - Oncogenic microRNA precursor
   - Anti-miR ASOs (antagomirs) are in clinical development
   - Simple hairpin structure — good positive control

### How to adapt these structures:

For each new PDB target:
1. Download the PDB from rcsb.org
2. Extract C3' atom coordinates (one bead per nucleotide) using a script like:
   ```python
   # Extract coarse-grained beads from PDB
   for line in open('structure.pdb'):
       if line.startswith('ATOM') and "C3'" in line:
           # Extract residue name → map to type 1-4
           # Extract x, y, z coordinates
   ```
3. Identify the hairpin region and design a complementary ASO
4. Use `generate_all_configs.py` as a template (modify `create_aso_sequence`)
5. The force field parameters (base/rna pair_coeff) remain the same

---

## Analysis Tips

After simulations complete, key analyses to run:

- **Radius of gyration** vs time (complex compactness) — already logged
- **RMSD** of ASO relative to hairpin (binding stability)
- **Contact maps** between ASO beads and hairpin beads
- **Binding/unbinding events** — count how many free ASOs dock during simulation
- **Temperature-dependent Rg** — plot average Rg vs T for melting curve
- **Concentration-dependent binding** — fraction of bound ASOs vs [ASO]

---

## Troubleshooting

**"pair_style base/rna not found"**: Your LAMMPS build needs this custom pair style compiled in. If you're on a shared cluster, make sure you're using your custom `lmp_amd` binary.

**Simulation crashes at start**: Usually means overlapping atoms. The minimization step should fix this, but if a config has severe clashes, try increasing the box size or reducing `n_free_aso`.

**Very slow performance**: If wall time exceeds 24h for 4µs, check that you're NOT using 32 tasks. With 4 tasks and ~1000 atoms, 4µs should complete in ~8-12 hours.
