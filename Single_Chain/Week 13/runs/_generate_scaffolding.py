#!/usr/bin/env python3
"""
Generates Week 13 run scaffolding files for all active targets.
Run once from the runs/ directory — safe to re-run (overwrites).
"""

from pathlib import Path
import random

RUNS_DIR = Path(__file__).resolve().parent
WEEK13   = RUNS_DIR.parent

TARGETS = {
    "1ANR": {
        "desc":     "HIV-1 TAR RNA cis-acting regulatory element",
        "nt":       29,
        "seq":      "GGCAGAUCUGAGCCUGGGAGCUCUCUGCC",
        "resrange": "17–45",
        "caveats":  "NMR ensemble — MODEL 1 used. 29 nt with known bulge loop.",
        "seed1":    11111, "seed2": 22222,
    },
    "1E95": {
        "desc":     "SRV-1 RNA pseudoknot involved in -1 frameshifting",
        "nt":       36,
        "seq":      "GCGGCCAGCUCCAGGCCGCCAAACAAUAUGGAGCAC",
        "resrange": "1–36",
        "caveats":  "Solution NMR. Clean geometry, no caveats.",
        "seed1":    33333, "seed2": 44444,
    },
    "1P5O": {
        "desc":     "HCV IRES domain II",
        "nt":       77,
        "seq":      "GGCUGUGAGGAACUACUGUCUUCACGCAGAAAGCGUCUAGCCAUGGCGUUAGUAUGAGUGUCGUGCAGCCUCCAGCC",
        "resrange": "1–77",
        "caveats":  "Largest target (77 nt). Run time may need 12–24 h. Check box size.",
        "seed1":    55555, "seed2": 66666,
    },
    "1RNK": {
        "desc":     "RNA pseudoknot causing efficient -1 programmed frameshifting",
        "nt":       34,
        "seq":      "GGCGCAGUGGGCUAGCGCCACUCAAAAGGCCCAU",
        "resrange": "1–34",
        "caveats":  "Clean pseudoknot. No caveats.",
        "seed1":    77777, "seed2": 88888,
    },
    "1XJR": {
        "desc":     "Rigorously conserved RNA element within SARS coronavirus",
        "nt":       46,
        "seq":      "GAGUUCACCGAGGCCACGCGGAGUACGAUCGAGGGUACAGUGAAUU",
        "resrange": "2–47",
        "caveats":  "GTP modified residue at position 1 was excluded by pipeline (resrange 2–47). "
                    "Actual usable chain is 46 nt. Confirm this is acceptable before ASO design.",
        "seed1":    12321, "seed2": 45654,
    },
    "7LYJ": {
        "desc":     "SARS-CoV-2 frameshifting pseudoknot RNA",
        "nt":       66,
        "seq":      "CGGUGUAAGUGCAGCCCGUCUUACACCGUGCGGCACAGCGGAAACGCUGAUGUCGUAUACAGGGCU",
        "resrange": "1–66",
        "caveats":  "Primary thesis target for ASO binding. 66 nt. Verify folded structure "
                    "matches literature pseudoknot topology before ASO placement.",
        "seed1":    98765, "seed2": 43210,
    },
}

FF_BLOCK = """\
bond_style  harmonic
bond_coeff  1 7.5 5.9

angle_style harmonic
angle_coeff 1 5.0 150.0

pair_style  hybrid/overlay lj/cut 10.0 base/rna 18.0
pair_modify shift yes

pair_coeff * * lj/cut 2.0 8.9089871814

pair_coeff 1 1 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 1 2 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 1 3 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 1 4 base/rna 3.33333 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 2 2 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 2 3 base/rna 5.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 2 4 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 3 3 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 3 4 base/rna 3.33333 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 4 4 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345\
"""

MODULES = """\
module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6\
"""


def nvt_target(name, meta):
    return f"""\
# =================================================================
# Week 13  |  Target-only NVT equilibration + production
# Target   : {name}  ({meta['desc']})
# Sequence : {meta['seq']}
# Length   : {meta['nt']} nt  (residues {meta['resrange']})
# Ensemble : NVT — Langevin thermostat
# Prepared : Week 13 prepare_target pipeline
# -----------------------------------------------------------------
# TODOs before launching:
#   [ ] Confirm T_target is appropriate for this RNA
#   [ ] Replace seed values with fresh random integers
#   [ ] Verify box size is adequate (current: ±500.6 Å default)
#   [ ] Review dump frequency vs available disk quota
# =================================================================

units       real
boundary    p p p
atom_style  full

timestep    10
special_bonds lj 0.0 0.0 1.0
neighbor    15.0 bin
neigh_modify every 10 delay 0
comm_style  tiled

# --- Variables ---------------------------------------------------
variable df       string inputs/{name}.dat
variable T_target equal  300.0          # TODO: adjust temperature (K)
variable tag      string {name}_target

shell mkdir -p outputs logs

# --- Read structure ----------------------------------------------
read_data   ${{df}}
log         outputs/log.${{tag}}.lammps

# --- Force field -------------------------------------------------
{FF_BLOCK}

# --- Minimization ------------------------------------------------
minimize 1.0e-4 1.0e-6 1000 10000
reset_timestep 0

# --- Output setup ------------------------------------------------
dump d1 all custom 50000 outputs/${{tag}}.lammpstrj id mol type q xu yu zu
dump_modify d1 sort id

thermo_style custom step pe ke etotal temp press density
thermo       10000
thermo_modify flush yes

variable vpe equal pe
variable vke equal ke
variable vT  equal temp
variable vE  equal etotal
variable vRg equal gyration(all)

fix favg all ave/time 10000 10 100000 v_vpe v_vke v_vT v_vE v_vRg &
    file outputs/thermo_${{tag}}_averaged.dat

# --- NVT equilibration (50 ns) -----------------------------------
velocity all create ${{T_target}} {meta['seed1']} rot yes dist gaussian

fix fxlang all langevin ${{T_target}} ${{T_target}} 1000.0 {meta['seed2']}
fix fxnve  all nve
fix recenter all momentum 1000 linear 1 1 1

run 5000000

# --- Production (1 µs) -------------------------------------------
unfix favg
fix favg all ave/time 10000 10 100000 v_vpe v_vke v_vT v_vE v_vRg &
    file outputs/thermo_${{tag}}_production.dat

restart 10000000 outputs/backup_${{tag}}.*.restart
run 100000000

write_data outputs/${{tag}}_final.data
"""


def nvt_aso_template(name, meta):
    return f"""\
# =================================================================
# Week 13  |  ASO-binding NVT simulation TEMPLATE
# Target   : {name}  ({meta['desc']})
# Sequence : {meta['seq']}
# Length   : {meta['nt']} nt
# -----------------------------------------------------------------
# *** THIS FILE IS A TEMPLATE — NOT READY TO RUN ***
#
# Before this file can be used you must:
#   [ ] Build a combined RNA + ASO coarse-grained data file and
#       place it at:  inputs/{name}_aso_combined.dat
#       (see notes/{name}_notes.txt for required format)
#   [ ] Choose and validate ASO sequence and placement
#   [ ] Confirm total atom type count (RNA: 4 types; ASO may add more)
#   [ ] Add any ASO-specific pair_coeff lines below the RNA FF block
#   [ ] Set ASO molecule ID (mol 2 by convention from prior weeks)
#   [ ] Review and update seed values
#   [ ] Run target-only simulation first; confirm stable before adding ASO
# =================================================================

units       real
boundary    p p p
atom_style  full

timestep    10
special_bonds lj 0.0 0.0 1.0
neighbor    15.0 bin
neigh_modify every 10 delay 0
comm_style  tiled

# --- Variables ---------------------------------------------------
variable df       string inputs/{name}_aso_combined.dat  # TODO: create this file
variable T_target equal  300.0                           # TODO: adjust
variable tag      string {name}_aso_bind

shell mkdir -p outputs logs

# --- Read combined RNA+ASO structure -----------------------------
read_data   ${{df}}
log         outputs/log.${{tag}}.lammps

# --- Force field (RNA) -------------------------------------------
{FF_BLOCK}

# --- TODO: Add ASO pair_coeff lines here -------------------------
# Example from prior ASO work (verify before use):
#   pair_coeff <ASO_type> <RNA_type> base/rna <params>
# -----------------------------------------------------------------

# --- Minimization ------------------------------------------------
minimize 1.0e-4 1.0e-6 1000 10000
reset_timestep 0

# --- Output setup ------------------------------------------------
dump d1 all custom 50000 outputs/${{tag}}.lammpstrj id mol type q xu yu zu
dump_modify d1 sort id

thermo_style custom step pe ke etotal temp press density
thermo       10000
thermo_modify flush yes

variable vpe equal pe
variable vke equal ke
variable vT  equal temp
variable vE  equal etotal
variable vRg equal gyration(all)

fix favg all ave/time 10000 10 100000 v_vpe v_vke v_vT v_vE v_vRg &
    file outputs/thermo_${{tag}}_averaged.dat

# --- NVT equilibration (50 ns) -----------------------------------
velocity all create ${{T_target}} {meta['seed1']} rot yes dist gaussian

fix fxlang all langevin ${{T_target}} ${{T_target}} 1000.0 {meta['seed2']}
fix fxnve  all nve
fix recenter all momentum 1000 linear 1 1 1

run 5000000

# --- Production (1 µs) -------------------------------------------
unfix favg
fix favg all ave/time 10000 10 100000 v_vpe v_vke v_vT v_vE v_vRg &
    file outputs/thermo_${{tag}}_production.dat

restart 10000000 outputs/backup_${{tag}}.*.restart
run 100000000

write_data outputs/${{tag}}_final.data
"""


def slurm_target(name, meta):
    # Scale walltime with sequence length: <50 nt → 8h, else 12h
    time = "08:00:00" if meta["nt"] < 50 else "12:00:00"
    return f"""\
#!/bin/bash
#SBATCH --job-name={name}_target
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time={time}
#SBATCH --mail-type=all
#SBATCH --mail-user=nn9775@princeton.edu
#SBATCH --constraint=rh9
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

{MODULES}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun $HOME/.local/bin/lmp_amd -in inputs/NVT_{name}_target.in
"""


def slurm_aso(name, meta):
    return f"""\
#!/bin/bash
# *** TEMPLATE — not ready to submit until combined .dat is built ***
#SBATCH --job-name={name}_aso_bind
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=nn9775@princeton.edu
#SBATCH --constraint=rh9
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

{MODULES}

# TODO: Confirm the combined RNA+ASO data file exists at:
#   inputs/{name}_aso_combined.dat
# before submitting this job.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun $HOME/.local/bin/lmp_amd -in inputs/NVT_{name}_aso_bind.in
"""


def notes_file(name, meta):
    prep_dir = WEEK13 / "outputs" / name
    return f"""\
Week 13 — {name} run notes
{"=" * 60}

Target        : {name}
Description   : {meta['desc']}
Sequence      : {meta['seq']}
Length        : {meta['nt']} nt  (residues {meta['resrange']}, chain A)
Caveats       : {meta['caveats']}

Prepared outputs (do not modify):
  {prep_dir}/{name}_rna_full.pdb
  {prep_dir}/{name}_c3only.pdb
  {prep_dir}/{name}.dat           <- copied to inputs/{name}.dat
  {prep_dir}/{name}_sequence.txt
  {prep_dir}/{name}_metadata.json
  {prep_dir}/{name}_validation.txt

Files in this run folder:
  inputs/{name}.dat               — coarse-grained C3' LAMMPS data
  inputs/NVT_{name}_target.in     — target-only NVT input (edit seeds/T)
  inputs/NVT_{name}_aso_bind.in   — ASO-binding template (NOT ready)
  slurm/run_{name}_target.sh      — SLURM submission for target-only run
  slurm/run_{name}_aso_bind.sh    — SLURM template for ASO run (NOT ready)

TODOs before launching target-only simulation:
  [ ] Choose temperature (default 300 K; consider 250/275/325/350 K sweep)
  [ ] Replace seed values in NVT_{name}_target.in
  [ ] Confirm outputs/ disk quota is sufficient
      (1 µs dump at 50000-step interval ≈ estimate based on {meta['nt']} atoms)
  [ ] Submit: sbatch slurm/run_{name}_target.sh

TODOs before building ASO-binding simulation:
  [ ] Complete target-only run and confirm stable folded structure
  [ ] Design ASO sequence complementary to target
  [ ] Build combined RNA+ASO coarse-grained data file:
        inputs/{name}_aso_combined.dat
      Required format: same as existing ASO .dat files in Week 9/10
        - mol 1 = RNA ({meta['nt']} beads, atom types 1–4)
        - mol 2 = ASO (N beads, atom types as needed)
        - bonds and angles for both molecules
  [ ] Add ASO-specific pair_coeff lines to NVT_{name}_aso_bind.in
  [ ] Submit: sbatch slurm/run_{name}_aso_bind.sh
"""


def main():
    for name, meta in TARGETS.items():
        run_dir = RUNS_DIR / name

        (run_dir / "inputs" / f"NVT_{name}_target.in").write_text(nvt_target(name, meta))
        (run_dir / "inputs" / f"NVT_{name}_aso_bind.in").write_text(nvt_aso_template(name, meta))
        (run_dir / "slurm"  / f"run_{name}_target.sh").write_text(slurm_target(name, meta))
        (run_dir / "slurm"  / f"run_{name}_aso_bind.sh").write_text(slurm_aso(name, meta))
        (run_dir / "notes"  / f"{name}_notes.txt").write_text(notes_file(name, meta))

        print(f"[OK] {name}")

    print(f"\nAll scaffolding written under {RUNS_DIR}")


if __name__ == "__main__":
    main()
