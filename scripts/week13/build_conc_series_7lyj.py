#!/usr/bin/env python3
"""
Build and submit concentration series ASO-binding runs for 7LYJ.
n_aso = 5, 10, 20, 50, 100, 200, 300, 400, 500

Box scaled as 80 * n^(1/3) to maintain constant ASO concentration.
ntasks scaled with atom count for fastest runtime.
"""

import random
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path("/scratch/gpfs/JERELLE/nilbert/scripts/week13")
RUNS_7LYJ   = Path("/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 13/runs/7LYJ")
RNA_DAT     = RUNS_7LYJ / "inputs/7LYJ.dat"
ASO_SEQ     = "CGGUGUAAGA"

# n_aso → (box_half Å, ntasks, walltime_h)
# box_half = round(80 * n^(1/3), -1)   [rounded to nearest 10]
# atoms    = 66 + n*10
# ntasks   = min(32, max(4, atoms // 100))  → rounded to power of 2
CONFIGS = {
    5:   (140,  4,  24),
    10:  (175,  4,  24),
    20:  (220,  8,  36),
    50:  (300,  8,  48),
    100: (375, 16,  48),
    200: (470, 32,  48),
    300: (540, 32,  72),
    400: (590, 32,  72),
    500: (640, 32,  72),
}

rng = random.Random(20260418)
SEEDS = {n: (rng.randint(10000, 99999), rng.randint(10000, 99999))
         for n in sorted(CONFIGS)}


def build_dat(n_aso, box_half):
    out = RUNS_7LYJ / f"inputs/7LYJ_aso_n{n_aso}_combined.dat"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "build_target_aso_dat.py"),
        "--rna-dat", str(RNA_DAT),
        "--aso-seq", ASO_SEQ,
        "--n-aso",   str(n_aso),
        "--box-half", str(float(box_half)),
        "--offset",  "20",
        "--out",     str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"build_target_aso_dat failed for n={n_aso}:\n{r.stderr}")
    print(f"  [DAT] {r.stdout.strip()}")


def write_in(n_aso, box_half, seed_v, seed_l):
    tag      = f"7LYJ_aso_n{n_aso}"
    dat_name = f"7LYJ_aso_n{n_aso}_combined.dat"
    n_atoms  = 66 + n_aso * 10

    content = f"""\
# =================================================================
# Week 13  |  ASO concentration series  n_aso={n_aso}
# Target   : 7LYJ  (SARS-CoV-2 frameshifting pseudoknot, 66 nt)
# ASO      : {ASO_SEQ}  (10-mer, inter-stem hub pos 19-28)
# n_aso    : {n_aso}  ({n_atoms} total beads)
# Box      : +/-{box_half} A  (constant-concentration scaling)
# Seeds    : velocity={seed_v}  langevin={seed_l}
# mol 1=primary ASO | mol 2=RNA | mol 3..{n_aso + 1}=free ASOs
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
variable df       string inputs/{dat_name}
variable T_target equal  300.0
variable tag      string {tag}

shell mkdir -p outputs logs

# --- Read structure ----------------------------------------------
read_data   ${{df}}
log         outputs/log.${{tag}}.lammps

# --- Force field -------------------------------------------------
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
pair_coeff 4 4 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345

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
velocity all create ${{T_target}} {seed_v} rot yes dist gaussian

fix fxlang all langevin ${{T_target}} ${{T_target}} 1000.0 {seed_l}
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
    path = RUNS_7LYJ / f"inputs/NVT_{tag}.in"
    path.write_text(content)
    print(f"  [IN]  {path.name}")
    return path


def write_sh(n_aso, ntasks, walltime_h):
    tag = f"7LYJ_aso_n{n_aso}"
    hh  = f"{walltime_h:02d}:00:00"
    content = f"""\
#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time={hh}
#SBATCH --mail-type=all
#SBATCH --mail-user=nn9775@princeton.edu
#SBATCH --constraint=rh9
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun $HOME/.local/bin/lmp_amd -in inputs/NVT_{tag}.in
"""
    path = RUNS_7LYJ / f"slurm/run_{tag}.sh"
    path.write_text(content)
    print(f"  [SH]  {path.name}")
    return path


def main():
    print("=== 7LYJ ASO concentration series ===\n")
    submitted = []

    for n_aso in sorted(CONFIGS):
        box_half, ntasks, walltime_h = CONFIGS[n_aso]
        seed_v, seed_l = SEEDS[n_aso]
        n_atoms = 66 + n_aso * 10
        print(f"--- n={n_aso:>3} | {n_atoms:>4} atoms | box ±{box_half} Å "
              f"| {ntasks} tasks | {walltime_h}h ---")
        build_dat(n_aso, box_half)
        write_in(n_aso, box_half, seed_v, seed_l)
        sh = write_sh(n_aso, ntasks, walltime_h)
        r = subprocess.run(["sbatch", str(sh)],
                           capture_output=True, text=True, cwd=str(RUNS_7LYJ))
        if r.returncode == 0:
            jid = r.stdout.strip().split()[-1]
            submitted.append((n_aso, ntasks, walltime_h, jid))
            print(f"  [JOB] {jid}\n")
        else:
            print(f"  [ERR] {r.stderr.strip()}\n")

    print("=== Submitted jobs ===")
    print(f"{'n_aso':>6}  {'tasks':>5}  {'walltime':>8}  {'job_id':>10}")
    for n_aso, ntasks, wt, jid in submitted:
        print(f"{n_aso:>6}  {ntasks:>5}  {wt:>6}h     {jid:>10}")


if __name__ == "__main__":
    main()
