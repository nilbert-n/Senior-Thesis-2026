#!/usr/bin/env python3
"""
Small-box 7LYJ concentration series: n=5, 20, 50 in a fixed ±100 Å box.
Shorter encounter time (L² ~ 40000 Å², well within 1 µs diffusion range).
"""

import random
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path("/scratch/gpfs/JERELLE/nilbert/scripts/week13")
RUNS_7LYJ   = Path("/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 13/runs/7LYJ")
RNA_DAT     = RUNS_7LYJ / "inputs/7LYJ.dat"
ASO_SEQ     = "CGGUGUAAGA"

BOX_HALF = 100.0
OFFSET   = 25.0  # primary ASO offset — safely inside ±100 box for 10-mer

# n_aso → (ntasks, walltime_h)
CONFIGS = {
    5:  (4,  12),
    20: (4,  18),
    50: (8,  24),
}

rng = random.Random(202604190)
SEEDS = {n: (rng.randint(10000, 99999), rng.randint(10000, 99999))
         for n in sorted(CONFIGS)}

FF_PAIR_COEFFS = """\
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
pair_coeff 4 4 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345"""


def build_dat(n_aso):
    out = RUNS_7LYJ / f"inputs/7LYJ_aso_n{n_aso}_smallbox_combined.dat"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "build_target_aso_dat.py"),
        "--rna-dat",  str(RNA_DAT),
        "--aso-seq",  ASO_SEQ,
        "--n-aso",    str(n_aso),
        "--box-half", str(BOX_HALF),
        "--offset",   str(OFFSET),
        "--out",      str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"build failed n={n_aso}:\n{r.stderr}")
    print(f"  [DAT] {r.stdout.strip()}")


def write_in(n_aso, seed_v, seed_l):
    tag      = f"7LYJ_aso_n{n_aso}_smallbox"
    dat_name = f"7LYJ_aso_n{n_aso}_smallbox_combined.dat"
    content = f"""\
# =================================================================
# Week 13  |  7LYJ small-box concentration scan  n_aso={n_aso}
# Target   : 7LYJ (66 nt)  ASO: {ASO_SEQ} (10-mer)
# Box      : +/-{int(BOX_HALF)} A (fixed — concentration scales with N)
# Seeds    : velocity={seed_v}  langevin={seed_l}
# mol 1=primary ASO | mol 2=RNA | mol 3..{n_aso+1}=free ASOs
# =================================================================

units       real
boundary    p p p
atom_style  full

timestep    10
special_bonds lj 0.0 0.0 1.0
neighbor    15.0 bin
neigh_modify every 10 delay 0
comm_style  tiled

variable df       string inputs/{dat_name}
variable T_target equal  300.0
variable tag      string {tag}

shell mkdir -p outputs logs

read_data   ${{df}}
log         outputs/log.${{tag}}.lammps

bond_style  harmonic
bond_coeff  1 7.5 5.9

angle_style harmonic
angle_coeff 1 5.0 150.0

pair_style  hybrid/overlay lj/cut 10.0 base/rna 18.0
pair_modify shift yes

{FF_PAIR_COEFFS}

minimize 1.0e-4 1.0e-6 1000 10000
reset_timestep 0

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

velocity all create ${{T_target}} {seed_v} rot yes dist gaussian

fix fxlang all langevin ${{T_target}} ${{T_target}} 1000.0 {seed_l}
fix fxnve  all nve
fix recenter all momentum 1000 linear 1 1 1

run 5000000

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
    tag = f"7LYJ_aso_n{n_aso}_smallbox"
    content = f"""\
#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time={walltime_h:02d}:00:00
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
    print(f"=== 7LYJ small-box (±{int(BOX_HALF)} Å) concentration series ===\n")
    submitted = []

    for n_aso in sorted(CONFIGS):
        ntasks, walltime_h = CONFIGS[n_aso]
        seed_v, seed_l = SEEDS[n_aso]
        print(f"--- n={n_aso} | ntasks={ntasks} | walltime={walltime_h}h ---")
        build_dat(n_aso)
        write_in(n_aso, seed_v, seed_l)
        sh = write_sh(n_aso, ntasks, walltime_h)
        r = subprocess.run(["sbatch", str(sh)], capture_output=True, text=True,
                           cwd=str(RUNS_7LYJ))
        if r.returncode == 0:
            jid = r.stdout.strip().split()[-1]
            submitted.append((n_aso, jid))
            print(f"  [JOB] {jid}\n")
        else:
            print(f"  [ERR] {r.stderr.strip()}\n")

    print("=== Submitted ===")
    for n, jid in submitted:
        print(f"  n={n:>3}  job {jid}")


if __name__ == "__main__":
    main()
