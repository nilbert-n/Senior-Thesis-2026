#!/usr/bin/env python3
"""
Build and submit n_aso=50 ASO-binding runs for 1ANR, 1E95, 1P5O, 1RNK, 1XJR.
Box: ±300 Å (same as 7LYJ n=50 — constant-concentration convention).
ASO: 10-mer reverse complement targeting the highest-contact mid-sequence region.
"""

import random
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path("/scratch/gpfs/JERELLE/nilbert/scripts/week13")
WEEK13_RUNS = Path("/scratch/gpfs/JERELLE/nilbert/Single_Chain/Week 13/runs")

N_ASO    = 50
BOX_HALF = 300.0
NTASKS   = 8
WALLTIME = "48:00:00"

# Reverse-complement a sequence string (RNA alphabet A/C/G/U)
def rc(seq):
    comp = str.maketrans("ACGU", "UGCA")
    return seq.translate(comp)[::-1]

# Target regions chosen as the structurally engaged mid-sequence junction:
#   1ANR  (29 nt, HIV-1 TAR)               : bulge-adjacent upper stem
#   1E95  (36 nt, SRV-1 pseudoknot)        : central stem-loop junction
#   1P5O  (77 nt, HCV IRES domain II)      : internal loop / junction
#   1RNK  (34 nt, -1 frameshift pseudoknot): junction stem
#   1XJR  (47 nt, SARS conserved element)  : inter-stem region
#
# Indexing is 1-based; end is inclusive.
TARGETS = {
    "1ANR": {
        "seq":       "GGCAGAUCUGAGCCUGGGAGCUCUCUGCC",
        "nt":         29,
        "aso_region": (11, 20),   # AGCCUGGGAG — bulge-adjacent upper stem
        "desc":       "HIV-1 TAR RNA",
        "dat":        WEEK13_RUNS / "1ANR/inputs/1ANR.dat",
    },
    "1E95": {
        "seq":       "GCGGCCAGCUCCAGGCCGCCAAACAAUAUGGAGCAC",
        "nt":         36,
        "aso_region": (14, 23),   # GGCCGCCAAA — central stem-loop junction
        "desc":       "SRV-1 frameshifting pseudoknot",
        "dat":        WEEK13_RUNS / "1E95/inputs/1E95.dat",
    },
    "1P5O": {
        "seq":       "GGCUGUGAGGAACUACUGUCUUCACGCAGAAAGCGUCUAGCCAUGGCGUUAGUAUGAGUGUCGUGCAGCCUCCAGCC",
        "nt":         77,
        "aso_region": (20, 29),   # CUUCACGCAG — internal loop / junction
        "desc":       "HCV IRES domain II",
        "dat":        WEEK13_RUNS / "1P5O/inputs/1P5O.dat",
    },
    "1RNK": {
        "seq":       "GGCGCAGUGGGCUAGCGCCACUCAAAAGGCCCAU",
        "nt":         34,
        "aso_region": (13, 22),   # UAGCGCCACU — junction stem
        "desc":       "-1 frameshifting pseudoknot",
        "dat":        WEEK13_RUNS / "1RNK/inputs/1RNK.dat",
    },
    "1XJR": {
        "seq":       "GGAGUUCACCGAGGCCACGCGGAGUACGAUCGAGGGUACAGUGAAUU",
        "nt":         47,
        "aso_region": (19, 28),   # CGCGGAGUAC — inter-stem region
        "desc":       "SARS conserved RNA element",
        "dat":        WEEK13_RUNS / "1XJR/inputs/1XJR.dat",
    },
}

# Unique seeds per target
rng = random.Random(202604182)
SEEDS = {t: (rng.randint(10000, 99999), rng.randint(10000, 99999))
         for t in TARGETS}

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


def aso_for_target(info):
    s, e = info["aso_region"]  # 1-based inclusive
    subseq = info["seq"][s-1:e]
    return rc(subseq), subseq


def build_dat(target, aso_seq, rna_dat):
    runs_dir = WEEK13_RUNS / target
    out = runs_dir / f"inputs/{target}_aso_n{N_ASO}_combined.dat"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "build_target_aso_dat.py"),
        "--rna-dat", str(rna_dat),
        "--aso-seq", aso_seq,
        "--n-aso",   str(N_ASO),
        "--box-half", str(BOX_HALF),
        "--offset",  "20",
        "--out",     str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{target} build_target_aso_dat failed:\n{r.stderr}")
    print(f"  [DAT] {r.stdout.strip()}")
    return out


def write_in(target, info, aso_seq, rna_subseq, seed_v, seed_l):
    s, e = info["aso_region"]
    tag      = f"{target}_aso_n{N_ASO}"
    dat_name = f"{target}_aso_n{N_ASO}_combined.dat"
    n_rna    = info["nt"]
    n_atoms  = n_rna + N_ASO * 10

    content = f"""\
# =================================================================
# Week 13  |  ASO-binding NVT  n_aso={N_ASO}
# Target   : {target}  ({info['desc']}, {n_rna} nt)
# ASO      : {aso_seq}  (10-mer, RC of pos {s}-{e}: {rna_subseq})
# n_aso    : {N_ASO}  ({n_atoms} total beads)
# Box      : +/-{int(BOX_HALF)} A
# Seeds    : velocity={seed_v}  langevin={seed_l}
# mol 1=primary ASO | mol 2=RNA | mol 3..{N_ASO + 1}=free ASOs
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

{FF_PAIR_COEFFS}

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
    path = WEEK13_RUNS / target / f"inputs/NVT_{tag}.in"
    path.write_text(content)
    print(f"  [IN]  {path.name}")
    return path


def write_sh(target):
    tag = f"{target}_aso_n{N_ASO}"
    content = f"""\
#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks={NTASKS}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time={WALLTIME}
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
    path = WEEK13_RUNS / target / f"slurm/run_{tag}.sh"
    path.write_text(content)
    print(f"  [SH]  {path.name}")
    return path


def main():
    print(f"=== n_aso={N_ASO} ASO-binding runs for 5 targets ===\n")
    submitted = []

    for target, info in TARGETS.items():
        aso_seq, rna_subseq = aso_for_target(info)
        seed_v, seed_l = SEEDS[target]
        s, e = info["aso_region"]
        print(f"--- {target} ({info['nt']} nt) | ASO {aso_seq} (RC pos {s}-{e}: {rna_subseq}) ---")

        build_dat(target, aso_seq, info["dat"])
        write_in(target, info, aso_seq, rna_subseq, seed_v, seed_l)
        sh = write_sh(target)

        r = subprocess.run(
            ["sbatch", str(sh)],
            capture_output=True, text=True,
            cwd=str(WEEK13_RUNS / target),
        )
        if r.returncode == 0:
            jid = r.stdout.strip().split()[-1]
            submitted.append((target, aso_seq, jid))
            print(f"  [JOB] {jid}\n")
        else:
            print(f"  [ERR] {r.stderr.strip()}\n")

    print("=== Submitted ===")
    print(f"{'target':>6}  {'ASO':>12}  {'job_id':>10}")
    for target, aso_seq, jid in submitted:
        print(f"{target:>6}  {aso_seq:>12}  {jid:>10}")


if __name__ == "__main__":
    main()
