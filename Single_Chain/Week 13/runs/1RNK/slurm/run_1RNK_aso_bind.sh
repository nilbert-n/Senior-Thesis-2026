#!/bin/bash
# *** TEMPLATE — not ready to submit until combined .dat is built ***
#SBATCH --job-name=1RNK_aso_bind
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

module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

# TODO: Confirm the combined RNA+ASO data file exists at:
#   inputs/1RNK_aso_combined.dat
# before submitting this job.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun $HOME/.local/bin/lmp_amd -in inputs/NVT_1RNK_aso_bind.in
