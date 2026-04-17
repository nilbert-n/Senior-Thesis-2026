#!/bin/bash
#SBATCH --job-name=ASO100_275K_restart
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
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

mkdir -p outputs logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Tasks: $SLURM_NTASKS"

srun $HOME/.local/bin/lmp_amd -in inputs/NVT_100ASO_275K_restart.in

echo "Job finished: $(date)"
