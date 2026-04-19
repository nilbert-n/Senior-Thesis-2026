#!/bin/bash
#SBATCH --job-name=1XJR_aso_n50
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=48:00:00
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

srun $HOME/.local/bin/lmp_amd -in inputs/NVT_1XJR_aso_n50.in
