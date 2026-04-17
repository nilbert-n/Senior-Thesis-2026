#!/bin/bash
#SBATCH --job-name=IDP_Rg        
#SBATCH --nodes=1                
#SBATCH --ntasks=32              # CHANGED: Use 32 cores (MPI tasks) instead of 1
#SBATCH --cpus-per-task=1        # Keep at 1 (Pure MPI is best here)
#SBATCH --mem-per-cpu=1G         
#SBATCH --time=12:00:00          # CHANGED: Give it 12 hours to ensure it finishes
#SBATCH --mail-type=all        
#SBATCH --mail-user=nn9775@princeton.edu
#SBATCH --constraint=rh9

module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

mkdir -p outputs logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# srun will automatically use the 32 cores requested above
srun $HOME/.local/bin/lmp_amd -in NVE_4us.in