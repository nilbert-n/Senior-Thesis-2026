#!/bin/bash
#SBATCH --job-name=IDP_Rg        # Job name for identification purposes during execution
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=2:30:00
#SBATCH --mail-type=all        # send email when job begins
##SBATCH --mail-type=end          # send email when job ends
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

srun $HOME/.local/bin/lmp_amd -in NVE.in