#!/bin/bash
#SBATCH --job-name=IDP_Rg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=03:00:00
#SBATCH --mail-type=all
##SBATCH --mail-type=end
#SBATCH --mail-user=nn9775@princeton.edu
#SBATCH --constraint=rh9
#SBATCH --array=1-8
#SBATCH -o logs/slurm.%A.%a.out
#SBATCH -e logs/slurm.%A.%a.err

module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

mkdir -p outputs_sa logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Build file list once (from current dir)
if [ ! -s cfg.lst ]; then
  ls -1 config_*.dat > cfg.lst
fi

# Pick the Nth file for this array task
DAT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cfg.lst)
if [ -z "$DAT" ]; then
  echo "No entry for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in cfg.lst"; exit 1
fi
echo "Running SA on: $DAT"

# Launch LAMMPS (expects sa_one.in to 'read_data ${df}')
srun "$HOME/.local/bin/lmp_amd" -in sa_one.in -var df "$DAT"
