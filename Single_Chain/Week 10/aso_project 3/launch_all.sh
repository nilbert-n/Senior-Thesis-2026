#!/bin/bash
# Master launcher - submits all simulation jobs
# Run: bash launch_all.sh

echo '=== ASO-Hairpin Simulation Campaign ==='
echo 'Submitting 18 jobs...'

JOB=$(sbatch slurm/run_25ASO_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_25ASO_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_50ASO_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_50ASO_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_100ASO_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_100ASO_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_200ASO_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_200ASO_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_100ASO_250K.sh | awk '{print $4}')
echo "Submitted slurm/run_100ASO_250K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_100ASO_275K.sh | awk '{print $4}')
echo "Submitted slurm/run_100ASO_275K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_100ASO_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_100ASO_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_100ASO_325K.sh | awk '{print $4}')
echo "Submitted slurm/run_100ASO_325K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_100ASO_350K.sh | awk '{print $4}')
echo "Submitted slurm/run_100ASO_350K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_truncated_unmodified_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_truncated_unmodified_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_extended_12mer_unmodified_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_extended_12mer_unmodified_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_extended_14mer_unmodified_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_extended_14mer_unmodified_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_mismatch_G6A_unmodified_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_mismatch_G6A_unmodified_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_mismatch_U5C_unmodified_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_mismatch_U5C_unmodified_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_all_purine_unmodified_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_all_purine_unmodified_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_scrambled_unmodified_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_scrambled_unmodified_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_unmodified_AAtoCC_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_unmodified_AAtoCC_300K.sh -> Job $JOB"

JOB=$(sbatch slurm/run_unmodified_loop_GG_to_AA_300K.sh | awk '{print $4}')
echo "Submitted slurm/run_unmodified_loop_GG_to_AA_300K.sh -> Job $JOB"

echo 'All jobs submitted!'
echo 'Monitor with: squeue -u $USER'
