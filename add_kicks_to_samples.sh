#!/bin/bash
#SBATCH --job-name=bh_kick_samples
#SBATCH --array=0-99
#SBATCH --time=2:00:00
#SBATCH --output=bh_kick_samples_%a.out
#SBATCH --mem-per-cpu=1000

source ~/.bash_profile
conda activate parallel_bilby

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

FNAME=$(printf "/fred/oz117/avajpeyi/projects/phase-marginalisation-test/jobs/out_hundred_injections_gstar/out_injection_${SLURM_ARRAY_TASK_ID}/result/injection_${SLURM_ARRAY_TASK_ID}_0_posterior_samples.dat")

echo "Adding kicks to " "$FNAME"

python calculate_kick_vel_from_samples.py $FNAME

