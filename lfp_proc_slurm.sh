#!/bin/bash
#SBATCH -J lfp_proc
#SBATCH -t 30:00
#SBATCH --mem=80G
#SBATCH --array=0-5

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o lfp_proc-%j.out
#SBATCH -e lfp_proc-%j.out

python l5_proc_slurm.py $SLURM_ARRAY_TASK_ID

scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats