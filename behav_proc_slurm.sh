#!/bin/bash
#SBATCH -J behav_proc
#SBATCH -t 5:00
#SBATCH --mem=80G
#SBATCH --array=0-5

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o behav_proc-%j.out
#SBATCH -e behav_proc-%j.out

python behav_proc_slurm.py $SLURM_ARRAY_TASK_ID

scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats