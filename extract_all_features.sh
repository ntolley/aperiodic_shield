#!/bin/bash
#SBATCH -J lfp_proc
#SBATCH -t 60:00
#SBATCH --mem=80G
#SBATCH --ntasks=20

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o extract_features-%j.out
#SBATCH -e extract_features-%j.out

python extract_all_features.py