#!/bin/bash -l
#SBATCH --qos urgent
#SBATCH -t 10:00:00
#SBATCH -A desi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheydenr@ucsc.edu
#SBATCH -N 1
#SBATCH -J randoms
##SBATCH -L SCRATCH
#SBATCH -C cpu
##SBATCH -C haswell
#SBATCH --output=/global/u2/s/sven/code/lensingWithoutBorders/logs/log_randoms.%j.oe

# module load python

eval "$(conda shell.bash hook)"
conda activate py38

which python

python /global/u2/s/sven/code/lensingWithoutBorders/plotting/randoms.py config_plots_hscy1.conf

