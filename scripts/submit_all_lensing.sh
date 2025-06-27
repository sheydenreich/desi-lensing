#!/bin/bash -l
#SBATCH --qos regular
#SBATCH -t 8:00:00
#SBATCH -A desi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheydenr@ucsc.edu
#SBATCH -N 1
#SBATCH -J lensing_all
#SBATCH -L SCRATCH
#SBATCH -C gpu
#SBATCH --output=/global/u2/s/sven/code/lensingWithoutBorders/logs/log_lensing_all.%j.oe

# module load python

eval "$(conda shell.bash hook)"
conda activate desi-lensing

export HDF5_USE_FILE_LOCKING=FALSE

desi-lensing --verbose compute deltasigma --galaxy-type BGS_BRIGHT,LRG --save-precomputed --no-blinding --source-surveys DES,KiDS --output-dir /pscratch/sd/s/sven/desi_lensing_gpu/
