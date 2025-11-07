#!/bin/bash -l
#SBATCH --qos regular
#SBATCH -t 24:00:00
#SBATCH -A desi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheydenr@ucsc.edu
#SBATCH -N 1
#SBATCH -J lensing_all
##SBATCH -L SCRATCH
#SBATCH -C gpu
#SBATCH --array=0-3
#SBATCH --output=/global/u2/s/sven/code/lensingWithoutBorders/logs/log_lensing_all.%A_%a.oe

# module load python
eval "$(conda shell.bash hook)"
conda activate desi-lensing

export HDF5_USE_FILE_LOCKING=FALSE

# check DECADE on login33
# Select which job to run based on SLURM array task ID
case $SLURM_ARRAY_TASK_ID in
    0)
        # Job 0: BGS_BRIGHT with DES,HSCY3 source surveys
        echo "Running Job 0: BGS_BRIGHT with DES,HSCY3"
        desi-lensing --verbose compute deltasigma --catalogue-path /global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/DR2/ \
        --output-dir /global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensing_measurements/DR2/ \
        --n-jobs 1 --source-surveys DECADE --randoms 0 --randoms-ratio 20 --bgs-version v1 --lrg-version v1 \
        --release loa --galaxy-type BGS_BRIGHT
        ;;
    1)
        # Job 1: LRG with all source surveys
        echo "Running Job 1: LRG with all source surveys"
        desi-lensing --verbose compute deltasigma --catalogue-path /global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/DR2/ \
        --output-dir /global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensing_measurements/DR2/ \
        --n-jobs 4 --source-surveys DECADE_NGC,DECADE_SGC,DECADE,KiDS,DES,HSCY3 --randoms 0 --randoms-ratio 20 --bgs-version v1 --lrg-version v1 \
        --release loa --galaxy-type LRG
        ;;
    2)
        # Job 2: BGS_BRIGHT magnitude cuts
        echo "Running Job 2: BGS_BRIGHT magnitude cuts"
        desi-lensing compute deltasigma --catalogue-path /global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/DR2/nonKP_magcut/ \
        --output-dir /global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensing_measurements/DR2/nonKP_magcut/ \
        --n-jobs 4 --source-surveys DECADE_NGC,DECADE_SGC,DECADE,KiDS,DES,HSCY3 --randoms-ratio 20 --randoms 0 --bgs-version v1 \
        --release loa --galaxy-type BGS_BRIGHT
        ;;
    3)
        # Job 3: B-modes computation
        echo "Running Job 3: B-modes computation"
        desi-lensing compute deltasigma --catalogue-path /global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/DR2/ \
        --output-dir /global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensing_measurements/DR2/ \
        --n-jobs 4 --source-surveys DECADE_NGC,DECADE_SGC,DECADE,KiDS,DES,HSCY3 --randoms-ratio 20 --randoms 0 --bgs-version v1 --lrg-version v1 \
        --release loa --galaxy-type BGS_BRIGHT,LRG --bmodes --no-blinding
        ;;
    *)
        echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac
