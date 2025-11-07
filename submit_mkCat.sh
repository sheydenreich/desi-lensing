###################################################################################
#BGS CLUSTERING CATALOGUES 

#Instructions for generating clustering and random catalogues for the BGS using the 
#`mkCat_subsamp.py` script, starting from the `full_HPmapcut` catalogues produced by Ashley.

#This script will create a clustering catalogue for the BGS_BRIGHT sample that is selected by targetIDs.

#To run this script, on the command line enter:
#./submit_mkCat.sh
#You might need to first change the permissions by doing
#chmod 755 submit_mkCat.sh

#Set the number of random files to use
export NUM_RAND=4
#First, load the DESI environment.
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

#Finally, run the `mkCat_subsamp.py` script to generate the catalogues. This will generate the 
#file `$OUTPUT_DIR/BGS_BRIGHT_clustering.dat.fits`, with similar random files.
#`--clusd y` - set to `y` to make the clustering catalogue 
#`--clusran y` - set to `y` to make the random catalogue(s)
#`--minr 0` - first random file is set to file number `0`
#`--maxr 4` - last random file is set to file number `0`. Increase this if more randoms are used
#`--bgs_zmin -0.0033` - Minimum redshift limit (default is 0.01)
#`--bgs_zmax 1.0` - Maximum redshift limit (default is 0.5)
#`--splitGC y` - Creates NGC and SGC files, which are needed when calculating the clustering
#`--compmd altmtl` - This argument is recommended for bitweights
#`--nz y` - This is needed to refactor the weights
#`--splitGC y` - Create output files split into NGC and SGC. This is needed if you want to run `xirunpc.py`
#By default it uses the `full_HPmapcut` files
#To use the `full` files instead, include (with the empty string) --use_map_veto ''

# srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi 
# python scripts/mkCat_subsamp.py --input_tracer BGS_BRIGHT --maxr $NUM_RAND --mkfulldat y --clusd y --clusran y --nz y --splitGC y --ccut targetIDs \
#         --imsys_clus y --imsys_clus_ran y --targetIDs '/pscratch/sd/s/sven/holden_lensing/holden_desi_targetids.fits' --compmd not_altmtl \
#         --outdir $SCRATCH/holden_lensing/v1.5/

# python /global/homes/s/sven/code/LSS/scripts/mkCat_subsamp.py --input_tracer BGS_BRIGHT --maxr $NUM_RAND --mkfulldat y --clusd y --clusran y --nz y --splitGC y --ccut targetIDs \
#         --imsys_clus y --imsys_clus_ran y --compmd not_altmtl --targetIDs 'bgs_bright_magcut.fits' --verspec loa-v1 \
#         --outdir $SCRATCH/BGS_BRIGHT_magcut/

# python /global/homes/s/sven/code/LSS/scripts/mkCat_subsamp.py --input_tracer BGS_BRIGHT --maxr $NUM_RAND --mkfulldat y --clusd y --clusran y --nz y --splitGC y --ccut targetIDs \
#         --imsys_clus y --imsys_clus_ran y --compmd not_altmtl --targetIDs 'bgs_bright_magcut_onecut.fits' --verspec loa-v1 \
#         --outdir $SCRATCH/BGS_BRIGHT/

python /global/homes/s/sven/code/LSS/scripts/mkCat_subsamp.py --input_tracer LRG --maxr $NUM_RAND --mkfulldat y --clusd y --clusran y --nz y --splitGC y --ccut noccut \
        --imsys_clus y --imsys_clus_ran y --compmd not_altmtl --verspec loa-v1 \
        --outdir $SCRATCH/LRG/
