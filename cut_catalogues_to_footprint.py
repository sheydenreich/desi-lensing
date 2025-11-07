from copy import deepcopy
import os
import numpy as np
import sys
import healpy as hp
from glob import glob
from astropy.table import Table
import argparse

def clean_filename(filename):
    return filename.replace("targetIDs","").replace("noccut","")

# version = "v0.4.5"
def cut_to_footprint(table, _lensing_survey, galaxy_type, ra_col="RA", dec_col="DEC", verbose=False, copy=True,
                    footfile_fpath = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/footprints_desiy3/",nside=1024):
    if(copy):
        table = deepcopy(table)
    # print(f"Cutting {galaxy_type} to {lensing_survey}")
    if "decade" in _lensing_survey:
        lensing_survey = _lensing_survey.split("_")[0]
        ngcsgc = _lensing_survey.split("_")[1]
    else:
        lensing_survey = _lensing_survey
        ngcsgc = None
    if galaxy_type=="LRG+ELG":
        footpix = np.zeros(hp.nside2npix(nside))
        for galaxy_loadname in ["LRG","ELG"]:
            footfile = footfile_fpath+f"{lensing_survey}_Y3{galaxy_loadname}_nside{nside}.fits"
            footpix += hp.fitsfunc.read_map(footfile)
    else:
        if(galaxy_type=="LRG"):
            galaxy_loadname="LRG"
        elif galaxy_type in ["BGS","BGS_BRIGHT"]:
            galaxy_loadname="BGS"
        elif galaxy_type == "ELG":
            galaxy_loadname="ELG"
        else:
            raise ValueError(f"galaxy_type {galaxy_type} not recognized")
        footfile = footfile_fpath+f"{lensing_survey}_Y3{galaxy_loadname}_nside{nside}.fits"
        footpix = hp.fitsfunc.read_map(footfile)
    if ngcsgc is not None:
        if ngcsgc == "ngc":
            footpix *= hp.read_map("/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/model_inputs_desiy3/galactic_caps/ngc_only.fits")
        elif ngcsgc == "sgc":
            footpix *= hp.read_map("/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/model_inputs_desiy3/galactic_caps/sgc_only.fits")
        else:
            raise ValueError(f"ngcsgc {ngcsgc} not recognized")
    phi,theta = np.radians(table[ra_col]),np.radians(90.-table[dec_col])
    ipix = hp.ang2pix(nside,theta,phi,nest=False)
    cut = (footpix[ipix] > 0.)
    ngals_before = len(table[ra_col])
    table = table[cut]
    ngals_after = len(table[ra_col])
    if(verbose):
        print(f"{ngals_after} galaxies remaining after cuting DESI {galaxy_type} to {_lensing_survey}. Before cut: {ngals_before}")
    return table

import numpy as np
from copy import deepcopy
import healpy as hp

def radec_to_healpix(ra,dec,nside=1024):
    phi,theta = np.radians(ra),np.radians(90.-dec)
    ipix = hp.ang2pix(nside,theta,phi,nest=False)
    return ipix

def healpix_to_radec(ipix,nside=1024):
    theta,phi = hp.pix2ang(nside,ipix,nest=False)
    ra,dec = np.degrees(phi),90.-np.degrees(theta)
    return ra,dec

def cut_to_joint_footprint(table_1,table_2,desi_galaxy_type=None,source_survey=None,ra_col="RA",dec_col="DEC",
                           verbose=False,copy=True,footfile_path="/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/footprints_desiy3/",
                           nside=1024):
    if(copy):
        table_1 = deepcopy(table_1)
        table_2 = deepcopy(table_2)

    ipix_1 = radec_to_healpix(table_1[ra_col],table_1[dec_col],nside=nside)
    ipix_2 = radec_to_healpix(table_2[ra_col],table_2[dec_col],nside=nside)

    if desi_galaxy_type is not None:
        assert nside==1024, "Pre-computed healpix maps are only available for nside=1024"
        assert source_survey is not None, "source_survey must be provided to load Pre-computed healpix map"

        if(verbose):
            print(f"Cutting {desi_galaxy_type} and {source_survey} to joint footprint!")

        if desi_galaxy_type=="LRG+ELG":
            footpix = np.zeros(hp.nside2npix(nside),dtype=bool)
            for desi_galaxy_type in ["LRG","ELG"]:
                footfile = footfile_path+f"{source_survey}_Y3{desi_galaxy_type}_nside{nside}.fits"
                footpix += hp.fitsfunc.read_map(footfile)
        else:
            if(galaxy_type=="LRG"):
                galaxy_loadname="LRG"
            elif galaxy_type in ["BGS","BGS_BRIGHT"]:
                galaxy_loadname="BGS"
            elif galaxy_type == "ELG":
                galaxy_loadname="ELG"
            else:
                raise ValueError(f"galaxy_type {galaxy_type} not recognized")
            footfile = footfile_path+f"{lensing_survey}_Y3{galaxy_loadname}_nside{nside}.fits"
            footpix = hp.fitsfunc.read_map(footfile)

    else:
        if(verbose):
            print("Computing footprint from table_1")
        footpix_1 = np.zeros(hp.nside2npix(nside),dtype=bool)
        footpix_1[ipix_1] = True
        if(verbose):
            print("Computing footprint from table_2")
        footpix_2 = np.zeros(hp.nside2npix(nside),dtype=bool)
        footpix_2[ipix_2] = True
        footpix = footpix_1*footpix_2

    cut_1 = footpix[ipix_1]
    cut_2 = footpix[ipix_2]

    if(verbose):
        print(f"{np.sum(cut_1)}/{len(cut_1)} galaxies remaining in table_1 after cuting to joint footprint")
        print(f"{np.sum(cut_2)}/{len(cut_2)} galaxies remaining in table_2 after cuttring to joint footprint")
    table_1 = table_1[cut_1]
    table_2 = table_2[cut_2]

    return table_1,table_2


if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Cut catalogues to footprint')
    parser.add_argument('input_path', help='Input path containing FITS files')
    parser.add_argument('--output_path', '-o', help='Output path (if not specified, uses input path)')
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path if args.output_path is not None else input_path
    
    # Ensure paths end with '/'
    if not input_path.endswith('/'):
        input_path += '/'
    if not output_path.endswith('/'):
        output_path += '/'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    
    files = glob(input_path+'*.fits')
    print(files)
    if "pip" in input_path.split('/')[-2]:  # Check parent directory name
        files = [file for file in files if "clustering" in file]
    else:
        files = [file for file in files if ("clustering" in os.path.basename(file) or "full_HPmapcut" in os.path.basename(file))]
        files = [file for file in files if not "ELG" in os.path.basename(file)]

    for fpath_load in files:
        filename = os.path.basename(fpath_load)
        if "NGC" in filename or "SGC" in filename:
            continue

        print("Reading "+fpath_load)
        galcat_data = Table.read(fpath_load)
        galaxy_type = clean_filename(filename.split("_")[0])
        for lensing_survey in ["decade_ngc","decade_sgc","kids1000N","desy3","hscy3"]:
            fpath_save_cut = output_path+f"{lensing_survey}/"
            os.makedirs(fpath_save_cut,exist_ok=True)
            data_cat_processed_cut = cut_to_footprint(galcat_data,lensing_survey,galaxy_type,verbose=True)
            print("Writing "+fpath_save_cut+clean_filename(filename))
            data_cat_processed_cut.write(fpath_save_cut+clean_filename(filename),overwrite=True)

