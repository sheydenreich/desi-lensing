from copy import deepcopy
import numpy as np
import healpy as hp
from astropy.table import Table
import os

from itertools import chain, combinations

import sys

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

lensingsurvey_name_dict = {'sdss':'sdss',
                           'hscy1':'hscy1',
                           'des':'desy3',
                           'kids':'kids1000N',
                           'hscy3':'hscy3',
                           'decade':'decade'}


def get_mask_footprint(_source_survey,galaxy_type,ras,dec,fpath="/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/footprints_desiy3/",
                      nside=1024):
    # if galaxy_type in ["BGS","BGS_BRIGHT"]:
    #     fgal = "BGS"
    # elif galaxy_type in ["LRG"]:
    #     fgal = "LRG"
    # else:
    #     raise ValueError("Invalid galaxy type")
    if "decade" in _source_survey.lower():
        source_survey = _source_survey.split("_")[0]
        ngcsgc = _source_survey.split("_")[1]
    else:
        source_survey = _source_survey
        ngcsgc = None
    if type(galaxy_type) == str:
        footfile = fpath+f"{lensingsurvey_name_dict[source_survey.lower()]}_Y3{galaxy_type[:3]}_nside{nside}.fits"
        footpix = hp.fitsfunc.read_map(footfile)
    else:
        footpix = False
        for gtype in galaxy_type:
            footfile = fpath+f"{lensingsurvey_name_dict[source_survey.lower()]}_Y3{gtype[:3]}_nside{nside}.fits"
            _footpix = hp.fitsfunc.read_map(footfile)
            footpix = np.logical_or(footpix,_footpix)
    if ngcsgc is not None:
        if ngcsgc.lower() == "ngc":

            ngcsgc_mask = hp.read_map("/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/model_inputs_desiy3/galactic_caps/ngc_only.fits")
        elif ngcsgc.lower() == "sgc":
            ngcsgc_mask = hp.read_map("/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/model_inputs_desiy3/galactic_caps/sgc_only.fits")
        else:
            raise ValueError(f"ngcsgc {ngcsgc} not recognized")
        footpix = np.logical_and(footpix,ngcsgc_mask)
    phi,theta = np.radians(ras),np.radians(90.-dec)
    ipix = hp.ang2pix(nside,theta,phi,nest=False)
    cut = (footpix[ipix] > 0.)
    return cut

def cut_to_footprint(table, lensing_survey, galaxy_type, ra_col="RA", dec_col="DEC", verbose=False, copy=True,
                    footfile_fpath = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/footprints_desiy3/",nside=1024):
    if(copy):
        table = deepcopy(table)
    if(galaxy_type=="LRG"):
        galaxy_loadname="LRG"
    if galaxy_type in ["BGS","BGS_BRIGHT"]:
        galaxy_loadname="BGS"
    if galaxy_type == "ELG":
        galaxy_loadname="ELG"
    footfile = footfile_fpath+f"{lensing_survey}_Y3{galaxy_loadname}_nside{nside}.fits"
    footpix = hp.fitsfunc.read_map(footfile)
    phi,theta = np.radians(table[ra_col]),np.radians(90.-table[dec_col])
    ipix = hp.ang2pix(nside,theta,phi,nest=False)
    cut = (footpix[ipix] > 0.)
    ngals_before = len(table[ra_col])
    table = table[cut]
    ngals_after = len(table[ra_col])
    if(verbose):
        print(f"{ngals_after} galaxies remaining after cuting DESI {galaxy_type} to {lensing_survey}. Before cut: {ngals_before}")
    return table


fpath = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensingsurvey_catalogues/"
fpath_load = {"KiDS":fpath+"{}kids/kids_cat{}.fits",
             "HSCY1":fpath+"{}hsc/hsc_cat{}.fits",
             "DES":fpath+"{}desy3/desy3_cat_newsompz{}.fits",
             "HSCY3":fpath+"{}hscy3/hscy3_cat{}.fits",
             "DECADE_NGC":fpath+"{}decade/decade_ngc_cat{}.hdf5",
             "DECADE_SGC":fpath+"{}decade/decade_sgc_cat{}.hdf5",
             }

for source_survey in ["DECADE_NGC","DECADE_SGC","KiDS","DES","HSCY3"]:
# for source_survey in ["DES"]:
    for galaxy_types in powerset(["BGS","LRG","ELG"]):
        print("Doing",source_survey,galaxy_types)
        tab = Table.read(fpath_load[source_survey].format("","").replace("_newsompz",""))
        mask = get_mask_footprint(source_survey,galaxy_types,tab["RA"],tab["Dec"])
        tab = tab[mask]
        savestring = ""
        for galaxy_type in galaxy_types:
            savestring += f"_{galaxy_type}"
        # print(os.path.basename(fpath_load[source_survey].format("cut_catalogues/","_"+galaxy_type)))
        print(fpath_load[source_survey].format("cut_catalogues/",savestring))
        # print(os.path.dirname(fpath_load[source_survey].format("cut_catalogues/","_"+galaxy_type)))
        os.makedirs(os.path.dirname(fpath_load[source_survey].format("cut_catalogues/","_"+galaxy_type)),exist_ok=True)
        tab.write(fpath_load[source_survey].format("cut_catalogues/",savestring),overwrite=True)
