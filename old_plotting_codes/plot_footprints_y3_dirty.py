import configparser
import sys
import os
import skymapper as skm
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger
from data_handler import get_last_mtime
sys.path.append(os.path.abspath('..'))
from load_catalogues import get_lens_table,get_source_table
import astropy.units as u
from datetime import datetime

script_name = 'plot_footprint'


def place_gals_on_hpmap(lens_table, nside):
    theta = np.radians(90 - lens_table["DEC"])  # HEALPix uses colatitude (90 - Dec)
    phi = np.radians(lens_table["RA"])  # RA in radians
    pixels = hp.ang2pix(nside, theta, phi, lonlat=False)

    # Create a HEALPix map and count galaxies per pixel
    healpix_map = np.zeros(hp.nside2npix(nside), dtype=int)
    np.add.at(healpix_map, pixels, 1)

    return healpix_map


def plot_footprint(config):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    vertices_path = clean_read(config,script_name,'vertices_path',split=False)
    os.makedirs(vertices_path,exist_ok=True)

    xpos_list = [clean_read(config,script_name,'pos_'+x,split=True,convert_to_float=True)[0] for x in survey_list]
    ypos_list = [clean_read(config,script_name,'pos_'+x,split=True,convert_to_float=True)[1] for x in survey_list]
    alphas_list = [clean_read(config,script_name,'alpha_'+x,split=False,convert_to_float=True) for x in survey_list]

    smoothing = clean_read(config,script_name,'smoothing',split=False,convert_to_float=True)*u.deg
    sep = clean_read(config,script_name,'sep',split=False,convert_to_float=True)
    nside = clean_read(config,script_name,'nside',split=False,convert_to_int=True)
    os.makedirs(savepath,exist_ok=True)
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    for galaxy_type in ["BGS_BRIGHT","LRG"]:
        fig = plt.figure(figsize = (18, 9))
        proj = skm.Hammer()
        footprint = skm.Map(proj, facecolor = 'white', ax = fig.gca())
        sep = 30 # number of graticules per degree
        footprint.grid(sep = sep)

        # 4) add data to the map, e.g.
        # make density plot
        nside = 64

        pixels, rap, decp, vertices = skm.healpix.getGrid(
            nside, return_vertices = True) # returns positional information of grid

        # lens_table = get_lens_table(galaxy_type,None,None,versions=versions, logger=logger)[0]
        from astropy.table import Table
        lens_table = Table.read(f"/global/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v1.1/{galaxy_type}_full_HPmapcut.dat.fits")
        lens_table = lens_table[lens_table['ZWARN'] < 100]
        lens_table.keep_columns(['RA','DEC'])
        hpmap_observed = place_gals_on_hpmap(lens_table, nside)

        full_table = Table.read(f"/global/cfs/cdirs/desi/survey/catalogs/main/LSS/{galaxy_type}targetsDR9v1.1.1.fits")
        full_table.keep_columns(['RA','DEC'])
        hpmap_full = place_gals_on_hpmap(full_table, nside)



        # hpmap_ratio = np.where(hpmap_full > 0, hpmap_observed/hpmap_full, np.nan)
        hpmap_ratio = hpmap_observed*1./hpmap_full +1e-2

        # mappable = footprint.density(lens_table['RA'], lens_table['DEC'], nside=nside)
        # print(np.histogram(hpmap_ratio[np.isfinite(hpmap_ratio)]))
        # print(np.sum(hpmap_ratio==0),np.sum(hpmap_full==0),np.sum(hpmap_observed==0),np.sum(hpmap_observed==0)-np.sum(hpmap_full==0))
        # print(len(hpmap_ratio)-np.isnan(hpmap_ratio).sum(),len(hpmap_ratio)-(hpmap_full==0).sum(),len(hpmap_ratio)-(hpmap_observed==0).sum())
        # print(np.isnan(hpmap_ratio).sum(),(hpmap_full==0).sum(),(hpmap_observed==0).sum())
        # continue
        mappable = footprint.healpix(hpmap_ratio, vmin=-1e-2, vmax=1+1e-2, cmap='viridis')
        # mappable = footprint.healpix(hpmap_full, vmin=0, vmax=np.max(hpmap_full), cmap='viridis')

        cb = footprint.colorbar(mappable, cb_label=galaxy_type[:3]+" #spec / #phot")
        savename = galaxy_type[:3]
        # mappable = footprint.density(lrg_table['ra'], lrg_table['dec'], nside=nside)
        # cb = footprint.colorbar(mappable, cb_label="LRG $n_g$ [arcmin$^{-2}$]")


        for survey,posx,posy,color,alpha in zip(survey_list+["Unions"],xpos_list+[-20],ypos_list+[60],color_list+["firebrick"],alphas_list+[0.9]):
            savename += "_"+survey
            try:
                _fpath_load = vertices_path+"pix_{}_nside_{}_smooth_{}.npy".format(survey,nside,smoothing.value).replace(" ","")
                pix = np.load(_fpath_load)
                logger.info("Loaded "+_fpath_load+f" from {get_last_mtime(_fpath_load)}")
            except FileNotFoundError:
                logger.info("ReComputing "+survey)
                if survey == "Unions":
                    unions_pixel = np.load("/global/cfs/cdirs/desicollab/users/sven/unions/ipix_mask_UNIONS_fromgalpos.npy")
                    # unions_mask = np.zeros(hp.nside2npix(nside=2**12),dtype=bool)
                    # unions_mask[unions_pixel]=1
                    theta,phi = hp.pix2ang(nside=2**12,ipix=unions_pixel,lonlat=False)
                    # Convert angles to RA, Dec
                    ra = np.degrees(phi)  # Right Ascension (in degrees)
                    dec = 90 - np.degrees(theta)  # Declination (in degrees)
                    table = Table()
                    table['ra'] = ra
                    table['dec'] = dec
                else:
                    table = get_source_table(survey,"BGS_BRIGHT",cut_catalogues_to_DESI=False,logger=logger)[0]
                pix,vert,_ = put_survey_on_grid(table["ra"],table["dec"],
                                        rap,decp,pixels,vertices,
                                            smoothing=smoothing)
                np.save(vertices_path+"vert_{}_nside_{}_smooth_{}".format(survey,nside,smoothing.value).replace(" ",""),
                    vert)
                np.save(vertices_path+"pix_{}_nside_{}_smooth_{}".format(survey,nside,smoothing.value).replace(" ",""),
                    pix)

            myvert = vertices[get_boundary_mask(pix,nside,niter=1)]
            logger.info(f"Plotting {survey}, {len(myvert)}")

            footprint.vertex(myvert, facecolors = color, alpha=alpha)
            txt = footprint.ax.text(
                np.deg2rad(posx), np.deg2rad(posy), 
                survey,
                size = 20,
                color=color,
                horizontalalignment = 'center', 
                verticalalignment = 'bottom')
            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
        footprint.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+'footprint_{}_y3.png'.format(savename), 
                        dpi = 300, transparent = True, bbox_inches="tight", pad_inches = 0)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    plot_footprint(config)