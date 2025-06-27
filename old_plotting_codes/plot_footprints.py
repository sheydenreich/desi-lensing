import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,get_mask,get_vertices_from_pixels
import healpy as hp
from data_handler import get_last_mtime
sys.path.append(os.path.abspath('..'))
from load_catalogues import get_lens_table,get_source_table
import astropy.units as u
from datetime import datetime

script_name = 'plot_footprint'

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

    # DIRTY: SDSS is ACT
    sdss_idx = survey_list.index("SDSS")
    survey_list[sdss_idx] = "ACT"
    xpos_list[sdss_idx] = -68
    ypos_list[sdss_idx] = 15
    # color_list[sdss_idx] = "orange"

    for galaxy_type in ["BGS_BRIGHT","LRG"]:
        fig = plt.figure(figsize = (18, 9))
        proj = skm.Hammer()
        footprint = skm.Map(proj, facecolor = 'white', ax = fig.gca())
        sep = 30 # number of graticules per degree
        footprint.grid(sep = sep)

        # 4) add data to the map, e.g.
        # make density plot
        nside = 256

        pixels, rap, decp, vertices = skm.healpix.getGrid(
            nside, return_vertices = True) # returns positional information of grid

        lens_table = get_lens_table(galaxy_type,None,None,versions=versions, logger=logger)[0]

        mappable = footprint.density(lens_table['ra'], lens_table['dec'], nside=nside, cmap="Greys")
        cb = footprint.colorbar(mappable, cb_label=galaxy_type[:3]+" $n_g$ [arcmin$^{-2}$]")
        savename = galaxy_type[:3]
        # mappable = footprint.density(lrg_table['ra'], lrg_table['dec'], nside=nside)
        # cb = footprint.colorbar(mappable, cb_label="LRG $n_g$ [arcmin$^{-2}$]")


        for survey,posx,posy,color,alpha in zip(survey_list,xpos_list,ypos_list,color_list,alphas_list):
            savename += "_"+survey
            try:
                _fpath_load = vertices_path+"pix_{}_nside_{}_smooth_{}.npy".format(survey,nside,smoothing.value).replace(" ","")
                pix = np.load(_fpath_load)
                vert = np.load(vertices_path+"vert_{}_nside_{}_smooth_{}.npy".format(survey,nside,smoothing.value).replace(" ",""))
                logger.info("Loaded "+_fpath_load+f" from {get_last_mtime(_fpath_load)}")
            except FileNotFoundError:
                logger.info("ReComputing "+survey)
                if survey != "ACT":
                    table = get_source_table(survey,"BGS_BRIGHT",cut_catalogues_to_DESI=False,logger=logger)[0]
                    pix,vert,_ = put_survey_on_grid(table["ra"],table["dec"],
                                            rap,decp,pixels,vertices,
                                                smoothing=smoothing)
                else:
                    hpmap = get_mask("AdvACT")
                    # scale up to nside
                    hpmap = hp.ud_grade(hpmap,nside)
                    pix = np.where(hpmap)[0]
                    vert = vertices[pix]
                np.save(vertices_path+"vert_{}_nside_{}_smooth_{}".format(survey,nside,smoothing.value).replace(" ",""),
                    vert)
                np.save(vertices_path+"pix_{}_nside_{}_smooth_{}".format(survey,nside,smoothing.value).replace(" ",""),
                    pix)

            myvert = vertices[get_boundary_mask(pix,nside,niter=2)]
            logger.info(f"Plotting {survey}, {len(myvert)}")

            footprint.vertex(myvert, facecolors = color, alpha=alpha)
            footprint.vertex(vert, facecolors = color, alpha=alpha*0.25)
            txt = footprint.ax.text(
                np.deg2rad(posx), np.deg2rad(posy), 
                survey,
                size = 20,
                color=color,
                horizontalalignment = 'center', 
                verticalalignment = 'bottom')
            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
        footprint.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+'footprint_{}.pdf'.format(savename), 
                        dpi = 300, transparent = False, bbox_inches="tight")#, pad_inches = 0)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    plot_footprint(config)