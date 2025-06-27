import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,get_mean
from data_handler import get_last_mtime
sys.path.append(os.path.abspath('..'))
from load_catalogues import get_lens_table,get_source_table
import astropy.units as u

script_name = 'plot_split_footprints'

def plot_split_footprints(config):

    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    savepath = clean_read(config,'general','savepath',split=False)
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


    _columns_to_inspect = clean_read(config,script_name,'columns_to_inspect',split=True)
    _columns_to_inspect_LRG = clean_read(config,script_name,'columns_to_inspect',split=True)
    # _columns_to_inspect_LRG.remove("ABSMAG_RP0")

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    for galaxy_type in ["BGS_BRIGHT","LRG"]:
        if(galaxy_type=="BGS_BRIGHT"):
            columns_to_inspect = _columns_to_inspect
        elif(galaxy_type=="LRG"):
            columns_to_inspect = _columns_to_inspect_LRG

        mybgs_table = get_lens_table(galaxy_type,None,None,None,
                                    columns_to_add = columns_to_inspect,
                                    convert_to_dsigma_table=False,
                                    versions=versions,
                                    logger=logger)[0]

        pixels, rap, decp, vertices = skm.healpix.getGrid(
            nside, return_vertices = True) # returns positional information of grid

        dpix,dvert,dmask = put_survey_on_grid(mybgs_table["RA"],mybgs_table["DEC"],
                                rap,decp,pixels,vertices)


        for col in (["Z"]+columns_to_inspect):
            if col not in mybgs_table.colnames:
                import warnings
                warnings.warn("Column {} not in table".format(col))
                continue
            if col in ["PROB_OBS","logM"]:
                _mytable = mybgs_table[mybgs_table[col]>0]
            else:
                _mytable = mybgs_table
            fig = plt.figure(figsize = (18, 9))
            proj = skm.Hammer()
            footprint = skm.Map(proj, facecolor = 'white', ax = fig.gca())
            footprint.grid(sep = sep)

            logger.info("Looking at "+col)
            meancolstr = r"$\langle$"+col+r"$\rangle$"
            full_color,color_weights = get_mean(_mytable["RA"],_mytable["DEC"],_mytable[col],nside,
                                                        print_counts=True)
            mycolor = full_color[dmask]

            nanmask_full_color = ~np.isnan(full_color)
            nanmask_mycolor = ~np.isnan(mycolor)

            mappable = footprint.vertex(dvert,color=mycolor,
                                    cmap = "YlOrRd")

            cb = footprint.colorbar(mappable, cb_label=galaxy_type[:3]+meancolstr)
            savename = galaxy_type[:3]+"_split_"+col
            for survey,posx,posy,color,alpha in zip(survey_list,xpos_list,ypos_list,color_list,alphas_list):
                savename += "_"+survey
                try:
                    _fpath_pix = vertices_path+"pix_{}_nside_{}_smooth_{}.npy".format(survey,nside,smoothing.value).replace(" ","")
                    pix = np.load(vertices_path+"pix_{}_nside_{}_smooth_{}.npy".format(survey,nside,smoothing.value).replace(" ",""))
                    vert = np.load(vertices_path+"vert_{}_nside_{}_smooth_{}.npy".format(survey,nside,smoothing.value).replace(" ",""))
                    mask = np.load(vertices_path+"mask_{}_nside_{}_smooth_{}.npy".format(survey,nside,smoothing.value).replace(" ",""))
                    logger.info(f"Loaded {_fpath_pix} from {get_last_mtime(_fpath_pix)}")
                except FileNotFoundError:
                    logger.info("ReComputing "+survey)
                    table = get_source_table(survey,"BGS_BRIGHT",cut_catalogues_to_DESI=False,
                                             logger=logger)[0]
                    pix,vert,mask = put_survey_on_grid(table["ra"],table["dec"],
                                            rap,decp,pixels,vertices,
                                                smoothing=smoothing)
                    np.save(vertices_path+"vert_{}_nside_{}_smooth_{}".format(survey,nside,smoothing.value).replace(" ",""),
                        vert)
                    np.save(vertices_path+"pix_{}_nside_{}_smooth_{}".format(survey,nside,smoothing.value).replace(" ",""),
                        pix)
                    np.save(vertices_path+"mask_{}_nside_{}_smooth_{}".format(survey,nside,smoothing.value).replace(" ",""),
                        mask)

                myvert = vertices[get_boundary_mask(pix,nside,niter=1)]
                logger.info("Plotting"+survey)
                overlapmask = (mask & dmask & nanmask_full_color)
                overlapval = np.average(full_color[overlapmask],weights=color_weights[overlapmask])
                overlapstd = np.sqrt(np.average((full_color[overlapmask]-overlapval)**2,
                                                weights=color_weights[overlapmask]))

                footprint.vertex(myvert, facecolors = color, alpha=alpha, 
                                label=survey+": "+meancolstr+"= {:.2f}$\\,\\pm\\,${:.2f}".format(overlapval,overlapstd))
                txt = footprint.ax.text(
                    np.deg2rad(posx), np.deg2rad(posy), 
                    survey,
                    size = 20,
                    color=color,
                    horizontalalignment = 'center', 
                    verticalalignment = 'bottom')
                txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

            fig.legend(fontsize='x-large',
                    loc='upper left')
            
            footprint.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+'footprint_{}.png'.format(savename), 
                            dpi = 300, transparent = True, bbox_inches="tight", pad_inches = 0)
            plt.close()

if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    plot_split_footprints(config)