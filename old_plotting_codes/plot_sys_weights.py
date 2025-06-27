
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
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

script_name = 'plot_sys_weights'
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def plot_sys_weights(config):
    # "Viridis-like" colormap with white background

    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    savepath = clean_read(config,'general','savepath',split=False)
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)



    bgs_table = get_lens_table("BGS_BRIGHT",None,None,None,
                            columns_to_add = ["WEIGHT_SYS","WEIGHT_SYS_ORIG"],
                            convert_to_dsigma_table=False,
                            versions=versions,
                            logger=logger)[0]

    lrg_table = get_lens_table("LRG",None,None,None,
                        columns_to_add = ["WEIGHT_SYS","WEIGHT_SYS_ORIG"],
                        convert_to_dsigma_table=False,
                        versions=versions,
                        logger=logger)[0]

    
    mask_bgs_sys = np.isfinite(bgs_table['WEIGHT_SYS'])
    mask_bgs_sys_orig = np.isfinite(bgs_table['WEIGHT_SYS_ORIG'])
    bgs_table = bgs_table[mask_bgs_sys & mask_bgs_sys_orig]
    logger.info(f"WEIGHT_SYS had {np.sum(~mask_bgs_sys)} NaNs")
    logger.info(f"WEIGHT_SYS_ORIG had {np.sum(~mask_bgs_sys_orig)} NaNs")



    def using_mpl_scatter_density(fig, nrow, ncol, idx, x, y):
        ax = fig.add_subplot(nrow, ncol, idx, projection='scatter_density')
        density = ax.scatter_density(x, y, cmap=white_viridis)
        return ax
        # fig.colorbar(density, label='Number of points per pixel')
    fig = plt.figure(figsize=(10,6))
    ax0 = using_mpl_scatter_density(fig,1,2,1,bgs_table['WEIGHT_SYS'],bgs_table['WEIGHT_SYS_ORIG'])
    ax1 = using_mpl_scatter_density(fig,1,2,2,lrg_table['WEIGHT_SYS'],lrg_table['WEIGHT_SYS_ORIG'])
    ax0.set_xlim(0.5,1.5)
    ax0.set_ylim(0.5,1.5)
    ax0.set_title("BGS_BRIGHT")
    ax1.set_xlim(0.5,1.5)
    ax1.set_ylim(0.5,1.5)
    ax1.set_title("LRG")
    ax0.set_xlabel('WEIGHT_SYS')
    ax0.set_ylabel('WEIGHT_SYS_ORIG')
    ax1.set_xlabel('WEIGHT_SYS')
    fig.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+'sys_weights.png',
                dpi=300)
        
if __name__ == '__main__':
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    plot_sys_weights(config)