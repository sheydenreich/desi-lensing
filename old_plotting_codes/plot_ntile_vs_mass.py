import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,clean_read,get_logger
sys.path.append(os.path.abspath('..'))
from load_catalogues import get_lens_table,get_source_table
import astropy.units as u
from datetime import datetime

script_name = 'plot_ntile_vs_mass'

mass_column = "LOGMSTAR"

def plot_ntile_vs_mass(config):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    fig,ax = plt.subplots(1,2,figsize=(12,6))
    for x,galtype in enumerate(["BGS_BRIGHT","LRG"]):
        tab,_ = get_lens_table(galtype,None,None,dsigma_additional_columns=[mass_column,"NTILE"],
                            versions=versions,logger=logger)

        ntiles = np.unique(tab["NTILE"])
        means = np.zeros(len(ntiles))
        means_w = np.zeros(len(ntiles))
        stds = np.zeros(len(ntiles))
        for i in range(len(ntiles)):
            mask = ((tab["NTILE"]==ntiles[i]) &\
                    np.isfinite(tab[mass_column]) &\
                    (tab[mass_column]>0) &\
                    np.isfinite(tab["w_sys"]))
            _tab = tab[mask]
            means[i] = np.average(_tab[mass_column])
            means_w[i] = np.average(_tab[mass_column],weights=_tab["w_sys"])
            stds[i] = np.std(_tab[mass_column])/np.sqrt(np.sum(mask))
        ax[x].errorbar(ntiles,means,stds,fmt='o',label='unweighted',
                    color="C0")
        ax[x].errorbar(ntiles+0.1,means_w,stds,fmt='o',label='weighted',
                    color="C1")
        ax[x].set_ylabel(r"$\log(M_*/M_\odot)$")
        ax[x].set_xlabel("NTILE")
        ax[x].set_title(galtype)
    ax[1].legend()

    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+'ntile_vs_mass.png', 
                dpi = 300, transparent = True, bbox_inches="tight", pad_inches = 0)
    plt.close()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    plot_ntile_vs_mass(config)
