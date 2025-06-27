import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,add_colorbar_legend,\
                                initialize_gridspec_figure
import astropy.units as u
from datetime import datetime
from data_handler import load_covariance_chris,get_rp_chris,get_allowed_bins,get_number_of_source_bins,get_bins_mask,\
                        load_data_and_covariance_notomo,load_data_and_covariance_tomo,get_number_of_lens_bins,combine_datavectors,\
                        get_number_of_radial_bins,get_reference_datavector_of_galtype,get_scales_mask,get_deltasigma_amplitudes,\
                        get_pvalue,get_rp_from_deg,get_scales_mask_from_degrees,generate_bmode_tomo_latex_table_from_dict,get_ntot,full_covariance_bin_mask,\
                        load_clustering_measurements,load_mstar_complete_clustering_measurements
import matplotlib.gridspec as gridspec
from copy import deepcopy
                        

script_name = 'plot_wp'

def plot_wp(config,mstar_complete=False):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    galaxy_types = clean_read(config,'general','galaxy_types',split=True)
    statistic = clean_read(config,'general','statistic',split=False)

    min_deg = clean_read(config,'general','min_deg',split=False,convert_to_float=True)
    max_deg = clean_read(config,'general','max_deg',split=False,convert_to_float=True)
    rp_pivot = clean_read(config,'general','rp_pivot',split=False,convert_to_float=True)
    scales_list = clean_read(config,'general','analyzed_scales',split=True)

    linlog = clean_read(config,script_name,'linlog',split=False)
    if linlog.lower() == 'log':
        logscale = True
    elif linlog.lower() == 'lin':
        logscale = False
    else:
        raise ValueError("angular_scale must be either 'log' or 'lin'")

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    if mstar_complete:
        n_LRG_bins = 0
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    transparent_background = clean_read(config,'general','transparent_background',split=False,convert_to_bool=True)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    fig,ax,gs = initialize_gridspec_figure((7.24,2.5),
                                    1,
                                    n_total_bins,hspace=0,wspace=0
                                    )

    ax[0,0].set_xscale('log')
    legends = ["NTILE split {}".format(i+1) for i in range(4)]
    legends += ["All"]
    color_list = color_list[:len(legends)-1]+['k']
    add_colorbar_legend(fig,ax,gs,color_list,legends)

    dats = {}
    covs = {}
    for gt,galaxy_type in enumerate(galaxy_types):
        n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
        if mstar_complete and galaxy_type == "LRG":
            n_lens_bins = 0
        for lens_bin in range(n_lens_bins):
            if not mstar_complete:
                for idx_ntile,NTILE in enumerate(list(range(4))+[None]):
                    rp,data,cov = load_clustering_measurements(galaxy_type,lens_bin,NTILE)
                    if NTILE is None:
                        ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].errorbar(rp,#*np.exp(0.05*(idx_ntile-2.5)),
                                                rp*data,rp*np.sqrt(np.diag(cov)),
                                                    fmt='o',color=color_list[idx_ntile],
                                                    markersize=2,zorder=10)
                        np.savetxt("/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/wp_measurements/results/covariances/" + \
                                   f"covariance_wp_{galaxy_type}_{lens_bin}.dat",cov)
                    else:
                        ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].errorbar(rp*np.exp(0.05*(idx_ntile-2.5)),
                            rp*data,rp*np.sqrt(np.diag(cov)),
                                ls=':',color=color_list[idx_ntile])
            else:
                rp,data,cov = load_mstar_complete_clustering_measurements(galaxy_type,lens_bin)
                ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].errorbar(rp,
                        rp*data,rp*np.sqrt(np.diag(cov)),
                            fmt='o',color=color_list[-1])

            ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_xscale('log')
                

            ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")
            ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_xlabel(r"$r_p\,[\mathrm{Mpc/h}]$")
            if(gt==0):
                if logscale:
                    ax[0,0].set_ylabel(f"$w_p(r_p)$")
                    
                else:
                    ax[0,0].set_ylabel(f"$r_p\\times w_p(r_p)$")
                    if mstar_complete:
                        ax[0,0].set_ylim(0,300)
                    else:
                        ax[0,0].set_ylim(0,230)
                        # ax[0,0].set_ylim(0,300)

    plot_scalecuts(ax,min_deg,max_deg,rp_pivot,galaxy_types if not mstar_complete else ["BGS_BRIGHT"],config,statistic)
    plt.tight_layout()
    if mstar_complete:
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"wp_mstar_complete.png",
                dpi=300,transparent=transparent_background,bbox_inches="tight")
    else:
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"wp_datavector_ntile.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
    plt.close()
    return

def plot_scalecuts(ax,degmin,degmax,rpcut,galaxy_types,config,shared_axes=True,tomo=False):
    statistic = clean_read(config,'general','statistic',split=False)
    if(shared_axes):
        axmin,axmax = ax[0,0].get_xlim()

    for j,galaxy_type in enumerate(galaxy_types):
        niter = range(ax.shape[0])
        n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
        for xiter in niter:
            for i in range(n_lens_bins):
                if(statistic=="deltasigma"):
                    rpmin,rpmax = get_rp_from_deg(degmin,degmax,galaxy_type,i,config)
                    ax[xiter,i+j*config.getint('general','N_BGS_BRIGHT_bins')].axvline(rpcut,color='k',linestyle=':')
                else:
                    rpmin=degmin
                    rpmax=degmax
                if not shared_axes:
                    axmin,axmax = ax[0,i+j*config.getint('general','N_BGS_BRIGHT_bins')].get_xlim()
                ax[xiter,i+j*config.getint('general','N_BGS_BRIGHT_bins')].axvspan(axmin,rpmin,color='gray',alpha=0.5)
                ax[xiter,i+j*config.getint('general','N_BGS_BRIGHT_bins')].axvspan(rpmax,axmax,color='gray',alpha=0.5)
                if not shared_axes:
                    ax[xiter,i+j*config.getint('general','N_BGS_BRIGHT_bins')].set_xlim(axmin,axmax)
    if shared_axes:
        ax[0,0].set_xlim(axmin,axmax)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")

    # plot_wp(config,mstar_complete=True)
    plot_wp(config)