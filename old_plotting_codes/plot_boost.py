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
                        get_precomputed_table,load_dv_johannes
from dsigma.stacking import boost_factor
import matplotlib.gridspec as gridspec
from copy import deepcopy

plt.rcParams['errorbar.capsize'] = 1.5  # Default cap size for error bars
plt.rcParams['lines.linewidth'] = 0.5  # Default line width (affects error bars too)
plt.rcParams['lines.markersize'] = 1.5

script_name = 'plot_boost'



def plot_boost(config,plot=True,datavec=None):
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

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    linlog = clean_read(config,script_name,'linlog',split=False)
    if linlog.lower() == 'log':
        logscale = True
    elif linlog.lower() == 'lin':
        logscale = False
    else:
        raise ValueError("angular_scale must be either 'log' or 'lin'")

    transparent_background = clean_read(config,'general','transparent_background',split=False,convert_to_bool=True)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_bmodes_tomo',__name__)


    offset = clean_read(config,script_name,'offset',split=False,convert_to_float=True)

    if "HSCY3" in survey_list:
        hscy3=True
    elif "HSCY1" in survey_list:
        hscy3=False
    else:
        raise ValueError("HSCY3 or HSCY1 must be in survey")


    # gs = gridspec.GridSpec(get_number_of_lens_bins(galaxy_types[0]),len(galaxy_types)+1,
    #                         width_ratios = [20]*len(galaxy_types)+[1])
    if(plot):
        fig,ax,gs = initialize_gridspec_figure((7.24,7.24/n_total_bins*len(survey_list)),
                            len(survey_list),
                            n_total_bins,
                            hspace=0,wspace=0)
        add_colorbar_legend(fig,ax,gs,color_list,[f"Bin {i+1}" for i in range(5)])

    for gt,galaxy_type in enumerate(galaxy_types):
        n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
        # full_cov = load_covariance_chris(galaxy_type,"all_y3" if hscy3 else "all_y1",
        #                     statistic,chris_path,pure_noise=True,include_sdss=True)
        
        for lens_bin in range(n_lens_bins):
            n_radial_bins = get_number_of_radial_bins(galaxy_type,survey_list[0],None)
            rp = get_rp_chris(galaxy_type,survey_list[0],chris_path,
                                statistic,logger)[:n_radial_bins]
            for ss,source_survey in enumerate(survey_list):
                ax_x = lens_bin+gt*n_BGS_BRIGHT_bins

                if(ss==0):
                    ax[ss,ax_x].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")
            

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                if source_survey != "SDSS":
                    systematics_factor = 1.+load_dv_johannes(galaxy_type,source_survey,chris_path,
                                                            'deltasigma',logger,
                                                            systype='boost')/100
                else:
                    systematics_factor = np.nan*np.ones(3*len(rp))

                for idb,myBin in enumerate(allowed_bins):
                    from time import time
                    start_time = time()
                    print(f"plotting {galaxy_type} {source_survey} bin l{lens_bin} s{myBin}")
                    if os.path.exists(f"/pscratch/sd/s/sven/lensingWithoutBorders/data/boost/boost_factor_{galaxy_type}_{source_survey}_l{lens_bin}_s{myBin}.npy"):
                        boost = np.load(f"/pscratch/sd/s/sven/lensingWithoutBorders/data/boost/boost_factor_{galaxy_type}_{source_survey}_l{lens_bin}_s{myBin}.npy")
                    else:
                        table_l,table_r = get_precomputed_table(galaxy_type,source_survey,data_path,version,
                                                                statistic,lens_bin,myBin,randoms=True,boost=False)
                        print(f"table loaded, calculating boost factor, took {time()-start_time} seconds")
                        start_time = time()
                        boost = boost_factor(table_l,table_r)
                        print(f"boost factor calculated, took {time()-start_time} seconds")
                        np.save(f"/pscratch/sd/s/sven/lensingWithoutBorders/data/boost/boost_factor_{galaxy_type}_{source_survey}_l{lens_bin}_s{myBin}.npy",boost)
                        del table_l,table_r
                        import gc
                        gc.collect()

                    bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[myBin])

                    ax[ss,ax_x].plot(rp,boost,color=color_list[myBin])
                    ax[ss,ax_x].plot(rp,systematics_factor[bin_mask],color=color_list[myBin],ls="--")
                    ax[ss,ax_x].set_xscale('log')



                    if(statistic=="deltasigma"):

                        if(ax_x==0):
                            if logscale:
                                ax[ss,ax_x].set_ylabel(f"{source_survey}\n boost factor")
                            else:
                                ax[ss,ax_x].set_ylabel(f"{source_survey}\n boost factor")
                            # ax[ss,ax_x].set_ylim(-2,2)
                        if(ss==len(survey_list)-1):
                            ax[ss,ax_x].set_xlabel(r"$r_p\,[\mathrm{Mpc/h}]$")

                    else:
                        if(ax_x==0):
                            ax[ss,ax_x].set_ylabel(f"{source_survey}\n $\\theta\\times\\gamma_\\mathrm{{t}}(\\theta)$")
                        ax[lens_bin,gt].set_ylim(-5e-5,5e-5)
                        if(lens_bin==n_lens_bins-1):
                            ax[lens_bin,gt].set_xlabel(r"$\theta\,[deg]$")
            





    if(plot):
        plot_scalecuts(ax,min_deg,max_deg,rp_pivot,galaxy_types,config,tomo=True)
        plt.tight_layout()
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_boost_tomo.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
    # return pvalues,chisqs

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
                    # print(galaxy_type,", bin ",i+1,", rpmin",rpmin," Mpc/h, rpmax",rpmax, " Mpc/h")
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

    plot_boost(config,plot=True)
