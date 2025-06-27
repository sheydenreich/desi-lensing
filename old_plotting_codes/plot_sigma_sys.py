import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,add_colorbar_legend,initialize_gridspec_figure
import astropy.units as u
from datetime import datetime
from data_handler import load_covariance_chris,get_rp_chris,get_allowed_bins,get_number_of_source_bins,get_bins_mask,\
                        load_data_and_covariance_notomo,load_data_and_covariance_tomo,get_number_of_lens_bins,combine_datavectors,\
                        get_number_of_radial_bins,get_reference_datavector,get_scales_mask,get_deltasigma_amplitudes,\
                        get_reference_datavector_of_galtype,get_scales_mask_from_degrees,calculate_sigma_sys
import matplotlib.gridspec as gridspec

script_name = 'sigma_sys'

def plot_sigma_sys(config,plot,logger="create"):
    version = clean_read(config,'general','version',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    if("SDSS" in survey_list):
        idx_sdss = survey_list.index("SDSS")
        survey_list.remove("SDSS")
        import warnings
        warnings.warn("SDSS not available here!")
        color_list.pop(idx_sdss)

    galaxy_types = clean_read(config,'general','galaxy_types',split=True)
    scales_list = clean_read(config,'general','analyzed_scales',split=True)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)
    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins
    transparent_background = config.getboolean('general','transparent_background')

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    # print("Got B")
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    if logger == "create":
        logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)


    systematics_list = clean_read(config,'secondary_effects','systematics_list',split=True)


    if(plot):
        if logger is not None:
            logger.info("Preparing plot")
        fig,ax,gs = initialize_gridspec_figure((7.24,7.24/n_total_bins*len(scales_list)),
                            len(scales_list),
                            n_total_bins,
                            hspace=0,wspace=0)

    sigma_sys_list = np.load(savepath_slope_values + os.sep + version + os.sep +"source_redshift_slope"+ os.sep + "source_redshift_slope" + "_tomo_sigma_sys_list.npy",allow_pickle=True).item()
    reduced_chisq_list = np.load(savepath_slope_values + os.sep + version + os.sep +"source_redshift_slope"+ os.sep + "source_redshift_slope" + "_tomo_reduced_chisq_list.npy",allow_pickle=True).item()

    fstr = ""
    for systematic in systematics_list:
        fstr += f"_{systematic}"

    amplitude_list = np.load(savepath_slope_values + os.sep + version + os.sep +"secondary_effects"+ os.sep + f"lens_amplitudes_combined_systematics{fstr}.npy",allow_pickle=True).item()
    error_list = np.load(savepath_slope_values + os.sep + version + os.sep +"secondary_effects"+ os.sep + f"lens_amplitude_errors_combined_systematics{fstr}.npy",allow_pickle=True).item()

    for scale,scaletitle in enumerate(scales_list):
        for gt,galaxy_type in enumerate(galaxy_types):
            n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
            for lens_bin in range(n_lens_bins):
                ax_y = lens_bin+gt*n_BGS_BRIGHT_bins
                ax_x = scale
                ax[0,ax_y].set_xlim(-0.1,len(survey_list)+0.75)
                ax[ax_x,0].set_ylim(0,0.18)

                sigma_sys = np.array(sigma_sys_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"])
                if np.isfinite(sigma_sys[2]):
                    sigma_sys = np.nan_to_num(sigma_sys,copy=True,nan=0)
                reduced_chisq = reduced_chisq_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"]
                ax[ax_x,ax_y].errorbar(len(survey_list),sigma_sys[2],yerr=np.array([sigma_sys[2]-sigma_sys[0],
                                                                             sigma_sys[4]-sigma_sys[2]]).reshape((2,1)),fmt='o',color='blue')
                ax[ax_x,ax_y].errorbar(len(survey_list),sigma_sys[2],yerr=np.array([sigma_sys[2]-sigma_sys[1],
                                                                sigma_sys[3]-sigma_sys[2]]).reshape((2,1)),fmt='o',color='red')
                
                if np.isfinite(reduced_chisq):
                    ax[ax_x,ax_y].text(0.5,0.95,f"$\\chi^2_\\nu={reduced_chisq:.2f}$",
                                    transform=ax[ax_x,ax_y].transAxes,
                                    horizontalalignment='center',
                                    verticalalignment='top')
                
                # print(galaxy_type,scaletitle,lens_bin,sigma_sys)
                for ss,source_survey in enumerate(survey_list):
                    try:
                        amplitudes = np.array(amplitude_list[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}"])
                        errors = np.array(error_list[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}"])
                        sigma_sys_mock = np.sqrt(np.average((amplitudes)**2,weights=1/errors**2))
                        sigma_stat = np.sqrt(np.mean(errors**2))
                        # print(galaxy_type,scaletitle,lens_bin,source_survey,sigma_sys_mock,sigma_stat)
                        ax[ax_x,ax_y].plot(ss,sigma_sys_mock,'^',color=color_list[ss])
                        print(sigma_sys_mock)
                        ax[ax_x,ax_y].plot(ss,sigma_stat,'v',color=color_list[ss])
                    except KeyError as ke:
                        # print(ke)
                        pass
                if(ax_x==len(scales_list)-1):
                    ax[ax_x,ax_y].set_xticks(np.arange(len(survey_list)+1))
                    ax[ax_x,ax_y].set_xticklabels([*survey_list,''],rotation=45)
                if(ax_x==0):
                    ax[ax_x,ax_y].set_title(f"{galaxy_type[:3]} bin {lens_bin+1}")
                if(ax_y==0):
                    ax[ax_x,ax_y].set_ylabel(f"$\\sigma_{{\\Delta\\Sigma}}$\n {scaletitle}")
    ax[0,-2].errorbar([],[],fmt='o',color='red',label=r'$\sigma_\mathrm{sys}^\mathrm{measured}$')
    ax[0,-2].plot([],[],'^',color='black',label=r'$\sigma_\mathrm{sys}^\mathrm{mock}$')
    ax[0,-2].plot([],[],'v',color='black',label=r'$\sigma_\mathrm{stat}$')
    ax[0,-2].legend(loc='lower right')


    plt.tight_layout()
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"sigma_sys.png",
                dpi=300,transparent=transparent_background,bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")

    plot_sigma_sys(config,plot=True,logger=None)