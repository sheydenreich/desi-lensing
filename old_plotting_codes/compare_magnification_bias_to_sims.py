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
                        get_number_of_radial_bins,get_reference_datavector,get_scales_mask,get_deltasigma_amplitudes,\
                        get_reference_datavector_of_galtype,load_dv_johannes,get_scales_mask_from_degrees,load_randoms_values
import matplotlib.gridspec as gridspec

script_name = 'secondary_effects'

def compare_magbias(config,plot,datavec=None,logger="create",all_zsource=None):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)
    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    if("SDSS" in survey_list):
        idx_sdss = survey_list.index("SDSS")
        survey_list.remove("SDSS")
        import warnings
        warnings.warn("SDSS not available here!")
        color_list.pop(idx_sdss)
    galaxy_types = clean_read(config,'general','galaxy_types',split=True)

    min_deg = clean_read(config,'general','min_deg',split=False,convert_to_float=True)
    max_deg = clean_read(config,'general','max_deg',split=False,convert_to_float=True)
    rp_pivot = clean_read(config,'general','rp_pivot',split=False,convert_to_float=True)
    scales_list = clean_read(config,'general','analyzed_scales',split=True)

    rp = clean_read(config,'general','rp',split=True,convert_to_float=True)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)
    slope_uncertainty = clean_read(config,script_name,'slope_uncertainty',split=False)

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    # print("Got B")
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    if logger == "create":
        logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_tomo',__name__)

    if logger is not None:
        logger.info("Preparing plot")
    fig,ax,gs = initialize_gridspec_figure((7.24,6),
                                    len(survey_list),
                                    n_total_bins,hspace=0,wspace=0
                                    )

    add_colorbar_legend(fig,ax,gs,color_list,[f"Bin {i}" for i in range(1,6)])

    for gt,galaxy_type in enumerate(galaxy_types):
        n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
        for lens_bin in range(n_lens_bins):
            n_radial_bins = get_number_of_radial_bins(galaxy_type,survey_list[0],None)
            scales_mask = get_scales_mask_from_degrees(rp,'all scales',min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
            n_used_bins = np.sum(scales_mask)

            for ss,source_survey in enumerate(survey_list):
                _,_,_,mycov,full_zlens,full_zsource,_,magbias = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path,"deltasigma",
                                                                                versions,logger=logger,
                                                                                return_additional_quantity='magnification_bias')


                magbias_measured = load_dv_johannes(galaxy_type,source_survey,chris_path,
                                                        'deltasigma',logger,dvtype='absolute',
                                                        systype='lens_magnification')
                dmonly_measured = load_dv_johannes(galaxy_type,source_survey,chris_path,
                                                        'deltasigma',logger,dvtype='absolute',
                                                        systype='gravitational')
                
                # magbias_measured -= dmonly_measured

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                n_source_bins = len(allowed_bins)
                ax_x = lens_bin+gt*n_BGS_BRIGHT_bins

                if ss==0:
                    ax[ss,ax_x].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")
                if ax_x==0:
                    ax[ss,ax_x].set_ylabel(source_survey + "\n $r_p\\,\\Delta\\Sigma_\\mathrm{LSS}$")
                if ss==len(survey_list)-1:
                    ax[ss,ax_x].set_xlabel("$r_p$ [Mpc/h]")

                for source_bin in allowed_bins:
                    bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,source_bin)
                    _magbias = magbias[bin_mask]
                    _magbias_measured = magbias_measured[bin_mask]
                    ax[ss,ax_x].plot(rp,rp*_magbias,color=color_list[source_bin],ls='--')
                    ax[ss,ax_x].plot(rp,rp*_magbias_measured,color=color_list[source_bin],ls='-')
                    # ax[ss,ax_x].plot(rp,_magbias/_magbias_measured,color=color_list[source_bin],ls='-')
                    ax[ss,ax_x].set_xscale('log')
                    ax[ss,ax_x].set_ylim(-0.05,0.5)



    plt.tight_layout()
    fstr = ""
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"magbias_compare.png",
                dpi=300,transparent=False,bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")

    compare_magbias(config,True)