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

script_name = 'boost_factor'

def boost_factor_tomo(config,plot,logger="create"):
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
    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False)


    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    # print("Got B")
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    if logger == "create":
        logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_tomo',__name__)

    p_list_randoms, V_list_randoms, key_list_randoms = load_randoms_values(config)
    p_randoms_std = np.std(p_list_randoms,axis=0)
    V_randoms_std = np.zeros((*p_randoms_std.shape,p_randoms_std.shape[-1]))
    for i in range(p_list_randoms.shape[1]):
        V_randoms_std[i] = np.cov(p_list_randoms[:,i,:].T)

    all_lens_amplitudes = {}
    all_lens_amplitude_errors = {}

    fig,ax,gs = initialize_gridspec_figure((2*len(survey_list),
                                2*len(galaxy_types)*get_number_of_lens_bins(galaxy_types[0])),len(galaxy_types)*get_number_of_lens_bins(galaxy_types[0]),
                        len(survey_list))
    add_colorbar_legend(fig,ax,gs,color_list,["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5"])

    for gt,galaxy_type in enumerate(galaxy_types):
        n_lens_bins = get_number_of_lens_bins(galaxy_type)
        for lens_bin in range(n_lens_bins):
            scales_mask = get_scales_mask_from_degrees(rp,'all scales',min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)


            for ss,source_survey in enumerate(survey_list):
                _,_,_,mycov,full_zlens,full_zsource,_,boostfactor = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path,"deltasigma",
                                                                                versions,logger=logger,boost=True,
                                                                                return_additional_quantity='b')
                logger.info("Using reference datavector")
                # full_data = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
                # full_data = np.tile(full_data,get_number_of_source_bins(source_survey)*n_lens_bins)



                if(use_theory_covariance):
                    full_cov = load_covariance_chris(galaxy_type,source_survey,"deltasigma",
                                                    chris_path)
                else:
                    full_cov = mycov

                # apply systematic to datavector

                systematics_factor = 1.+load_dv_johannes(galaxy_type,source_survey,chris_path,
                                                        'deltasigma',logger,
                                                        systype='boost')/100

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                n_source_bins = len(allowed_bins)
                if(n_source_bins==0):
                    if logger is not None:
                        logger.info(f"Skipping {galaxy_type} {source_survey} {lens_bin}")
                for source_bin in allowed_bins:
                    bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[source_bin])
                    ax[gt*3+lens_bin,ss].plot(rp,systematics_factor[bin_mask],color=color_list[source_bin])
                    ax[gt*3+lens_bin,ss].plot(rp,boostfactor[bin_mask],marker='x',ls="",color=color_list[source_bin])#,yerr=np.sqrt(np.diag(full_cov)[bin_mask])
                    ax[gt*3+lens_bin,ss].set_xscale('log')


                if(gt*3+lens_bin==0):
                    ax[gt*3+lens_bin,ss].set_title(source_survey)
                if(ss==0):
                    ax[gt*3+lens_bin,ss].set_ylabel(f"$\Delta\Sigma$, {galaxy_type} bin {lens_bin+1}")
                if(gt*3+lens_bin==len(galaxy_types)*n_lens_bins-1):
                    ax[gt*3+lens_bin,ss].set_xlabel(r"$r_p$")

    ax[0,0].plot([],[],color='black',label='Simulated')
    ax[0,0].plot([],[],color='black',ls='',marker='x',label='Measured')
    ax[0,0].legend()

    if(plot):
        plt.tight_layout()
        fstr = ""
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"boost_factor.png",
                    dpi=300,transparent=False,bbox_inches="tight")
        plt.close()
    return None

if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    boost_factor_tomo(config,plot=True)
