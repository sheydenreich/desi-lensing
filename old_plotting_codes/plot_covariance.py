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
                        get_pvalue,get_rp_from_deg,get_scales_mask_from_degrees,generate_bmode_tomo_latex_table_from_dict,get_ntot,full_covariance_bin_mask
import matplotlib.gridspec as gridspec
from copy import deepcopy
                        

script_name = 'covariance'

def normalize_covariance(cov):
    return cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))


def plot_full_covmat(config,bmodes=False,include_sdss=True):
    config = deepcopy(config)
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

    transparent_background = clean_read(config,'general','transparent_background',split=False,convert_to_bool=True)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_bmodes_notomo',__name__)

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    if "HSCY3" in survey_list:
        hscy3=True
    elif "HSCY1" in survey_list:
        hscy3=False
    else:
        raise ValueError("HSCY3 or HSCY1 must be in survey")

    for gt,galaxy_type in enumerate(galaxy_types):
        n_lens_bins = 3
        full_cov = load_covariance_chris(galaxy_type,"all_y3" if hscy3 else "all_y1",statistic,
                                chris_path,pure_noise=bmodes,
                                include_sdss=include_sdss)
        fig,ax = plt.subplots(1,1,figsize=(7.24*2,7.24*2))
        im = ax.imshow(normalize_covariance(full_cov),cmap='RdBu_r',vmin=-1,vmax=1)
        # cbar = plt.colorbar(im,ax=ax)
        # cbar.set_label("Normalized covariance")
        # ax.set_title(f"{galaxy_type} {statistic} {'Bmodes' if bmodes else 'Data'}")

        if "HSCY3" in survey_list:
            mysurveys = ["KiDS","DES","HSCY3"]
        elif "HSCY1" in survey_list:
            mysurveys = ["KiDS","DES","HSCY1"]
        else:
            raise ValueError("HSCY3 or HSCY1 must be in survey")
        if(include_sdss):
            mysurveys.append("SDSS")

        survey_sep = 0

        for ss,source_survey in enumerate(mysurveys):
            source_bins = get_number_of_source_bins(source_survey)
            n_radial_bins = get_number_of_radial_bins(galaxy_type,source_survey,None)
            survey_sep_before = survey_sep
            if ss!=0:
                plt.axvline(survey_sep-0.5,color='k',linestyle='-')
                plt.axhline(survey_sep-0.5,color='k',linestyle='-')
            for i in range(n_lens_bins):
                if i!=0:
                    plt.axhline(survey_sep-0.5,color='k',linestyle='--')
                    plt.axvline(survey_sep-0.5,color='k',linestyle='--')
                for j in range(source_bins):
                    if j!=0:
                        plt.axhline(survey_sep-0.5,color='k',linestyle=':')
                        plt.axvline(survey_sep-0.5,color='k',linestyle=':')
                    survey_sep += n_radial_bins
                    
            ax.text((survey_sep_before+0.5*(survey_sep-survey_sep_before))/full_cov.shape[0],1.02,
                    source_survey,transform=ax.transAxes,fontsize=20,horizontalalignment='center')



        plt.tight_layout()
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_{galaxy_type}_full_cov_bmodes_{bmodes}.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
        plt.close()

def plot_covmat(config,bmodes=False,include_sdss=True):
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

    transparent_background = clean_read(config,'general','transparent_background',split=False,convert_to_bool=True)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_bmodes_notomo',__name__)

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    if "HSCY3" in survey_list:
        hscy3=True
    elif "HSCY1" in survey_list:
        hscy3=False
    else:
        raise ValueError("HSCY3 or HSCY1 must be in survey")

    for gt,galaxy_type in enumerate(galaxy_types):
        n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
        full_cov = load_covariance_chris(galaxy_type,"all_y3" if hscy3 else "all_y1",statistic,
                                chris_path,pure_noise=bmodes,
                                include_sdss=include_sdss)
        fig,ax = plt.subplots(1,1,figsize=(7.24*2,7.24*2))
        covariance_mask = np.zeros(full_cov.shape[0],dtype=bool)

        # cbar = plt.colorbar(im,ax=ax)
        # cbar.set_label("Normalized covariance")
        # ax.set_title(f"{galaxy_type} {statistic} {'Bmodes' if bmodes else 'Data'}")

        if "HSCY3" in survey_list:
            mysurveys = ["KiDS","DES","HSCY3"]
        elif "HSCY1" in survey_list:
            mysurveys = ["KiDS","DES","HSCY1"]
        else:
            raise ValueError("HSCY3 or HSCY1 must be in survey")
        if(include_sdss):
            mysurveys.append("SDSS")

        survey_sep = 0
        survey_seps = [0]

        for ss,source_survey in enumerate(mysurveys):
            source_bins = get_number_of_source_bins(source_survey)
            n_radial_bins = get_number_of_radial_bins(galaxy_type,source_survey,None)
            if ss!=0:
                plt.axvline(survey_sep-0.5,color='k',linestyle='-')
                plt.axhline(survey_sep-0.5,color='k',linestyle='-')
            for lens_bin in range(n_lens_bins):
                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                if lens_bin!=0:
                    plt.axhline(survey_sep-0.5,color='k',linestyle='--')
                    plt.axvline(survey_sep-0.5,color='k',linestyle='--')
                for source_bin in allowed_bins:
                    if source_bin!=0:
                        plt.axhline(survey_sep-0.5,color='k',linestyle=':')
                        plt.axvline(survey_sep-0.5,color='k',linestyle=':')
                    full_mask = full_covariance_bin_mask(galaxy_type,source_survey,lens_bin,source_bin,include_sdss=include_sdss)
                    rp = get_rp_chris(galaxy_type,source_survey,chris_path,statistic,logger)[:n_radial_bins]
                    scales_mask = get_scales_mask_from_degrees(rp,'all scales',min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                    covariance_mask[full_mask] = scales_mask
                    survey_sep += np.sum(scales_mask)
            survey_seps.append(survey_sep)
            
        for ss,source_survey in enumerate(mysurveys):
        
            if(survey_seps[ss] < survey_seps[ss+1]):
                ax.text((survey_seps[ss]+0.5*(survey_seps[ss+1]-survey_seps[ss]))/np.sum(covariance_mask),1.02,
                        source_survey,transform=ax.transAxes,fontsize=20,horizontalalignment='center')
        im = ax.imshow(normalize_covariance(full_cov[covariance_mask][:,covariance_mask]),cmap='RdBu_r',vmin=-1,vmax=1)



        plt.tight_layout()
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_{galaxy_type}_cov_bmodes_{bmodes}.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
        plt.close()


def compare_jackknife(config,bmodes=False):
    config = deepcopy(config)
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

    transparent_background = clean_read(config,'general','transparent_background',split=False,convert_to_bool=True)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_bmodes_tomo',__name__)

    if "HSCY3" in survey_list:
        hscy3=True
    elif "HSCY1" in survey_list:
        hscy3=False
    else:
        raise ValueError("HSCY3 or HSCY1 must be in survey")
    
    if "SDSS" in survey_list:
        idx_sdss = survey_list.index('SDSS')
        survey_list.pop(idx_sdss)


    # gs = gridspec.GridSpec(get_number_of_lens_bins(galaxy_types[0]),len(galaxy_types)+1,
    #                         width_ratios = [20]*len(galaxy_types)+[1])

    fig,ax,gs = initialize_gridspec_figure((7.24,7.24/n_total_bins*len(survey_list)*1.2),
                        len(survey_list)*2,
                        n_total_bins,
                        hspace=0,wspace=0,
                        height_ratios = [2, 1]*len(survey_list),
                        )
    add_colorbar_legend(fig,ax,gs,color_list,[f"Bin {i+1}" for i in range(5)],skip=2)

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
          

                _,data,_,mycov,zlens,zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path,statistic,
                                                                                versions,logger=logger,
                                                                                correct_for_magnification_bias=not bmodes,
                                                                                bmodes=bmodes)
                
                cov = load_covariance_chris(galaxy_type,source_survey,statistic,
                                            chris_path,pure_noise=bmodes)
                
                # full_datvec = np.zeros((cov.shape[0]))
                # full_mask = np.zeros((cov.shape[0],len(scales_list)),dtype=bool)

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                for idb,myBin in enumerate(allowed_bins):
                    # full_bin_mask = full_covariance_bin_mask(galaxy_type,source_survey,lens_bin,myBin,
                    #                                     include_sdss=True)
                    source_bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[myBin])

                    errors = np.sqrt(np.diag(cov[source_bin_mask][:,source_bin_mask]))
                    myerrors = np.sqrt(np.diag(mycov[source_bin_mask][:,source_bin_mask]))
                    ax[2*ss,ax_x].plot(rp,rp*errors,color=color_list[myBin])
                    ax[2*ss,ax_x].plot(rp,rp*myerrors,color=color_list[myBin],linestyle='--')
                    ax[2*ss,ax_x].set_xscale('log')

                    ax[2*ss+1,ax_x].plot(rp,myerrors/errors,color=color_list[myBin])
                    ax[2*ss+1,ax_x].axhline(1,color='k',linestyle='--')



                    if(statistic=="deltasigma"):

                        if(ax_x==0):
                            fstr = "_\\times" if bmodes else ""
                            ax[2*ss,ax_x].set_ylabel(f"{source_survey}\n $r_p\\times\\sigma_{{\\Delta\\Sigma{fstr}}}(r_p)$")
                            ax[2*ss+1,ax_x].set_ylabel(r"$\frac{\mathrm{Data}}{\mathrm{Theory}}$")
                            # ax[ss,ax_x].set_ylim(-2,2)
                        if(ss==len(survey_list)-1):
                            ax[2*ss,ax_x].set_xlabel(r"$r_p\,[\mathrm{Mpc/h}]$")

                    else:
                        if(ax_x==0):
                            ax[2*ss,ax_x].set_ylabel(f"{source_survey}\n $\\theta\\times\\gamma_\\mathrm{{t}}(\\theta)$")
                            # ax[lens_bin,gt].set_ylim(-5e-5,5e-5)
                        if(lens_bin==n_lens_bins-1):
                            ax[lens_bin,gt].set_xlabel(r"$\theta\,[deg]$")
            





    from plot_datavector import plot_scalecuts
    plot_scalecuts(ax,min_deg,max_deg,rp_pivot,galaxy_types,config,tomo=True)
    plt.tight_layout()
    ax[0,-2].plot([],[],color='k',label='Theory')
    ax[0,-2].plot([],[],color='k',linestyle='--',label='Jackknife')
    ax[0,-2].legend(loc='upper right')
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_jk_vs_theory_bmodes_{bmodes}.png",
                dpi=300,transparent=transparent_background,bbox_inches="tight")
    plt.close()
    # return pvalues,chisqs


if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")

    # plot_datavector_notomo(config)
    compare_jackknife(config,bmodes=False)
    compare_jackknife(config,bmodes=True)
    plot_covmat(config,bmodes=False,include_sdss=True)
    plot_covmat(config,bmodes=True,include_sdss=True)
    # plot_full_covmat(config,bmodes=False,include_sdss=True)
    # plot_full_covmat(config,bmodes=True,include_sdss=True)


    # savepath = clean_read(config,'general','savepath',split=False) + os.sep
    # savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    # version = clean_read(config,'general','version',split=False)
    # statistic = clean_read(config,'general','statistic',split=False)

    # mytab = generate_bmode_tomo_latex_table_from_dict(pvalues, config, caption=f"{config.get('general','statistic')} p-values for B-modes", precision=3)
    # fil = open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"tab_{statistic}_pvalues_bmodes_tomo.tex", "w")
    # fil.write(mytab)
    # fil.close()

    # mytab = generate_bmode_tomo_latex_table_from_dict(chisqs, config, caption=f"{config.get('general','statistic')} $\\chi^2$ for B-modes", precision=1)
    # fil = open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"tab_{statistic}_chisqs_bmodes_tomo.tex", "w")
    # fil.write(mytab)
    # fil.close()
