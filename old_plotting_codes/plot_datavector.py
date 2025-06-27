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

plt.rcParams['errorbar.capsize'] = 1.5  # Default cap size for error bars
plt.rcParams['lines.linewidth'] = 0.5  # Default line width (affects error bars too)
plt.rcParams['lines.markersize'] = 1.5

script_name = 'plot_datavector'

def plot_datavector_notomo(config,datavec=None):
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
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    transparent_background = clean_read(config,'general','transparent_background',split=False,convert_to_bool=True)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_tomo',__name__)


    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    if(use_theory_covariance):
        logger.info("Using theory covariance")
    else:
        logger.info("Using jackknife covariance")
    use_optimal_matrix = clean_read(config,script_name,'use_optimal_matrix',split=False,convert_to_bool=True)
    if(use_optimal_matrix):
        logger.info("Using optimal matrix for data compression")
    else:
        logger.info("Using full matrix for data compression")
    
    offset = clean_read(config,script_name,'offset',split=False,convert_to_float=True)

    fig,ax,gs = initialize_gridspec_figure((7.24,2.5),
                                    1,
                                    n_total_bins,hspace=0,wspace=0
                                    )

    ax[0,0].set_xscale('log')
    add_colorbar_legend(fig,ax,gs,color_list[:len(survey_list)],survey_list)
    for gt,galaxy_type in enumerate(galaxy_types):
        n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
        for lens_bin in range(n_lens_bins):
            n_radial_bins = get_number_of_radial_bins(galaxy_type,survey_list[0],None)
            rp = get_rp_chris(galaxy_type,survey_list[0],chris_path,
                                statistic,logger)[:n_radial_bins]
            zlenses = []
            for ss,source_survey in enumerate(survey_list):
                _,data,_,mycov,zlens,zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path,statistic,
                                                                                versions,logger=logger)
                if(datavec is not None):
                    logger.info(f"Using mock datavector for {galaxy_type} {source_survey}")
                    data = datavec[f"{galaxy_type}_{source_survey}"]

                if(use_theory_covariance):
                    cov = load_covariance_chris(galaxy_type,source_survey,statistic,
                                                    chris_path)
                else:
                    cov = mycov

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,allowed_bins)
                n_source_bins = len(allowed_bins)
                if(n_source_bins==0):
                    logger.info(f"Skipping {galaxy_type} {source_survey} {lens_bin}")
                    continue
                data = data[bin_mask]
                zsource = zsource[bin_mask]
                zlens = zlens[bin_mask]
                cov = cov[bin_mask][:,bin_mask]
                
                if(n_source_bins>1):
                    cdata,ccov = combine_datavectors(data,cov,optimal_matrix=use_optimal_matrix,
                                                    n_radial_bins=n_radial_bins)
                    czsource,_ = combine_datavectors(zsource,cov,optimal_matrix=use_optimal_matrix,
                                                    n_radial_bins=n_radial_bins)
                    czlens,_ = combine_datavectors(zlens,cov,optimal_matrix=use_optimal_matrix,
                                                    n_radial_bins=n_radial_bins)
                else:
                    cdata = data
                    ccov = cov
                    czsource = zsource
                    czlens = zlens
                ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].errorbar(rp*np.exp(offset*ss),
                                         rp*cdata,rp*np.sqrt(np.diag(ccov)),
                                            fmt='o',color=color_list[ss])
                ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_xscale('log')
                

                zlenses.append(np.mean(czlens))


                # if(lens_bin==0):
                ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")
                if(statistic=="deltasigma"):
                    if(gt==0):
                        if logscale:
                            ax[0,0].set_ylabel(f"$\\Delta\\Sigma(r_p)$")
                        
                        else:
                            ax[0,0].set_ylabel(f"$r_p\\times\\Delta\\Sigma(r_p)$")
                            ax[0,0].set_ylim(0,10)
                    # if(lens_bin==n_lens_bins-1):
                    ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_xlabel(r"$r_p\,[\mathrm{Mpc/h}]$")
                else:
                    if(gt==0):
                        ax[0,gt].set_ylabel(f"$\\theta\\times\\gamma_\\mathrm{{t}}$")
                        ax[0,gt].set_ylim(0,5e-4)
                    # if(lens_bin==n_lens_bins-1):
                    ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_xlabel(r"$\theta\,[\mathrm{deg}]$")

            reference_dv = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
            if logscale:
                ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].plot(rp,reference_dv,color='k',linestyle='--')
                ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_yscale('log')
            else:
                ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].plot(rp,rp*reference_dv,color='k',linestyle='--')
    plot_scalecuts(ax,min_deg,max_deg,rp_pivot,galaxy_types,config,statistic)
    plt.tight_layout()
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_datavector_notomo.png",
                dpi=300,transparent=transparent_background,bbox_inches="tight")
    plt.close()
    return

def plot_bmodes_notomo(config,plot=True,datavec=None):
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


    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    if(use_theory_covariance):
        logger.info("Using theory covariance")
    else:
        logger.info("Using jackknife covariance")
    use_optimal_matrix = clean_read(config,script_name,'use_optimal_matrix',split=False,convert_to_bool=True)
    if(use_optimal_matrix):
        logger.info("Using optimal matrix for data compression")
    else:
        logger.info("Using full matrix for data compression")
    
    offset = clean_read(config,script_name,'offset',split=False,convert_to_float=True)


    # gs = gridspec.GridSpec(get_number_of_lens_bins(galaxy_types[0]),len(galaxy_types)+1,
    #                         width_ratios = [20]*len(galaxy_types)+[1])
    if(plot):
        fig,ax,gs = initialize_gridspec_figure((7.24,2.5),
                                        1,n_total_bins,
                                        hspace=0,wspace=0
                                        )
    
        add_colorbar_legend(fig,ax,gs,color_list[:len(survey_list)],survey_list)
    pvalues = {}
    for gt,galaxy_type in enumerate(galaxy_types):
        n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
        for lens_bin in range(n_lens_bins):
            n_radial_bins = get_number_of_radial_bins(galaxy_type,survey_list[0],None)
            rp = get_rp_chris(galaxy_type,survey_list[0],chris_path,
                                statistic,logger)[:n_radial_bins]
            for ss,source_survey in enumerate(survey_list):
                _,data,_,mycov,zlens,zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path,statistic,
                                                                                versions,bmodes=True,logger=logger,
                                                                                correct_for_magnification_bias=False)
                if(datavec is not None):
                    logger.info(f"Using mock datavector for {galaxy_type} {source_survey}")
                    data = datavec[f"{galaxy_type}_{source_survey}"]

                if(use_theory_covariance):
                    cov = load_covariance_chris(galaxy_type,source_survey,statistic,
                                                    chris_path,pure_noise=True)
                else:
                    cov = mycov

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,allowed_bins)
                n_source_bins = len(allowed_bins)
                if(n_source_bins==0):
                    logger.info(f"Skipping {galaxy_type} {source_survey} {lens_bin}")
                    continue
                data = data[bin_mask]
                cov = cov[bin_mask][:,bin_mask]
                
                if(n_source_bins>1):
                    cdata,ccov = combine_datavectors(data,cov,optimal_matrix=use_optimal_matrix,
                                                    n_radial_bins=n_radial_bins)
                else:
                    cdata = data
                    ccov = cov

                for scales in scales_list:
                    scales_mask = get_scales_mask_from_degrees(rp,scales,min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                    _cdata = cdata[scales_mask]
                    _ccov = ccov[scales_mask][:,scales_mask]
                    pvalue = get_pvalue(_cdata,_ccov)
                    pvalues[f"{galaxy_type}_{source_survey}_{lens_bin}_{scales}"] = pvalue
                    chisq = np.einsum("i,ij,j",_cdata,np.linalg.inv(_ccov),_cdata)

                if(plot):
                    ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_xscale('log')

                    ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].errorbar(rp*np.exp(offset*ss),
                                            rp*cdata,rp*np.sqrt(np.diag(ccov)),
                                                fmt='o',color=color_list[ss])
                    ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].text(0.05,0.92-0.07*(len(survey_list)-ss-1),f"p={pvalue:.3f}, $\\chi^2$={chisq:.1f}",
                                         transform=ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].transAxes,
                                         color=color_list[ss],
                                         fontsize=8,
                                        path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')]
                                        )



                    # if(lens_bin==0):
                    ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")
                    if(statistic=="deltasigma"):

                        if(gt==0):
                            ax[0,0].set_ylabel(f"$r_p\\times\\Delta\\Sigma_\\times(r_p)$")
                            ax[0,0].set_ylim(-2,2)
                        # if(lens_bin==n_lens_bins-1):
                        ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_xlabel(r"$r_p\,[\mathrm{Mpc/h}]$")

                    else:
                        if(gt==0):
                            ax[0,0].set_ylabel(f"$r_p\\times\\gamma_\\times(\\theta)$")
                            ax[0,0].set_ylim(-5e-5,5e-5)
                        # if(lens_bin==n_lens_bins-1):
                        ax[0,lens_bin+n_BGS_BRIGHT_bins*gt].set_xlabel(r"$\theta\,[deg]$")


    if(plot):
        plot_scalecuts(ax,min_deg,max_deg,rp_pivot,galaxy_types,config)
        plt.tight_layout()
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_bmodes_notomo.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
        plt.close()
    return pvalues

def plot_bmodes_tomo(config,plot=True,datavec=None):
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


    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    if(use_theory_covariance):
        logger.info("Using theory covariance")
    else:
        logger.info("Using jackknife covariance")
    use_optimal_matrix = clean_read(config,script_name,'use_optimal_matrix',split=False,convert_to_bool=True)
    if(use_optimal_matrix):
        logger.info("Using optimal matrix for data compression")
    else:
        logger.info("Using full matrix for data compression")
    
    offset = clean_read(config,script_name,'offset',split=False,convert_to_float=True)

    if 'SDSS' in survey_list:
        idx_sdss = survey_list.index('SDSS')
        survey_list.pop(idx_sdss)


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
    pvalues = {}
    chisqs = {}
    for gt,galaxy_type in enumerate(galaxy_types):
        n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
        # full_cov = load_covariance_chris(galaxy_type,"all_y3" if hscy3 else "all_y1",
        #                     statistic,chris_path,pure_noise=True,include_sdss=True)
        
        for ss,source_survey in enumerate(survey_list):
            for lens_bin in range(n_lens_bins):
                ax_x = lens_bin+gt*n_BGS_BRIGHT_bins

                if(ss==0):
                    ax[ss,ax_x].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")

                # full_datvec = np.zeros((full_cov.shape[0]))
                # full_mask = np.zeros((full_cov.shape[0],len(scales_list)),dtype=bool)

                n_radial_bins = get_number_of_radial_bins(galaxy_type,survey_list[0],None)
                rp = get_rp_chris(galaxy_type,survey_list[0],chris_path,
                                    statistic,logger)[:n_radial_bins]
            

                _,data,_,mycov,zlens,zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path,statistic,
                                                                                versions,bmodes=True,logger=logger,
                                                                                correct_for_magnification_bias=False)
                
                cov = load_covariance_chris(galaxy_type,source_survey,statistic,
                                            chris_path,pure_noise=True)
                
                full_datvec = np.zeros((cov.shape[0]))
                full_mask = np.zeros((cov.shape[0],len(scales_list)),dtype=bool)

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                for idb,myBin in enumerate(allowed_bins):
                    # full_bin_mask = full_covariance_bin_mask(galaxy_type,source_survey,lens_bin,myBin,
                    #                                     include_sdss=True)
                    source_bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[myBin])
                    for idx_scale,scales in enumerate(scales_list):
                        scales_mask = get_scales_mask_from_degrees(rp,scales,min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                        
                        # full_datvec[full_bin_mask] = data[source_bin_mask]
                        # full_mask[full_bin_mask,idx_scale] = scales_mask
                        full_datvec[source_bin_mask] = data[source_bin_mask]
                        full_mask[source_bin_mask,idx_scale] = scales_mask
                    if(plot):
                        # errors = np.sqrt(np.diag(full_cov[full_bin_mask][:,full_bin_mask]))
                        errors = np.sqrt(np.diag(cov[source_bin_mask][:,source_bin_mask]))
                        ax[ss,ax_x].errorbar(rp*np.exp(offset*ss),
                                                rp*data[source_bin_mask],rp*errors,
                                                    fmt='o',color=color_list[myBin])
                        ax[ss,ax_x].set_xscale('log')
                        ax[ss,ax_x].axhline(0,color='k',linestyle='--')



                        if(statistic=="deltasigma"):

                            if(ax_x==0):
                                ax[ss,ax_x].set_ylabel(f"{source_survey}\n $r_p\\times\\Delta\\Sigma_\\times(r_p)$")
                                ax[ss,ax_x].set_ylim(-2,2)
                            if(ss==len(survey_list)-1):
                                ax[ss,ax_x].set_xlabel(r"$r_p\,[\mathrm{Mpc/h}]$")

                        else:
                            if(ax_x==0):
                                ax[ss,ax_x].set_ylabel(f"{source_survey}\n $\\theta\\times\\gamma_\\times(\\theta)$")
                                ax[lens_bin,gt].set_ylim(-5e-5,5e-5)
                            if(lens_bin==n_lens_bins-1):
                                ax[lens_bin,gt].set_xlabel(r"$\theta\,[deg]$")
            
                idx_all_scales = scales_list.index("all scales")
                # chisq_data = full_datvec[full_mask[:,idx_all_scales]]
                # chisq_cov = full_cov[full_mask[:,idx_all_scales]][:,full_mask[:,idx_all_scales]]
                chisq_data = full_datvec[full_mask[:,idx_all_scales]]
                chisq_cov = cov[full_mask[:,idx_all_scales]][:,full_mask[:,idx_all_scales]]
                pvalue = get_pvalue(chisq_data,chisq_cov)
                chisq = np.einsum("i,ij,j",chisq_data,np.linalg.inv(chisq_cov),chisq_data)

                # print(f"{galaxy_type} {source_survey} {lens_bin} p={pvalue:.3f}, chi2={chisq:.1f}, dof={len(chisq_data)}")
                if galaxy_type=="LRG" and source_survey=="HSCY3" and lens_bin==0:
                    np.savetxt("lrg_hscy3_bmodes.dat",chisq_data)
                    np.savetxt("lrg_hscy3_bmodes_cov.dat",chisq_cov)
                if np.isfinite(pvalue):
                    pvalues[f"{galaxy_type}_{source_survey}_{lens_bin}"] = pvalue
                    chisqs[f"{galaxy_type}_{source_survey}_{lens_bin}"] = chisq
                    ax[ss,gt*n_BGS_BRIGHT_bins+lens_bin].text(0.05,0.9,f"p={pvalue:.3f}, $\\chi^2$={chisq:.1f}",
                        transform=ax[ss,gt*n_BGS_BRIGHT_bins+lens_bin].transAxes,
                        color='k',fontsize=8,
                        path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')]
                        )




    if(plot):
        plot_scalecuts(ax,min_deg,max_deg,rp_pivot,galaxy_types,config,tomo=True)
        plt.tight_layout()
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_bmodes_tomo.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
        plt.close()
    return pvalues,chisqs

def plot_datavector_tomo(config,plot=True,datavec=None):
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


    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    if(use_theory_covariance):
        logger.info("Using theory covariance")
    else:
        logger.info("Using jackknife covariance")
    use_optimal_matrix = clean_read(config,script_name,'use_optimal_matrix',split=False,convert_to_bool=True)
    if(use_optimal_matrix):
        logger.info("Using optimal matrix for data compression")
    else:
        logger.info("Using full matrix for data compression")
    
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
            reference_dv = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
            for ss,source_survey in enumerate(survey_list):
                ax_x = lens_bin+gt*n_BGS_BRIGHT_bins

                if(ss==0):
                    ax[ss,ax_x].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")

                # full_datvec = np.zeros((full_cov.shape[0]))
                # full_mask = np.zeros((full_cov.shape[0],len(scales_list)),dtype=bool)

            

                _,data,_,mycov,zlens,zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path,statistic,
                                                                                versions,logger=logger,
                                                                                correct_for_magnification_bias=True)
                
                cov = load_covariance_chris(galaxy_type,source_survey,statistic,
                                            chris_path)
                
                # full_datvec = np.zeros((cov.shape[0]))
                # full_mask = np.zeros((cov.shape[0],len(scales_list)),dtype=bool)

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                for idb,myBin in enumerate(allowed_bins):
                    # full_bin_mask = full_covariance_bin_mask(galaxy_type,source_survey,lens_bin,myBin,
                    #                                     include_sdss=True)
                    source_bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[myBin])
                    # for idx_scale,scales in enumerate(scales_list):
                    scales_mask = get_scales_mask_from_degrees(rp,'all scales',min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                        
                    tempdata = data[source_bin_mask][scales_mask]
                    tempcov = cov[source_bin_mask][:,source_bin_mask][scales_mask][:,scales_mask]
                    tempref = reference_dv[scales_mask]
                    print(f"S/N: {galaxy_type} l{lens_bin+1} {source_survey} s{myBin+1}: ", \
                          f"{np.sqrt(np.einsum('i,ij,j',tempdata,np.linalg.inv(tempcov),tempref)):.2f}")
                    print(f"rmin: {rp[scales_mask][0]:.2f} Mpc/h, rmax: {rp[scales_mask][-1]:.2f} Mpc/h, ndata: {len(tempdata)}")
                        # full_mask[full_bin_mask,idx_scale] = scales_mask
                        # full_datvec[source_bin_mask] = data[source_bin_mask]
                        # full_mask[source_bin_mask,idx_scale] = scales_mask
                    if(plot):
                        # errors = np.sqrt(np.diag(full_cov[full_bin_mask][:,full_bin_mask]))
                        errors = np.sqrt(np.diag(cov[source_bin_mask][:,source_bin_mask]))
                        if logscale:
                            ax[ss,ax_x].errorbar(rp*np.exp(offset*ss),
                                                    data[source_bin_mask],errors,
                                                        fmt='o',color=color_list[myBin])
                            ax[ss,ax_x].plot(rp,reference_dv,color='k',linestyle='--')
                            ax[ss,ax_x].set_yscale('log')
                            ax[ss,ax_x].set_ylim(5e-2,1e+2)

                        else:
                            ax[ss,ax_x].errorbar(rp*np.exp(offset*ss),
                                                    rp*data[source_bin_mask],rp*errors,
                                                        fmt='o',color=color_list[myBin])
                            ax[ss,ax_x].plot(rp,rp*reference_dv,color='k',linestyle='--')
                            ax[ss,ax_x].set_ylim(-1,16)
                        ax[ss,ax_x].set_xscale('log')



                        if(statistic=="deltasigma"):

                            if(ax_x==0):
                                if logscale:
                                    ax[ss,ax_x].set_ylabel(f"{source_survey}\n $\\Delta\\Sigma(r_p)$")
                                else:
                                    ax[ss,ax_x].set_ylabel(f"{source_survey}\n $r_p\\times\\Delta\\Sigma(r_p)$")
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
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_datavector_tomo.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
        for ss in range(len(survey_list)):
            for ax_x in range(n_total_bins):
                ax[ss,ax_x].set_yticks([])
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_datavector_tomo_noticks.png",
                    dpi=300,transparent=False,bbox_inches="tight")
        plt.close()


def plot_different_cosmologies_tomo(config,plot=True,datavec=None,
                                    cosmology_names=["wCDM","WMAP9"],
                                    marker_list={"wCDM":"o","WMAP9":"^"}):
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


    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    if(use_theory_covariance):
        logger.info("Using theory covariance")
    else:
        logger.info("Using jackknife covariance")
    use_optimal_matrix = clean_read(config,script_name,'use_optimal_matrix',split=False,convert_to_bool=True)
    if(use_optimal_matrix):
        logger.info("Using optimal matrix for data compression")
    else:
        logger.info("Using full matrix for data compression")
    
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
            reference_dv = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
            for ss,source_survey in enumerate(survey_list):
                ax_x = lens_bin+gt*n_BGS_BRIGHT_bins

                if(ss==0):
                    ax[ss,ax_x].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")

                # full_datvec = np.zeros((full_cov.shape[0]))
                # full_mask = np.zeros((full_cov.shape[0],len(scales_list)),dtype=bool)

            

                _,data,_,mycov,zlens,zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path,statistic,
                                                                                versions,logger=logger,
                                                                                correct_for_magnification_bias=False)
                altdata = {}
                for cosmology_name in cosmology_names:
                    altdata[cosmology_name] = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path+os.sep+cosmology_name+os.sep,statistic,
                                                                                versions,logger=logger,
                                                                                correct_for_magnification_bias=False,
                                                                                skip_on_error=True)[1]
                
                cov = load_covariance_chris(galaxy_type,source_survey,statistic,
                                            chris_path)
                
                # full_datvec = np.zeros((cov.shape[0]))
                # full_mask = np.zeros((cov.shape[0],len(scales_list)),dtype=bool)

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                for idb,myBin in enumerate(allowed_bins):
                    # full_bin_mask = full_covariance_bin_mask(galaxy_type,source_survey,lens_bin,myBin,
                    #                                     include_sdss=True)
                    source_bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[myBin])
                    # for idx_scale,scales in enumerate(scales_list):
                    scales_mask = get_scales_mask_from_degrees(rp,'all scales',min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                        
                    if(plot):
                        # errors = np.sqrt(np.diag(full_cov[full_bin_mask][:,full_bin_mask]))
                        errors = np.sqrt(np.diag(cov[source_bin_mask][:,source_bin_mask]))
                        if logscale:
                            raise NotImplementedError("logscale not implemented for different cosmologies")
                        else:
                            for cosmology_name in cosmology_names:
                                ax[ss,ax_x].plot(rp,rp*(altdata[cosmology_name][source_bin_mask]-data[source_bin_mask]),color=color_list[myBin],
                                                 marker=marker_list[cosmology_name],linestyle='')
                            ax[ss,ax_x].plot(rp,-rp*errors,color=color_list[myBin],ls=':')
                            ax[ss,ax_x].plot(rp,rp*errors,color=color_list[myBin],ls=':')
                            ax[ss,ax_x].set_ylim(-3,3)

                            # ax[ss,ax_x].plot(rp,rp*reference_dv,color='k',linestyle='--')
                            # ax[ss,ax_x].set_ylim(-1,16)
                        ax[ss,ax_x].set_xscale('log')



                        if(statistic=="deltasigma"):

                            if(ax_x==0):
                                if logscale:
                                    ax[ss,ax_x].set_ylabel(f"{source_survey}\n $\\Delta\\Sigma(r_p)$")
                                else:
                                    ax[ss,ax_x].set_ylabel(f"{source_survey}\n $r_p\\times\\Delta\\Sigma(r_p)$")
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
        for cosmology_name in cosmology_names:
            if cosmology_name == "wCDM":
                label = "$w_0w_a$CDM"
            elif cosmology_name == "WMAP9":
                label = r"low-$\Omega_m$"
            ax[0,-2].plot([],[],color='k',marker=marker_list[cosmology_name],label=label,ls="")
        ax[0,-2].plot([],[],color='k',linestyle=':',label='error')
        ax[0,-2].legend()
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_datavector_different_cosmologies_tomo.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")

def plot_randoms_tomo(config,plot=True,datavec=None):
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


    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    if(use_theory_covariance):
        logger.info("Using theory covariance")
    else:
        logger.info("Using jackknife covariance")
    use_optimal_matrix = clean_read(config,script_name,'use_optimal_matrix',split=False,convert_to_bool=True)
    if(use_optimal_matrix):
        logger.info("Using optimal matrix for data compression")
    else:
        logger.info("Using full matrix for data compression")
    
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
            # reference_dv = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
            for ss,source_survey in enumerate(survey_list):
                ax_x = lens_bin+gt*n_BGS_BRIGHT_bins

                if(ss==0):
                    ax[ss,ax_x].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")

                # full_datvec = np.zeros((full_cov.shape[0]))
                # full_mask = np.zeros((full_cov.shape[0],len(scales_list)),dtype=bool)

            

                _,_,_,mycov,zlens,zsource,_,data = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                data_path,statistic,
                                                                                versions,logger=logger,
                                                                                correct_for_magnification_bias=True,
                                                                                return_additional_quantity='ds_r' if statistic=='deltasigma' else 'et_r')
                
                cov = load_covariance_chris(galaxy_type,source_survey,statistic,
                                            chris_path)
                
                # full_datvec = np.zeros((cov.shape[0]))
                # full_mask = np.zeros((cov.shape[0],len(scales_list)),dtype=bool)

                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                for idb,myBin in enumerate(allowed_bins):
                    # full_bin_mask = full_covariance_bin_mask(galaxy_type,source_survey,lens_bin,myBin,
                    #                                     include_sdss=True)
                    source_bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[myBin])
                    # for idx_scale,scales in enumerate(scales_list):
                        # scales_mask = get_scales_mask_from_degrees(rp,scales,min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                        
                        # full_datvec[full_bin_mask] = data[source_bin_mask]
                        # full_mask[full_bin_mask,idx_scale] = scales_mask
                        # full_datvec[source_bin_mask] = data[source_bin_mask]
                        # full_mask[source_bin_mask,idx_scale] = scales_mask
                    if(plot):
                        # errors = np.sqrt(np.diag(full_cov[full_bin_mask][:,full_bin_mask]))
                        errors = np.sqrt(np.diag(cov[source_bin_mask][:,source_bin_mask]))
                        ax[ss,ax_x].errorbar(rp*np.exp(offset*ss),
                                                rp*data[source_bin_mask],rp*errors,
                                                    fmt='o',color=color_list[myBin])
                        # ax[ss,ax_x].plot(rp,rp*reference_dv,color='k',linestyle='--')
                        ax[ss,ax_x].set_xscale('log')
                        ax[ss,ax_x].axvline(5,color='k',linestyle='-',lw=0.5)



                        if(statistic=="deltasigma"):

                            if(ax_x==0):
                                ax[ss,ax_x].set_ylabel(f"{source_survey}\n $r_p\\times\\Delta\\Sigma(r_p) \\,[10^6\\mathrm{{M_\\odot/pc}}]$")
                                # ax[ss,ax_x].set_ylim(-2,2)
                            if(ss==len(survey_list)-1):
                                ax[ss,ax_x].set_xlabel(r"$r_p\,[\mathrm{Mpc/h}]$")

                        else:
                            if(ax_x==0):
                                ax[ss,ax_x].set_ylabel(f"{source_survey}\n $\\theta\\times\\gamma_\\mathrm{{t}}(\\theta)$")
                                # ax[lens_bin,gt].set_ylim(-5e-5,5e-5)
                            if(lens_bin==n_lens_bins-1):
                                ax[lens_bin,gt].set_xlabel(r"$\theta\,[deg]$")
            





    if(plot):
        plot_scalecuts(ax,min_deg,max_deg,rp_pivot,galaxy_types,config,tomo=True)
        for ss in range(len(survey_list)):
            ax[ss,ax_x].set_ylim(-2.5,2.5)

        plt.tight_layout()
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_randoms_tomo.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
        plt.close()

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

    # plot_datavector_notomo(config)
    # plot_bmodes_notomo(config)
    plot_datavector_tomo(config,plot=True)
    plot_randoms_tomo(config,plot=True)
    pvalues, chisqs = plot_bmodes_tomo(config,plot=True)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    version = clean_read(config,'general','version',split=False)
    statistic = clean_read(config,'general','statistic',split=False)

    mytab = generate_bmode_tomo_latex_table_from_dict(pvalues, config, caption=f"{config.get('general','statistic')} p-values for B-modes", precision=3)
    fil = open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"tab_{statistic}_pvalues_bmodes_tomo.tex", "w")
    fil.write(mytab)
    fil.close()

    mytab = generate_bmode_tomo_latex_table_from_dict(chisqs, config, caption=f"{config.get('general','statistic')} $\\chi^2$ for B-modes", precision=1)
    fil = open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"tab_{statistic}_chisqs_bmodes_tomo.tex", "w")
    fil.write(mytab)
    fil.close()

    plot_different_cosmologies_tomo(config,plot=True)
