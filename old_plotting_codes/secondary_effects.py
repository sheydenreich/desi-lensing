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


def hide_errorbars(errlines,reverse=False):
    for line in errlines:
        for err in line[2]:
            err.set_visible(reverse)


def all_secondary_effects_tomo(config,plot,datavec=None,logger="create",all_zsource=None,allbins=False):
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

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    # print("Got B")
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    if logger == "create":
        logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_tomo',__name__)

    systematics_list = clean_read(config,script_name,'systematics_list',split=True)

    slope_color = clean_read(config,script_name,'slope_color',split=False)
    slope_uncertainty = clean_read(config,script_name,'slope_uncertainty',split=False)

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    # if(use_theory_covariance):
    #     logger.info("Using theory covariance")
    # else:
    #     logger.info("Using jackknife covariance")
    if(plot):
        if logger is not None:
            logger.info("Preparing plot")
        fig,ax,gs = initialize_gridspec_figure((7.24,7.24/n_total_bins*len(scales_list)),
                            len(scales_list),
                            n_total_bins,
                            hspace=0,wspace=0)
        add_colorbar_legend(fig,ax,gs,color_list,survey_list)
        errlines = []
        p_values = {}
        # ax[-1,0].errorbar([],[],[],fmt='o',color='k',label="Reference data")
        # ax[-1,0].errorbar([],[],[],fmt='^',color='k',label="Systematics-infused data")
        # ax[-1,0].legend()

    p_list_randoms, V_list_randoms, key_list_randoms = load_randoms_values(config)
    p_randoms_std = np.std(p_list_randoms,axis=0)
    V_randoms_std = np.zeros((*p_randoms_std.shape,p_randoms_std.shape[-1]))
    for i in range(p_list_randoms.shape[1]):
        V_randoms_std[i] = np.cov(p_list_randoms[:,i,:].T)

    all_zsource_return = {}
    all_lens_amplitudes = {}
    all_lens_amplitude_errors = {}

    for scale,scaletitle in enumerate(scales_list):
        for gt,galaxy_type in enumerate(galaxy_types):
            n_lens_bins = 3
            for lens_bin in range(config.getint('general','N_'+galaxy_type+'_bins')):
                n_radial_bins = get_number_of_radial_bins(galaxy_type,survey_list[0],None)
                scales_mask = get_scales_mask_from_degrees(rp,scaletitle,min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                n_used_bins = np.sum(scales_mask)
                
                dvs = []
                dvs_sys = []
                covs = []
                zsources = []
                zlenses = []
                survey_indices = []

                for ss,source_survey in enumerate(survey_list):
                    _,_,_,mycov,full_zlens,full_zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                    data_path,"deltasigma",
                                                                                    versions,logger=logger,
                                                                                    only_allowed_bins=True)
                    logger.info("Using reference datavector")
                    full_data = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
                    full_data = np.tile(full_data,get_number_of_source_bins(source_survey)*n_lens_bins)


                    if(datavec is not None):
                        if logger is not None:
                            logger.info(f"Using mock datavector for {galaxy_type} {source_survey}")
                        full_data = datavec[f"{galaxy_type}_{source_survey}"]

                    if(use_theory_covariance):
                        full_cov = load_covariance_chris(galaxy_type,source_survey,"deltasigma",
                                                        chris_path)
                    else:
                        full_cov = mycov

                    # apply all systematics to datavector
                    systematics_factor = 1.
                    for systematic in systematics_list:
                        systematics_factor *= (1+load_dv_johannes(galaxy_type,source_survey,chris_path,
                                                               'deltasigma',logger,
                                                               systype=systematic)/100)
                        # if not np.all(np.isfinite(systematics_factor)):
                        #     print(f"Failed to load {systematic} for {galaxy_type} {source_survey}")
                        #     sys.exit()

                    full_data_sys = full_data * systematics_factor

                    allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                    if allbins:
                        allowed_bins = np.arange(get_number_of_source_bins(source_survey))
                    n_source_bins = len(allowed_bins)
                    if(n_source_bins==0):
                        if logger is not None:
                            logger.info(f"Skipping {galaxy_type} {source_survey} {lens_bin}")
                    for source_bin in allowed_bins:
                        bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[source_bin])

                        data = full_data[bin_mask][scales_mask]
                        data_sys = full_data_sys[bin_mask][scales_mask]

                        zsource = full_zsource[bin_mask][scales_mask]
                        zlens = full_zlens[bin_mask][scales_mask]
                        cov = full_cov[bin_mask][:,bin_mask][scales_mask][:,scales_mask]


                        if allbins:
                            if np.mean(zsource) < np.mean(zlens):
                                continue

                        assert np.all(np.isfinite(data))
                        assert np.all(np.isfinite(zsource))
                        assert np.all(np.isfinite(zlens))
                        assert np.all(np.isfinite(cov))

                        dvs.append(data)
                        dvs_sys.append(data_sys)
                        covs.append(cov)

                        zsources.append(np.mean(zsource))
                        zlenses.append(np.mean(zlens))
                        survey_indices.append(ss)
                reference_dv = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
                reference_dv = reference_dv[scales_mask]
                dvs = np.array(dvs)
                dvs_sys = np.array(dvs_sys)
                covs = np.array(covs)
                # print(dvs.shape,dvs_sys.shape,np.concatenate((dvs,dvs_sys)).shape)
                # print(covs.shape,np.concatenate((covs,covs)).shape)

                zsources = np.array(zsources)
                zlenses = np.array(zlenses)
                if all_zsource is None:
                    all_zsource_return[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = zsources
                else:
                    zsources = all_zsource[f"{galaxy_type}_{scaletitle}_{lens_bin}"]
                if(len(dvs_sys)==0):
                    continue

                if not np.all(np.isfinite(dvs_sys)):
                    mask = np.all(np.isfinite(dvs_sys),axis=1)
                    dvs_sys = dvs_sys[mask]
                    zsources = zsources[mask]
                    covs = covs[mask]
                lensamp,lensamperr,_ = get_deltasigma_amplitudes(dvs_sys,
                                                                covs,
                                                                reference_dv)
                if not np.all(np.isfinite(lensamp)):
                    print(f"{galaxy_type}_{scaletitle}_{lens_bin}")
                    print(lensamp)
                    print(lensamperr)
                    print(dvs_sys)
                    print(covs)
                    sys.exit()
                
                n_dvs = len(dvs)
                
                if(len(dvs)>2):
                    p,V = np.polyfit(zsources,lensamp,1,
                                    w=1/lensamperr,cov=True)
                    if not np.all(np.isfinite(p)):
                        print(f"{galaxy_type}_{scaletitle}_{lens_bin}")
                        print(p)
                        print(V)
                        print(zsources,lensamp,1/lensamperr)
                    # p_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = p
                    # V_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = V
                elif(len(dvs)==2):
                    p = np.polyfit(zsources,lensamp,1,
                                    w=1/lensamperr)
                    # p_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = p
                    # V_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = np.zeros((2,2))+np.nan
                else:
                    p = np.zeros(2)+np.nan
                    V = np.zeros((2,2))+np.nan

                if(plot):
                    n_dvs = len(dvs_sys)
                    ax_x = lens_bin+gt*n_lens_bins
                    ax[scale,ax_x].axhline(0,color='k',ls=':')
                    xarr = np.linspace(np.min(zsources),np.max(zsources),100)
                    y = xarr*p[0]+p[1]
                    ax[scale,ax_x].plot(xarr,y,color=slope_color,
                                        linestyle='--')
                    print(f"{galaxy_type}_{scaletitle}_{lens_bin}",p)
                    p_values[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = p

                    if np.isfinite(p[0]):
                        if slope_uncertainty == "covariance":
                            slope_uncertainty_val = np.sqrt(V[0,0])
                        elif slope_uncertainty == "randoms":
                            idx = np.where(key_list_randoms==f"{galaxy_type}_{scaletitle}_{lens_bin}")[0]
                            if not(len(idx)==1):
                                print(f"Found {len(idx)} matches for {galaxy_type}_{scaletitle}_{lens_bin}, available keys: {key_list_randoms}")
                                sys.exit()
                            slope_uncertainty_val = p_randoms_std[idx][0][0]
                            # print(slope_uncertainty_val,type(slope_uncertainty_val))

                        elif slope_uncertainty == "randoms_covariance":
                            raise NotImplementedError


                        ax[scale,ax_x].text(0.5,0.85,f"$\\beta={p[0]:.3f}\\pm {slope_uncertainty_val:.3f}$",
                                            transform=ax[scale,ax_x].transAxes,horizontalalignment="center",
                                            fontsize=8)

                    for x in range(n_dvs):
                        source_survey = survey_list[survey_indices[x]]
                        color = color_list[survey_indices[x]]
                        if not f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}" in all_lens_amplitudes.keys():
                            all_lens_amplitudes[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}"] = []
                            all_lens_amplitude_errors[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}"] = []
                        all_lens_amplitudes[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}"].append(lensamp[x])
                        all_lens_amplitude_errors[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}"].append(lensamperr[x])

                        # ax[scale,ax_x].errorbar(zsources[x],lensamp[x],lensamperr[x],fmt='o',
                        #                 color=color)
                        errlines.append(ax[scale,ax_x].errorbar(zsources[x],lensamp[x],lensamperr[x],fmt='^',
                                        color=color))
                        if(x==0):
                            if(scale==0):
                                ax[scale,ax_x].set_title(f"{galaxy_type[:3]} bin {lens_bin+1}")
                            if(ax_x==0):
                                ax[scale,ax_x].set_ylabel(f"$A_\\mathrm{{lens}}$ \n {scaletitle}")
                            if(scale==len(scales_list)-1):
                                ax[scale,ax_x].set_xlabel(r"$\langle z_{\mathrm{source}}\rangle$")

    if(plot):
        hide_errorbars(errlines)
        for scale in range(len(scales_list)):
            for ax_x in range(n_total_bins):
                ax[scale,ax_x].relim()
                ax[scale,ax_x].autoscale()
        hide_errorbars(errlines,reverse=True)

        plt.tight_layout()
        fstr = ""
        for systematic in systematics_list:
            fstr += f"_{systematic}"
        if allbins:
            plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"systematics{fstr}_allbins.png",
                    dpi=300,transparent=config.getboolean("general","transparent_background"),bbox_inches="tight")
            np.savez(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"systematics{fstr}_allbins",
                     p_values)

        else:
            plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"systematics{fstr}.png",
                        dpi=300,transparent=config.getboolean("general","transparent_background"),bbox_inches="tight")
            np.savez(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"systematics{fstr}",
                        p_values)
        plt.close()
    return all_lens_amplitudes,all_lens_amplitude_errors


def secondary_effects_tomo(config,plot,datavec=None,logger="create",all_zsource=None,allbins=False):
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

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    # print("Got B")
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    if logger == "create":
        logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_tomo',__name__)

    systematics_list = clean_read(config,script_name,'systematics_list',split=True)
    slope_color = clean_read(config,script_name,'slope_color',split=False)

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)

    p_list_randoms, V_list_randoms, key_list_randoms = load_randoms_values(config)
    p_randoms_std = np.std(p_list_randoms,axis=0)
    V_randoms_std = np.zeros((*p_randoms_std.shape,p_randoms_std.shape[-1]))
    for i in range(p_list_randoms.shape[1]):
        V_randoms_std[i] = np.cov(p_list_randoms[:,i,:].T)

    all_lens_amplitudes = {}
    all_lens_amplitude_errors = {}

    for systematic in systematics_list:
        print("Plotting ",systematic)
        if(plot):
            if logger is not None:
                logger.info("Preparing plot")
            fig,ax,gs = initialize_gridspec_figure((7.24,7.24/n_total_bins*len(scales_list)),
                                len(scales_list),
                                n_total_bins,
                                hspace=0,wspace=0)
            add_colorbar_legend(fig,ax,gs,color_list,survey_list)
            errlines = []
            # ax[-1,0].errorbar([],[],[],fmt='o',color='k',label="Reference data")
            # ax[-1,0].errorbar([],[],[],fmt='^',color='k',label="Systematics-infused data")
            # ax[-1,0].legend()

        all_zsource_return = {}
        for scale,scaletitle in enumerate(scales_list):
            for gt,galaxy_type in enumerate(galaxy_types):
                n_lens_bins = get_number_of_lens_bins(galaxy_type)
                for lens_bin in range(config.getint('general','N_'+galaxy_type+'_bins')):
                    n_radial_bins = get_number_of_radial_bins(galaxy_type,survey_list[0],None)
                    scales_mask = get_scales_mask_from_degrees(rp,scaletitle,min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                    n_used_bins = np.sum(scales_mask)
                    
                    dvs = []
                    dvs_sys = []
                    covs = []
                    zsources = []
                    zlenses = []
                    survey_indices = []

                    for ss,source_survey in enumerate(survey_list):
                        _,_,_,mycov,full_zlens,full_zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                        data_path,"deltasigma",
                                                                                        versions,logger=logger,
                                                                                        only_allowed_bins=True)
                        logger.info("Using reference datavector")
                        full_data = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
                        full_data = np.tile(full_data,get_number_of_source_bins(source_survey)*n_lens_bins)


                        if(datavec is not None):
                            if logger is not None:
                                logger.info(f"Using mock datavector for {galaxy_type} {source_survey}")
                            full_data = datavec[f"{galaxy_type}_{source_survey}"]

                        if(use_theory_covariance):
                            full_cov = load_covariance_chris(galaxy_type,source_survey,"deltasigma",
                                                            chris_path)
                        else:
                            full_cov = mycov

                        # apply systematic to datavector

                        systematics_factor = 1.+load_dv_johannes(galaxy_type,source_survey,chris_path,
                                                                'deltasigma',logger,
                                                                systype=systematic)/100
                        full_data_sys = full_data * systematics_factor

                        allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                        if allbins:
                            allowed_bins = np.arange(get_number_of_source_bins(source_survey))
                        n_source_bins = len(allowed_bins)
                        if(n_source_bins==0):
                            if logger is not None:
                                logger.info(f"Skipping {galaxy_type} {source_survey} {lens_bin}")
                        for source_bin in allowed_bins:
                            bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[source_bin])

                            data = full_data[bin_mask][scales_mask]
                            data_sys = full_data_sys[bin_mask][scales_mask]

                            zsource = full_zsource[bin_mask][scales_mask]
                            zlens = full_zlens[bin_mask][scales_mask]
                            cov = full_cov[bin_mask][:,bin_mask][scales_mask][:,scales_mask]

                            assert np.all(np.isfinite(data))

                            assert np.all(np.isfinite(zsource))
                            assert np.all(np.isfinite(zlens))
                            assert np.all(np.isfinite(cov))

                            if allbins:
                                if np.mean(zsource) < np.mean(zlens):
                                    continue


                            dvs.append(data)
                            dvs_sys.append(data_sys)
                            covs.append(cov)

                            zsources.append(np.mean(zsource))
                            zlenses.append(np.mean(zlens))
                            survey_indices.append(ss)
                    reference_dv = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
                    reference_dv = reference_dv[scales_mask]
                    dvs = np.array(dvs)
                    dvs_sys = np.array(dvs_sys)
                    covs = np.array(covs)
                    # print(dvs.shape,dvs_sys.shape,np.concatenate((dvs,dvs_sys)).shape)
                    # print(covs.shape,np.concatenate((covs,covs)).shape)

                    zsources = np.array(zsources)
                    zlenses = np.array(zlenses)
                    if all_zsource is None:
                        all_zsource_return[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = zsources
                    else:
                        zsources = all_zsource[f"{galaxy_type}_{scaletitle}_{lens_bin}"]
                    if(len(dvs_sys)==0):
                        continue
                    lensamp,lensamperr,_ = get_deltasigma_amplitudes(dvs_sys,
                                                                    covs,
                                                                    reference_dv)
                    # lensamp,lensamperr,_ = get_deltasigma_amplitudes(np.concatenate((dvs,dvs_sys)),
                    #                                                 np.concatenate((covs,covs)),
                    #                                                 reference_dv)
                    n_dvs = len(dvs)
                    
                    if(len(dvs)>2):
                        p,V = np.polyfit(zsources,lensamp,1,
                                        w=1/lensamperr,cov=True)
                        # p_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = p
                        # V_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = V
                    elif(len(dvs)==2):
                        p = np.polyfit(zsources,lensamp,1,
                                        w=1/lensamperr)
                        # p_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = p
                        # V_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = np.zeros((2,2))+np.nan
                    else:
                        p = np.zeros(2)+np.nan
                        V = np.zeros((2,2))+np.nan

                    
                    if(plot):
                        ax_x = lens_bin+gt*n_lens_bins
                        ax[scale,ax_x].axhline(0,color='k',ls=':')

                        xarr = np.linspace(np.min(zsources),np.max(zsources),100)
                        y = xarr*p[0]+p[1]
                        ax[scale,ax_x].plot(xarr,y,color=slope_color,
                                            linestyle='--')
                        if np.isfinite(p[0]):
                            if slope_uncertainty == "covariance":
                                slope_uncertainty_val = np.sqrt(V[0,0])
                            elif slope_uncertainty == "randoms":
                                idx = np.where(key_list_randoms==f"{galaxy_type}_{scaletitle}_{lens_bin}")[0]
                                if not(len(idx)==1):
                                    print(f"Found {len(idx)} matches for {galaxy_type}_{scaletitle}_{lens_bin}, available keys: {key_list_randoms}")
                                    sys.exit()
                                slope_uncertainty_val = p_randoms_std[idx][0][0]
                                # print(slope_uncertainty_val,type(slope_uncertainty_val))

                            elif slope_uncertainty == "randoms_covariance":
                                raise NotImplementedError

                            ax[scale,ax_x].text(0.5,0.85,f"$\\beta={p[0]:.3f}\\pm {slope_uncertainty_val:.3f}$",
                                                transform=ax[scale,ax_x].transAxes,horizontalalignment="center",
                                                fontsize=8)

                        for x in range(n_dvs):
                            source_survey = survey_list[survey_indices[x]]
                            if not f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}_{systematic}" in all_lens_amplitudes.keys():
                                all_lens_amplitudes[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}_{systematic}"] = []
                                all_lens_amplitude_errors[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}_{systematic}"] = []
                            all_lens_amplitudes[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}_{systematic}"].append(lensamp[x])
                            all_lens_amplitude_errors[f"{galaxy_type}_{scaletitle}_{lens_bin}_{source_survey}_{systematic}"].append(lensamperr[x])

                            color = color_list[survey_indices[x]]

                            # ax[scale,ax_x].errorbar(zsources[x],lensamp[x],lensamperr[x],fmt='o',
                            #                 color=color)
                            errlines.append(ax[scale,ax_x].errorbar(zsources[x],lensamp[x],lensamperr[x],fmt='^',
                                            color=color))
                            if(x==0):
                                if(scale==0):
                                    ax[scale,ax_x].set_title(f"{galaxy_type[:3]} Bin {lens_bin+1}")
                                if(ax_x==0):
                                    ax[scale,ax_x].set_ylabel(f"$A_\\mathrm{{lens}}$ \n {scaletitle}")
                                if(scale==len(scales_list)-1):
                                    ax[scale,ax_x].set_xlabel(r"$\langle z_{\mathrm{source}}\rangle$")

        if(plot):
            hide_errorbars(errlines)
            for scale in range(len(scales_list)):
                for ax_x in range(n_total_bins):
                    ax[scale,ax_x].relim()
                    ax[scale,ax_x].autoscale()
            hide_errorbars(errlines,reverse=True)

            plt.tight_layout()
            fstr = ""
            if allbins:
                plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"systematics_{systematic}_allbins.png",
                        dpi=300,transparent=config.getboolean("general","transparent_background"),bbox_inches="tight")
            else:
                plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"systematics_{systematic}.png",
                            dpi=300,transparent=config.getboolean("general","transparent_background"),bbox_inches="tight")
            plt.close()
    return all_lens_amplitudes,all_lens_amplitude_errors

if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")

    version = clean_read(config,'general','version',split=False)
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    systematics_list = clean_read(config,script_name,'systematics_list',split=True)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)
    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    os.makedirs(savepath_slope_values,exist_ok=True)

    # all_secondary_effects_tomo(config,plot=True,allbins=True)
    

    lensamps,lensamperrs = all_secondary_effects_tomo(config,plot=True)
    fstr = ""
    for systematic in systematics_list:
        fstr += f"_{systematic}"
    np.save(savepath_slope_values+f"lens_amplitudes_combined_systematics{fstr}",lensamps)
    np.save(savepath_slope_values+f"lens_amplitude_errors_combined_systematics{fstr}",lensamperrs)

    # secondary_effects_tomo(config,plot=True,allbins=True)


    lensamps,lensamperrs = secondary_effects_tomo(config,plot=True)
    np.save(savepath_slope_values+f"lens_amplitudes_single_systematics",lensamps)
    np.save(savepath_slope_values+f"lens_amplitude_errors_single_systematics",lensamperrs)
