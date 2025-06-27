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
                        get_reference_datavector_of_galtype,get_scales_mask_from_degrees,get_split_value
import matplotlib.gridspec as gridspec
                        

script_name = 'splits'

def plot_all_splits(config,plot,datavec=None,covariances=None,verbose=True):
    all_splits = clean_read(config,script_name,'splits_to_consider',split=True)
    n_splits = clean_read(config,script_name,'n_splits',split=False,convert_to_int=True)
    p_list = {}
    V_list = {}
    for split_by in all_splits:
        if(verbose):
            print("Computing split "+split_by)
        # try:
        if(True):
            _p_list,_V_list = plot_split(config,plot,split_by,n_splits=n_splits,datavec=datavec,covariances=covariances)
            for key in list(_p_list.keys()):
                p_list[split_by+"_"+key] = _p_list[key]
                V_list[split_by+"_"+key] = _V_list[key]
        # except Exception as e:
        #     print("Could not compute split "+split_by)
        #     print(e)
    return p_list,V_list

def plot_split(config,plot,split_by,n_splits,datavec=None,covariances=None):
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
    critical_sigma = clean_read(config,'general','critical_sigma',split=False,convert_to_float=True)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    transparent_background = clean_read(config,'general','transparent_background',split=False,convert_to_bool=True)

    # xscale = clean_read(config,script_name,'xscale',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)


    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    if(use_theory_covariance):
        raise NotImplementedError
        logger.info("Using theory covariance")
    else:
        logger.info("Using jackknife covariance")
    use_optimal_matrix = clean_read(config,script_name,'use_optimal_matrix',split=False,convert_to_bool=True)
    if(use_optimal_matrix):
        logger.info("Using optimal matrix for data compression")
    else:
        logger.info("Using diagonalized matrix for data compression")


    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)
    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    os.makedirs(savepath_slope_values,exist_ok=True)
    plot_slope = clean_read(config,script_name,'plot_slope',split=False,convert_to_bool=True)
    plot_slope_uncertainty = clean_read(config,script_name,'plot_slope_uncertainty',split=False,convert_to_bool=True)
    slope_color = clean_read(config,script_name,'slope_color',split=False)
    slope_uncertainty = clean_read(config,script_name,'slope_uncertainty',split=False)
    assert slope_uncertainty in ["covariance","randoms","randoms_covariance"], f"slope_uncertainty {slope_uncertainty} must be one of covariance, randoms, randoms_covariance"

    if(plot):
        print("Plotting")
        fig,ax,gs = initialize_gridspec_figure((7.24,7.24/n_total_bins*len(scales_list)),
                            len(scales_list),
                            n_total_bins,
                            hspace=0,wspace=0)
        add_colorbar_legend(fig,ax,gs,color_list,survey_list)
        if slope_uncertainty == "randoms":
            p_list_randoms = np.load(savepath_slope_values+os.sep+"splits_p_arr.npy")
            key_list_randoms = np.load(savepath_slope_values+os.sep+"splits_keys.npy",allow_pickle=True)
            p_randoms_std = np.std(p_list_randoms,axis=0)
            V_randoms_std = np.zeros((*p_randoms_std.shape,p_randoms_std.shape[-1]))
            for i in range(p_list_randoms.shape[1]):
                V_randoms_std[i] = np.cov(p_list_randoms[:,i,:].T)
            # print(V_randoms_std.shape,p_randoms_std.shape)

    p_list = {}
    V_list = {}

    read_data = (plot or (not use_theory_covariance and covariances is None) or (datavec is None))
    if(read_data):
        print("Reading data!")
        print("*"*50)

    for scale,scaletitle in enumerate(scales_list):
        for gt,galaxy_type in enumerate(galaxy_types):
            n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')
            # splits by NTILE work a bit differently, since equal-number splits are not possible!
            if split_by.lower() == "ntile":
                n_splits = clean_read(config,script_name,f'n_ntile_{galaxy_type[:3]}',split=False,convert_to_int=True)
                n_splits_computed = clean_read(config,script_name,f'n_ntile_computed_{galaxy_type[:3]}',split=False,convert_to_int=True)
            else:
                n_splits_computed = n_splits

            for lens_bin in range(n_lens_bins):
                n_radial_bins = get_number_of_radial_bins(galaxy_type,survey_list[0],None)
                rp = get_rp_chris(galaxy_type,survey_list[0],chris_path,
                                    statistic,logger)[:n_radial_bins]
                scales_mask = get_scales_mask_from_degrees(rp,scaletitle,min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                n_used_bins = np.sum(scales_mask)
                
                dvs = {}
                covs = {}
                split_vals = {}

                # if(read_data):
                #     zsources = np.zeros((len(survey_list)))
                #     zlenses = np.zeros((len(survey_list)))

                # source_survey_mask = np.ones((len(survey_list)),dtype=bool)
                for ss,source_survey in enumerate(survey_list):
                    allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                    if(len(allowed_bins)==0):
                        logger.info(f"Skipping {galaxy_type} {source_survey} {lens_bin}")
                        continue
                    # if(source_survey.lower()=='sdss' and galaxy_type.lower()=='lrg'):
                        # logger.info(f"Skipping {galaxy_type} {source_survey} {lens_bin}")
                        # continue
                    for split in range(n_splits_computed):
                        if(read_data):
                            _,data,_,mycov,zlens,zsource,_ = load_data_and_covariance_notomo(galaxy_type,source_survey,
                                                                                            data_path,statistic,
                                                                                            versions,split_by=split_by,
                                                                                            n_splits=n_splits,
                                                                                            split=split)
                            data = data[lens_bin]
                            mycov = mycov[lens_bin]
                            # print(data.shape,mycov.shape)
                        if(covariances is not None):
                            mycov = covariances[f"{galaxy_type}_{source_survey}_{split_by}_{split}_of_{n_splits}"][lens_bin]
                        if(datavec is not None):
                            logger.info(f"Using mock datavector for {galaxy_type} {source_survey}")
                            data = datavec[f"{galaxy_type}_{source_survey}_{split_by}_{split}_of_{n_splits}"][lens_bin]

                        if(use_theory_covariance):
                            cov = load_covariance_chris(galaxy_type,source_survey,statistic,
                                                            chris_path)
                        else:
                            cov = mycov

                        covs[f'{ss}_{split}'] = cov[scales_mask][:,scales_mask]
                        dvs[f'{ss}_{split}'] = data[scales_mask]
                        # if split_by.lower() == "ntile":
                        #     split_vals[f'{ss}_{split}'] = split
                        # else:
                        split_vals[f'{ss}_{split}'] = get_split_value(galaxy_type,source_survey,
                                                                    data_path,statistic,versions,
                                                                    lens_bin,
                                                                    split_by=split_by,n_splits=n_splits,
                                                                    split=split)

                    
                reference_dv = get_reference_datavector_of_galtype(config,rp,
                                                                   galaxy_type,
                                                                   lens_bin)
                reference_dv = reference_dv[scales_mask]
                all_keys = list(dvs.keys())
                dvs_arr = np.array([dvs[key] for key in all_keys])
                covs_arr = np.array([covs[key] for key in all_keys])
                splitvals_arr = np.array([split_vals[key] for key in all_keys])
                lensamp = np.zeros(len(all_keys))+np.nan
                lensamperr = np.zeros((len(all_keys)))+np.nan
                # if(np.any(np.isnan(dvs_arr))):
                    # print("WARNING: nans in dvs_arr")
                if(np.any(np.isnan(covs_arr))):
                    # print("WARNING: nans in covs_arr")
                    mask_nan = np.any(np.isnan(covs_arr),axis=(1,2))
                    # print(covs_arr.shape,mask_nan.shape)
                    dvs_arr = dvs_arr[~mask_nan]
                    covs_arr = covs_arr[~mask_nan]
                    splitvals_arr = splitvals_arr[~mask_nan]
                    all_keys = np.array(all_keys)[~mask_nan]
                if(np.any(np.isnan(splitvals_arr))):
                    print("WARNING: nans in splitvals_arr")

                lensamp,lensamperr,_ = get_deltasigma_amplitudes(dvs_arr,covs_arr,reference_dv)

                p,V = np.polyfit(splitvals_arr,lensamp,1,
                                 w=1/lensamperr**2,cov=True)
                p_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = p
                V_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = V

                if(plot):
                    for x,key in enumerate(all_keys):
                        ss,split = key.split("_")
                        ss = int(ss)
                        split = int(split)
                        ax_x = lens_bin+gt*n_BGS_BRIGHT_bins
                        ax[scale,ax_x].errorbar(splitvals_arr[x],lensamp[x],lensamperr[x],fmt='o',
                                        color=color_list[ss])
                        if(x==0):
                            if(scale==0):
                                ax[scale,ax_x].set_title(f"{galaxy_type[:3]} bin {lens_bin+1}")
                            if(ax_x==0):
                                ax[scale,ax_x].set_ylabel(f"$A_\\mathrm{{\\Delta\\Sigma}}$,\n {scaletitle}")
                            if(scale==len(scales_list)-1):
                                ax[scale,ax_x].set_xlabel(f"{split_by}")
                    if slope_uncertainty == "covariance":
                        slope_uncertainty_val = np.sqrt(V[0,0])
                    elif slope_uncertainty == "randoms":
                        idx = np.where(key_list_randoms==f"{split_by}_{galaxy_type}_{scaletitle}_{lens_bin}")[0]
                        if not(len(idx)==1):
                            print(f"Found {len(idx)} matches for {split_by}_{galaxy_type}_{scaletitle}_{lens_bin}, available keys: {key_list_randoms}")
                            sys.exit()
                        slope_uncertainty_val = p_randoms_std[idx][0][0]
                        # print(slope_uncertainty_val,type(slope_uncertainty_val))

                    elif slope_uncertainty == "randoms_covariance":
                        raise NotImplementedError
                    
                    if(plot_slope):
                        xarr = np.linspace(np.min(splitvals_arr),np.max(splitvals_arr),100)
                        y = xarr*p[0]+p[1]
                        ax[scale,ax_x].plot(xarr,y,color=slope_color,
                                            linestyle='--')
                        if(plot_slope_uncertainty):

                            if slope_uncertainty == "covariance":
                                slope_covmat = V
                            elif slope_uncertainty == "randoms":
                                slope_covmat = V_randoms_std[idx][0]
                                # print(slope_covmat.shape)
                            elif slope_uncertainty == "randoms_covariance":
                                raise NotImplementedError
                            dy = np.sqrt((xarr**2 * slope_covmat[0, 0]) + slope_covmat[1, 1] + 2 * xarr * slope_covmat[0, 1])
                            ax[scale,ax_x].fill_between(xarr, y - dy, y + dy, color='gray', alpha=0.5)

                    if np.isfinite(p[0]):
                        if(np.abs(p[0])> critical_sigma*slope_uncertainty_val):
                            bbox=dict(facecolor='gray', edgecolor='red', boxstyle='round,pad=0.2', alpha=0.2)
                            ax[scale,ax_x].text(0.5,0.85,f"$\\beta={p[0]:.2f}\\pm {slope_uncertainty_val:.2f}$",
                                            transform=ax[scale,ax_x].transAxes,horizontalalignment="center",bbox=bbox,
                                            fontsize=8)
                        else:
                            ax[scale,ax_x].text(0.5,0.85,f"$\\beta={p[0]:.2f}\\pm {slope_uncertainty_val:.2f}$",
                                            transform=ax[scale,ax_x].transAxes,horizontalalignment="center",
                                            fontsize=8)

    if(plot):
        plt.tight_layout()
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_split_by_{split_by}.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
        plt.close()
    return p_list,V_list

if __name__=="__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    p_list,V_list = plot_all_splits(config,plot=True)

    version = clean_read(config,'general','version',split=False)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    os.makedirs(savepath_slope_values,exist_ok=True)
    np.save(savepath_slope_values+os.sep+script_name+"_data_p_list",p_list)
    np.save(savepath_slope_values+os.sep+script_name+"_data_V_list",V_list)

