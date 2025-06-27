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
                        get_pvalue,get_rp_from_deg,get_scales_mask_from_degrees,generate_bmode_tomo_latex_table_from_dict,get_ntot,full_covariance_bin_mask,get_reference_datavector
import matplotlib.gridspec as gridspec
from copy import deepcopy
from astropy.cosmology import Planck18

from scipy.optimize import minimize

def fit_log10_Mmin(log10_Mmin,data,inverse_covariance,rp,z):
    emulated_data = get_reference_datavector(rp,z,Mmin=10**log10_Mmin)
    if np.any(np.isnan(emulated_data)):
        return np.inf
    delta = data-emulated_data
    chi2 = np.dot(delta,np.dot(inverse_covariance,delta))
    return chi2

script_name = 'darkemu_Mmin'
def fit_mmin_darkemu(config,plot=True):
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

    n_BGS_BRIGHT_bins = 3
    n_LRG_bins = 3
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    transparent_background = clean_read(config,'general','transparent_background',split=False,convert_to_bool=True)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_tomo',__name__)

    if "HSCY3" in survey_list:
        hscy3=True
    else:
        hscy3=False
    if 'SDSS' in survey_list:
        idx_sdss = survey_list.index('SDSS')
        survey_list.pop(idx_sdss)

    all_Mmin = {}
    if(plot):
        fig,ax,gs = initialize_gridspec_figure((7.24,7.24/n_total_bins),
                            1,
                            n_total_bins,
                            hspace=0,wspace=0,
                            add_cbar=False)
        # add_colorbar_legend(fig,ax,gs,color_list,[f"Bin {i+1}" for i in range(5)])

    for gt,galaxy_type in enumerate(galaxy_types):
        full_cov = load_covariance_chris(galaxy_type,"all_y3" if hscy3 else "all_y1",statistic,
                            chris_path,pure_noise=True)
                

        all_Mmin[galaxy_type] = np.zeros(3)
        for lens_bin in range(3):
            full_datvec = np.zeros((full_cov.shape[0]))
            all_rp = np.zeros((full_cov.shape[0]))
            full_mask = np.zeros((full_cov.shape[0]),dtype=bool)
            full_zlens = np.zeros((full_cov.shape[0]))

            for ss,source_survey in enumerate(survey_list):
                allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                full_bin_mask = full_covariance_bin_mask(galaxy_type,source_survey,lens_bin,allowed_bins)
            
                rp = get_rp_chris(galaxy_type,source_survey,chris_path,'deltasigma')[:15]

                source_bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,allowed_bins)
                scales_mask = get_scales_mask_from_degrees(rp,'all scales',min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                _,data,_,_,zlens,_,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,data_path,'deltasigma',versions)
                
                
                full_datvec[full_bin_mask] = data[source_bin_mask]
                full_zlens[full_bin_mask] = zlens[source_bin_mask]
                all_rp[full_bin_mask] = np.tile(rp,len(allowed_bins))
                full_mask[full_bin_mask] = np.tile(scales_mask,len(allowed_bins))
                assert np.all(np.isfinite(full_datvec[full_mask]))
            
            print(f'Fitting Mmin for {galaxy_type} bin {lens_bin}')
            print(f'Full mask shape: {full_mask.shape}, number of unmasked scales: {np.sum(full_mask)}')
            initial_guess = 12.5
            # Call the minimize function
            inverse_covariance = np.linalg.inv(full_cov[full_mask][:,full_mask])
            mydata = full_datvec[full_mask]
            myrp = all_rp[full_mask]
            z = np.mean(full_zlens[full_mask])
            result = minimize(fit_log10_Mmin, initial_guess, args=(mydata, inverse_covariance, myrp, z),
                              bounds=[(11,14)],method='L-BFGS-B')
            assert result.success
            all_Mmin[galaxy_type][lens_bin] = 10**result.x[0]
            print(f'Minimized Mmin: {all_Mmin[galaxy_type][lens_bin]:.2e}')

            if(plot):
                ax_x = gt*n_BGS_BRIGHT_bins + lens_bin
                ax[0,ax_x].scatter(myrp,myrp*mydata,label='Data')
                ax[0,ax_x].plot(rp,rp*get_reference_datavector(rp,z,Mmin=all_Mmin[galaxy_type][lens_bin]),
                              label='Emulated',color='k',linestyle='--')
                ax[0,ax_x].set_xscale('log')
                ax[0,ax_x].set_title(f'{galaxy_type[:3]} bin {lens_bin+1}')
                ax[0,ax_x].set_xlabel(r'$r_p$ [Mpc]')
                if(ax_x==0):
                    ax[0,ax_x].set_ylabel(r'$r_p \Delta\Sigma$ [M$_\odot$/pc$^2$]')

    if(plot):
        fig.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f'{script_name}_tomo.png',
                    bbox_inches='tight',dpi=300,transparent=transparent_background)

    np.savez(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f'{script_name}_Mmin.npz',**all_Mmin)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    fit_mmin_darkemu(config)