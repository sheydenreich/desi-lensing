import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from dsigma.physics import effective_critical_surface_density
from dsigma.precompute import photo_z_dilution_factor
from scipy.optimize import brentq

from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,add_colorbar_legend,initialize_gridspec_figure
import astropy.units as u
from datetime import datetime
from data_handler import load_covariance_chris,get_rp_chris,get_allowed_bins,get_number_of_source_bins,get_bins_mask,\
                        load_data_and_covariance_notomo,load_data_and_covariance_tomo,get_number_of_lens_bins,combine_datavectors,\
                        get_number_of_radial_bins,get_reference_datavector,get_scales_mask,get_deltasigma_amplitudes,\
                        get_reference_datavector_of_galtype,get_scales_mask_from_degrees,calculate_sigma_sys
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath('../'))
from load_catalogues import read_nofz,get_lens_table,get_source_table

from astropy.cosmology import Planck18 as desicosmo
cosmo = desicosmo.clone(name='Planck18_h1', H0=100)

def hide_errorbars(errlines,reverse=False):
    for line in errlines:
        for err in line[2]:
            err.set_visible(reverse)


# TODO: Make this go away. refractor shit!
def get_lens_bins(galaxy_type):
    if galaxy_type[:3] == "BGS":
        return np.array([0.1,0.2,0.3,0.4])
    elif galaxy_type[:3] == "LRG":
        return np.array([0.4,0.6,0.8,1.1])

def lens_amplitude_modification_deltaz(z_lens_mean,z_arr,dndz_arr,dz,cosmology,table_c=None,lens_source_cut=None):
    """
    Compute the shift in the DeltaSigma amplitude induced by a shift in the source redshift distribution.
    """
    if table_c is None:
        # fiducial critical surface density
        sigmacrit_fiducial = effective_critical_surface_density(z_lens_mean,z_arr,dndz_arr,cosmology)
        # shifted redshift distribution
        z_arr_shifted = z_arr + dz
        # mask for valid redshifts
        mask = (z_arr_shifted >= 0)
        # shifted critical surface density
        sigmacrit_shifted = effective_critical_surface_density(z_lens_mean,z_arr_shifted[mask],dndz_arr[mask],cosmology)
        # compute the shift in the amplitude
        amplitude_shift = sigmacrit_shifted/sigmacrit_fiducial
    else:
        f_bias_fid = photo_z_dilution_factor(z_lens_mean,table_c,cosmology,lens_source_cut=None)
        table_c_shifted = table_c.copy()
        table_c_shifted['z_true'] = table_c['z_true'] + dz
        mask = table_c_shifted['z_true'] >= 0
        if not np.any(mask):
            return 1
        f_bias_shifted = photo_z_dilution_factor(z_lens_mean,table_c_shifted[mask],cosmology,lens_source_cut=None)
        amplitude_shift = f_bias_shifted/f_bias_fid
    return amplitude_shift

def find_dz_shift_from_amplitude(current_value,target_value,dz_range,z_lens_mean,z_arr,dndz_arr,cosmology,table_c=None,lens_source_cut=None):
    target_func = lambda dz: lens_amplitude_modification_deltaz(z_lens_mean,z_arr,dndz_arr,dz,cosmology,table_c=table_c,lens_source_cut=lens_source_cut)*current_value-target_value
        # Using brentq to find the root, where the function changes sign.
    try:
        dz_solution = brentq(target_func, *dz_range)
    except ValueError as ve:
        print(ve)
        dz_solution = np.nan
    return dz_solution




script_name = 'delta_z_effects'

def delta_z_effects_tomo(config,plot,datavec=None,logger="create",all_zsource=None):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)
    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    galaxy_types = clean_read(config,'general','galaxy_types',split=True)

    min_deg = clean_read(config,'general','min_deg',split=False,convert_to_float=True)
    max_deg = clean_read(config,'general','max_deg',split=False,convert_to_float=True)
    rp_pivot = clean_read(config,'general','rp_pivot',split=False,convert_to_float=True)
    scales_list = clean_read(config,'general','analyzed_scales',split=True)

    rp = clean_read(config,'general','rp',split=True,convert_to_float=True)


    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    if logger == "create":
        logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_tomo',__name__)

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    if 'SDSS' in survey_list:
        idx_sdss = survey_list.index('SDSS')
        survey_list.pop(idx_sdss)
        color_list.pop(idx_sdss)

    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    # if(use_theory_covariance):
    #     logger.info("Using theory covariance")
    # else:
    #     logger.info("Using jackknife covariance")

    include_fbias = clean_read(config,script_name,'include_fbias',split=False,convert_to_bool=True)
    dz_min = clean_read(config,script_name,'dz_min',split=False,convert_to_float=True)
    dz_max = clean_read(config,script_name,'dz_max',split=False,convert_to_float=True)

    if(plot):
        if logger is not None:
            logger.info("Preparing plot")
        fig,ax,gs = initialize_gridspec_figure((7.24,7.24/n_total_bins*len(scales_list)),
                            len(scales_list),
                            n_total_bins,
                            hspace=0,wspace=0)
        add_colorbar_legend(fig, ax, gs, color_list[:len(survey_list)],survey_list)


    p_list = {}
    V_list = {}
    sigma_sys_list = {}
    reduced_chisq_list = {}
    read_data = (plot or (not use_theory_covariance) or (datavec is None))
    errlines = []
    if(read_data):
        print("Reading data!")
        print("*"*50)
    if all_zsource is None:
        all_zsource_return = {}
    for scale,scaletitle in enumerate(scales_list):
        for gt,galaxy_type in enumerate(galaxy_types):
            n_lens_bins = config.getint('general',f'N_{galaxy_type}_bins')


            for lens_bin in range(n_lens_bins):
                n_radial_bins = get_number_of_radial_bins(galaxy_type,survey_list[0],None)
                scales_mask = get_scales_mask_from_degrees(rp,scaletitle,min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)
                n_used_bins = np.sum(scales_mask)
                
                dvs = []
                covs = []
                if(read_data):
                    zsources = []
                    zlenses = []
                survey_indices = []
                survey_bins = []

                for ss,source_survey in enumerate(survey_list):
                    allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                    if len(allowed_bins)==0:
                        continue
                    if(read_data):
                        _,full_data,_,mycov,full_zlens,full_zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                        data_path,"deltasigma",
                                                                                        versions,logger=logger)


                    if(datavec is not None):
                        if logger is not None:
                            logger.info(f"Using mock datavector for {galaxy_type} {source_survey}")
                        full_data = datavec[f"{galaxy_type}_{source_survey}"]

                    if(use_theory_covariance):
                        full_cov = load_covariance_chris(galaxy_type,source_survey,"deltasigma",
                                                        chris_path)
                    else:
                        full_cov = mycov

                    n_source_bins = len(allowed_bins)
                    if(n_source_bins==0):
                        if logger is not None:
                            logger.info(f"Skipping {galaxy_type} {source_survey} {lens_bin}")
                    for source_bin in allowed_bins:
                        bin_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,[source_bin])

                        data = full_data[bin_mask][scales_mask]
                        if(read_data):
                            zsource = full_zsource[bin_mask][scales_mask]
                            zlens = full_zlens[bin_mask][scales_mask]
                        cov = full_cov[bin_mask][:,bin_mask][scales_mask][:,scales_mask]

                        assert np.all(np.isfinite(data))
                        if(read_data):
                            assert np.all(np.isfinite(zsource))
                            assert np.all(np.isfinite(zlens))
                        assert np.all(np.isfinite(cov))

                        dvs.append(data)
                        covs.append(cov)
                        if(read_data):
                            zsources.append(np.mean(zsource))
                            zlenses.append(np.mean(zlens))
                        survey_indices.append(ss)
                        survey_bins.append(source_bin)
                reference_dv = get_reference_datavector_of_galtype(config,rp,galaxy_type,lens_bin)
                reference_dv = reference_dv[scales_mask]
                dvs = np.array(dvs)
                covs = np.array(covs)
                if(read_data):
                    zsources = np.array(zsources)
                    zlenses = np.array(zlenses)
                    if all_zsource is None:
                        all_zsource_return[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = zsources
                else:
                    zsources = all_zsource[f"{galaxy_type}_{scaletitle}_{lens_bin}"]
                lensamp,lensamperr,_ = get_deltasigma_amplitudes(dvs,covs,reference_dv,substract_mean=False)

                if(plot):
                    ax_x = lens_bin+gt*n_BGS_BRIGHT_bins
                    for x in range(len(dvs)):
                        source_survey = survey_list[survey_indices[x]]
                        color = color_list[survey_indices[x]]

                        full_lens_table,_ = get_lens_table(galaxy_type,source_survey,None,None,
                                    versions=versions)
                        redshift_bins = get_lens_bins(galaxy_type)
                        lens_table = full_lens_table[np.logical_and(full_lens_table['z']>=redshift_bins[lens_bin],
                                                                    full_lens_table['z']<redshift_bins[lens_bin+1])]
                        z_lens_mean = np.average(lens_table['z'],weights=lens_table['w_sys'])
                        if include_fbias:
                            _,_,precompute_kwargs,_ = get_source_table(source_survey,galaxy_type)
                            if 'table_c' in precompute_kwargs.keys():
                                table_c = precompute_kwargs['table_c']
                            else:
                                table_c = None
                        else:
                            table_c = None
                        source_bin = survey_bins[x]
                        nofz = read_nofz(source_survey)

                        mean_lensamp = np.average(lensamp,weights=1/lensamperr**2)
                        
                        dz_lower = find_dz_shift_from_amplitude(lensamp[x]-lensamperr[x],mean_lensamp,
                                                                (dz_min,dz_max),z_lens_mean,
                                                                nofz['z'],nofz['n'][:,source_bin],
                                                                cosmo,table_c=table_c)
                        if np.isnan(dz_lower):
                            dz_lower = dz_min
                        dz_upper = find_dz_shift_from_amplitude(lensamp[x]+lensamperr[x],mean_lensamp,
                                                                (dz_min,dz_max),z_lens_mean,
                                                                nofz['z'],nofz['n'][:,source_bin],
                                                                cosmo,table_c=table_c)
                        if np.isnan(dz_upper):
                            dz_upper = dz_max
                            
                        dz = find_dz_shift_from_amplitude(lensamp[x],mean_lensamp,
                                                            (dz_min,dz_max),z_lens_mean,
                                                            nofz['z'],nofz['n'][:,source_bin],
                                                            cosmo,table_c=table_c)
                        
                        # if not dz_lower < dz_upper:
                        #     print("dz_lower",dz_lower)
                        #     print("dz_upper",dz_upper)
                        # else:
                        #     print("we're good")

                        errlines.append(ax[scale,ax_x].errorbar(zsources[x],dz,yerr=[[dz-dz_lower],[dz_upper-dz]],
                                                fmt='o',color=color))
                        ax[scale,ax_x].axhline(0,color='black',linestyle=':')

                        if(x==0):
                            if(scale==0):
                                ax[scale,ax_x].set_title(f"{galaxy_type[:3]} bin {lens_bin+1}")
                            if(ax_x==0):
                                ax[scale,ax_x].set_ylabel(f"$\\Delta z_{{\\mathrm{{source}}}}$ \n {scaletitle}")
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
        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"delta_z_fbias_{include_fbias}.png",
                    dpi=300,transparent=True,bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")


    delta_z_effects_tomo(config,plot=True,logger=None)