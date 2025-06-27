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
                        get_reference_datavector_of_galtype,get_scales_mask_from_degrees,calculate_sigma_sys,load_data_table_notomo
import matplotlib.gridspec as gridspec
from astropy.table import Table
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_catalogues import get_source_table,read_nofz
import healpy as hp
from astropy.cosmology import FlatLambdaCDM


cosmology = FlatLambdaCDM(H0=100,Om0=0.3)

from dsigma.physics import effective_critical_surface_density

script_name = 'shapenoise'

def correct_stacking_bias(tab_result,scalar_shear_response_correction=False,
                          matrix_shear_response_correction=False,
                          shear_responsivity_correction=False,
                          hsc_selection_bias_correction=False,
                          photo_z_dilution_correction=None,
                          boost_correction = None,
                          random_subtraction = None,
                          verbose=True):
    correction_factor = 1
    if scalar_shear_response_correction:
        if(verbose):
            print("correcting scalar shear response")
        correction_factor /= np.average(tab_result['1+m'],weights=tab_result['n_pairs'])

    if matrix_shear_response_correction:
        if(verbose):
            print("correcting matrix shear response")
        correction_factor /= np.average(tab_result['R_t'],weights=tab_result['n_pairs'])

    if shear_responsivity_correction:
        if(verbose):
            print("correcting shear responsivity")
        correction_factor /= np.average(tab_result['2R'],weights=tab_result['n_pairs'])

    if hsc_selection_bias_correction:
        if(verbose):
            print("correcting HSC selection bias")
        correction_factor *= np.average(tab_result['1+m_sel'],weights=tab_result['n_pairs'])
    
    return correction_factor


def cut_to_footprint(hpmap,tab_s,ra_col="ra",dec_col="dec"):
    from copy import deepcopy
    nside = hp.get_nside(hpmap)
    table = deepcopy(tab_s)
    phi,theta = np.radians(table[ra_col]),np.radians(90.-table[dec_col])
    ipix = hp.ang2pix(nside,theta,phi,nest=False)
    cut = (hpmap[ipix] > 1e-10)
    table = table[cut]
    return table


def calculate_shapenoise(config,logger="create"):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)
    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    galaxy_types = clean_read(config,'general','galaxy_types',split=True)


    all_splits = clean_read(config,'splits','splits_to_consider',split=True)

    rp = clean_read(config,'general','rp',split=True,convert_to_float=True)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    # print("Got B")
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    if logger == "create":
        logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)



    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    os.makedirs(savepath_slope_values,exist_ok=True)

    for galaxy_type in galaxy_types:
        for survey in survey_list:
            if(survey=="SDSS"):
                continue
            logger.info("Calculating shapenoise for survey: {}, overlap with DESI {}".format(survey,galaxy_type))
            tab_s,kwargs,precompute_kwargs,stacking_kwargs = get_source_table(survey,galaxy_type,config=config)
            nofz = read_nofz(survey)

            for split_by in all_splits:
                if split_by.lower() == "ntile":
                    n_splits = clean_read(config,'splits',f'n_ntile_{galaxy_type[:3]}',split=False,convert_to_int=True)
                    n_splits_computed = clean_read(config,'splits',f'n_ntile_computed_{galaxy_type[:3]}',split=False,convert_to_int=True)
                else:
                    n_splits = clean_read(config,'splits','n_splits',split=False,convert_to_int=True)
                    n_splits_computed = n_splits

                for split in range(n_splits_computed):

                    logger.info("Calculating shapenoise for split: %s"%split_by)
                    hpmap = np.loadtxt(data_path + versions[galaxy_type] + '/healpix_maps/' + survey + '/' + f'hpmap_{galaxy_type}_split_{split_by}_{split}_of_{n_splits}.dat')
                    
                    tab_s_split = cut_to_footprint(hpmap,tab_s)
                    tab_l_split = Table.read(data_path + versions[galaxy_type] + '/split_tables/' + survey + '/' + split_by + '/' + galaxy_type + f'_split_{split_by}_{split}_of_{n_splits}.fits')

                    nofz_effective = np.zeros(len(nofz['z']))
                    for i in range(len(nofz['n'].shape[1])):
                        nofz_effective += nofz['n'][:,i] * np.sum(tab_s_split['w'][tab_s_split['z_bin']==i])


                    for lens_bin in range(get_number_of_lens_bins(galaxy_type)):
                        tab_s_result = load_data_table_notomo(galaxy_type,survey,data_path,"gammat",lens_bin,versions,
                                                                    split_by=split_by,split=split,n_splits=n_splits,logger=logger)
                        
                        
                        correction_factor = correct_stacking_bias(tab_s_result,**stacking_kwargs)

                        shapenoise = np.average(np.sqrt(tab_s_split["e_1"]**2+tab_s_split["e_2"]**2),weights=tab_s_split["w"])
                        shapenoise *= correction_factor


                        lens_bins = clean_read(config,"general",galaxy_type+"_bins",True,convert_to_float=True)

                        mask_lens = ((tab_l_split['z']<lens_bins[lens_bin+1]) & (tab_l_split['z']>=lens_bins[lens_bin]))

                        lens_nofz, lens_nofz_edges = np.histogram(tab_l_split['z'][mask_lens],bins=nofz['z'],weights=tab_l_split['w'][mask_lens])
                        lens_nofz_means = (lens_nofz_edges[1:]+lens_nofz_edges[:-1])/2.

                        sigmacrit_eff = effective_critical_surface_density(lens_nofz_means,nofz['z'],nofz_effective,cosmology)
                        sigmacrit_eff = np.average(sigmacrit_eff,weights=lens_nofz)

                        np.save(savepath_slope_values + f"shapenoise_{survey}_{galaxy_type}_{lens_bin}_{split_by}_{split}_of_{n_splits}.npy",np.array([shapenoise,sigmacrit_eff]))
                        print(f"Saved shapenoise {shapenoise:.2f} to {savepath_slope_values}/shapenoise_{survey}_{galaxy_type}_{lens_bin}_{split_by}_{split}_of_{n_splits}.npy")



def plot_shapenoise_vs_jackknife(config,logger="create"):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)
    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    galaxy_types = clean_read(config,'general','galaxy_types',split=True)


    all_splits = clean_read(config,'splits','splits_to_consider',split=True)

    rp = clean_read(config,'general','rp',split=True,convert_to_float=True)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    # print("Got B")
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    if logger == "create":
        logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)



    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    os.makedirs(savepath_slope_values,exist_ok=True)

    fig,ax,gs = initialize_gridspec_figure((2*len(all_splits),
                                2*len(galaxy_types)*get_number_of_lens_bins(galaxy_types[0])),
                                len(galaxy_types)*get_number_of_lens_bins(galaxy_types[0]),
                        len(all_splits))
    add_colorbar_legend(fig,ax,gs,color_list,survey_list)

    for gt,galaxy_type in enumerate(galaxy_types):
        for survey in survey_list:
            for sb,split_by in enumerate(all_splits):
                if split_by.lower() == "ntile":
                    n_splits = clean_read(config,'splits',f'n_ntile_{galaxy_type[:3]}',split=False,convert_to_int=True)
                    n_splits_computed = clean_read(config,'splits',f'n_ntile_computed_{galaxy_type[:3]}',split=False,convert_to_int=True)
                else:
                    n_splits = clean_read(config,'splits','n_splits',split=False,convert_to_int=True)
                    n_splits_computed = n_splits

                for split in range(n_splits_computed):
                    for lens_bin in range(get_number_of_lens_bins(galaxy_type)):
                        tab_s_result = load_data_table_notomo(galaxy_type,survey,data_path,"gammat",lens_bin,versions,
                                                                    split_by=split_by,split=split,n_splits=n_splits,logger=logger)
                        
                        shapenoise,sigmacrit_eff = np.load(savepath_slope_values + f"shapenoise_{survey}_{galaxy_type}_{lens_bin}_{split_by}_{split}_of_{n_splits}.npy")

                        ax[3*gt+lens_bin,sb].plot(rp,rp*shapenoise/np.sqrt(tab_s_result["n_pairs"])*sigmacrit_eff,color=color_list[gt],linestyle='-')
                        ax[3*gt+lens_bin,sb].set_xscale("log")
                        ax[3*gt+lens_bin,sb].plot(rp,tab_s_result['ds_err'],color=color_list[gt],linestyle='--')
    ax[0,0].plot([],[],color='k',linestyle='-',label='shapenoise')
    ax[0,0].plot([],[],color='k',linestyle='--',label='jackknife error')
    ax[0,0].legend(loc='upper right')
    fig.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"shapenoise_vs_jackknife.png",dpi=300)

                        # print(f"Saved shapenoise {shapenoise:.2f} to {savepath_slope_values}/shapenoise_{survey}_{galaxy_type}_{lens_bin}_{split_by}_{split}_of_{n_splits}.npy")

if __name__=="__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    calculate_shapenoise(config)
    plot_shapenoise_vs_jackknife(config)