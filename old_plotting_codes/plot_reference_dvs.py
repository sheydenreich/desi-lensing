import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,add_colorbar_legend,add_colorbar_legend,initialize_gridspec_figure
import astropy.units as u
from datetime import datetime
from data_handler import load_covariance_chris,get_rp_chris,get_allowed_bins,get_number_of_source_bins,get_bins_mask,\
                        load_data_and_covariance_notomo,load_data_and_covariance_tomo,get_number_of_lens_bins,combine_datavectors,\
                        get_number_of_radial_bins,get_reference_datavector,get_scales_mask,get_deltasigma_amplitudes,\
                        get_reference_datavector_of_galtype,get_scales_mask_from_degrees
import matplotlib.gridspec as gridspec
sys.path.insert(0,os.path.abspath('..'))
from load_catalogues import read_nofz

script_name = 'reference_dvs'

def write_dict_to_file(filename, mydict):
    if not mydict:
        print("The linked list is empty.")
        return

    # Open the file in write mode
    with open(filename, 'w') as file:
        # Extract keys for the header
        keys = list(mydict.keys())
        file.write(','.join(keys) + '\n')

        # Get the length of the arrays (assuming all arrays have the same length)
        array_length = len(mydict[keys[0]])

        # Write the data rows
        for i in range(array_length):
            row = [str(mydict[key][i]) for key in keys]
            file.write(','.join(row) + '\n')

def plot_reference_dvs(config):
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

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    lensbins = {'BGS_BRIGHT':clean_read(config,'general','BGS_BRIGHT_bins',True,convert_to_float=True),
                'LRG':clean_read(config,'general','LRG_bins',True,convert_to_float=True)}

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    reference_dv_models = clean_read(config,script_name,'models',split=True)

    fig,ax,gs = initialize_gridspec_figure((9,15),3,2,hspace=0,wspace=0)
    add_colorbar_legend(fig,ax,gs,color_list[:len(reference_dv_models)],reference_dv_models)

    n_bgs_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_lrg_bins = config.getint('general','N_LRG_bins')
    n_bins_plot = {'BGS_BRIGHT':n_bgs_bins,'LRG':n_lrg_bins,'total':n_bgs_bins+n_lrg_bins}

    ref_dvs = {}

    for gt,galaxy_type in enumerate(galaxy_types):
        rp = get_rp_chris(galaxy_type,"KiDS",chris_path,"deltasigma")
        if gt==0:
            ref_dvs['rp'] = rp[:15]

        for lens_bin in range(n_bins_plot[galaxy_type]):
            bin_mask = get_bins_mask(galaxy_type,"KiDS",lens_bin,4)

            for ref_dv,ref_dv_modelname in enumerate(reference_dv_models):
                ref_dv_model = get_reference_datavector_of_galtype(config,rp[bin_mask],galaxy_type,lens_bin,datavector_type=ref_dv_modelname)
                ref_dvs[f'{galaxy_type}_{lens_bin+1}_{ref_dv_modelname}'] = ref_dv_model
                ax[lens_bin,gt].plot(rp[bin_mask],rp[bin_mask]*ref_dv_model,color=color_list[ref_dv],label=ref_dv_modelname)
            if(gt==0):
                ax[lens_bin,gt].set_ylabel(r"$\Delta\Sigma(r_p)$")
            ax[lens_bin,gt].set_xscale("log")
        ax[0,gt].set_title(f"{galaxy_type}")
        ax[-1,gt].set_xlabel("$r_p$")

    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+"reference_datavectors.png",
                dpi=300,transparent=False,bbox_inches="tight")
    write_dict_to_file(savepath+os.sep+version+os.sep+savepath_addon+os.sep+"reference_datavectors.txt",ref_dvs)
    
if __name__=="__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    plot_reference_dvs(config)
