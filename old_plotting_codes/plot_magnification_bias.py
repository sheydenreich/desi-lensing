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
                        get_pvalue,get_rp_from_deg,get_scales_mask_from_degrees,generate_bmode_tomo_latex_table_from_dict
import matplotlib.gridspec as gridspec
                        

import json



def format_keys(key):
    key = key.replace("_"," ")
    key = key.replace(">","$>$")
    key = key.replace("<","$<$")
    return key

def generate_individual_cuts_latex_table(data, category):
    first_key = list(data[f"{category}_0"].keys())[0]
    latex = "\\begin{tabular}{l|c|c|c|c|c|c|}\n"
    latex += "Category "
    for bin in range(1,4):
        latex += " & \\multicolumn{2}{c|}{Bin "+str(bin)+"}"
    latex += "\\\\\n"
    # latex += "& \\multicolumn{2}{c|}{" + str(data[f"{category}_0"][first_key][-1]) + "} & \\multicolumn{2}{c|}{" + str(data[f"{category}_1"][first_key][-1]) + "} & \\multicolumn{2}{c}{" + str(data[f"{category}_2"][first_key][-1]) + "} \\\\\n\\hline\n"
    for i in range(3):
        latex += "& \# cut & $\\alpha$ "
    latex += "\\\\\n\\hline\n"

    for key in data[f"{category}_0"].keys():
        bin_1 = f"{data[f'{category}_0'][key][3]} & {data[f'{category}_0'][key][0]:.2f}"
        bin_2 = f"{data[f'{category}_1'][key][3]} & {data[f'{category}_1'][key][0]:.2f}"
        bin_3 = f"{data[f'{category}_2'][key][3]} & {data[f'{category}_2'][key][0]:.2f}"
        if(np.any(np.array([data[f'{category}_{i}'][key][0] for i in range(3)])>0.5)):
            latex += f"\\textbf{{{format_keys(key)}}} & {bin_1} & {bin_2} & {bin_3} \\\\\n"
        else:
            latex += f"{format_keys(key)} & {bin_1} & {bin_2} & {bin_3} \\\\\n"

    latex += "\\hline\n"
    latex += "\# Galaxies in bin "
    for bin in range(3):
        latex += f" & {data[f'{category}_{bin}'][first_key][-1]} & -- "
    

    latex += "\\end{tabular}\n"
    return latex

def generate_magnification_bias_latex_table(data,magbias_sys_dict = None):
    galaxy_types = data.keys()
    latex = "\\begin{tabular}{l|c|c}\n"

    for gt in galaxy_types:
        latex += f" & {gt}".replace("_"," ")
    latex += "\\\\\n\\hline\n"

    for i in range(3):
        latex += f"Bin {i+1} "
        for gt in galaxy_types:
            fstr = ""
            if magbias_sys_dict is not None:
                fstr = f"\\,{{\\color{{blue}} \\pm {magbias_sys_dict[f'{gt}_{i}']:.2f} }}"
            latex += f" & ${data[gt]['alphas'][i]:.2f} \\pm {data[gt]['alphas_error'][i]:.2f}{fstr}$"
        latex += "\\\\\n"

    latex += "\\end{tabular}\n"
    return latex


script_name = 'magnification_bias'

def plot_magnification_bias(config):
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
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    data_path = chris_path + os.sep + 'magnification_bias_DESI/' + version + os.sep

    with open(data_path+"DESI_magnification_bias.json","r") as f:
        alphas = json.load(f)
    with open(data_path+"DESI_magnification_bias_no_secondary_cuts.json","r") as f:
        alphas_no_secondary_cuts = json.load(f)
    alphas_buzzard = [0.9145633761955659, 1.5806488673171675, 2.0206666166515483, 2.58339234742153, 2.259265182806545, 1.8994795623320744]

    fig,ax = plt.subplots(1,2,figsize=(10,5))
    offset=0.05
    for gt,galaxy_type in enumerate(["BGS_BRIGHT","LRG"]):
        for lens_bin in range(3):
            ax[gt].errorbar(0+lens_bin-offset,alphas[galaxy_type]['simple_alphas'][lens_bin],
                            yerr=alphas[galaxy_type]['simple_alphas_error'][lens_bin],fmt='^',label="simple alpha" if lens_bin==0 else None,
                            color="C0")
            ax[gt].errorbar(0.1+lens_bin-offset,alphas_no_secondary_cuts[galaxy_type]['simple_alphas'][lens_bin],
                            yerr=alphas_no_secondary_cuts[galaxy_type]['simple_alphas_error'][lens_bin],fmt='v',label="simple alpha, no secondary cuts" if lens_bin==0 else None,
                            color="C0")
            
            ax[gt].errorbar(0.25+lens_bin-offset,alphas[galaxy_type]['alphas'][lens_bin],
                            yerr=alphas[galaxy_type]['alphas_error'][lens_bin],fmt='^',label="alpha fit" if lens_bin==0 else None,
                            color="C1")
            ax[gt].errorbar(0.35+lens_bin-offset,alphas_no_secondary_cuts[galaxy_type]['alphas'][lens_bin],
                            yerr=alphas_no_secondary_cuts[galaxy_type]['alphas_error'][lens_bin],fmt='v',label="alpha fit, no secondary cuts" if lens_bin==0 else None,
                            color="C1")
            ax[gt].plot(-0.15+lens_bin-offset,alphas_buzzard[lens_bin+3*gt],'o',label="Buzzard" if lens_bin==0 else None,color="k")
            ax[gt].set_xticks([0,1,2])
            ax[gt].set_xticklabels(["Bin 1","Bin 2","Bin 3"])
            ax[gt].set_xlabel("Lens bin")
            ax[gt].set_title(galaxy_type)

    ax[1].legend()
    ax[0].set_ylabel(r"$\alpha_\mathrm{L}$")

    plt.tight_layout()
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+"magnification_bias.png",transparent=transparent_background,
                bbox_inches='tight',dpi=300)
    

def plot_magnification_bias_profiles(config):
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
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    data_path = chris_path + os.sep + 'magnification_bias_DESI/' + version + os.sep

    with open(data_path+"DESI_magnification_bias.json","r") as f:
        alphas = json.load(f)
    with open(data_path+"DESI_magnification_bias_no_secondary_cuts.json","r") as f:
        alphas_no_secondary_cuts = json.load(f)
    with open(data_path+"DESI_magnification_bias_exponentialprofile.json","r") as f:
        alphas_exponential = json.load(f)
    with open(data_path+"DESI_magnification_bias_nofibermagnificationcorrection.json","r") as f:
        alphas_nofibermagnificationcorrection = json.load(f)
    
    alphas_buzzard = [0.9145633761955659, 1.5806488673171675, 2.0206666166515483, 2.58339234742153, 2.259265182806545, 1.8994795623320744]


    magbias_sys_errors = {}

    fig,ax = plt.subplots(1,2,figsize=(10,5))
    offset=0.15
    for gt,galaxy_type in enumerate(["BGS_BRIGHT","LRG"]):
        for lens_bin in range(3):
            # ax[gt].errorbar(0+lens_bin-offset,alphas[galaxy_type]['simple_alphas'][lens_bin],
            #                 yerr=alphas[galaxy_type]['simple_alphas_error'][lens_bin],fmt='^',label="simple alpha" if lens_bin==0 else None,
            #                 color="C0")
            # ax[gt].errorbar(0.1+lens_bin-offset,alphas_no_secondary_cuts[galaxy_type]['simple_alphas'][lens_bin],
            #                 yerr=alphas_no_secondary_cuts[galaxy_type]['simple_alphas_error'][lens_bin],fmt='v',label="simple alpha, no secondary cuts" if lens_bin==0 else None,
            #                 color="C0")
            
            ax[gt].errorbar(0+lens_bin-offset,alphas[galaxy_type]['alphas'][lens_bin],
                            yerr=alphas[galaxy_type]['alphas_error'][lens_bin],fmt='^',
                            label="deVaucouleurs profile" if lens_bin==0 else None,
                            color="C1")
            ax[gt].errorbar(0.1+lens_bin-offset,alphas_exponential[galaxy_type]['alphas'][lens_bin],
                yerr=alphas_exponential[galaxy_type]['alphas_error'][lens_bin],fmt='^',
                label="exponential profile" if lens_bin==0 else None,
                color="C2")
            
            ax[gt].errorbar(0.2+lens_bin-offset,alphas_nofibermagnificationcorrection[galaxy_type]['alphas'][lens_bin],
                yerr=alphas_nofibermagnificationcorrection[galaxy_type]['alphas_error'][lens_bin],fmt='^',
                label="no fiber correction" if lens_bin==0 else None,
                color="C3")
            
            magbias_sys_errors[f"{galaxy_type}_{lens_bin}"] = np.std([alphas[galaxy_type]['alphas'][lens_bin],
                                                                      alphas_exponential[galaxy_type]['alphas'][lens_bin],
                                                                      alphas_nofibermagnificationcorrection[galaxy_type]['alphas'][lens_bin]])

            # ax[gt].errorbar(0.1+lens_bin-offset,alphas_no_secondary_cuts[galaxy_type]['alphas'][lens_bin],
            #                 yerr=alphas_no_secondary_cuts[galaxy_type]['alphas_error'][lens_bin],fmt='v',label="alpha, no secondary cuts" if lens_bin==0 else None,
            #                 color="C1")
            


            ax[gt].plot(-0.1+lens_bin-offset,alphas_buzzard[lens_bin+3*gt],'o',label="Buzzard" if lens_bin==0 else None,color="k")
            ax[gt].set_xticks([0,1,2])
            ax[gt].set_xticklabels(["Bin 1","Bin 2","Bin 3"])
            ax[gt].set_xlabel("Lens bin")
            ax[gt].set_title(galaxy_type)

    ax[1].legend()
    ax[0].set_ylabel(r"$\alpha_\mathrm{L}$")

    plt.tight_layout()
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+"magnification_bias_profiles.png",transparent=transparent_background,
                bbox_inches='tight',dpi=300)

    return magbias_sys_errors


def make_magnification_bias_table(config,magbias_sys_dict):
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
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    data_path = chris_path + os.sep + 'magnification_bias_DESI/' + version + os.sep

    with open(data_path+"DESI_magnification_bias.json","r") as f:
        alphas = json.load(f)


    latex_table = generate_magnification_bias_latex_table(alphas, magbias_sys_dict)
    with open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"magnification_bias_table.tex","w") as f:
        f.write(latex_table)

def make_magnification_bias_tables_individual_cuts(config):
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
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    data_path = chris_path + os.sep + 'magnification_bias_DESI/' + version + os.sep

    with open(data_path+"DESI_magnification_bias_individual_cuts.json","r") as f:
        alphas_individual = json.load(f)

    for galaxy_type in galaxy_types:
        latex_table = generate_individual_cuts_latex_table(alphas_individual, galaxy_type)
        with open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"magnification_bias_individual_cuts_{galaxy_type}.tex","w") as f:
            f.write(latex_table)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    plot_magnification_bias(config)
    magbias_sys_dict = plot_magnification_bias_profiles(config)
    make_magnification_bias_table(config,magbias_sys_dict)
    make_magnification_bias_tables_individual_cuts(config)
