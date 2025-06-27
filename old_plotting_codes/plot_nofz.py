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

script_name = 'nofz'

def plot_nofz(config):
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

    fig,ax,gs = initialize_gridspec_figure((7.24,12),6,len(survey_list),hspace=0,wspace=0)
    for x,source_survey in enumerate(survey_list):
        ax[0,x].set_xlim(0,1.7)

        ax[-1,x].set_xlabel("$z$")
        ax[0,x].set_title(source_survey)
        tab = read_nofz(source_survey)
        for _lbin in range(6):
            if(_lbin<3):
                galaxy_type = "BGS_BRIGHT"
                lbin = _lbin
            else:
                galaxy_type = "LRG"
                lbin = _lbin-3
            zmin = lensbins[galaxy_type][lbin]
            zmax = lensbins[galaxy_type][lbin+1]
            if(x==0):
                ax[_lbin,x].set_ylabel(f"{galaxy_type} bin {lbin+1}")
            key = f"{source_survey.lower()}_{galaxy_type}_l{lbin}"

            ax[_lbin,x].axvspan(zmin,zmax,color="gray",alpha=0.5)
            

            # ax[_lbin,x].text(0.5,0.94,"SN <10 / SN >10 / SN full",
            #                 transform = ax[_lbin,x].transAxes,
            #                 color="black",horizontalalignment="center")
            for i in range(get_number_of_source_bins(source_survey.lower())):
                if(i in get_allowed_bins(galaxy_type,source_survey,lbin)):
                    ls = "-"
                else:
                    if(i in get_allowed_bins(galaxy_type,source_survey,lbin,conservative_cut=False)):
                        ls = "--"
                    else:
                        ls = ":"
                ax[_lbin,x].plot(tab['z'],tab['n'][:,i],linestyle=ls,color=f"C{i}")
    add_colorbar_legend(fig,ax,gs,["C"+str(i) for i in range(5)],[f"Bin {i}" for i in range(1,6)])
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+"nofz_DESIxLensingSurveys_manypanels.png",
                dpi=300,transparent=True,bbox_inches="tight")
    

def plot_nofz_onepanel(config):
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

    fig,ax,gs = initialize_gridspec_figure((7.24,2),1,len(survey_list),hspace=0,wspace=0)
    for x,source_survey in enumerate(survey_list):
        ax[0,x].set_xlim(0,1.7)
        ax[0,x].set_ylim(0,5.9)

        ax[0,x].set_xlabel("$z$")
        ax[0,x].set_title(source_survey)
        tab = read_nofz(source_survey)
        for _lbin in range(6):
            if(_lbin<3):
                galaxy_type = "BGS_BRIGHT"
                lbin = _lbin
            else:
                galaxy_type = "LRG"
                lbin = _lbin-3
            zmin = lensbins[galaxy_type][lbin]
            zmax = lensbins[galaxy_type][lbin+1]
            key = f"{source_survey.lower()}_{galaxy_type}_l{lbin}"

            ax[0,x].axvline(zmin,ls="--" if lbin==0 else ":",color="black",lw=0.75)
            if _lbin==5:
                ax[0,x].axvline(zmax,ls="--",color="black",lw=0.75)
            # ax[x].vlines(zmax,0,1.7,ls=":" if lbin==0 else ":",color="black")

            # ax[_lbin,x].text(0.5,0.94,"SN <10 / SN >10 / SN full",
            #                 transform = ax[_lbin,x].transAxes,
            #                 color="black",horizontalalignment="center")
            for i in range(get_number_of_source_bins(source_survey.lower())):
                # if(i in get_allowed_bins(galaxy_type,source_survey,lbin)):
                #     ls = "-"
                # else:
                #     if(i in get_allowed_bins(galaxy_type,source_survey,lbin,conservative_cut=False)):
                #         ls = "--"
                #     else:
                #         ls = ":"

                ax[0,x].plot(tab['z'],tab['n'][:,i],linestyle="-",color=f"C{i}")
    add_colorbar_legend(fig,ax,gs,["C"+str(i) for i in range(5)],[f"Bin {i}" for i in range(1,6)])
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+"nofz_DESIxLensingSurveys.png",
                dpi=300,transparent=False,bbox_inches="tight")

def generate_latex_table(config):
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    galaxy_types = ["BGS_BRIGHT", "LRG"]
    
    # Start LaTeX table
    latex = "\\begin{tabular}{ll|ccc|ccc}\n\\hline\n"
    latex += " & \\multicolumn{3}{c|}{BGS BRIGHT} & \\multicolumn{3}{c}{LRG} \\\\\n"
    latex += "& & Bin 1 & Bin 2 & Bin 3 & Bin 1 & Bin 2 & Bin 3 \\\\\n\\hline\n"
    
    for survey in survey_list:
        n_source_bins = get_number_of_source_bins(survey.lower())
        # Add survey name as multirow
        latex += f"\\multirow{{{n_source_bins}}}{{*}}{{{survey}}}"
        
        for source_bin in range(n_source_bins):
            if source_bin > 0:
                latex += " "  # Space for alignment with multirow
            latex += f" & Bin {source_bin + 1} "
            
            # Loop through galaxy types and their bins
            for gtype in galaxy_types:
                for lens_bin in range(3):  # Each type has 3 bins
                    # Check if bin is allowed
                    is_allowed_conservative = source_bin in get_allowed_bins(gtype, survey, lens_bin)
                    is_allowed_less_conservative = source_bin in get_allowed_bins(gtype, survey, lens_bin, conservative_cut=False)
                    
                    if is_allowed_conservative:
                        symbol = "c"
                    elif is_allowed_less_conservative:
                        symbol = "lc"
                    else:
                        symbol = "x"
                    
                    latex += f"& {symbol} "
            latex += "\\\\\n"
        latex += "\\hline\n"
    
    latex += "\\end{tabular}\n"
    
    # Save to file
    version = clean_read(config,'general','version',split=False)
    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    with open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+"source_lens_bins.tex", "w") as f:
        f.write(latex)

if __name__=="__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    # plot_nofz(config)
    plot_nofz_onepanel(config)
    generate_latex_table(config)
