import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,initialize_gridspec_figure,add_colorbar_legend
import astropy.units as u
from datetime import datetime
from data_handler import load_covariance_chris,get_rp_chris,get_allowed_bins,get_number_of_source_bins,get_bins_mask,\
                        load_data_and_covariance_notomo,load_data_and_covariance_tomo,get_number_of_lens_bins,combine_datavectors,\
                        get_number_of_radial_bins

script_name = 'plot_notomo_measurement_vs_compressed'

def plot_notomo_measurement_vs_compressed(config,bmodes=False):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    for statistic in ['deltasigma','gammat']:
        fig,ax,gs = initialize_gridspec_figure((6,10),6,2,add_cbar=True,
                              height_ratios=[1,0.3,1,0.3,1,0.3]
                              )
        add_colorbar_legend(fig,ax,gs,color_list,survey_list,start=0,skip=2)
        for gt,galaxy_type in enumerate(["BGS_BRIGHT","LRG"]):
            for index,color,source_survey in zip(np.arange(len(color_list)),color_list,survey_list):
                data = load_data_and_covariance_notomo(galaxy_type,source_survey,
                                                    fpath=data_path,
                                                    statistic=statistic,
                                                    versions=versions,
                                                    logger=logger,
                                                    bmodes=bmodes,
                                                    correct_for_magnification_bias=not bmodes)[1]
                cov = load_covariance_chris(galaxy_type,source_survey,
                                            statistic,chris_path,logger=logger,
                                            pure_noise=bmodes)
                _,data_tomo,error_tomo,_,_,_,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                    fpath=data_path,
                                                    statistic=statistic,
                                                    versions=versions,
                                                    logger=logger,
                                                    bmodes=bmodes,
                                                    correct_for_magnification_bias=not bmodes)
                _rp = get_rp_chris(galaxy_type,source_survey,chris_path,statistic,logger=logger)
                for lens_bin in range(get_number_of_lens_bins(galaxy_type)):
                    rp = _rp[:get_number_of_radial_bins(galaxy_type,source_survey,lens_bin)]
                    allowed_bins = get_allowed_bins(galaxy_type,source_survey,lens_bin)
                    if len(allowed_bins)==0:
                        continue
                    bins_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,allowed_bins)
                    compressed_dv,error = combine_datavectors(data_tomo[bins_mask],
                                                        cov[bins_mask][:,bins_mask],
                                                        optimal_matrix=False)
                    
                    if(bmodes):
                        scale = 1
                    else:
                        scale = rp
                    # compressed_dv_opt,error_opt = combine_datavectors(data_tomo[bins_mask],
                    #                 cov[bins_mask][:,bins_mask],
                    #                 optimal_matrix=True)
                    ax[2*lens_bin,gt].errorbar(rp,scale*compressed_dv,
                                            yerr=scale*np.sqrt(np.diag(error)),
                                            color=color,fmt='+',label=source_survey+" compressed")
                    # ax[lens_bin,gt].errorbar(rp*1.2,rp*compressed_dv_opt,
                    #                         yerr=rp*np.sqrt(np.diag(error_opt)),
                    #                         color=color,fmt='x',label=source_survey+" compressed with optimal matrix")
                    ax[2*lens_bin,gt].errorbar(rp*1.1,scale*data[lens_bin],
                                            yerr=scale*np.sqrt(np.diag(error)),
                                            color=color,fmt='x',label=source_survey+" nontomographic")
                    ax[2*lens_bin,gt].set_xscale('log')
                    # ax[2*lens_bin,gt].set_ylim(-5,10)
                    # ax[2*lens_bin+1,gt].errorbar(rp*1.1,(compressed_dv_opt-data[lens_bin])/np.sqrt(np.diag(error)),np.sqrt(np.diag(error_opt))/np.sqrt(np.diag(error)),fmt='x')
                    ax[2*lens_bin+1,gt].plot(rp,(compressed_dv-data[lens_bin])/np.sqrt(np.diag(error)),
                                             marker='o',ls="",color=color)
                    ax[2*lens_bin+1,gt].set_ylim(-3,3)
                    ax[2*lens_bin+1,gt].axhspan(-1,1,color='gray',alpha=0.5)
                    ax[2*lens_bin+1,gt].axhline(0,color='k',ls=':')

                    if(bmodes):
                        compressed_chisq = np.einsum("i,ij,j",compressed_dv,np.linalg.inv(error),compressed_dv)
                        notomo_chisq = np.einsum("i,ij,j",data[lens_bin],np.linalg.inv(error),data[lens_bin])
                        ax[2*lens_bin,gt].text(0.1,0.9-0.1*index,f"$\\chi^2$ comp/meas: {compressed_chisq:.1f}/{notomo_chisq:.1f}",transform=ax[2*lens_bin,gt].transAxes,
                                               color=color)
                        

                    
                    if(gt==0):
                        if(bmodes):
                            ax[2*lens_bin,gt].set_ylabel(f"$\\Delta\\Sigma_\\times(R_p)$, bin {lens_bin}")
                        else:
                            ax[2*lens_bin,gt].set_ylabel(f"$R_p\\times\\Delta\\Sigma(R_p)$, bin {lens_bin}")
                    if(lens_bin==0):
                        ax[lens_bin,gt].set_title(f"{galaxy_type}")

                ax[2*lens_bin+1,gt].set_xlabel(r"$R_p$ [Mpc/h]")
        # ax[0,0].legend(loc='upper right')
        plt.tight_layout()
        if(bmodes):
            plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_bmodes_notomo_vs_compressed.png",bbox_inches='tight')

        else:
            plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_notomo_vs_compressed.png",bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")

    plot_notomo_measurement_vs_compressed(config)
    plot_notomo_measurement_vs_compressed(config,bmodes=True)

