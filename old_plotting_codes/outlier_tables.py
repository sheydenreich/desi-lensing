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
                        get_reference_datavector_of_galtype,get_scales_mask_from_degrees,generate_redshift_slope_latex_table_from_dict,generate_splits_latex_table_from_dict
import matplotlib.gridspec as gridspec
from scipy.stats import binom,norm

def get_binomial_significance(config,n_total):
    allowed_sigma = clean_read(config,'general','critical_sigma',split=False,convert_to_float=True)
    probability = 1 - norm.cdf(allowed_sigma)
    return lambda x: 1-binom.cdf(x-1, n_total, probability)

def analyze_redshift_slope(config,script_name = 'source_redshift_slope',make_table=True):
    version = clean_read(config,'general','version',split=False)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)

    statistic = clean_read(config,"general","statistic",split=False)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep

    data_slopes = np.load(savepath_slope_values+os.sep+script_name+"_tomo_data_p_list.npy",allow_pickle=True).item()
    random_slopes = np.load(savepath_slope_values+os.sep+"redshift_slope_tomo_p_arr.npy")
    random_keys = np.load(savepath_slope_values+os.sep+"redshift_slope_tomo_keys.npy",allow_pickle=True)

    allowed_sigma = clean_read(config,'general','critical_sigma',split=False,convert_to_float=True)


    error_dict = {}
    data_dict = {}
    for key in data_slopes.keys():
        error_dict[key] = np.std(random_slopes[:,random_keys==key,0])
        data_dict[key] = data_slopes[key][0]

    if(make_table):
        mytab = generate_redshift_slope_latex_table_from_dict(data_dict,error_dict,config,"Measured source redshift slopes",precision=3)

        fil = open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"tab_{statistic}_{script_name}_slopes.tex", "w")
        fil.write(mytab)
        fil.close()

    return data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma

def analyze_splits(config,script_name = 'splits',make_table=True):
    version = clean_read(config,'general','version',split=False)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)

    statistic = clean_read(config,"general","statistic",split=False)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep

    data_slopes = np.load(savepath_slope_values+os.sep+script_name+"_data_p_list.npy",allow_pickle=True).item()
    random_slopes = np.load(savepath_slope_values+os.sep+"splits_p_arr.npy")
    random_keys = np.load(savepath_slope_values+os.sep+"splits_keys.npy",allow_pickle=True)

    allowed_sigma = clean_read(config,'general','critical_sigma',split=False,convert_to_float=True)


    error_dict = {}
    data_dict = {}
    for key in data_slopes.keys():
        error_dict[key] = np.std(random_slopes[:,random_keys==key,0])
        data_dict[key] = data_slopes[key][0]

    if(make_table):
        mytab = generate_splits_latex_table_from_dict(data_dict,error_dict,config,"Measured slopes for splits",precision=3)

        fil = open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"tab_{statistic}_{script_name}_slopes.tex", "w")
        fil.write(mytab)
        fil.close()

    return data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma



def plot_significance(config,data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma,ax):
    # figure out where the data has significant outliers
    n_total = len(data_slopes.keys())
    where_significant = np.zeros(n_total,dtype=bool)
    for x,key in enumerate(data_dict.keys()):
        if(np.abs(data_dict[key]) > allowed_sigma*error_dict[key]):
            where_significant[x] = True

    n_significant = np.sum(where_significant)

    # figure out the incidence of outliers in the randoms
    n_randoms = random_slopes.shape[0]
    random_significance = np.zeros(n_randoms,dtype=int)
    for x,key in enumerate(data_dict.keys()):
        _randoms = random_slopes[:,random_keys==key,0][:,0]
        mask = (np.abs(_randoms) > allowed_sigma*error_dict[key])
        random_significance[mask] += 1

    random_significance_fraction = np.bincount(random_significance,minlength=10)/n_randoms
    random_significance_fraction_cumulative = np.cumsum(random_significance_fraction[::-1])[::-1]
    # plot the distribution of randoms
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)

    binom_fnc = get_binomial_significance(config,n_total)
    ax.step(np.arange(10),random_significance_fraction_cumulative,where='mid',label=f"Outliers in {n_randoms} randoms")
    ax.step(np.arange(10),binom_fnc(np.arange(10)),where='mid',label='Binomial distribution')
    ax.text(0.05,0.95,f"Allowed significance: {allowed_sigma:.1f}",transform=ax.transAxes,fontsize=10)
    ax.text(0.05,0.9,f"Total measurements: {n_total}",transform=ax.transAxes,fontsize=10)
    assert np.allclose([random_significance_fraction_cumulative[0],binom_fnc(0)],1)

    ax.axvline(n_significant,color='k',linestyle='--',label=f"Data: {n_significant} outliers")

    # ax.set_yscale('log')
    # ax.legend()
    # plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_source_redshift_slopes_significance.png",dpi=300,
    #             bbox_inches='tight')

def analyze_significances(config,make_tables=True):
    fig,ax = plt.subplots(2,1,figsize=(4,9),sharex=True)
    data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma = analyze_splits(config,make_table=make_tables)
    plot_significance(config,data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma,ax[0])
    data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma = analyze_redshift_slope(config,make_table=make_tables)
    plot_significance(config,data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma,ax[1])
    ax[0].set_ylabel("$P(n\\geq N_\\mathrm{{out}})$")
    ax[1].set_ylabel("$P(n\\geq N_\\mathrm{{out}})$")
    ax[0].set_title("Lens inhomogeneity slopes")
    ax[1].set_title("Source redshift slopes")
    # ax[0].set_xlabel("Number of outliers $n$")
    ax[1].set_xlabel("Number of outliers $n$")
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    ax[0].set_ylim(1e-5,1.1)
    ax[1].set_ylim(1e-5,1.1)

    version = clean_read(config,'general','version',split=False)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)

    statistic = clean_read(config,"general","statistic",split=False)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,'randoms','savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon,exist_ok=True)
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_slopes_significance.png",dpi=300,
                bbox_inches='tight')
    
    plt.close()

def plot_one_outlier_graph(config,data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma_array,ax):
    # figure out where the data has significant outliers
    n_total = len(data_slopes.keys())
    where_significant = np.zeros((len(allowed_sigma_array),n_total),dtype=bool)
    for x,key in enumerate(data_dict.keys()):
        outlier_mask = np.abs(data_dict[key]) > allowed_sigma_array*error_dict[key]
        where_significant[outlier_mask,x] = True

    n_significant = np.sum(where_significant,axis=1)

    # figure out the incidence of outliers in the randoms
    n_randoms = random_slopes.shape[0]
    random_significance = np.zeros((len(allowed_sigma_array),n_randoms),dtype=int)
    for x,key in enumerate(data_dict.keys()):
        _randoms = random_slopes[:,random_keys==key,0][:,0]
        mask = (np.abs(_randoms)[None,:] > (allowed_sigma_array*error_dict[key])[:,None])
        random_significance[mask] += 1

    random_significance_fraction = np.sum(random_significance,axis=1).astype(float)/n_randoms
    random_significance_std = np.std(random_significance,axis=1)/np.sqrt(n_randoms)

    random_atleast_one = np.sum(random_significance>0,axis=1).astype(float)/n_randoms
    random_atleast_one_std = np.std(random_significance>0,axis=1)/np.sqrt(n_randoms)

    ax.plot(allowed_sigma_array,n_significant,label=f"Outliers in data")
    ax.fill_between(allowed_sigma_array,random_significance_fraction-random_significance_std,random_significance_fraction+random_significance_std,alpha=0.5,
                    color="gray")
    ax.plot(allowed_sigma_array,random_significance_fraction,color="k",label=f"Average outliers from {n_randoms} randoms")
    ax.plot(allowed_sigma_array,random_significance_fraction+random_significance_std,color="k",linestyle="--",lw=0.5)
    ax.plot(allowed_sigma_array,random_significance_fraction-random_significance_std,color="k",linestyle="--",lw=0.5)

    ax.fill_between(allowed_sigma_array,random_atleast_one-random_atleast_one_std,random_atleast_one+random_atleast_one_std,alpha=0.5,
                    color="red")
    ax.plot(allowed_sigma_array,random_atleast_one,color="red",linestyle="--",label=f"Probability of at least one outlier")
    ax.plot(allowed_sigma_array,random_atleast_one+random_atleast_one_std,color="red",linestyle="--",lw=0.5)
    ax.plot(allowed_sigma_array,random_atleast_one-random_atleast_one_std,color="red",linestyle="--",lw=0.5)


def plot_outlier_graph(config):
    fig,ax = plt.subplots(2,1,figsize=(4,9),sharex=True)
    allowed_sigma_array = np.linspace(1,5,400)
    data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma = analyze_splits(config,make_table=False)
    plot_one_outlier_graph(config,data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma_array,ax[0])
    data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma = analyze_redshift_slope(config,make_table=False)
    plot_one_outlier_graph(config,data_slopes,data_dict,error_dict,random_slopes,random_keys,allowed_sigma_array,ax[1])
    ax[0].set_ylabel("Number of outliers")
    ax[1].set_ylabel("Number of outliers")
    ax[0].set_title("Lens inhomogeneity slopes")
    ax[1].set_title("Source redshift slopes")
    # ax[0].set_xlabel("Number of outliers $n$")
    ax[1].set_xlabel("Allowed significance")
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    ax[0].set_ylim(1e-5,15*1.1)
    ax[1].set_ylim(1e-5,15*1.1)

    version = clean_read(config,'general','version',split=False)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)

    statistic = clean_read(config,"general","statistic",split=False)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,'randoms','savepath_addon',split=False)

    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon,exist_ok=True)
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"{statistic}_slopes_number_of_outliers.png",dpi=300,
                bbox_inches='tight')
    
    plt.close()



if __name__=="__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    plot_outlier_graph(config)
    analyze_significances(config)

