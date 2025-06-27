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
                        get_reference_datavector_of_galtype,get_scales_mask_from_degrees,calculate_sigma_sys,read_npz_file
import matplotlib.gridspec as gridspec
from matplotlib.patheffects import withStroke



script_name = 'source_redshift_slope'

def hide_errorbars(errlines,reverse=False):
    for line in errlines:
        for err in line[2]:
            err.set_visible(reverse)


def apply_deltaz_shifts(galaxy_type,version,full_data,full_zlens,full_zsource,
                        hscy3_deltaz_shifts):
    
    from astropy.cosmology import Planck18 as desicosmo
    cosmo = desicosmo.clone(name='Planck18_h1', H0=100)

    from dsigma.physics import effective_critical_surface_density
    n_data = len(full_data)
    data = np.copy(full_data)
    # zlens = np.copy(full_zlens)
    zsource = np.copy(full_zsource)
    assert n_data % 3 == 0, "N must be a multiple of 3 for equal thirds."
    third = n_data // 3

    assert third % 4 == 0, "N must be a multiple of 12 for equal fourths."
    fourth = third // 4

    # Create the array with integer type
    zbins_lens = np.concatenate((np.zeros(third, dtype=int), 
                             np.ones(third, dtype=int), 
                             np.full(third, 2, dtype=int)))

    zbins_source = np.tile(np.concatenate((np.zeros(fourth, dtype=int),
                                    np.ones(fourth, dtype=int),
                                    np.full(fourth, 2, dtype=int),
                                    np.full(fourth, 3, dtype=int))),3)
    
    sys.path.append('/global/homes/s/sven/code/lensingWithoutBorders/')
    from load_catalogues import read_nofz, get_lens_nofz

    nofz_lens = get_lens_nofz(galaxy_type,"HSCY3",version)
    nofz_source = read_nofz("HSCY3")

    for lens_bin in range(3):
        for source_bin in range(4):
            # fiducial critical surface density
            sigmacrit_fiducial = effective_critical_surface_density(nofz_lens['z_mid'],
                                                                    nofz_source['z'],
                                                                    nofz_source['n'][:,source_bin],cosmo)
            # shifted redshift distribution
            z_arr_shifted = nofz_source['z'] + hscy3_deltaz_shifts[source_bin]
            # mask for valid redshifts
            mask = (z_arr_shifted >= 0)
            # shifted critical surface density
            sigmacrit_shifted = effective_critical_surface_density(nofz_lens['z_mid'],z_arr_shifted[mask],
                                                                   nofz_source['n'][:,source_bin][mask],cosmo)
            # compute the shift in the amplitude
            amplitude_shift = np.average(sigmacrit_shifted,weights=nofz_lens[f'n_{lens_bin+1}'])/np.average(sigmacrit_fiducial,weights=nofz_lens[f'n_{lens_bin+1}'])

            mask = (zbins_lens == lens_bin) & (zbins_source == source_bin)
            data[mask] *= amplitude_shift
            zsource[mask] += hscy3_deltaz_shifts[source_bin]
            print(f"{galaxy_type}  l{lens_bin+1}, s{source_bin+1}, amplitude shift: {amplitude_shift:.3f}, zshift: {hscy3_deltaz_shifts[source_bin]:.3f}")

    return data,full_zlens,zsource

    



def source_redshift_slope_tomo(config,plot,datavec=None,logger="create",all_zsource=None, 
                               compute_sigma_sys=True,allbins=False,hscy3_deltaz_shifts=None):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)
    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    galaxy_types = clean_read(config,'general','galaxy_types',split=True)
    critical_sigma = clean_read(config,'general','critical_sigma',split=False,convert_to_float=True)

    min_deg = clean_read(config,'general','min_deg',split=False,convert_to_float=True)
    max_deg = clean_read(config,'general','max_deg',split=False,convert_to_float=True)
    rp_pivot = clean_read(config,'general','rp_pivot',split=False,convert_to_float=True)
    scales_list = clean_read(config,'general','analyzed_scales',split=True)
    transparent_background = clean_read(config,'general','transparent_background',split=False,convert_to_bool=True)

    n_BGS_BRIGHT_bins = config.getint('general','N_BGS_BRIGHT_bins')
    n_LRG_bins = config.getint('general','N_LRG_bins')
    n_total_bins = n_BGS_BRIGHT_bins + n_LRG_bins

    rp = clean_read(config,'general','rp',split=True,convert_to_float=True)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)

    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)

    add_secondary_effects_slope = clean_read(config,script_name,'add_secondary_effects',split=False,convert_to_bool=True)

    # print("Got B")
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    if logger == "create":
        logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name+'_tomo',__name__)


    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance',split=False,convert_to_bool=True)
    # if(use_theory_covariance):
    #     logger.info("Using theory covariance")
    # else:
    #     logger.info("Using jackknife covariance")
    # if "SDSS" in survey_list:
        # idx = survey_list.index("SDSS")
        # survey_list.pop(idx)
        # color_list.pop(idx)

    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    os.makedirs(savepath_slope_values,exist_ok=True)
    plot_slope = clean_read(config,script_name,'plot_slope',split=False,convert_to_bool=True)
    plot_slope_uncertainty = clean_read(config,script_name,'plot_slope_uncertainty',split=False,convert_to_bool=True)
    slope_color = clean_read(config,script_name,'slope_color',split=False)
    slope_uncertainty = clean_read(config,script_name,'slope_uncertainty',split=False)

    boost_factor = clean_read(config,'general','boost_factor',split=False,convert_to_bool=True)

    assert slope_uncertainty in ["covariance","randoms","randoms_covariance"], f"slope_uncertainty {slope_uncertainty} must be one of covariance, randoms, randoms_covariance"
    if(plot):
        if logger is not None:
            logger.info("Preparing plot")
        fig,ax,gs = initialize_gridspec_figure((7.24,7.24/n_total_bins*len(scales_list)),
                            len(scales_list),
                            n_total_bins,
                            hspace=0,wspace=0)
        add_colorbar_legend(fig,ax,gs,color_list,survey_list)
        fil = open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"ds_amplitudes_tomo_allbins_{allbins}.txt","w")
        fil.write("# galaxy_type lens_bin source_survey source_bin scale A_DS A_DS_err z_source\n")

        errlines = []
        if slope_uncertainty == "randoms":
            fstr = ""
            if allbins:
                fstr += "_allbins"
            if hscy3_deltaz_shifts is not None:
                fstr += "_hscy3_deltaz_shifts"
            p_list_randoms = np.load(savepath_slope_values+os.sep+f"redshift_slope_tomo_p_arr{fstr}.npy")
            key_list_randoms = np.load(savepath_slope_values+os.sep+f"redshift_slope_tomo_keys{fstr}.npy",allow_pickle=True)
            p_randoms_std = np.std(p_list_randoms,axis=0)
            V_randoms_std = np.zeros((*p_randoms_std.shape,p_randoms_std.shape[-1]))
            for i in range(p_list_randoms.shape[1]):
                V_randoms_std[i] = np.cov(p_list_randoms[:,i,:].T)

        if add_secondary_effects_slope:
            if allbins:
                secondary_effects_slopes = dict(read_npz_file(savepath+os.sep+version+os.sep+"secondary_effects/systematics_intrinsic_alignment_source_magnification_boost_boost_source_reduced_shear_allbins.npz")['arr_0'].item())
            else:
                secondary_effects_slopes = dict(read_npz_file(savepath+os.sep+version+os.sep+"secondary_effects/systematics_intrinsic_alignment_source_magnification_boost_boost_source_reduced_shear.npz")['arr_0'].item())


    p_list = {}
    V_list = {}
    sigma_sys_list = {}
    reduced_chisq_list = {}
    samples_list = {}
    read_data = (plot or (not use_theory_covariance) or (datavec is None))
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
                descriptors = []

                for ss,source_survey in enumerate(survey_list):
                    if(read_data):
                        _,full_data,_,mycov,full_zlens,full_zsource,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                                        data_path,"deltasigma",
                                                                                        versions,logger=logger,
                                                                                        boost=boost_factor)
                        if hscy3_deltaz_shifts is not None and source_survey == "HSCY3":
                            full_data,full_zlens,full_zsource = apply_deltaz_shifts(galaxy_type,version,full_data,full_zlens,full_zsource,
                                                                                    hscy3_deltaz_shifts)


                    if(datavec is not None):
                        if logger is not None:
                            logger.info(f"Using mock datavector for {galaxy_type} {source_survey}")
                        full_data = datavec[f"{galaxy_type}_{source_survey}"]

                    if(use_theory_covariance):
                        full_cov = load_covariance_chris(galaxy_type,source_survey,"deltasigma",
                                                        chris_path)
                    else:
                        full_cov = mycov

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
                        if(read_data):
                            zsource = full_zsource[bin_mask][scales_mask]
                            zlens = full_zlens[bin_mask][scales_mask]
                        cov = full_cov[bin_mask][:,bin_mask][scales_mask][:,scales_mask]
                        if allbins:
                            # print("Allbins", np.mean(zsource),np.mean(zlens))
                            if np.mean(zsource) < np.mean(zlens):
                                continue
                            if not np.all(np.isfinite(zsource)) or not np.all(np.isfinite(zlens)):
                                print("Not all data finite in ",galaxy_type,source_survey,lens_bin,source_bin)
                                print(zsource,zlens)
                                continue
                        if not np.all(np.isfinite(data)):
                            print("Not all data finite in ",galaxy_type,source_survey,lens_bin,source_bin)
                            print(data)
                            raise ValueError("Not all data finite")
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
                        descriptors.append(f"{galaxy_type} {lens_bin+1} {source_survey} {source_bin+1} {scaletitle.replace(' ','_')}")
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
                if(len(dvs)>2):
                    p,V = np.polyfit(zsources,lensamp,1,
                                    w=1/lensamperr,cov=True)
                    p_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = p
                    V_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = V
                elif(len(dvs)==2):
                    p = np.polyfit(zsources,lensamp,1,
                                    w=1/lensamperr)
                    p_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = p
                    V_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = np.zeros((2,2))+np.nan
                else:
                    p = np.zeros(2)+np.nan
                    V = np.zeros((2,2))+np.nan
                if(compute_sigma_sys):
                    reduced_chisq,sigma_sys,samples = calculate_sigma_sys(lensamp,lensamperr,method=config[script_name]['sigma_sys_method'])
                    reduced_chisq_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = reduced_chisq
                    sigma_sys_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = sigma_sys
                    samples_list[f"{galaxy_type}_{scaletitle}_{lens_bin}"] = samples

                if(plot):
                    ax_x = lens_bin+gt*n_BGS_BRIGHT_bins
                    if np.isfinite(sigma_sys[2]):
                        upper_sigma_sys = sigma_sys[3]-sigma_sys[2]
                        lower_sigma_sys = sigma_sys[2]-sigma_sys[1]
                        if np.isnan(lower_sigma_sys):
                            ax[scale,ax_x].text(0.5,0.3,f"$\\sigma_{{\\mathrm{{sys}}}}\\leq {sigma_sys[2]:.2f}+{upper_sigma_sys:.2f}$",
                                                transform=ax[scale,ax_x].transAxes,horizontalalignment="center",
                                                fontsize=8,path_effects=[withStroke(linewidth=1, foreground='white')] )

                        else:
                            ax[scale,ax_x].text(0.5,0.3,f"$\\sigma_{{\\mathrm{{sys}}}}={sigma_sys[2]:.2f}^{{+{upper_sigma_sys:.2f}}}_{{-{lower_sigma_sys:.2f}}}$",
                                                transform=ax[scale,ax_x].transAxes,horizontalalignment="center",
                                                fontsize=8,path_effects=[withStroke(linewidth=1, foreground='white')] )
                    for x in range(len(dvs)):
                        source_survey = survey_list[survey_indices[x]]
                        color = color_list[survey_indices[x]]

                        errlines.append(ax[scale,ax_x].errorbar(zsources[x],lensamp[x],lensamperr[x],fmt='o',
                                        color=color))
                        randfac = 1.2342987
                        fil.write(f"{descriptors[x]} {randfac*lensamp[x]} {randfac*lensamperr[x]} {zsources[x]}\n")
                        if(x==0):
                            if(scale==0):
                                ax[scale,ax_x].set_title(f"{galaxy_type[:3]} bin {lens_bin+1}")
                            if(ax_x==0):
                                ax[scale,ax_x].set_ylabel(f"$A_\\mathrm{{\\Delta\\Sigma}}$, {scaletitle}",
                                                          fontsize=8)
                            if(scale==len(scales_list)-1):
                                ax[scale,ax_x].set_xlabel(r"$\langle z_{\mathrm{source}}\rangle$")
                    if slope_uncertainty == "covariance":
                        slope_uncertainty_val = np.sqrt(V[0,0])
                    elif slope_uncertainty == "randoms":
                        idx = np.where(key_list_randoms==f"{galaxy_type}_{scaletitle}_{lens_bin}")[0]
                        if len(idx)==1:
                            slope_uncertainty_val = p_randoms_std[idx][0][0]

                        else:
                            # print(f"Found {len(idx)} matches for {galaxy_type}_{scaletitle}_{lens_bin}, available keys: {key_list_randoms}")
                            slope_uncertainty_val = np.nan
                    elif slope_uncertainty == "randoms_covariance":
                        raise NotImplementedError
                    
                    if(plot_slope):
                        xarr = np.linspace(np.min(zsources),np.max(zsources),100)
                        y = xarr*p[0]+p[1]
                        ax[scale,ax_x].plot(xarr,y,color=slope_color,
                                            linestyle='--')
                        if(plot_slope_uncertainty):

                            if slope_uncertainty == "covariance":
                                slope_covmat = V
                            elif slope_uncertainty == "randoms":
                                if len(idx)==1:
                                    slope_covmat = V_randoms_std[idx][0]
                                else:
                                    slope_covmat = np.zeros((2,2))+np.nan
                            elif slope_uncertainty == "randoms_covariance":
                                raise NotImplementedError
                            dy = np.sqrt((xarr**2 * slope_covmat[0, 0]) + slope_covmat[1, 1] + 2 * xarr * slope_covmat[0, 1])
                            ax[scale,ax_x].fill_between(xarr, y - dy, y + dy, color='gray', alpha=0.5)
                        
                        if add_secondary_effects_slope:
                            p_secondary = secondary_effects_slopes[f"{galaxy_type}_{scaletitle}_{lens_bin}"]
                            y_secondary = xarr*p_secondary[0]+p_secondary[1]
                            y_secondary -= np.mean(y_secondary)
                            ax[scale,ax_x].plot(xarr,y_secondary+np.mean(y),color='black',
                                            linestyle='--')

                    if np.isfinite(p[0]):
                        slope_uncertainty_val = np.sqrt(slope_covmat[0,0])
                        if(np.abs(p[0])> critical_sigma*slope_uncertainty_val):
                            bbox=dict(facecolor='gray', edgecolor='red', boxstyle='round,pad=0.2', alpha=0.2)
                            ax[scale,ax_x].text(0.5,0.15,f"$\\beta={p[0]:.2f}\\pm {slope_uncertainty_val:.2f}$",
                                            transform=ax[scale,ax_x].transAxes,horizontalalignment="center",bbox=bbox,
                                            fontsize=8,path_effects=[withStroke(linewidth=1, foreground='white')] )
                        else:
                            ax[scale,ax_x].text(0.5,0.15,f"$\\beta={p[0]:.2f}\\pm {slope_uncertainty_val:.2f}$",
                                            transform=ax[scale,ax_x].transAxes,horizontalalignment="center",
                                            fontsize=8,path_effects=[withStroke(linewidth=1, foreground='white')])

    if(plot):
        fil.close()
        hide_errorbars(errlines)
        for scale in range(len(scales_list)):
            for ax_x in range(n_total_bins):
                ax[scale,ax_x].relim()
                ax[scale,ax_x].autoscale()
        hide_errorbars(errlines,reverse=True)
        plt.tight_layout()
        fstr = ""
        if allbins:
            fstr += "_allbins"
        if hscy3_deltaz_shifts is not None:
            fstr += "_hscy3_deltaz_shifts"
        if "SDSS" in survey_list:
            fstr += "_SDSS"
            transparent_background = False

        plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+f"source_redshift_slope_tomo{fstr}.png",
                    dpi=300,transparent=transparent_background,bbox_inches="tight")
        plt.close()
    if all_zsource is None:
        return p_list,V_list,all_zsource_return,sigma_sys_list,reduced_chisq_list,samples_list
    else:
        return p_list,V_list,all_zsource,sigma_sys_list,reduced_chisq_list



if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    source_redshift_slope_tomo(config,plot=True,hscy3_deltaz_shifts=[0,0,0.115,0.192])
    # sys.exit()

    # source_redshift_slope_tomo(config,plot=True,allbins=True)

    version = clean_read(config,'general','version',split=False)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    os.makedirs(savepath_slope_values,exist_ok=True)

    # p_list,V_list = source_redshift_slope_notomo(config,plot=True)
    # np.save(savepath_slope_values+os.sep+script_name+"_notomo_data_p_list",p_list)
    # np.save(savepath_slope_values+os.sep+script_name+"_notomo_data_V_list",V_list)

    # print(p_list)
    # print(V_list)
    p_list,V_list,_,sigma_sys_list,reduced_chisq_list,samples_list = source_redshift_slope_tomo(config,plot=True)
    if "SDSS" in config['general']['lensing_surveys']:
        fstr = "_SDSS"
    else:
        fstr = ""
    np.save(savepath_slope_values+os.sep+script_name+"_tomo_data_p_list"+fstr,p_list)
    np.save(savepath_slope_values+os.sep+script_name+"_tomo_data_V_list"+fstr,V_list)
    np.save(savepath_slope_values+os.sep+script_name+"_tomo_sigma_sys_list"+fstr,sigma_sys_list)
    np.save(savepath_slope_values+os.sep+script_name+"_tomo_reduced_chisq_list"+fstr,reduced_chisq_list)
    np.save(savepath_slope_values+os.sep+script_name+"_tomo_samples_list"+fstr,samples_list)
    # print(p_list)
    # print(V_list)

