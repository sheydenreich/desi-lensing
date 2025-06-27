import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import configparser
import sys
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,add_colorbar_legend
from data_handler import load_covariance_chris,get_rp_chris,get_allowed_bins,get_number_of_source_bins,get_bins_mask,\
                        load_data_and_covariance_notomo,load_data_and_covariance_tomo,get_number_of_lens_bins,combine_datavectors,\
                        get_number_of_radial_bins,get_reference_datavector,get_scales_mask,get_deltasigma_amplitudes,load_mock_DV_chris,\
                        get_reference_datavector_of_galtype,get_ntot

from source_redshift_slope import source_redshift_slope_tomo #,source_redshift_slope_notomo
from plot_splits import plot_all_splits
from multiprocessing import Pool
from tqdm import tqdm
# from jax import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

                        

script_name = 'randoms'

def prepare_randoms_datavector(config,use_theory_covariance,datavector_type,
                               account_for_cross_covariance=True,
                                pure_noise=False,split_by=None,
                                split=None,n_splits=4,galaxy_types=None,
                                logger=None,tomographic=True):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    if galaxy_types is None:
        galaxy_types = clean_read(config,'general','galaxy_types',split=True)
    statistic = clean_read(config,'general','statistic',split=False)

    assert datavector_type in ['zero','emulator','chris','measured'], "datavector_type must be one of 'zero','emulator','chris','measured'"

    if(use_theory_covariance and (split_by is not None)):
        if logger is not None:
            logger.warning("Using theory covariance and splitting by "+split_by+" is not supported, setting use_theory_covariance to False")
        else:
            import warnings
            warnings.warn("Using theory covariance and splitting by "+split_by+" is not supported, setting use_theory_covariance to False")
        use_theory_covariance = False
    if(use_theory_covariance):
        if logger is not None:
            logger.info("Using theory covariance")
    else:
        if logger is not None:
            logger.info("Using jackknife covariance")
    
    datavectors = {}
    covariances = {}
    for galaxy_type in galaxy_types:
        for source_survey in survey_list:
            if tomographic:
                _,full_data,_,cov,_,_,_ = load_data_and_covariance_tomo(galaxy_type,source_survey,
                                                                    data_path,statistic,
                                                                    versions,bmodes=pure_noise,
                                                                    split_by=split_by,split=split,n_splits=n_splits,
                                                                    logger=logger)
            else:
                _,full_data,_,cov,_,_,_ = load_data_and_covariance_notomo(galaxy_type,source_survey,
                                                                    data_path,statistic,
                                                                    versions,bmodes=pure_noise,
                                                                    split_by=split_by,split=split,n_splits=n_splits,
                                                                    logger=logger)
            if(use_theory_covariance):
                cov = load_covariance_chris(galaxy_type,source_survey,statistic,
                                              chris_path,pure_noise=pure_noise,
                                              split_type=split_by,split=split,
                                              logger=logger)
            if(datavector_type == 'zero'):
                datavector = np.zeros_like(full_data)
            elif(datavector_type == 'emulator'):
                datavector = np.zeros_like(full_data)
                n_radial_bins = get_number_of_radial_bins(galaxy_type,source_survey,0)
                if statistic != "deltasigma":
                    raise NotImplementedError("Emulator datavector only implemented for deltasigma")
                rp = get_rp_chris(galaxy_type,source_survey,
                                  chris_path,statistic,logger=logger)[:n_radial_bins]
                counter = 0
                for lens_bin in range(get_number_of_lens_bins(galaxy_type)):
                    _datavector = get_reference_datavector_of_galtype(config,rp,
                                                                      galaxy_type,lens_bin)
                    n_source_bins = get_number_of_source_bins(source_survey)
                    for source_bin in range(n_source_bins):
                        datavector[counter*n_radial_bins:(counter+1)*n_radial_bins] = _datavector
                        counter += 1
                assert not np.any(np.isclose(datavector,0))
            elif(datavector_type == 'chris'):
                datavector = load_mock_DV_chris(galaxy_type,source_survey,chris_path,
                                                statistic,logger)
            elif(datavector_type == 'measured'):
                datavector = full_data
            else:
                raise ValueError(f"datavector_type {datavector_type} not known")
            datavectors[f"{galaxy_type}_{source_survey}"] = datavector

            # if not account_for_cross_covariance:
            covariances[f"{galaxy_type}_{source_survey}"] = cov
        if account_for_cross_covariance:
            
            if any("hscy3" == item.lower() for item in survey_list):
                assert not any("hscy1" == item.lower() for item in survey_list), "Cannot have both HSC Y1 and Y3 in survey list"
                full_cov = load_covariance_chris(galaxy_type,"all_y3",statistic,
                                                chris_path,pure_noise=pure_noise,
                                                split_type=split_by,split=split,
                                                logger=logger)
                hscy3 = True
            elif any("hscy1" == item.lower() for item in survey_list):
                assert not any("hscy3" == item.lower() for item in survey_list), "Cannot have both HSC Y1 and Y3 in survey list"
                full_cov = load_covariance_chris(galaxy_type,"all_y1",statistic,
                                                chris_path,pure_noise=pure_noise,
                                                split_type=split_by,split=split,
                                                logger=logger)
                hscy3 = False
            else:
                raise ValueError("Must have either HSC Y1 or Y3 in survey list")
            covariances[f"{galaxy_type}_all"] = full_cov
            if hscy3:
                datavectors[f"{galaxy_type}_all"] = np.concatenate([datavectors[f"{galaxy_type}_{source_survey}"] for source_survey in ["KiDS","DES","HSCY3"]],axis=0)
            else:
                datavectors[f"{galaxy_type}_all"] = np.concatenate([datavectors[f"{galaxy_type}_{source_survey}"] for source_survey in ["KiDS","DES","HSCY1"]],axis=0)
    return datavectors,covariances

def generate_randoms_datavectors(datavectors,covariances,n_randoms,n_processes,method="numpy",random_seed=0):
    randoms = {}
    np.random.seed(random_seed)
    # jnp_key = random.PRNGKey(np.random.randint(2**32-1))

    survey_keys = list(datavectors.keys())
    if np.any(["all" in survey_key for survey_key in survey_keys]):
        # check if this is a joint covariance situation
        combined_cov = True
        draw_keys = [survey_key for survey_key in survey_keys if "all" in survey_key]
    else:
        combined_cov = False
        draw_keys = survey_keys
    if method == "jax":
        for key in draw_keys:
            if(random_seed == 0):
                print(key,covariances[key].shape)
            # if(len(datavectors[key].shape) == 2):
            #     randoms[key] = np.zeros_like(datavectors[key])
            #     for i in range(datavectors[key].shape[0]):
            #         if np.any(np.isnan(covariances[key][i])):
            #             randoms[key][i] = np.zeros_like(datavectors[key][i]) + np.nan
            #         else:
            #             randoms[key][i] = random.multivariate_normal(jnp_key,datavectors[key][i],
            #                                                             covariances[key][i])
            # else:
            randoms[key] = random.multivariate_normal(jnp_key,datavectors[key],
                                                        covariances[key],shape=[n_randoms])
            if(random_seed == 0):
                print("Generated ",key,randoms[key].shape)
                if(np.any(np.isnan(randoms[key]))):
                    print("Nans in datvec:",np.any(np.isnan(datavectors[key])),"Nans in cov:",np.any(np.isnan(covariances[key])))
        return randoms
    elif method == "numpy":
        for key in draw_keys:
            if(len(datavectors[key].shape) == 2):
                randoms[key] = np.zeros([n_randoms,*datavectors[key].shape])
                for i in range(datavectors[key].shape[0]):
                    if np.any(np.isnan(covariances[key][i])):
                        randoms[key][:,i] = np.zeros_like(datavectors[key][i]) + np.nan
                    else:
                        randoms[key][:,i] = np.random.multivariate_normal(datavectors[key][i],
                                                                        covariances[key][i],
                                                                        size=n_randoms)
            else:
                randoms[key] = np.random.multivariate_normal(datavectors[key],
                                                            covariances[key],
                                                            size=n_randoms)
            if(random_seed == 0):
                print("Generated ",key,randoms[key].shape)
                if(np.any(np.isnan(randoms[key]))):
                    print("Nans in datvec:",np.any(np.isnan(datavectors[key])),"Nans in cov:",np.any(np.isnan(covariances[key])))
            if combined_cov:
                # split the _all into the individual surveys
                for survey_key in survey_keys:
                    # skip the ones that have "all"
                    if "all" in survey_key:
                        continue
                    # only take the ones that are the same except for the "all"-keyword
                    if not survey_key.split("_")[:-1] == key.split("_")[:-1]:
                        continue

                    print("Getting ",survey_key," from ",key)
                    galaxy_type = survey_key.split("_")[0]
                    if "kids" in survey_key.lower():
                        randoms[survey_key] = randoms[key][:,:get_ntot(galaxy_type,"kids")]
                    elif "des" in survey_key.lower():
                        randoms[survey_key] = randoms[key][:,get_ntot(galaxy_type,"kids"):get_ntot(galaxy_type,"kids")+get_ntot(galaxy_type,"des")]
                    elif "hsc" in survey_key.lower():
                        randoms[survey_key] = randoms[key][:,get_ntot(galaxy_type,"kids")+get_ntot(galaxy_type,"des"):]
                        assert randoms[survey_key].shape[1] == get_ntot(galaxy_type,"hscy1") or randoms[survey_key].shape[0] == get_ntot(galaxy_type,"hscy3") ,\
                            f"Randoms shape {randoms[survey_key].shape} does not match expected shape {get_ntot(galaxy_type,'hscy1')} or {get_ntot(galaxy_type,'hscy3')}"
                    elif "sdss" in survey_key.lower():
                        continue
                    else:
                        raise ValueError("Unknown survey: "+survey_key)
        return randoms

def splits_wrapper(config,datavecs,covariances,random_seed,logger):
    try:
        return *plot_all_splits(config,False,datavecs,covariances,True if random_seed==0 else False),random_seed
    except Exception as e:
        # Log the exception
        import traceback
        logger.error(f"Error in job {random_seed}: {e}\n{traceback.format_exc()}")
        return None  # Return a specific value or raise the exception

def source_redshift_slope_tomo_wrapper(config,datavecs,covariances,zsource,random_seed,logger,kwargs_dict={}):
    try:
        result = source_redshift_slope_tomo(config,False,datavecs,logger if random_seed==0 else None,zsource,False,**kwargs_dict)[:2]
        return *result,random_seed
    except Exception as e:
        # Log the exception
        import traceback
        logger.error(f"Error in job {random_seed}: {e}\n{traceback.format_exc()}")
        return None  # Return a specific value or raise the exception


def generate_random_source_redshift_slope_tomo(config,kwargs_dict={},fstr=""):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    galaxy_types = clean_read(config,'general','galaxy_types',split=True)
    statistic = clean_read(config,'general','statistic',split=False)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)
    boost_factor = clean_read(config,'general','boost_factor',split=False,convert_to_bool=True)

    savepath_addon = clean_read(config,'source_redshift_slope','savepath_addon',split=False)
    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    os.makedirs(savepath_slope_values,exist_ok=True)

    logger = get_logger(savepath_slope_values,script_name+'_tomo',__name__)

    n_processes = clean_read(config,script_name,'n_processes',split=False,convert_to_int=True)
    n_randoms = clean_read(config,script_name,'n_randoms',split=False,convert_to_int=True)
    verbose = clean_read(config,script_name,'verbose',split=False,convert_to_bool=True)
    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance_source_redshift_slope',split=False,convert_to_bool=True)
    datavector_type = clean_read(config,script_name,'datavector_type_source_redshift_slope',split=False)

    print("Preparing datavectors and covariances")
    datavectors,covariances = prepare_randoms_datavector(config,use_theory_covariance=use_theory_covariance,
                                                        datavector_type=datavector_type,
                                                        logger=logger)
    print("Done preparing datavectors and covariances, drawing randoms.")
    randoms_datvecs = generate_randoms_datavectors(datavectors,covariances,n_randoms,n_processes,method="numpy",random_seed=0)
    print("Done.")

    # jnp_key = random.PRNGKey(np.random.randint(2**32-1))
    datp,datV,all_zsource,_,_ = source_redshift_slope_tomo(config,False,None,logger,compute_sigma_sys=False,**kwargs_dict)
    print("Testing run with seed 0")
    p,V,i = source_redshift_slope_tomo_wrapper(config,datavectors,covariances,all_zsource,0,logger,kwargs_dict)
    print(p)
    all_keys = list(p.keys())
    with Pool(n_processes) as p:
        # if(verbose):
        #     niter = tqdm(niter)
        jobs = []
        p_arr = np.zeros((n_randoms,len(all_keys),2))
        V_arr = np.zeros((n_randoms,len(all_keys),2,2))
        with tqdm(total=n_randoms) as pbar:
            def fill_arrays(result):
                # print(result)
                p,V,i = result
                pbar.update(1)
                # print(i,i/n_randoms,n_randoms)
                for k,key in enumerate(all_keys):
                    p_arr[i,k,:] = p[key]
                    V_arr[i,k,:,:] = V[key]
            for i in range(n_randoms):
                randoms_datvec = {key:randoms_datvecs[key][i] for key in randoms_datvecs.keys()}
                # if i<5:
                    # print(randoms_datvec.keys(),np.all([np.all(np.isfinite(randoms_datvec[key])) for key in randoms_datvec.keys()]))
                jobs.append(p.apply_async(source_redshift_slope_tomo_wrapper,[config,randoms_datvec,covariances,all_zsource,i,logger,kwargs_dict],
                                        callback = fill_arrays))
            for job in jobs:
                job.wait()
            p.close()
            p.join()
        # for i,job in tqdm(enumerate(jobs),total=n_randoms):
        #     p_arr[i,:],V_arr[i,:,:] = job.get()
    print("Saving as ",savepath_slope_values+os.sep+f"redshift_slope_tomo_p_arr{fstr}.npy")
    np.save(savepath_slope_values+os.sep+f"redshift_slope_tomo_p_arr{fstr}.npy",p_arr)
    np.save(savepath_slope_values+os.sep+f"redshift_slope_tomo_V_arr{fstr}.npy",V_arr)
    np.save(savepath_slope_values+os.sep+f"redshift_slope_tomo_keys{fstr}.npy",all_keys,allow_pickle=True)


def generate_random_splits(config):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    data_path = clean_read(config,'general','data_path',split=False)
    chris_path = clean_read(config,'general','chris_path',split=False)
    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    galaxy_types = clean_read(config,'general','galaxy_types',split=True)
    statistic = clean_read(config,'general','statistic',split=False)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)

    savepath_addon = clean_read(config,'splits','savepath_addon',split=False)
    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    os.makedirs(savepath_slope_values,exist_ok=True)

    logger = get_logger(savepath_slope_values,script_name+'_tomo',__name__)

    n_processes = clean_read(config,script_name,'n_processes',split=False,convert_to_int=True)
    n_randoms = clean_read(config,script_name,'n_randoms',split=False,convert_to_int=True)
    verbose = clean_read(config,script_name,'verbose',split=False,convert_to_bool=True)
    use_theory_covariance = clean_read(config,script_name,'use_theory_covariance_splits',split=False,convert_to_bool=True)
    datavector_type = clean_read(config,script_name,'datavector_type_splits',split=False)

    print("Preparing datavectors and covariances")
    all_splits = clean_read(config,'splits','splits_to_consider',split=True)
    all_datavectors = {}
    all_covariances = {}
    for split_by in all_splits:
        if split_by.lower() == 'ntile':
            for galaxy_type in galaxy_types:
                n_splits = clean_read(config,'splits',f'n_ntile_{galaxy_type[:3]}',split=False,convert_to_int=True)
                n_splits_computed = clean_read(config,'splits',f'n_ntile_computed_{galaxy_type[:3]}',split=False,convert_to_int=True)
                for split in range(n_splits_computed):
                    datavectors,covariances = prepare_randoms_datavector(config,account_for_cross_covariance=False,
                                                                        use_theory_covariance=use_theory_covariance,
                                                                        datavector_type=datavector_type,
                                                                        logger=logger,split_by=split_by,split=split,n_splits=n_splits,
                                                                        galaxy_types=[galaxy_type],
                                                                        tomographic=False)
                    for key in datavectors.keys():
                        all_datavectors[key+f"_{split_by}_{split}_of_{n_splits}"] = datavectors[key]
                        all_covariances[key+f"_{split_by}_{split}_of_{n_splits}"] = covariances[key]
        else:
            n_splits = clean_read(config,'splits','n_splits',split=False,convert_to_int=True)
            for split in range(n_splits):
                datavectors,covariances = prepare_randoms_datavector(config,account_for_cross_covariance=False,
                                                                     use_theory_covariance=use_theory_covariance,
                                                                    datavector_type=datavector_type,
                                                                    logger=logger,split_by=split_by,split=split,n_splits=n_splits,
                                                                    tomographic=False)
                for key in datavectors.keys():
                    all_datavectors[key+f"_{split_by}_{split}_of_{n_splits}"] = datavectors[key]
                    all_covariances[key+f"_{split_by}_{split}_of_{n_splits}"] = covariances[key]

    print("Done preparing datavectors and covariances, drawing randoms.")
    randoms_datvecs = generate_randoms_datavectors(all_datavectors,all_covariances,n_randoms,n_processes,method="numpy",random_seed=0)
    print("Done.")
    print(randoms_datvecs.keys())
    # print(datavectors.keys())

    print("Testing run with seed 0")
    p,V,i = splits_wrapper(config,all_datavectors,all_covariances,0,logger)
    all_keys = list(p.keys())
    with Pool(n_processes) as p:
        jobs = []
        p_arr = np.zeros((n_randoms,len(all_keys),2))
        V_arr = np.zeros((n_randoms,len(all_keys),2,2))
        with tqdm(total=n_randoms) as pbar:
            def fill_arrays(result):
                p,V,i = result
                pbar.update(1)
                for k,key in enumerate(all_keys):
                    p_arr[i,k,:] = p[key]
                    V_arr[i,k,:,:] = V[key]
            for i in range(n_randoms):
                jobs.append(p.apply_async(splits_wrapper,[config,{key:randoms_datvecs[key][i] for key in randoms_datvecs.keys()},all_covariances,i,logger],
                                        callback = fill_arrays))
            p.close()
            p.join()
        # for i,job in tqdm(enumerate(jobs),total=n_randoms):
        #     p_arr[i,:],V_arr[i,:,:] = job.get()
    print("Saving as ",savepath_slope_values+os.sep+f"splits_p_arr.npy")
    np.save(savepath_slope_values+os.sep+f"splits_p_arr",p_arr)
    np.save(savepath_slope_values+os.sep+f"splits_V_arr",V_arr)
    np.save(savepath_slope_values+os.sep+f"splits_keys",all_keys,allow_pickle=True)




if __name__=="__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("config_plots.conf")
    
    # generate_random_source_redshift_slope_tomo(config,{"hscy3_deltaz_shifts":[0,0,0.115,0.192]},"_hscy3_deltaz_shifts")
    generate_random_source_redshift_slope_tomo(config)
    generate_random_splits(config)
    # generate_random_source_redshift_slope_tomo(config,{"allbins":True},"_allbins")
    # dvs,covs = prepare_randoms_datavector(config,use_theory_covariance=True,datavector_type="emulator")
    # allkeys = list(dvs.keys())
    # for key in allkeys:
    #     if not "all" in key:
    #         print("Removing "  + key)
    #         dvs.pop(key)
    #         covs.pop(key)

    # randoms = generate_randoms_datavectors(dvs,covs,100000,1,method="numpy")


    # for key in randoms.keys():
        # print(key,randoms[key].shape)