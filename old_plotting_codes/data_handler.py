import numpy as np
from astropy.table import Table
import os
from datetime import datetime
from scipy.stats import chi2
try:
    from plotting_utilities import clean_read
except ModuleNotFoundError:
    from .plotting_utilities import clean_read
import nautilus
from getdist import MCSamples
import os

from astropy.cosmology import Planck18,FlatLambdaCDM
cosmo = Planck18.clone(name="Planck18_h1",H0=100)
import astropy.units as u

from scipy.stats import chi2
from scipy.optimize import minimize,minimize_scalar
from scipy.interpolate import interp1d


import sys
sys.path.insert(0,"/global/homes/s/sven/code/dark_emulator_public/")
from dark_emulator import darkemu
emu = darkemu.base_class()
omb = 0.0224
omc = 0.120
omde = 0.6847
lnAs = 3.045
ns = 0.965
w = -1.


cparam = np.array([omb,omc,omde,lnAs,ns,w])
emu.set_cosmology(cparam)

omnu = 0.00064
omm = 1-omde
h = np.sqrt((omb+omc+omnu)/omm)
assert np.isclose(h,Planck18.h,rtol=1e-2)

def get_last_mtime(path):
    assert os.path.exists(path), f"File {path} does not exist"
    assert os.path.isfile(path), f"Path {path} is not a file"
    return datetime.fromtimestamp(os.path.getmtime(path))

def get_reference_datavector(rp,z,Mmin=5e+12):
    dsigma = emu.get_DeltaSigma_massthreshold(rp/h,Mmin,z)
    return dsigma

def is_table_masked(table):
    return any(getattr(col, 'mask', None) is not None for col in table.columns.values())

def get_lens_bins(galaxy_type):
    if galaxy_type[:3] == "BGS":
        return np.array([0.1,0.2,0.3,0.4])
    elif galaxy_type[:3] == "LRG":
        return np.array([0.4,0.6,0.8,1.1])
    else:
        raise ValueError(f"Invalid galaxy type {galaxy_type}, allowed: BGS, LRG")

def read_npz_file(file_like):
    with np.load(file_like,allow_pickle=True) as data:
        content = {file: data[file] for file in data.files}
    return content

def get_Mmin(config,galaxy_type,lens_bin):
    if not hasattr(get_Mmin, "file_content"):
        get_Mmin.file_content = read_npz_file(config.get('general','savepath')+os.sep+\
                                                    config.get('general','version')+os.sep+\
                                                    config.get('darkemu_Mmin','savepath_addon')+os.sep+\
                                                    'darkemu_Mmin_Mmin.npz')
    return get_Mmin.file_content[f"{galaxy_type}"][lens_bin]

def get_reference_datavector_of_galtype(config,rp,
                                        galaxy_type,
                                        lens_bin,
                                        datavector_type="default"):
    if datavector_type == "buzzard":
        chris_path = clean_read(config,'general','chris_path',split=False)
        return np.load(chris_path + "reference_datavectors/" + f"deltasigma_{galaxy_type}_l{lens_bin}_mean_datavector.npy")
    elif datavector_type == "darkemu":
        Mmin = get_Mmin(config,galaxy_type,lens_bin)
        z = get_zlens(galaxy_type,lens_bin,config)
        return get_reference_datavector(rp*h,z,Mmin)
    elif datavector_type == "alexie_hod":
        dat = np.loadtxt(config['general']['chris_path']+'reference_datavectors/BOSS/hod_model_saito2016_mdpl2.txt')
        z = get_zlens(galaxy_type,lens_bin,config)
        rp_interp = dat[:,0]*0.7/(1+z)
        ds_interp = dat[:,1]/0.7
        return interp1d(rp_interp,ds_interp,fill_value=np.nan,bounds_error=False)(rp)
    elif datavector_type == "tabcorr":
        chris_path = clean_read(config,'general','chris_path',split=False)
        return np.load(chris_path+f"lensing_measurements/hod_params/{galaxy_type}_{lens_bin}_ds.npy")
    elif datavector_type in ["default","abacus"]:
        chris_path = clean_read(config,'general','chris_path',split=False)
        return np.load(chris_path + "reference_datavectors/abacus/" + f"ds_{galaxy_type[:3]}_l{lens_bin}.npy")
    else:
        raise ValueError(f"Invalid datavector type {datavector_type}, allowed: default, buzzard, darkemu, alexie_hod")

def get_pvalue(data,cov):
    dof = len(data)
    chisq = np.einsum('i,ij,j',data,np.linalg.inv(cov),data)
    pval = 1-chi2.cdf(chisq,dof)
    return pval

def get_scales_mask(rp,analyzed_scales,min_rp,max_rp,rp_pivot):
    if analyzed_scales.lower() == "small scales":
        mask = (rp<rp_pivot) & (rp>min_rp)
    elif analyzed_scales.lower() == "large scales":
        mask = (rp>rp_pivot) & (rp<max_rp)
    elif analyzed_scales.lower() == "all scales":
        mask = (rp>min_rp) & (rp<max_rp)
    else:
        raise ValueError(f"Invalid analyzed scales {analyzed_scales.lower()}, allowed: small scales, large scales, all scales")
    return mask

def get_zlens(galaxy_type,lens_bin,config):
    lens_bins = clean_read(config,"general",galaxy_type+"_bins",True,convert_to_float=True)
    return (lens_bins[lens_bin]+lens_bins[lens_bin+1])/2

def get_rp_from_deg(min_deg,max_deg,galaxy_type,lens_bin,config):
    zlens = get_zlens(galaxy_type,lens_bin,config)
    dist = cosmo.comoving_transverse_distance(zlens).to(u.Mpc).value
    min_rp = dist*np.deg2rad(min_deg)
    max_rp = dist*np.deg2rad(max_deg)
    return min_rp,max_rp

def get_split_value(galaxy_type,source_survey,
                    data_path,statistic,
                    versions,lens_bin,split_by,split,n_splits=4,source_bin=None,boost=False,
                    fstr="split_value_{}_{}_zmin_{:.1f}_zmax_{:.1f}_blindA_boost_{}_split_{}_{}_of_{}.txt",
                    fstr_tomo="split_value_{}_{}_zmin_{:.1f}_zmax_{:.1f}_lenszbin_{}_blindA_boost_{}_split_{}_{}_of_{}.txt"):
    if(galaxy_type=='BGS_BRIGHT'):
        z_bins = [0.1,0.2,0.3,0.4]
    elif(galaxy_type=='LRG'):
        z_bins = [0.4,0.6,0.8,1.1]

    if source_bin is not None:
        dat = np.loadtxt(data_path+versions[galaxy_type]+os.sep+'splits/'+source_survey+os.sep+fstr_tomo.format(statistic,galaxy_type,
                                                                z_bins[lens_bin],z_bins[lens_bin+1],
                                                                source_bin,
                                                                boost,
                                                                split_by,
                                                                split,
                                                                n_splits))

    else:
        dat = np.loadtxt(data_path+versions[galaxy_type]+os.sep+'splits/'+source_survey+os.sep+fstr.format(statistic,galaxy_type,
                                                                z_bins[lens_bin],z_bins[lens_bin+1],
                                                                boost,
                                                                split_by,
                                                                split,
                                                                n_splits))

    return dat[0]



def get_scales_mask_from_degrees(rp,analyzed_scales,min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config):
    statistic = clean_read(config,"general","statistic",False)
    if(statistic=="deltasigma"):
        min_rp, max_rp = get_rp_from_deg(min_deg,max_deg,galaxy_type,lens_bin,config)
        # print(min_rp,max_rp)
        # print(rp)
    else:
        min_rp, max_rp = min_deg,max_deg
    if analyzed_scales.lower() == "small scales":
        mask = (rp<rp_pivot) & (rp>min_rp)
    elif analyzed_scales.lower() == "large scales":
        mask = (rp>rp_pivot) & (rp<max_rp)
    elif analyzed_scales.lower() == "all scales":
        mask = (rp>min_rp) & (rp<max_rp)
    else:
        raise ValueError(f"Invalid analyzed scales {analyzed_scales.lower()}, allowed: small scales, large scales, all scales")
    return mask

def get_deltasigma_amplitudes_list(datavectors,covariances,reference_datavector):
    keys = list(datavectors.keys())
    assert list(covariances.keys()) == keys, "Data and covariance must have the same keys!"
    assert list(reference_datavector.keys()) == keys, "Data and reference_datavector must have the same keys!"
    results = {}
    errors = {}
    weights = {}
    for key in keys:
        r,e,w = get_deltasigma_amplitudes(datavectors[key],covariances[key],reference_datavector[key])
        results[key] = r
        errors[key] = e
        weights[key] = w
    return results,errors,weights
        
def get_deltasigma_amplitudes(datavectors,covariances,reference_datavector,substract_mean=True):
    datavector_length = datavectors.shape[1]
    assert(datavector_length==covariances.shape[1]==covariances.shape[2]==len(reference_datavector))
    assert(datavectors.shape[0]==covariances.shape[0])
    weights = np.matmul(np.linalg.inv(np.sum(covariances,axis=0)),reference_datavector)
    errors = np.sqrt(np.einsum("i,kij,j->k",weights,covariances,weights))/\
        np.sum(weights*reference_datavector)
    amplitudes = np.sum(weights[None,:]*datavectors,axis=1)/\
                np.sum(weights*reference_datavector)
    if substract_mean:
        amplitudes -= np.mean(amplitudes)
    return amplitudes,errors,weights

# def get_minimum_deltasigma_amplitude(datavectors,covariances,reference_datavector,
#                                     start=0):
#     datavector_length = datavectors.shape[1]
#     all_rp = [np.arange(start,start+i+1) for i in range(datavector_length-start)]
#     bestrp = None
#     lowamp = np.inf
#     all_err = np.zeros((3,datavector_length-start))
#     for x,rp in enumerate(all_rp):
#         _,err,_ = get_deltasigma_amplitudes(datavectors[:,rp],
#                                             covariances[:,rp][:,:,rp],
#                                             reference_datavector[rp])
#         if(np.mean(err)<lowamp):
#             lowamp = np.mean(err)
#             bestrp = rp
#         all_err[:,x] = (err)
#     return rp,all_err

# scales_mask = get_scales_mask_from_degrees(rp,scales,min_deg,max_deg,rp_pivot,galaxy_type,lens_bin,config)

def full_covariance_bin_mask(galaxy_type,source_survey,lens_bin,source_bin,include_sdss=False):
    ntot_kids = get_ntot(galaxy_type,"kids")
    ntot_des = get_ntot(galaxy_type,"des")
    ntot_hsc = get_ntot(galaxy_type,"hscy3")
    ntot_sdss = get_ntot(galaxy_type,"sdss")
    if include_sdss:
        mask = np.zeros(ntot_kids+ntot_des+ntot_hsc+ntot_sdss,dtype=bool)
    else:
        mask = np.zeros(ntot_kids+ntot_des+ntot_hsc,dtype=bool)

    bins_mask = get_bins_mask(galaxy_type,source_survey,lens_bin,source_bin)
    if source_survey.lower()=="kids":
        mask[:ntot_kids] = bins_mask
    elif source_survey.lower()=="des":
        mask[ntot_kids:ntot_kids+ntot_des] = bins_mask
    elif source_survey.lower()[:3]=="hsc":
        mask[ntot_kids+ntot_des:ntot_kids+ntot_des+ntot_hsc] = bins_mask
    elif source_survey.lower()=="sdss":
        assert include_sdss
        mask[ntot_kids+ntot_des+ntot_hsc:] = bins_mask
    return mask


def load_covariance_chris(galaxy_type,source_survey,statistic,
                          fpath,
                          pure_noise=False,split_type=None,split=None,
                          logger=None,include_sdss=False):
    if split_type is not None:
        raise ValueError("Splits covariance not implemented")
        
    if galaxy_type in ["BGS","BGS_BRIGHT","bgs"]:
        fgal = "bgs"
    elif(galaxy_type in ["LRG","lrg"]):
        fgal = "lrg"
    else:
        raise ValueError("Invalid galaxy type")

    if(source_survey.lower()=="kids"):
        fsurv = "kids1000"
    elif(source_survey.lower()=="des"):
        fsurv = "desy3"
    elif(source_survey.lower()=="hscy1"):
        fsurv = "hscy1"
    elif(source_survey.lower()=="hscy3"):
        fsurv = "hscy3"
    elif(source_survey.lower()=="all_y1"):
        fsurv = "kids1000desy3hscy1_"
    elif(source_survey.lower()=="all_y3"):
        fsurv = "kids1000desy3hscy3_"
    else:
        fsurv = source_survey.lower()
    if(statistic=="deltasigma"):
        if pure_noise:
            fstat = "dx"
        else:
            fstat = "ds"
        fstr = f"model_inputs_desiy1/{fstat}covcorr_{fsurv}desiy1{fgal}_pzwei.dat"
    elif(statistic=="gammat"):
        if pure_noise:
            fstat = "gx"
        else:
            fstat = "gt"
        fstr = f"model_inputs_desiy1/{fstat}covcorr_{fsurv}desiy1{fgal}.dat"
    else:
        raise ValueError("Invalid statistic")
    if logger is not None:
        logger.info(f"Loading covariance from {fpath+fstr} from {get_last_mtime(fpath+fstr)}")
    fil = np.loadtxt(fpath+fstr,
                    skiprows = 1)
    ntot = get_ntot(galaxy_type,source_survey)
    fil = fil[:,-1].reshape(ntot,ntot)
    if include_sdss:
        assert source_survey.lower() == "all_y1" or source_survey.lower() == "all_y3"
        sdss_cov = load_covariance_chris(galaxy_type,"sdss",statistic,fpath,
                                         pure_noise=pure_noise,split_type=split_type,
                                         split=split,logger=logger)
        fil = np.pad(fil,((0,sdss_cov.shape[0]),(0,sdss_cov.shape[0])),mode="constant")
        fil[-sdss_cov.shape[0]:,-sdss_cov.shape[0]:] = sdss_cov
    return fil

def load_dv_johannes(galaxy_type,source_survey,fpath,statistic,logger=None,col = "value",dvtype="relative",
                    systype="intrinsic_alignment"):
    assert dvtype in ["relative","absolute"]
    if(statistic=="deltasigma"):
        fstat = "ds"
    elif(statistic=="gammat"):
        fstat = "gt"
    else:
        raise ValueError("Invalid statistic")

    if galaxy_type in ["BGS","BGS_BRIGHT","bgs"]:
        fgal = "bgs"
    elif(galaxy_type in ["LRG","lrg"]):
        fgal = "lrg"
    else:
        raise ValueError("Invalid galaxy type")

    fsurv = source_survey.lower()
    if "hsc" in fsurv:
        fsurv = "hsc"
 
    fdat = fpath+f"systematic_biases/{dvtype}/{systype}_{fstat}_{fsurv}.csv"
    if logger is not None:
        logger.info(f"Loading mock DV {fdat} from {get_last_mtime(fdat)}")
    fil = Table.read(fdat)
    if fgal=="bgs":
        dv = fil[col][:len(fil)//2]
    elif(fgal=="lrg"):
        dv = fil[col][len(fil)//2:]
    return dv


def load_mock_DV_chris(galaxy_type,source_survey,fpath,statistic,logger=None):
    if(statistic=="deltasigma"):
        fstat = "ds"
        append = "_pzwei"
    elif(statistic=="gammat"):
        fstat = "gt"
        append = ""
    else:
        raise ValueError("Invalid statistic")

    if galaxy_type in ["BGS","BGS_BRIGHT","bgs"]:
        fgal = "bgs"
    elif(galaxy_type in ["LRG","lrg"]):
        fgal = "lrg"
    else:
        raise ValueError("Invalid galaxy type")

    if(source_survey.lower()=="kids"):
        fsurv = "kids1000"
    elif(source_survey.lower()=="des"):
        fsurv = "desy3"
    else:
        fsurv = source_survey.lower()

    if logger is not None:
        logger.info(f"Loading mock DV from {fpath+f'/{fstat}modvec_{fsurv}desiy1{fgal}{append}.dat'} from {get_last_mtime(fpath+f'/{fstat}modvec_{fsurv}desiy1{fgal}{append}.dat')}")
    fil = np.loadtxt(fpath+f"model_inputs_desiy1/{fstat}modvec_{fsurv}desiy1{fgal}{append}.dat")
    dv = fil[:,1]
    return dv

def get_rp_chris(galaxy_type,source_survey,fpath,statistic,logger=None):
    if(statistic=="deltasigma"):
        fstat = "ds"
        append = ""
    elif(statistic=="gammat"):
        fstat = "gt"
        append = ""
    else:
        raise ValueError("Invalid statistic")

    if galaxy_type in ["BGS","BGS_BRIGHT","bgs"]:
        fgal = "bgs"
    elif(galaxy_type in ["LRG","lrg"]):
        fgal = "lrg"
    else:
        raise ValueError("Invalid galaxy type")

    if(source_survey.lower()=="kids"):
        fsurv = "kids1000"
    elif(source_survey.lower()=="des"):
        fsurv = "desy3"
    elif(source_survey.lower()=="hsc"):
        fsurv = "hscy1"
    else:
        fsurv = source_survey.lower()

    if logger is not None:
        logger.info(f"Loading rp from {fpath+f'model_inputs_desiy1/bin_{fstat}_{fsurv}desiy1{fgal}{append}.dat'} from {get_last_mtime(fpath+f'model_inputs_desiy1/bin_{fstat}_{fsurv}desiy1{fgal}{append}.dat')}")
    fil = np.loadtxt(fpath+f"model_inputs_desiy1/bin_{fstat}_{fsurv}desiy1{fgal}{append}.dat")
    rp = fil[:,-1]
    return rp

def load_mstar_complete_clustering_measurements(galaxy_type,lens_bin,
                                                weight_type = "pip_angular_bitwise",
                                                lens_bin_redshifts={"BGS_BRIGHT":[0.1,0.2,0.3,0.4],
                                                                    "LRG":[0.4,0.6,0.8,1.1]},
                                                                    njacks = {"BGS_BRIGHT":64,"LRG":64}):
    from pycorr import TwoPointCorrelationFunction, project_to_wp
    rmin = 0.08
    rmax = 80
    binning = 2

    lens_bins = lens_bin_redshifts[galaxy_type]
    result = TwoPointCorrelationFunction.load(f'/pscratch/sd/m/mcdemart/rppi/v1.5pip/{int(10*lens_bins[lens_bin]):d}_{int(10*lens_bins[lens_bin+1]):d}/rppi/allcounts_{galaxy_type}-21.5_GCcomb_{lens_bin_redshifts[galaxy_type][lens_bin]:.1f}_{lens_bin_redshifts[galaxy_type][lens_bin+1]:.1f}_{weight_type}_log_njack{njacks[galaxy_type]}_nran4_split20.npy')
    result = result[::binning,::]
    result.select((rmin, rmax))
    s, wp, cov = project_to_wp(result, pimax=100)
    return s,wp,cov



def load_clustering_measurements(galaxy_type,lens_bin,NTILE=None,
                                 weight_type = "default_angular_bitwise",
                                 lens_bin_redshifts={"BGS_BRIGHT":[0.1,0.2,0.3,0.4],
                                            "LRG":[0.4,0.6,0.8,1.1]},
                                njacks = {"BGS_BRIGHT":64,"LRG":64},
                                rmin=0.08,rmax=80,binning=1,pimax=100,
                                measurements = "sven"):
    from pycorr import TwoPointCorrelationFunction, project_to_wp

    if measurements == "sven":
        basepath = "/pscratch/sd/s/sven/desi/wp_measured/"
    # basepath = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/wp_measurements/results/"
        if NTILE is None:
            ntile_path = ""
        else:
            ntile_path = f"split_{NTILE+1}/"
        filepath = basepath + f"bin_{lens_bin+1}/" + ntile_path + f"rppi/allcounts_{galaxy_type}_GCcomb_{lens_bin_redshifts[galaxy_type][lens_bin]:.1f}_{lens_bin_redshifts[galaxy_type][lens_bin+1]:.1f}_{weight_type}_log_njack{njacks[galaxy_type]}_nran4_split20.npy"
        try:
            result = TwoPointCorrelationFunction.load(filepath)
        except FileNotFoundError as e:
            try:
                result = TwoPointCorrelationFunction.load(filepath.replace(f"log_njack{njacks[galaxy_type]}","log_njack32"))
                print(f"64 Jackknife regions not available, loaded {filepath.replace(f'log_njack{njacks[galaxy_type]}','log_njack32')}")
            except FileNotFoundError:
                print(e)
                return np.zeros(15)*np.nan,np.zeros(15)*np.nan,np.zeros((15,15))*np.nan
    elif measurements in["zechang_old","zechang_new"]:
        if NTILE is not None:
            raise ValueError("NTILE not supported for zechang measurements as of now")
        basepath = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/wp_measurements_pimax40/results/"
        if measurements == "zechang_new":
            basepath += f"{galaxy_type[:3]}_out/".lower()
        basepath += "rppi/"
        filepath = basepath + f"allcounts_{galaxy_type}_GCcomb_{lens_bin_redshifts[galaxy_type][lens_bin]:.1f}_{lens_bin_redshifts[galaxy_type][lens_bin+1]:.1f}_{weight_type}_log_njack{njacks[galaxy_type]}_nran4_split20.npy"
        result = TwoPointCorrelationFunction.load(filepath)

    result = result[::binning,::]
    result.select((rmin, rmax))
    s, wp, cov = project_to_wp(result, pimax=pimax)
    return s,wp,cov


def get_allowed_bins(galaxy_type,source_survey,lens_bin,conservative_cut=True):
    if conservative_cut:
        allowed_bins =  {'kids_BGS_BRIGHT_l0': [3, 4],
                        'kids_BGS_BRIGHT_l1': [3, 4],
                        'kids_BGS_BRIGHT_l2': [3, 4],
                        'kids_LRG_l0': [3, 4],
                        'kids_LRG_l1': [],
                        'kids_LRG_l2': [],
                        'des_BGS_BRIGHT_l0': [2, 3],
                        'des_BGS_BRIGHT_l1': [2, 3],
                        'des_BGS_BRIGHT_l2': [2, 3],
                        'des_LRG_l0': [3],
                        'des_LRG_l1': [],
                        'des_LRG_l2': [],
                        'hscy1_BGS_BRIGHT_l0': [1, 2, 3],
                        'hscy1_BGS_BRIGHT_l1': [1, 2, 3],
                        'hscy1_BGS_BRIGHT_l2': [1, 2, 3],
                        'hscy1_LRG_l0': [1, 2, 3],
                        'hscy1_LRG_l1': [2, 3],
                        'hscy1_LRG_l2': [],
                        'hscy3_BGS_BRIGHT_l0': [1, 2, 3],
                        'hscy3_BGS_BRIGHT_l1': [1, 2, 3],
                        'hscy3_BGS_BRIGHT_l2': [1, 2, 3],
                        'hscy3_LRG_l0': [2, 3],
                        'hscy3_LRG_l1': [2, 3],
                        'hscy3_LRG_l2': [],
                        'sdss_BGS_BRIGHT_l0': [0],
                        'sdss_BGS_BRIGHT_l1': [0],
                        'sdss_BGS_BRIGHT_l2': [0],
                        'sdss_LRG_l0': [],
                        'sdss_LRG_l1': [],
                        'sdss_LRG_l2': []}

    else:
        allowed_bins =  {'kids_BGS_BRIGHT_l0': [1, 2, 3, 4],
                        'kids_BGS_BRIGHT_l1': [2, 3, 4],
                        'kids_BGS_BRIGHT_l2': [3, 4],
                        'kids_LRG_l0': [3, 4],
                        'kids_LRG_l1': [],
                        'kids_LRG_l2': [],
                        'des_BGS_BRIGHT_l0': [0, 1, 2, 3],
                        'des_BGS_BRIGHT_l1': [1, 2, 3],
                        'des_BGS_BRIGHT_l2': [1, 2, 3],
                        'des_LRG_l0': [2,3],
                        'des_LRG_l1': [],
                        'des_LRG_l2': [],
                        'hscy1_BGS_BRIGHT_l0': [0, 1, 2, 3],
                        'hscy1_BGS_BRIGHT_l1': [0, 1, 2, 3],
                        'hscy1_BGS_BRIGHT_l2': [1, 2, 3],
                        'hscy1_LRG_l0': [1, 2, 3],
                        'hscy1_LRG_l1': [2, 3],
                        'hscy1_LRG_l2': [3],
                        'hscy3_BGS_BRIGHT_l0': [0, 1, 2, 3],
                        'hscy3_BGS_BRIGHT_l1': [0, 1, 2, 3],
                        'hscy3_BGS_BRIGHT_l2': [1, 2, 3],
                        'hscy3_LRG_l0': [1, 2, 3],
                        'hscy3_LRG_l1': [2, 3],
                        'hscy3_LRG_l2': [3],
                        'sdss_BGS_BRIGHT_l0': [0],
                        'sdss_BGS_BRIGHT_l1': [0],
                        'sdss_BGS_BRIGHT_l2': [0],
                        'sdss_LRG_l0': [],
                        'sdss_LRG_l1': [],
                        'sdss_LRG_l2': []}
    if source_survey is None and galaxy_type is None and lens_bin is None:
        return allowed_bins
    return allowed_bins[f'{source_survey.lower()}_{galaxy_type}_l{lens_bin}']

# def get_allowed_bins(galaxy_type,source_survey,lens_bin,conservative_cut=True):
#     if conservative_cut:
#         allowed_bins =  {'kids_BGS_BRIGHT_l0': [3, 4],
#                         'kids_BGS_BRIGHT_l1': [3, 4],
#                         'kids_BGS_BRIGHT_l2': [3, 4],
#                         'kids_LRG_l0': [3, 4],
#                         'kids_LRG_l1': [],
#                         'kids_LRG_l2': [],
#                         'des_BGS_BRIGHT_l0': [2, 3],
#                         'des_BGS_BRIGHT_l1': [2, 3],
#                         'des_BGS_BRIGHT_l2': [2, 3],
#                         'des_LRG_l0': [3],
#                         'des_LRG_l1': [],
#                         'des_LRG_l2': [],
#                         'hscy1_BGS_BRIGHT_l0': [1, 2, 3],
#                         'hscy1_BGS_BRIGHT_l1': [1, 2, 3],
#                         'hscy1_BGS_BRIGHT_l2': [1, 2, 3],
#                         'hscy1_LRG_l0': [1, 2, 3],
#                         'hscy1_LRG_l1': [2, 3],
#                         'hscy1_LRG_l2': [],
#                         'hscy3_BGS_BRIGHT_l0': [1, 2, 3],
#                         'hscy3_BGS_BRIGHT_l1': [1, 2, 3],
#                         'hscy3_BGS_BRIGHT_l2': [1, 2, 3],
#                         'hscy3_LRG_l0': [1, 2, 3],
#                         'hscy3_LRG_l1': [2, 3],
#                         'hscy3_LRG_l2': [],
#                         'sdss_BGS_BRIGHT_l0': [0],
#                         'sdss_BGS_BRIGHT_l1': [0],
#                         'sdss_BGS_BRIGHT_l2': [0],
#                         'sdss_LRG_l0': [],
#                         'sdss_LRG_l1': [],
#                         'sdss_LRG_l2': []}

#     else:
#         allowed_bins =  {'kids_BGS_BRIGHT_l0': [1, 2, 3, 4],
#                         'kids_BGS_BRIGHT_l1': [2, 3, 4],
#                         'kids_BGS_BRIGHT_l2': [3, 4],
#                         'kids_LRG_l0': [3, 4],
#                         'kids_LRG_l1': [],
#                         'kids_LRG_l2': [],
#                         'des_BGS_BRIGHT_l0': [1, 2, 3],
#                         'des_BGS_BRIGHT_l1': [1, 2, 3],
#                         'des_BGS_BRIGHT_l2': [2, 3],
#                         'des_LRG_l0': [3],
#                         'des_LRG_l1': [],
#                         'des_LRG_l2': [],
#                         'hscy1_BGS_BRIGHT_l0': [0, 1, 2, 3],
#                         'hscy1_BGS_BRIGHT_l1': [0, 1, 2, 3],
#                         'hscy1_BGS_BRIGHT_l2': [1, 2, 3],
#                         'hscy1_LRG_l0': [1, 2, 3],
#                         'hscy1_LRG_l1': [2, 3],
#                         'hscy1_LRG_l2': [3],
#                         'hscy3_BGS_BRIGHT_l0': [0, 1, 2, 3],
#                         'hscy3_BGS_BRIGHT_l1': [0, 1, 2, 3],
#                         'hscy3_BGS_BRIGHT_l2': [1, 2, 3],
#                         'hscy3_LRG_l0': [1, 2, 3],
#                         'hscy3_LRG_l1': [2, 3],
#                         'hscy3_LRG_l2': [3],
#                         'sdss_BGS_BRIGHT_l0': [0],
#                         'sdss_BGS_BRIGHT_l1': [0],
#                         'sdss_BGS_BRIGHT_l2': [0],
#                         'sdss_LRG_l0': [],
#                         'sdss_LRG_l1': [],
#                         'sdss_LRG_l2': []}
#     if source_survey is None and galaxy_type is None and lens_bin is None:
#         return allowed_bins
#     return allowed_bins[f'{source_survey.lower()}_{galaxy_type}_l{lens_bin}']

def get_number_of_source_bins(source_survey):
    n_source_bins = {'kids': 5,
                    'des': 4,
                    'hsc': 4,
                    'hscy1': 4,
                    'hscy3': 4,
                    'sdss': 1}
    return n_source_bins[source_survey.lower()]

def get_number_of_lens_bins(galaxy_type):
    if galaxy_type[:3] == "BGS":
        return 3
    elif galaxy_type[:3] == "LRG":
        return 3
    else:
        raise ValueError(f"Invalid galaxy type {galaxy_type}, allowed: BGS, LRG")


def get_number_of_radial_bins(galaxy_type,source_survey,lens_bin=None):
    return 15

def get_ntot(galaxy_type,source_survey):
    if source_survey == "all_y1":
        return get_ntot(galaxy_type,"kids")+get_ntot(galaxy_type,"des")+get_ntot(galaxy_type,"hscy1")
    elif source_survey == "all_y3":
        return get_ntot(galaxy_type,"kids")+get_ntot(galaxy_type,"des")+get_ntot(galaxy_type,"hscy3")
    else:
        return get_number_of_lens_bins(galaxy_type)*get_number_of_source_bins(source_survey)*get_number_of_radial_bins(galaxy_type,source_survey)

def get_bins_mask(galaxy_type,source_survey,lens_bin,source_bins):
    if not hasattr(source_bins,"__len__"):
        source_bins = [source_bins]
    ntot = get_ntot(galaxy_type,source_survey)
    mask_source = np.zeros(ntot,dtype=bool)
    mask_lensbin,mask_sourcebin = get_masks_lensbin_and_sourcebin(get_number_of_lens_bins(galaxy_type),
                                                                    get_number_of_source_bins(source_survey),
                                                                    get_number_of_radial_bins(galaxy_type,source_survey,lens_bin))
    for sbin in source_bins:
        mask_source |= (mask_sourcebin == sbin)
    mask = (mask_source & (mask_lensbin == lens_bin))
    return mask

def get_masks_lensbin_and_sourcebin(n_lens_bins,n_source_bins,n_radial_bins=15):
    ntot = n_lens_bins*n_source_bins*n_radial_bins
    mask_lensbin = np.zeros(ntot,dtype=int)
    mask_sourcebin = np.zeros(ntot,dtype=int)
    counter = 0
    for lensbin in range(n_lens_bins):
        for sourcebin in range(n_source_bins):
            mask_lensbin[counter*n_radial_bins:(counter+1)*n_radial_bins] = lensbin
            mask_sourcebin[counter*n_radial_bins:(counter+1)*n_radial_bins] = sourcebin
            counter += 1
    assert(counter*n_radial_bins == ntot)
    return mask_lensbin,mask_sourcebin


def load_data_table_notomo(galaxy_type,source_survey,fpath,statistic,lens_bin,
                        versions = {"BGS_BRIGHT":"v0.5.1",
                                    "LRG" : "v0.4.5"},
                        boost=False,
                        fstr = "{}_{}_zmin_{:.1f}_zmax_{:.1f}_blindA_boost_{}.{}",
                        fstr_split = "{}_{}_zmin_{:.1f}_zmax_{:.1f}_blindA_boost_{}_split_{}_{}_of_{}.{}",
                        bmodes = False, split_by=None, split=None, n_splits=4,logger=None,
                        correct_for_magnification_bias=False):
    fpath = fpath + versions[galaxy_type] + os.sep
    if(bmodes):
        bmodestr = "bmodes_"
    else:
        bmodestr = ""

    if(galaxy_type=='BGS_BRIGHT'):
        z_bins = [0.1,0.2,0.3,0.4]
    elif(galaxy_type=='LRG'):
        z_bins = [0.4,0.6,0.8,1.1]

    if split_by is not None:
        if logger is not None:
            logger.info(f"Loading {fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,z_bins[lens_bin],z_bins[lens_bin+1],boost,split_by,split,n_splits,'fits')} from {get_last_mtime(fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,z_bins[lens_bin],z_bins[lens_bin+1],boost,split_by,split,n_splits,'fits'))}")
        tab = Table.read(fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,
                                                            z_bins[lens_bin],z_bins[lens_bin+1],
                                                            boost,split_by,split,n_splits,"fits"))

    else:
        if logger is not None:
            logger.info(f"Loading {fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[lens_bin],z_bins[lens_bin+1],boost,'fits')} from {get_last_mtime(fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[lens_bin],z_bins[lens_bin+1],boost,'fits'))}")
        tab = Table.read(fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,
                                                            z_bins[lens_bin],z_bins[lens_bin+1],
                                                            boost,"fits"))
    if correct_for_magnification_bias:
        if statistic=="deltasigma":
            tab["ds"] = tab["ds"] - tab["magnification_bias"]
        elif statistic=="gammat":
            tab["et"] = tab["et"] - tab["magnification_bias"]
    return tab

def load_data_and_covariance_notomo(galaxy_type,source_survey,fpath,statistic,
                        versions = {"BGS_BRIGHT":"v0.5.1",
                                    "LRG" : "v0.4.5"},
                        boost=False,
                        fstr = "{}_{}_zmin_{:.1f}_zmax_{:.1f}_blindA_boost_{}.{}",
                        fstr_split = "{}_{}_zmin_{:.1f}_zmax_{:.1f}_blindA_boost_{}_split_{}_{}_of_{}.{}",
                        bmodes = False, split_by=None, split=None, n_splits=4,logger=None,
                        correct_for_magnification_bias=True):
    fpath = fpath + versions[galaxy_type] + os.sep
    if(bmodes):
        bmodestr = "bmodes_"
    else:
        bmodestr = ""

    if(statistic=='deltasigma'):
        key_s = 'ds'
        key_n = 'ds_err'
        key_r = 'rp'
    elif(statistic=='gammat'):
        key_s = 'et'
        key_n = 'et_err'
        key_r = 'rp'
    else:
        raise ValueError('invalid statistic : {}'.format(statistic))
    if(galaxy_type=='BGS_BRIGHT'):
        z_bins = [0.1,0.2,0.3,0.4]
    elif(galaxy_type=='LRG'):
        z_bins = [0.4,0.6,0.8,1.1]
    n_lens_bins = get_number_of_lens_bins(galaxy_type)
    n_radial_bins = get_number_of_radial_bins(galaxy_type,source_survey)

    results = np.zeros((n_lens_bins,n_radial_bins))
    noise = np.zeros((n_lens_bins,n_radial_bins))
    rp = np.zeros((n_lens_bins,n_radial_bins))
    covs = np.zeros((n_lens_bins,n_radial_bins,n_radial_bins))
    zlens = np.zeros((n_lens_bins,n_radial_bins))
    zsource = np.zeros((n_lens_bins,n_radial_bins))
    npairs = np.zeros((n_lens_bins,n_radial_bins))
    for i in range(n_lens_bins):
        try:
            if split_by is not None:
                if logger is not None:
                    logger.info(f"Loading {fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],boost,split_by,split,n_splits,'fits')} from {get_last_mtime(fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],boost,split_by,split,n_splits,'fits'))}")
                load_path_print = fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,
                                                                    z_bins[i],z_bins[i+1],
                                                                    boost,split_by,split,n_splits,"fits")
                tab = Table.read(fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,
                                                                    z_bins[i],z_bins[i+1],
                                                                    boost,split_by,split,n_splits,"fits"))

            else:
                if logger is not None:
                    logger.info(f"Loading {fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],boost,'fits')} from {get_last_mtime(fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],boost,'fits'))}")
                load_path_print = fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,
                                                                    z_bins[i],z_bins[i+1],
                                                                    boost,"fits")
                tab = Table.read(fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,
                                                                    z_bins[i],z_bins[i+1],
                                                                    boost,"fits"))
            if correct_for_magnification_bias:
                try:
                    magbias = tab["magnification_bias"]
                except KeyError:
                    raise ValueError(f"Magnification bias not found in {load_path_print}. Available keys: {tab.keys()}")
            else:
                magbias = np.zeros(len(tab[key_s]))
            results[i] = (tab[key_s]-magbias)
            noise[i] = tab[key_n]
            rp[i] = (tab[key_r])
            zlens[i] = tab["z_l"]
            zsource[i] = tab["z_s"]
            npairs[i] = tab["n_pairs"]

        except Exception as e:
            results[i] = np.zeros(n_radial_bins)+np.nan
            noise[i] = np.zeros(n_radial_bins)+np.nan
            rp[i] = np.zeros(n_radial_bins)+np.nan
            zlens[i] = np.zeros(n_radial_bins)+np.nan
            zsource[i] = np.zeros(n_radial_bins)+np.nan
            npairs[i] = np.zeros(n_radial_bins)+np.nan


        try:
            if split_by is not None:
                covs[i] = (np.loadtxt(fpath+'splits/'+source_survey+os.sep+bmodestr+"covariance_"+\
                                    fstr_split.format(statistic,galaxy_type,
                                                z_bins[i],z_bins[i+1],
                                                boost,split_by,split,n_splits,"dat")))
            else:
                covs[i] = (np.loadtxt(fpath+source_survey+os.sep+bmodestr+"covariance_"+\
                                    fstr.format(statistic,galaxy_type,
                                                z_bins[i],z_bins[i+1],
                                                boost,"dat")))

        except Exception as e:
            covs[i] = np.zeros((n_radial_bins,n_radial_bins))+np.nan
    return (rp),(results),noise,(covs),zlens,zsource,npairs

def load_data_and_covariance_tomo(galaxy_type,source_survey,fpath,statistic,
                        versions = {"BGS_BRIGHT":"v0.5.1",
                                    "LRG" : "v0.4.5"},
                        boost=False,
                       fstr = "{}_{}_zmin_{:.1f}_zmax_{:.1f}_lenszbin_{}_blindA_boost_{}.{}",
                        fstr_split = "{}_{}_zmin_{:.1f}_zmax_{:.1f}_lenszbin_{}_blindA_boost_{}_split_{}_{}_of_{}.{}",
                        bmodes = False, split_by=None, split=None, n_splits=4,
                        logger=None, return_additional_quantity = None,
                        correct_for_magnification_bias=True,
                        only_allowed_bins = True,
                        skip_on_error=False):
    fpath = fpath + versions[galaxy_type] + os.sep
    if(statistic=='deltasigma'):
        key_s = 'ds'
        key_n = 'ds_err'
        key_r = 'rp'
    elif(statistic=='gammat'):
        key_s = 'et'
        key_n = 'et_err'
        key_r = 'rp'
    else:
        raise ValueError('invalid statistic : {}'.format(statistic))
    if(galaxy_type=='BGS_BRIGHT'):
        z_bins = [0.1,0.2,0.3,0.4]
    elif(galaxy_type=='LRG'):
        z_bins = [0.4,0.6,0.8,1.1]
    if(bmodes):
        bmodestr = "bmodes_"
    else:
        bmodestr = ""
    if(split_by is not None):
        fpath = fpath + 'splits/'
    ntot = get_ntot(galaxy_type,source_survey)
    n_lens_bins = get_number_of_lens_bins(galaxy_type)
    n_source_bins = get_number_of_source_bins(source_survey)
    n_radial_bins = get_number_of_radial_bins(galaxy_type,source_survey)
    results = np.zeros(ntot)+np.nan
    noise = np.zeros(ntot)+np.nan
    rp = np.zeros(ntot)+np.nan
    covs = np.zeros((ntot,ntot))+np.nan
    zlens = np.zeros(ntot)+np.nan
    zsource = np.zeros(ntot)+np.nan
    npairs = np.zeros(ntot)+np.nan
    if return_additional_quantity is not None:
        additional_quantity = np.zeros(ntot)+np.nan
    counter = 0
    for i in range(n_lens_bins):
        allowed_bins = get_allowed_bins(galaxy_type,source_survey,i)
        for j in range(n_source_bins):
            if only_allowed_bins and j not in allowed_bins:
                results[counter*n_radial_bins:(counter+1)*n_radial_bins] = np.zeros(n_radial_bins)+np.nan
                noise[counter*n_radial_bins:(counter+1)*n_radial_bins] = np.zeros(n_radial_bins)+np.nan
                rp[counter*n_radial_bins:(counter+1)*n_radial_bins] = np.zeros(n_radial_bins)+np.nan
                zlens[counter*n_radial_bins:(counter+1)*n_radial_bins] = np.zeros(n_radial_bins)+np.nan
                zsource[counter*n_radial_bins:(counter+1)*n_radial_bins] = np.zeros(n_radial_bins)+np.nan
                npairs[counter*n_radial_bins:(counter+1)*n_radial_bins] = np.zeros(n_radial_bins)+np.nan
                if return_additional_quantity is not None:
                    additional_quantity[counter*n_radial_bins:(counter+1)*n_radial_bins] = np.zeros(n_radial_bins)+np.nan
                counter +=1
                continue


            if(split_by is not None):
                if logger is not None:
                    logger.info(f"Loading {fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],j,boost,split_by,split,n_splits,'fits')} from {get_last_mtime(fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],j,boost,split_by,split,n_splits,'fits'))}")
                load_path_print = fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,
                                                                        z_bins[i],z_bins[i+1],
                                                                        j,
                                                                        boost,
                                                                        split_by,
                                                                        split,
                                                                        n_splits,
                                                                        "fits")
                tab = Table.read(fpath+'splits/'+source_survey+os.sep+bmodestr+fstr_split.format(statistic,galaxy_type,
                                                                        z_bins[i],z_bins[i+1],
                                                                        j,
                                                                        boost,
                                                                        split_by,
                                                                        split,
                                                                        n_splits,
                                                                        "fits"))
                covs[counter*n_radial_bins:(counter+1)*n_radial_bins,counter*n_radial_bins:(counter+1)*n_radial_bins] = (np.loadtxt(fpath+'splits/'+source_survey+os.sep+bmodestr+"covariance_"+\
                    fstr_split.format(statistic,galaxy_type,
                                z_bins[i],z_bins[i+1],
                                j,
                                boost,
                                split_by,split,n_splits,"dat")))

            else:
                if skip_on_error:
                    try:
                        if logger is not None:
                            logger.info(f"Loading {fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],j,boost,'fits')} from {get_last_mtime(fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],j,boost,'fits'))}")
                            # if source_survey=="KiDS":
                                # print(f"Loading {fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],j,boost,'fits')}")
                        load_path_print = fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,
                                                        z_bins[i],z_bins[i+1],
                                                        j,
                                                        boost,"fits")
                        tab = Table.read(fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,
                                                        z_bins[i],z_bins[i+1],
                                                        j,
                                                        boost,"fits"))
                    except Exception as e:
                        if logger is not None:
                            load_path_print = fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,
                                                        z_bins[i],z_bins[i+1],
                                                        j,
                                                        boost,"fits")
                            logger.warning(f"Error loading {load_path_print}: {e}")
                        tab = Table()
                        tab[key_s] = np.zeros(n_radial_bins)+np.nan
                        tab[key_n] = np.zeros(n_radial_bins)+np.nan
                        tab[key_r] = np.zeros(n_radial_bins)+np.nan
                        tab["z_l"] = np.zeros(n_radial_bins)+np.nan
                        tab["z_s"] = np.zeros(n_radial_bins)+np.nan
                        tab["n_pairs"] = np.zeros(n_radial_bins)+np.nan
                        
                else:
                    if logger is not None:
                        logger.info(f"Loading {fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],j,boost,'fits')} from {get_last_mtime(fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],j,boost,'fits'))}")
                        # if source_survey=="KiDS":
                            # print(f"Loading {fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,z_bins[i],z_bins[i+1],j,boost,'fits')}")
                    load_path_print = fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,
                                                    z_bins[i],z_bins[i+1],
                                                    j,
                                                    boost,"fits")
                    tab = Table.read(fpath+source_survey+os.sep+bmodestr+fstr.format(statistic,galaxy_type,
                                                    z_bins[i],z_bins[i+1],
                                                    j,
                                                    boost,"fits"))

                try:
                    covs[counter*n_radial_bins:(counter+1)*n_radial_bins,counter*n_radial_bins:(counter+1)*n_radial_bins] = (np.loadtxt(fpath+source_survey+os.sep+bmodestr+"covariance_"+\
                    fstr.format(statistic,galaxy_type,
                                z_bins[i],z_bins[i+1],
                                j,
                                boost,"dat")))
                except FileNotFoundError as e:
                    # if logger is not None:
                    #     logger.warning(f"File not found: {load_path_print}. Setting covariance to zero")
                    # else:
                    #     print(f"File not found: {load_path_print}. Setting covariance to zero")
                    covs[counter*n_radial_bins:(counter+1)*n_radial_bins,counter*n_radial_bins:(counter+1)*n_radial_bins] = np.zeros((n_radial_bins,n_radial_bins))

            if correct_for_magnification_bias:
                try:
                    magbias = tab["magnification_bias"]
                except KeyError:
                    raise KeyError(f"Magnification bias not found in {load_path_print}. Available keys: {tab.keys()}")
            else:
                magbias = np.zeros(len(tab[key_s]))

            results[counter*n_radial_bins:(counter+1)*n_radial_bins] = (tab[key_s]-magbias)
            noise[counter*n_radial_bins:(counter+1)*n_radial_bins] = tab[key_n]
            rp[counter*n_radial_bins:(counter+1)*n_radial_bins] = (tab[key_r])
            zlens[counter*n_radial_bins:(counter+1)*n_radial_bins] = tab["z_l"]
            zsource[counter*n_radial_bins:(counter+1)*n_radial_bins] = tab["z_s"]
            npairs[counter*n_radial_bins:(counter+1)*n_radial_bins] = tab["n_pairs"]
            if return_additional_quantity is not None:
                additional_quantity[counter*n_radial_bins:(counter+1)*n_radial_bins] = tab[return_additional_quantity]
            counter += 1
    assert(counter*n_radial_bins==ntot)
    if return_additional_quantity is not None:
        return (rp),(results),noise,(covs),zlens,zsource,npairs,additional_quantity
    else:
        return (rp),(results),noise,(covs),zlens,zsource,npairs


def fit_slope_to_deltasigma_amplitudes(xvals,data,error,return_err = False,
                                    logger=None):
    keys = list(data.keys())
    assert list(error.keys()) == keys, "Data and covariance must have the same keys!"
    assert list(xvals.keys()) == keys, "Data and xvals must have the same keys!"
    results = {}
    if(return_err):
        errors = {}
    for key in keys:
        if logger is not None:
            logger.info(f"Fitting {key}")
        p,V = np.polyfit(xvals[key],data[key],1,cov=True,w=1./error[key])
        results[key] = p
        if(return_err):
            errors[key] = V
    if(return_err):
        return results,errors
    else:
        return results

def combine_datavectors(datavector,cov,
                    optimal_matrix=False,
                    n_radial_bins=15):
    assert np.all(np.isfinite(datavector)), "Datavector must be finite and not contan nans"
    assert np.all(np.isfinite(cov)), "Covariance must be finite and not contan nans"
    n_tomo_bins = len(datavector)//n_radial_bins
    Dmat = np.zeros((len(datavector),n_radial_bins))
    for i in range(n_radial_bins):
        Dmat[i::n_radial_bins,i]=1

    covinv = np.linalg.inv(cov)
    if(optimal_matrix):
        wei = np.matmul(covinv,Dmat)
    else:
        wei = np.matmul(np.diag(1/np.diag(cov)),Dmat)
    wei /= np.matmul(np.ones(covinv.shape),wei)

    covtot = np.einsum("ik,ij,jl->kl",wei,cov,wei)
    datatot = np.matmul(wei.T,datavector)
    return datatot,covtot

def generate_random_datavector(data,covariance,size=1,seed=None):
    if(seed is not None):
        np.random.seed(seed)
    return np.random.multivariate_normal(data,covariance,size=size)[0]

def load_randoms_values(config, randoms_type='source_redshift_slope'):
    version = clean_read(config,'general','version',split=False)
    savepath_slope_values = clean_read(config,'general','savepath_slope_values',split=False)
    savepath_addon = clean_read(config,randoms_type,'savepath_addon',split=False)
    savepath_slope_values = savepath_slope_values + os.sep + version + os.sep + savepath_addon + os.sep
    p_arr = np.load(savepath_slope_values+os.sep+"redshift_slope_tomo_p_arr.npy")
    V_arr = np.load(savepath_slope_values+os.sep+"redshift_slope_tomo_V_arr.npy")
    all_keys = np.load(savepath_slope_values+os.sep+"redshift_slope_tomo_keys.npy",allow_pickle=True)
    return p_arr,V_arr,all_keys

def generate_bmode_tomo_latex_table_from_dict(data_dict, config, caption="Example caption", precision=3):
    surveys = clean_read(config,'general','lensing_surveys',split=True)
    types = clean_read(config,'general','galaxy_types',split=True)
    scales = clean_read(config,'general','analyzed_scales',split=True)

    bins = ["Bin 1", "Bin 2", "Bin 3"]
    survey_bins = {"KiDS": 5, "DES": 4, "HSCY1": 4, "HSCY3": 4, "SDSS": 1}
    
    #     \\begin{table}
    # \\centering
    # \\caption{""" + caption + """}

    header = """
    \\begin{tabular}{|c|c|ccc|ccc|}
        \\hline
"""
    
    table_content = ""
    
    for scale in scales:
        table_content += f"        & & \\multicolumn{{6}}{{c|}}{{{scale}}} \\\\\n"
        table_content += "        \\cline{3-8}\n"
        table_content += "        & & \\multicolumn{3}{c|}{BGS} & \\multicolumn{3}{c|}{LRG} \\\\\n"
        table_content += "        \\cline{3-8}\n"
        table_content += "        & & Bin 1 & Bin 2 & Bin 3 & Bin 1 & Bin 2 & Bin 3 \\\\\n"
        table_content += "        \\hline\n"
        
        for survey in surveys:
            for survey_bin_num in range(1, survey_bins[survey] + 1):
                survey_bin = f"Bin {survey_bin_num}"
                if survey_bin_num == 1:
                    table_content += f"        \\multirow{{{survey_bins[survey]}}}{{*}}{{{survey}}} & {survey_bin} "
                else:
                    table_content += f"        & {survey_bin} "
                
                for type_ in types:
                    for bin_ in bins:
                        key = f"{type_}_{survey}_l{int(bin_[-1])-1}_s{survey_bin_num-1}_{scale}"
                        value = data_dict.get(key, "--")  # Fetch from dictionary or default to empty string
                        if(value=="--"):
                            table_content += f"& {value} "
                        elif(precision==3):
                            table_content += f"& {value:.3f} "
                        elif(precision==1):
                            table_content += f"& {value:.1f} "
                        else:
                            raise NotImplementedError("Only precision 1 and 3 are implemented")
                table_content += "\\\\\n"
            
            table_content += "        \\hline\n"
    
    footer = """
    \\end{tabular}
"""

    return header + table_content + footer

def table_text(value,error,allowed_sigma,precision,mean=0):
    if(value=="--"):
        return f"& {value} "
    else:
        if(np.abs(value)>1e-2):
            returnstr = f"{value:.{precision}f}\\pm {error:.{precision}f}"
        else:
            order_of_magnitude = int(np.abs(np.ceil(np.log10(np.abs(value)))))
            # returnstr = f"({value*10**order_of_magnitude:.{precision}f}\\pm {error*10**order_of_magnitude:.{precision}f})\\times 10^{{{order_of_magnitude}}}"
            returnstr = f"{value*10**order_of_magnitude:.{precision}f}\\pm {error*10**order_of_magnitude:.{precision}f}^*"
        if np.abs(value-mean) > allowed_sigma*error:
            # print(f"Found {value} +- {error} with mean {mean} and allowed sigma {allowed_sigma}")
            return f"& \\textcolor{{red}}{{${returnstr}$}}"
        else:
            # print(f"Found no error in {value} +- {error} with mean {mean} and allowed sigma {allowed_sigma}")
            return f"& ${returnstr}$"

def generate_redshift_slope_latex_table_from_dict(data_dict, error_dict, config, caption="Example caption", precision=3):
    surveys = clean_read(config,'general','lensing_surveys',split=True)
    types = clean_read(config,'general','galaxy_types',split=True)
    scales = clean_read(config,'general','analyzed_scales',split=True)
    allowed_sigma = clean_read(config,'general','critical_sigma',split=False,convert_to_float=True)

    bins = ["Bin 1", "Bin 2", "Bin 3"]
    
    header = """
    \\begin{tabular}{|c|c|c|c|c|}
        \\hline
"""
    
    table_content = ""
    table_content += " & & small scales & large scales & all scales \\\\\n"
    table_content += "        \\hline\n"
    for galaxy_type in types:
        for survey_bin_num in range(1,4):    
                survey_bin = f"Bin {survey_bin_num}"
                if survey_bin_num == 1:
                    table_content += f"        \\multirow{{3}}{{*}}{{{galaxy_type[:3]}}} & {survey_bin} "
                else:
                    table_content += f"        & {survey_bin} "
                
                for scale in scales:
                    key = f"{galaxy_type}_{scale}_{survey_bin_num-1}"
                    value = data_dict.get(key, "--")  # Fetch from dictionary or default to empty string
                    error = error_dict.get(key, "--")  # Fetch from dictionary or default to empty string
                    table_content += table_text(value,error,allowed_sigma,precision)
                table_content += "\\\\\n"
            
        table_content += "        \\hline\n"
    
    footer = """
    \\end{tabular}
"""

    return header + table_content + footer


def generate_splits_latex_table_from_dict(data_dict, error_dict, config, caption="Example caption", precision=3):
    surveys = clean_read(config,'general','lensing_surveys',split=True)
    types = clean_read(config,'general','galaxy_types',split=True)
    scales = clean_read(config,'general','analyzed_scales',split=True)
    allowed_sigma = clean_read(config,'general','critical_sigma',split=False,convert_to_float=True)

    all_splits = clean_read(config,'splits','splits_to_consider',split=True)

    bins = ["Bin 1", "Bin 2", "Bin 3"]
    
    header = """
    \\begin{tabular}{|c|c|c|c|c|}
        \\hline
"""
    
    table_content = ""
    table_content += " & & small scales & large scales & all scales \\\\\n"
    table_content += "        \\hline\n"
    for split in all_splits:
        table_content += "& & \\multicolumn{{3}}{{c|}}{{\\textbf{{{}}}}} \\\\\n".format(split.replace("_","\\_"))
        for galaxy_type in types:
            for survey_bin_num in range(1,4):    
                    survey_bin = f"Bin {survey_bin_num}"
                    if survey_bin_num == 1:
                        table_content += f"        \\multirow{{3}}{{*}}{{{galaxy_type[:3]}}} & {survey_bin} "
                    else:
                        table_content += f"        & {survey_bin} "
                    
                    for scale in scales:
                        key = f"{split}_{galaxy_type}_{scale}_{survey_bin_num-1}"
                        value = data_dict.get(key, "--")  # Fetch from dictionary or default to empty string
                        error = error_dict.get(key, "--")  # Fetch from dictionary or default to empty string
                        table_content += table_text(value,error,allowed_sigma,precision)
                    table_content += "\\\\\n"
                
            table_content += "        \\hline\n"
    
    footer = """
    \\end{tabular}
"""

    return header + table_content + footer

def minimizing_fnc(sigma_sys,amplitudes,statistical_errors,target_value):
    errors = np.sqrt(statistical_errors**2+sigma_sys**2)
    mean_amplitude = np.average(amplitudes,weights=1/errors**2)
    chisq = np.sum((amplitudes-mean_amplitude)**2/errors**2)
    return (chisq-target_value)**2

def find_best_fit_and_errors_brute(log_likelihood_func, n_sigma=1):
    from scipy.optimize import minimize,root_scalar
    from scipy.integrate import quad
    # from scipy.stats import chisq
    # Find the best-fit value by minimizing the negative log-likelihood
    result = minimize(lambda x: -log_likelihood_func(x), x0=1)
    best_fit = result.x[0]
    max_log_likelihood = log_likelihood_func(best_fit)
    
    # Define the likelihood ratio test statistic
    def likelihood_ratio_test(x):
        return -2 * (log_likelihood_func(x) - max_log_likelihood)
    
    # Find the critical value for the desired number of standard deviations
    # For a 1-dimensional likelihood function, the critical value is the square of the number of standard deviations
    critical_value = [0,0.68,0.95,0.997][n_sigma]
    
    # Find the parameter values that correspond to the critical value
    def find_error(critical_value):
        likelihood_fnc = lambda x: np.exp(log_likelihood_func(x))
        norm = quad(likelihood_fnc,-np.inf,np.inf)[0]
        likelihood_fnc = lambda x: np.exp(log_likelihood_func(x))/norm
        def objective_low(x):
            return quad(likelihood_fnc,-np.inf,x)[0] - (1-critical_value)/2
        def objective_up(x):
            return quad(likelihood_fnc,x,np.inf)[0] - (1-critical_value)/2
        try:
            lower_bound = root_scalar(objective_low,bracket = [-100,best_fit], method='brentq').root
        except ValueError:
            print(f"Failed to find lower bound for target value {critical_value}")
            lower_bound = 0
        try:
            upper_bound = root_scalar(objective_up,bracket = [best_fit,100], method='brentq').root
        except ValueError:
            print(f"Failed to find upper bound for target value {critical_value}")
            upper_bound = 100
        return lower_bound, upper_bound


    lower_bound, upper_bound = find_error(critical_value)
    
    return best_fit, lower_bound, upper_bound

def find_best_fit_and_errors(log_likelihood_func, n_sigma=1):
    from scipy.optimize import minimize,root_scalar
    # Find the best-fit value by minimizing the negative log-likelihood
    result = minimize(lambda x: -log_likelihood_func(x), x0=1)
    best_fit = result.x[0]
    max_log_likelihood = log_likelihood_func(best_fit)
    
    # Define the likelihood ratio test statistic
    def likelihood_ratio_test(x):
        return -2 * (log_likelihood_func(x) - max_log_likelihood)
    
    # Find the critical value for the desired number of standard deviations
    # For a 1-dimensional likelihood function, the critical value is the square of the number of standard deviations
    critical_value = n_sigma ** 2
    
    # Find the parameter values that correspond to the critical value
    def find_error(target_value):
        def objective(x):
            return likelihood_ratio_test(x) - target_value
        sol_upper = root_scalar(objective,bracket = [best_fit,best_fit+10], method='brentq')
        try:
            sol_lower = root_scalar(objective,bracket = [best_fit-10 , best_fit], method='brentq')
        except ValueError:
            print(f"Failed to find lower bound for target value {target_value}")
            return np.nan, sol_upper.root
        return sol_lower.root, sol_upper.root
    
    lower_bound, upper_bound = find_error(critical_value)
    
    return best_fit, lower_bound, upper_bound

def calculate_sigma_sys_bayesian(amplitudes, errors, confidence_sigma=[1, 2], verbose=False):
    print("Calculating sigma_sys using Bayesian method")

    prior = nautilus.Prior()
    parameters = ['Alens','sigmasys']

    prior.add_parameter('Alens',dist=(-1,1))
    prior.add_parameter('sigmasys',dist=(0,2))

    def log_likelihood(param_dict):
        total_error = np.sqrt(errors**2 + param_dict['sigmasys']**2)
        chisq = np.sum((amplitudes - param_dict['Alens'])**2 / total_error**2)
        return -0.5 * (chisq + np.sum(np.log(total_error**2)))

    sampler = nautilus.Sampler(prior,log_likelihood,pool=40)
    sampler.run(verbose=verbose)

    points, log_w, log_l = sampler.posterior()

    samples = MCSamples(samples=points, weights=np.exp(log_w), names=parameters, labels=parameters)

    # Get the stats for the chain
    stats = samples.getMargeStats()

    if(verbose):
        # Print the 1-sigma and 2-sigma uncertainties for each parameter
        for param in parameters:
            param_stats = stats.parWithName(param)
            mean = param_stats.mean
            lower_1sigma = param_stats.limits[0].lower
            upper_1sigma = param_stats.limits[0].upper
            lower_2sigma = param_stats.limits[1].lower
            upper_2sigma = param_stats.limits[1].upper

            print(f"{param}:")
            print(f"  Mean: {mean}")
            print(f"  1-sigma: {mean} -{mean-lower_1sigma} +{upper_1sigma-mean}")
            print(f"  2-sigma: {mean} -{mean-lower_2sigma} +{upper_2sigma-mean}")

    # meanA = stats.parWithName('Alens').mean
    meanA = np.average(amplitudes,weights=1/errors**2)
    delta = meanA - amplitudes
    reduced_chisq = np.sum(delta**2 / errors**2) / (len(amplitudes) - 1)
    param_stats = stats.parWithName('sigmasys')
    sigma_sys = param_stats.mean
    lower_1sigma = param_stats.limits[0].lower
    upper_1sigma = param_stats.limits[0].upper
    lower_2sigma = param_stats.limits[1].lower
    upper_2sigma = param_stats.limits[1].upper

    return reduced_chisq, [lower_2sigma, lower_1sigma, sigma_sys, upper_1sigma, upper_2sigma], [points, log_w, log_l]

def calculate_sigma_sys(amplitudes,errors,method="chisqpdf"):
    if method=="chisqpdf":
        return calculate_sigma_sys_chisqpdf(amplitudes,errors)
    elif method=="bayesian":
        return calculate_sigma_sys_bayesian(amplitudes,errors)
    elif method=="bayesian_brute":
        return calculate_sigma_sys_bayesian(amplitudes,errors,brute=True)
    else:
        raise ValueError(f"Method {method} not recognized")

def calculate_sigma_sys_chisqpdf(amplitudes,errors,confidence_regions = [0.68,0.95]):
    assert len(amplitudes)==len(errors), "Amplitudes and errors must have the same length"
    n_measurements = len(amplitudes)
    mean_amplitude = np.average(amplitudes,weights=1/errors**2)
    chisq = np.sum((amplitudes-mean_amplitude)**2/errors**2)
    reduced_chisq = chisq/(n_measurements-1)

    target_chisqs_upper = [chi2.ppf(0.5+confidence_region/2,n_measurements-1) for confidence_region in confidence_regions]
    target_chisqs_lower = [chi2.ppf(0.5-confidence_region/2,n_measurements-1) for confidence_region in confidence_regions[::-1]]
    target_chisqs = np.array(target_chisqs_lower + [n_measurements-1] + target_chisqs_upper)
    target_values = np.zeros_like(target_chisqs)
    for i,target_chisq in enumerate(target_chisqs):
        if (target_chisq >= chisq) or np.isnan(target_chisq):
            target_values[i] = np.nan
        else:
            # minval = minimize(minimizing_fnc,x0=[0.1],args=(amplitudes,errors,target_chisq),bounds=[(0,10)])
            # target_values[i] = minval.x[0] 
            minval = minimize_scalar(minimizing_fnc,args=(amplitudes,errors,target_chisq),bounds=[0,10],tol=1e-8)
            target_values[i] = minval.x

            assert minimizing_fnc(target_values[i],amplitudes,errors,target_chisq)<1e-4, "Minimization failed!"
                # print("Minimization failed! Amplitudes and errors:",amplitudes,errors)
                # assert minimizing_fnc(target_values[i],amplitudes,errors,target_chisq)<1e-1, "Minimization failed!"
    # print(target_chisqs,chisq,n_measurements)
    # print(target_values)
    return reduced_chisq,target_values[::-1],None

    # print(chi2.ppf(0.5,n_measurements-1),n_measurements)
def replace_strings(x,old,new,copy=True):
    if(copy):
        from copy import deepcopy
        x = deepcopy(x)
    for i in range(len(old)):
        x = x.replace(str(old[i]),str(new[i]))
    return x

def get_precomputed_table(galaxy_type,source_survey,fpath_save,version,statistic,lens_bin,source_bin=None,
                          savename_notomo="STAT_GALTYPE_zmin_ZMIN_zmax_ZMAX_blindBLIND_boost_BOOST",
                          savename_tomo="STAT_GALTYPE_zmin_ZMIN_zmax_ZMAX_lenszbin_SOURCEBIN_blindBLIND_boost_BOOST",
                          randoms=None,boost=True,split_by=None, split=None, n_splits=4):
    from astropy.table import join
    import pickle

    if source_survey is None:
        source_survey_str = ''
    else:
        source_survey_str = source_survey + '/'
    
    z_bins_lens = get_lens_bins(galaxy_type)
    

    if split_by is not None:
        table_l_split = Table.read(fpath_save + version + '/split_tables/' + source_survey_str + split_by + '/' + galaxy_type + f'_split_{split_by}_{split}_of_{n_splits}.hdf5')
        table_l_split.keep_columns(['TARGETID','z',split_by])
        if randoms is not None:
            table_r_split = Table.read(fpath_save + version + '/split_tables/' + source_survey_str + split_by + '/' + galaxy_type + f'_split_{split_by}_{split}_of_{n_splits}_randoms.hdf5')
            table_r_split.keep_columns(['TARGETID','z',split_by])

        mask_l = ((z_bins_lens[lens_bin] <= table_l_split['z']) &
                (table_l_split['z'] < z_bins_lens[lens_bin + 1]))

        table_l_part = table_l_split[mask_l]
        table_l_part.remove_column('z')
        if table_r_split is not None:
            mask_r = ((z_bins_lens[lens_bin] <= table_r_split['z']) &
                    (table_r_split['z'] < z_bins_lens[lens_bin + 1]))
            table_r_part = table_r_split[mask_r]
            table_r_part.remove_column('z')

        else:
            table_r_part = None

    if source_bin is not None:
        precomputed_tables_loadname = replace_strings(savename_tomo,
                ['STAT','GALTYPE','ZMIN','ZMAX','SOURCEBIN','BLIND','BOOST'],
                [statistic,galaxy_type,z_bins_lens[lens_bin],z_bins_lens[lens_bin+1],source_bin,
                'A',boost])
    else:
        precomputed_tables_loadname = replace_strings(savename_notomo,
                ['STAT','GALTYPE','ZMIN','ZMAX','BLIND','BOOST'],
                [statistic,galaxy_type,z_bins_lens[lens_bin],z_bins_lens[lens_bin+1],
                'A',boost])

    precomputed_table_l = Table.read(fpath_save + version + '/precomputed_tables/' + source_survey + '/' + precomputed_tables_loadname + '_l.hdf5')
    with open(fpath_save + version + '/precomputed_tables/' + source_survey + '/' + precomputed_tables_loadname + '_meta.pkl',"rb") as f:
        precomputed_tables_meta = pickle.load(f)
    precomputed_table_l.meta = precomputed_tables_meta
    
    if randoms is not None:
        precomputed_table_r = Table.read(fpath_save + version + '/precomputed_tables/' + source_survey + '/' + precomputed_tables_loadname + '_r.hdf5')
        precomputed_table_r.meta = precomputed_tables_meta
    else:
        precomputed_table_r = None

    if split_by is None:
        mask_l = ((z_bins_lens[lens_bin] <= precomputed_table_l['z']) &
                (precomputed_table_l['z'] < z_bins_lens[lens_bin + 1]))

        table_l_part = precomputed_table_l[mask_l]

        if precomputed_table_r is not None:
            mask_r = ((z_bins_lens[lens_bin] <= precomputed_table_r['z']) &
                    (precomputed_table_r['z'] < z_bins_lens[lens_bin + 1]))
            table_r_part = precomputed_table_r[mask_r]
    else:
        table_l_part = join(table_l_part,precomputed_table_l,keys='TARGETID',join_type='left')
        if randoms is not None:
            table_r_part = join(table_r_part,precomputed_table_r,keys='TARGETID',join_type='left')

    if is_table_masked(table_l_part):
        print(f"Masked table_l in split {split} of {split_by} for lens_bin {lens_bin}!")
        print(f"Table_l_split masked: {is_table_masked(table_l_split)}, precomputed_table_l masked: {is_table_masked(precomputed_table_l)}")
        targetids_l_split = set(table_l_split['TARGETID'])
        targetids_precomputed_l = set(precomputed_table_l['TARGETID'])
        targetids_diff = targetids_l_split - targetids_precomputed_l
        print(f"Targetids in table_l_split but not in precomputed_table_l: {len(targetids_diff)}")
        print(f"Overlapping targetids: {len(join(table_l_split,precomputed_table_l,keys='TARGETID',join_type='inner'))}")
        print(f"Length of tables: {len(table_l_split)}, {len(precomputed_table_l)}")
        # sys.exit(-1)
    if randoms is not None:
        if is_table_masked(table_r_part):
            print(f"Masked table_r in split {split} of {split_by} for lens_bin {lens_bin}!")
            print(f"Table_r_split masked: {is_table_masked(table_r_split)}, precomputed_table_r masked: {is_table_masked(precomputed_table_r)}")
            # sys.exit(-1)
        return table_l_part,table_r_part
    return table_l_part
