from randoms import prepare_randoms_datavector
from data_handler import get_scales_mask_from_degrees,clean_read,get_rp_chris,get_reference_datavector_of_galtype
import configparser
import sys
import numpy as np

def get_signal_to_noise(config,pivots=[2,8]):
    full_data, full_cov = prepare_randoms_datavector(config,use_theory_covariance=True,datavector_type="measured",
                                                     account_for_cross_covariance=True,pure_noise=False,split_by=None,
                                                     split=None,n_splits=4,logger=None,galaxy_types=["BGS_BRIGHT","LRG"])
    full_reference, _ = prepare_randoms_datavector(config,use_theory_covariance=True,datavector_type="emulator",
                                                     account_for_cross_covariance=True,pure_noise=False,split_by=None,
                                                     split=None,n_splits=4,logger=None,galaxy_types=["BGS_BRIGHT","LRG"])
    min_deg = clean_read(config,'general','min_deg',split=False,convert_to_float=True)
    max_deg = clean_read(config,'general','max_deg',split=False,convert_to_float=True)

    for galaxy_type in ["BGS_BRIGHT","LRG"]:
        for pivot in pivots:
            data = full_data[f"{galaxy_type}_all"]
            cov = full_cov[f"{galaxy_type}_all"]
            full_mask = np.zeros(len(data),dtype=bool)
            idx = 0
            source_masks = np.zeros((3,len(data)),dtype=bool)
            for i,source_survey in enumerate(clean_read(config,'general','lensing_surveys',split=True)[1:]):
                rp = get_rp_chris(galaxy_type,source_survey,clean_read(config,'general','chris_path',split=False),
                        "deltasigma",None)
                scales_mask = get_scales_mask_from_degrees(rp,'large scales',min_deg,max_deg,pivot,galaxy_type,0,config)
                full_mask[idx:idx+len(rp)] = scales_mask
                source_masks[i,idx:idx+len(rp)] = scales_mask
                full_mask = full_mask & np.isfinite(data)
                source_masks[i] = source_masks[i] & np.isfinite(data)
                idx += len(rp)
            _data = data[full_mask]
            _cov = cov[full_mask][:,full_mask]
            _ref = full_reference[f"{galaxy_type}_all"][full_mask]
            assert not np.isclose(_ref,0).any(), f"Reference datavector is zero for {galaxy_type} {pivot} Mpc/h"
            assert not np.isclose(np.diag(cov),0).any(), f"Covariance matrix is zero for {galaxy_type} {pivot} Mpc/h"
            assert not np.isclose(_data,0).any(), f"Datavector is zero for {galaxy_type} {pivot} Mpc/h"
            assert not np.isclose(_ref,_data).any(), f"Reference datavector is equal to datavector for {galaxy_type} {pivot} Mpc/h"
            sn = np.sqrt(np.einsum('i,ij,j',_data,np.linalg.inv(_cov),_ref))
            print(f"{galaxy_type} all {pivot} Mpc/h: {sn:.2f}, ndata: {len(data)}")
            for i,source_survey in enumerate(clean_read(config,'general','lensing_surveys',split=True)[1:]):
                _data = data[source_masks[i]]
                _cov = cov[source_masks[i]][:,source_masks[i]]
                _ref = full_reference[f"{galaxy_type}_all"][source_masks[i]]
                assert not np.isclose(_ref,0).any(), f"Reference datavector is zero for {galaxy_type} {pivot} Mpc/h"
                assert not np.isclose(np.diag(_cov),0).any(), f"Covariance matrix is zero for {galaxy_type} {pivot} Mpc/h"
                assert not np.isclose(_data,0).any(), f"Datavector is zero for {galaxy_type} {pivot} Mpc/h"
                assert not np.isclose(_ref,_data).any(), f"Reference datavector is equal to datavector for {galaxy_type} {pivot} Mpc/h"
                sn = np.sqrt(np.einsum('i,ij,j',_data,np.linalg.inv(_cov),_ref))
                print(f"{galaxy_type} {source_survey} {pivot} Mpc/h: {sn:.2f}, ndata: {len(_data)}")

    # return full_data, full_cov

if __name__ == "__main__":
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("config_plots.conf")
    get_signal_to_noise(config)
