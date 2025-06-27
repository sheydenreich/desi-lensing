import os
import numpy as np
from matplotlib import pyplot as plt
from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, utils, setup_logging, project_to_wp
from astropy.io import ascii


rmin = 0.08
rmax = 80
binning = 2

scales = ['0.1_0.2', '0.2_0.3', "0.3_0.4", "0.1_0.4"]

for scale in scales:
    plt.figure(figsize=(8, 4), dpi=200)
    try:
        result = TwoPointCorrelationFunction.load(f'/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/wp_measurements/results/NTILE/split_1/rppi/allcounts_BGS_BRIGHT_GCcomb_{scale}_pip_angular_bitwise_log_njack64_nran4_split20.npy')
        result= result[::binning,::]
        result.select((rmin, rmax))
        s, data, cov = project_to_wp(result, pimax=100)
        _ = plt.errorbar(s, s*data, s * np.diag(cov)**0.5, label='split-1', fmt='o-', color='blue', alpha=0.5)
    except:
        pass
    try:
        result = TwoPointCorrelationFunction.load(f'global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/wp_measurements/results/NTILE/split_2/rppi/allcounts_BGS_BRIGHT_GCcomb_{scale}_pip_angular_bitwise_log_njack64_nran4_split20.npy')
        result= result[::binning,::]
        result.select((rmin, rmax))
        s, data, cov = project_to_wp(result, pimax=100)
        _ = plt.errorbar(s, s*data, s * np.diag(cov)**0.5, label='split-2', fmt='s--', color='red', alpha=0.5)
    except:
        pass
    try:
        result = TwoPointCorrelationFunction.load(f'global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/wp_measurements/results/NTILE/split_3/rppi/allcounts_BGS_BRIGHT_GCcomb_{scale}_pip_angular_bitwise_log_njack64_nran4_split20.npy')
        result= result[::binning,::]
        result.select((rmin, rmax))
        s, data, cov = project_to_wp(result, pimax=100)
        _ = plt.errorbar(s, s*data, s * np.diag(cov)**0.5, label='split-3', fmt='^-', color='green', alpha=0.5)
    except:
        pass
    try:
        result = TwoPointCorrelationFunction.load(f'global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/wp_measurements/results/NTILE/split_4/rppi/allcounts_BGS_BRIGHT_GCcomb_{scale}_pip_angular_bitwise_log_njack64_nran4_split20.npy')
        result= result[::binning,::]
        result.select((rmin, rmax))
        s, data, cov = project_to_wp(result, pimax=100)
        _ = plt.errorbar(s, s*data, s*np.diag(cov)**0.5, label='split-4', fmt='h--', color='purple', alpha=0.5)
    except:
        pass
    _ = plt.semilogy()
    _ = plt.semilogx()
    _ = plt.xlim(xmax=200)
    _ = plt.xlabel(r'$r_{p} [\mathrm{Mpc}\cdot \mathrm{h}^{-1}]$', fontsize=10)
    _ = plt.ylabel(r'$r_{p}\times w_{p}(r_{p})$', fontsize=15)
    _ = plt.legend(loc='lower left', frameon=True, fontsize=10)
    plt.title(f"fullfootprint BGS_BRIGHT reshift: {scale}")
    plt.savefig(f'nsplit/fullfootprint_bgs_{scale}.png')
    plt.close()