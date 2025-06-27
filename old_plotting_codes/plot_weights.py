import configparser
import sys
import os
import matplotlib.pyplot as plt
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger
from data_handler import get_last_mtime
from astropy.io import fits
from astropy.table import Table,join
import numpy as np

redshift_bins = {}
redshift_bins['LRG'] = [0.4,0.6,0.8,1.1]
redshift_bins['BGS_BRIGHT'] = [0.1,0.2,0.3,0.4]

fname_add = {}
fname_add['LRG'] = 'LRG'
fname_add['BGS_BRIGHT'] = 'BGS_BRIGHT-21.5'

script_name = 'plot_weights'

def plot_weights(config):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    savepath = clean_read(config,'general','savepath',split=False) + os.sep

    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)


    os.makedirs(savepath,exist_ok=True)
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)

    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)


    fig,ax = plt.subplots(1,2,figsize=(12,6),sharey=True)

    for gt,galaxy_type in enumerate(["BGS_BRIGHT","LRG"]):

        # fname_LSS = f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1/unblinded/{fname_add[galaxy_type]}_clustering.dat.fits"

        fname_LSS = f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{versions[galaxy_type]}/unblinded/{fname_add[galaxy_type]}_clustering.dat.fits"
        # fname_LSS = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.1/unblinded/LRG_clustering.dat.fits"
        hdul_LSS = fits.open(fname_LSS)
        cols_LSS = hdul_LSS[1].columns
        dat_LSS = Table(hdul_LSS[1].data)

        fname_ours = f"/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/{versions[galaxy_type]}/{galaxy_type}_full.dat.fits"
        # fname_ours = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/v1.1/LRG_full.dat.fits"
        hdul_ours = fits.open(fname_ours)
        cols_ours = hdul_ours[1].columns
        dat_ours = Table(hdul_ours[1].data)

        dat_LSS.keep_columns(['TARGETID','WEIGHT_SYS','Z','RA','DEC'])
        dat_ours.keep_columns(['TARGETID','WEIGHT_SYS','Z','RA','DEC'])
        dat_LSS.rename_column('WEIGHT_SYS','WEIGHT_LSS')
        dat_LSS.rename_column('Z','Z_LSS')
        dat_LSS.rename_column('RA','RA_LSS')
        dat_LSS.rename_column('DEC','DEC_LSS')

        dat = join(dat_ours,dat_LSS,keys='TARGETID',join_type='outer')
        dat['Z_BIN'] = np.digitize(dat['Z'],bins=redshift_bins[galaxy_type])

        im = ax[gt].scatter(dat['WEIGHT_SYS'],dat['WEIGHT_LSS'],s=1,c=dat['Z_BIN'],cmap='viridis')
        ax[gt].set_xlabel('Our Imaging systematics weight')
        ax[gt].plot([0.85,1.25],[0.85,1.25],c='k',ls='--')
        ax[gt].set_title(galaxy_type)
    
    # plt.colorbar()
    ax[0].set_ylabel('LSS WEIGHT SYS')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="z-Bin")
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+'imsys_weights.png',dpi=300,transparent=True)

    fig,ax = plt.subplots(1,2,figsize=(12,6))

    for gt,galaxy_type in enumerate(["BGS_BRIGHT","LRG"]):

        # fname_LSS = f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1/unblinded/{fname_add[galaxy_type]}_clustering.dat.fits"

        fname_LSS = f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{versions[galaxy_type]}/unblinded/{fname_add[galaxy_type]}_clustering.dat.fits"
        # fname_LSS = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.1/unblinded/LRG_clustering.dat.fits"
        hdul_LSS = fits.open(fname_LSS)
        cols_LSS = hdul_LSS[1].columns
        dat_LSS = Table(hdul_LSS[1].data)

        fname_ours = f"/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/{versions[galaxy_type]}/{galaxy_type}_full.dat.fits"
        # fname_ours = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/v1.1/LRG_full.dat.fits"
        hdul_ours = fits.open(fname_ours)
        cols_ours = hdul_ours[1].columns
        dat_ours = Table(hdul_ours[1].data)

        dat_LSS.keep_columns(['TARGETID','WEIGHT_SYS','Z','RA','DEC'])
        dat_ours.keep_columns(['TARGETID','WEIGHT_SYS','Z','RA','DEC'])
        dat_LSS.rename_column('WEIGHT_SYS','WEIGHT_LSS')
        dat_LSS.rename_column('Z','Z_LSS')
        dat_LSS.rename_column('RA','RA_LSS')
        dat_LSS.rename_column('DEC','DEC_LSS')

        dat = join(dat_ours,dat_LSS,keys='TARGETID',join_type='outer')
        dat['Z_BIN'] = np.digitize(dat['Z'],bins=redshift_bins[galaxy_type])

        im = ax[gt].scatter(dat['WEIGHT_SYS'],dat['WEIGHT_LSS']-dat['WEIGHT_SYS'],s=1,c=dat['Z_BIN'],cmap='viridis')
        ax[gt].set_xlabel('Our Imaging systematics weight')
        # ax[gt].plot([0.85,1.25],[0.85,1.25],c='k',ls='--')
        ax[gt].axhline(0,c='k',ls='--')
        ax[gt].set_title(galaxy_type)
    
    # plt.colorbar()
    ax[0].set_ylabel('difference to WEIGHT LSS')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="z-Bin")
    plt.savefig(savepath+os.sep+version+os.sep+savepath_addon+os.sep+'imsys_weights_diff.png',dpi=300,transparent=True)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    plot_weights(config)