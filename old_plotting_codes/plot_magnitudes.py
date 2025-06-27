import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,using_mpl_scatter_density
from data_handler import get_last_mtime
sys.path.append(os.path.abspath('..'))
from load_catalogues import get_lens_table,get_source_table
from cataloguecreator import CatalogueCreator
from modules.versions import Version

import astropy.units as u
from datetime import datetime


script_name = 'plot_magnitudes'


def plot_magnitudes(config,add_kp3_cut=False):
    version = Version(clean_read(config,'general','version',split=False))
    versions = get_versions(version)

    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)


    os.makedirs(savepath,exist_ok=True)
    os.makedirs(savepath+os.sep+str(version)+os.sep+savepath_addon+os.sep,exist_ok=True)

    logger = get_logger(savepath+os.sep+str(version)+os.sep+savepath_addon+os.sep,script_name,__name__)

    mag_col = clean_read(config,script_name,'mag_col',split=False)
    fig = plt.figure(figsize=(5,3))

    galaxy_type = 'BGS_BRIGHT'

    catcreator = CatalogueCreator(galaxy_type,
                        catalogue_version=versions[galaxy_type],
                        verbose=True)
    lens_table = catcreator.get_data_catalogue(magnitude_cuts=None)
    magnitude_cuts = catcreator.get_magnitude_cuts()
    # magnitude_cuts = np.array([-21.5,-21.5,-21.5])

    lens_bins = catcreator.get_lens_bins()
    magnitude_mask = catcreator.get_magnitude_mask(lens_table,magnitude_cuts,lens_bins,mag_col=mag_col)



    # lens_table = get_lens_table(galaxy_type,None,None,versions=versions, logger=logger, dsigma_additional_columns=['ABSMAG_RP0'])[0]
    # ax.scatter(lens_table['Z_not4clus'][magnitude_mask],lens_table['ABSMAG_RP0'][magnitude_mask],s=1)
    # ax.scatter(lens_table['Z_not4clus'][~magnitude_mask],lens_table['ABSMAG_RP0'][~magnitude_mask],s=1)
    ax = using_mpl_scatter_density(fig,lens_table['Z_not4clus'],lens_table['ABSMAG_RP0'],
                                   cbar=True,cbar_label=r'$N_{\rm gal}$/pixel')

    survived_cut = np.sum(magnitude_mask)
    total = len(lens_table)
    survived_cut = survived_cut/total
    ax.text(0.05,0.95,f"{survived_cut*100:.1f}% remain",transform=ax.transAxes,
            ha='left',va='top',fontsize=15)
    print('total_gals:',np.sum(magnitude_mask))

    for i in range(len(magnitude_cuts)):
        plt.plot(lens_bins[i:i+2],[magnitude_cuts[i],magnitude_cuts[i]],color='k',linestyle='--',linewidth=2)
    for i in range(len(lens_bins)):
        plt.axvline(x=lens_bins[i],ymin=magnitude_cuts[i-1] if i!=0 else magnitude_cuts[0],color='k',linestyle=':',linewidth=2)
    if(add_kp3_cut):
        plt.axhline(-21.5,color='r',linestyle='-',linewidth=2)
    ax.set_ylim(-25,-17.5)
    ax.set_xlabel('z')
    ax.set_ylabel('Absolute R-band magnitude')


    # ax.set_title(galaxy_type[:3])

    ax.invert_yaxis()
    if add_kp3_cut:
        plt.savefig(savepath+os.sep+str(version)+os.sep+savepath_addon+os.sep+'absolute_magnitudes_kp3.png',
                dpi=300,transparent=True,bbox_inches='tight')
    else:
        plt.savefig(savepath+os.sep+str(version)+os.sep+savepath_addon+os.sep+'absolute_magnitudes.png',
                    dpi=300,transparent=True,bbox_inches='tight')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    add_kp3_cut = "--add_kp3_cut" in sys.argv
    plot_magnitudes(config,add_kp3_cut=add_kp3_cut)