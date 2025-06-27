#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from data_handler import load_covariance_chris,get_rp_chris,get_allowed_bins,get_number_of_source_bins,get_bins_mask,                        load_data_and_covariance_notomo,load_data_and_covariance_tomo,get_number_of_lens_bins,combine_datavectors,                        get_number_of_radial_bins,get_reference_datavector,get_scales_mask,get_deltasigma_amplitudes,                        get_reference_datavector_of_galtype,get_scales_mask_from_degrees,get_split_value,get_lens_bins
import matplotlib.gridspec as gridspec

from dsigma.jackknife import jackknife_resampling,jackknife_resampling_cross_covariance,compute_jackknife_fields,compress_jackknife_fields


sys.path.append('../')
from load_catalogues import get_lens_table,get_source_table,is_table_masked
from data_handler import get_allowed_bins

import pickle

from astropy.table import Table,join,hstack,vstack
script_name = 'splits'

from importlib import reload

from dsigma.stacking import excess_surface_density


# In[ ]:





# In[2]:


config = configparser.ConfigParser()
config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")

def replace_strings(x,old,new,copy=True):
    if(copy):
        from copy import deepcopy
        x = deepcopy(x)
    for i in range(len(old)):
        x = x.replace(str(old[i]),str(new[i]))
    return x


# In[3]:


def get_precomputed_table(galaxy_type,source_survey,fpath_save,version,statistic,lens_bin,
                          savename_notomo="STAT_GALTYPE_zmin_ZMIN_zmax_ZMAX_blindBLIND_boost_BOOST",randoms=None,boost=True,split_by=None, split=None, n_splits=4):
    if source_survey is None:
        source_survey_str = ''
    else:
        source_survey_str = source_survey + '/'
    
    z_bins_lens = get_lens_bins(galaxy_type)
    


    table_l_split = Table.read(fpath_save + version + '/split_tables/' + source_survey_str + split_by + '/' + galaxy_type + f'_split_{split_by}_{split}_of_{n_splits}.fits')
    table_l_split.keep_columns(['TARGETID','z',split_by])
    if randoms is not None:
        table_r_split = Table.read(fpath_save + version + '/split_tables/' + source_survey_str + split_by + '/' + galaxy_type + f'_split_{split_by}_{split}_of_{n_splits}_randoms.fits')
        table_r_split.keep_columns(['RANDOM_TARGETID','z',split_by])

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

    precomputed_tables_loadname = replace_strings(savename_notomo,
            ['STAT','GALTYPE','ZMIN','ZMAX','BLIND','BOOST'],
            [statistic,galaxy_type,z_bins_lens[lens_bin],z_bins_lens[lens_bin+1],
            'A',boost])

    precomputed_table_l = Table.read(fpath_save + version + '/precomputed_tables/' + source_survey + '/' + precomputed_tables_loadname + '_l.fits')
    with open("/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensing_measurements/v1.1/precomputed_tables/HSC/deltasigma_LRG_zmin_0.4_zmax_0.6_blindA_boost_True_meta.pkl","rb") as f:
        precomputed_tables_meta = pickle.load(f)
    precomputed_table_l.meta = precomputed_tables_meta
    
    if randoms is not None:
        precomputed_table_r = Table.read(fpath_save + version + '/precomputed_tables/' + source_survey + '/' + precomputed_tables_loadname + '_r.fits')
        precomputed_table_r.meta = precomputed_tables_meta
    else:
        precomputed_table_r = None
    table_l_part = join(table_l_part,precomputed_table_l,keys='TARGETID',join_type='left')
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
        table_r_part = join(table_r_part,precomputed_table_r,keys='RANDOM_TARGETID',join_type='left')
        if is_table_masked(table_r_part):
            print(f"Masked table_r in split {split} of {split_by} for lens_bin {lens_bin}!")
            print(f"Table_r_split masked: {is_table_masked(table_r_split)}, precomputed_table_r masked: {is_table_masked(precomputed_table_r)}")
            # sys.exit(-1)
    return table_l_part,table_r_part


# In[7]:


lens_bin = 0
tabs_l = []
tabs_r = []
for survey in ['HSC','KiDS']:
    for split_by in ['NTILE','STARDENS']:
        for split in range(4):
            tab_l_1,tab_r_1 = get_precomputed_table("BGS_BRIGHT",survey,
                                                    "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensing_measurements/",
                                                    "v1.2","deltasigma",lens_bin,
                                                    "STAT_GALTYPE_zmin_ZMIN_zmax_ZMAX_blindBLIND_boost_BOOST",
                                                    True,True,split_by,split)
            tab_l_1.remove_columns(['field_jk',split_by])
            tab_r_1.remove_columns(['field_jk',split_by])

            # centers = compute_jackknife_fields(tab_l_1,100)
            # compute_jackknife_fields(tab_r_1,centers)
            tabs_l.append(tab_l_1)
            tabs_r.append(tab_r_1)
joint_tab_l = vstack(tabs_l)
joint_tab_r = vstack(tabs_r)
centers = compute_jackknife_fields(joint_tab_l,100)
_ = compute_jackknife_fields(joint_tab_r,centers)


# In[8]:


is_table_masked(joint_tab_l),is_table_masked(joint_tab_r)


# In[13]:


for col in joint_tab_l.colnames:
    print(col,getattr(joint_tab_l[col],'mask',None))
# any(getattr(col, 'mask', None) is not None for col in table.columns.values())


# In[9]:


ctabs_l = []
ctabs_r = []
counter_l = 0
counter_r = 0
for i in range(len(tabs_l)):
    ltab_l = len(tabs_l[i])
    ltab_r = len(tabs_r[i])
    ctabs_l.append(compress_jackknife_fields(joint_tab_l[counter_l:counter_l+ltab_l]))
    ctabs_r.append(compress_jackknife_fields(joint_tab_r[counter_r:counter_r+ltab_r]))
    counter_l += ltab_l
    counter_r += ltab_r


# In[ ]:





# In[10]:


from importlib import reload
import dsigma.jackknife
import dsigma.stacking
reload(dsigma.jackknife)
reload(dsigma.stacking)
from dsigma.jackknife import jackknife_resampling,jackknife_resampling_cross_covariance,compute_jackknife_fields,compress_jackknife_fields


# In[11]:


# big_cov = jackknife_resampling_cross_covariance(excess_surface_density,ctabs_l,ctabs_r)
cov = jackknife_resampling(excess_surface_density,ctabs_l[0],ctabs_r[0])


# In[ ]:


for i in range(len(big_cov)//15):
    cov = jackknife_resampling(excess_surface_density,ctabs_l[i],ctabs_r[i])
    assert np.allclose(big_cov[15*i:15*(i+1),15*i:15*(i+1)],cov)


# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(20,20))

plt.imshow(big_cov/np.sqrt(np.diag(big_cov)[:,None]*np.diag(big_cov)[None,:]),vmin=-1,vmax=1,cmap='seismic')
for i in range(len(big_cov)//15):
    plt.axhline(i*15-0.5,color='black',ls='--')
    plt.axvline(i*15-0.5,color='black',ls='--')
for i in range(len(big_cov)//(4*15)):
    plt.axhline(i*4*15-0.5,color='black',ls='-')
    plt.axvline(i*4*15-0.5,color='black',ls='-')
for i in range(len(big_cov)//(4*15*2)+1):
    plt.axhline(i*4*15*2-0.5,color='black',ls='-',lw=2)
    plt.axvline(i*4*15*2-0.5,color='black',ls='-',lw=2)

plt.axis('off')
ax.text(0.25,1.1,'HSC',transform=ax.transAxes,fontsize=20,horizontalalignment='center')
ax.text(0.5+0.25,1.1,'KiDS',transform=ax.transAxes,fontsize=20,horizontalalignment='center')
ax.text(0.25/2,1.05,'NTILE',transform=ax.transAxes,fontsize=20,horizontalalignment='center')
ax.text(0.25/2+0.25,1.05,'STARDENS',transform=ax.transAxes,fontsize=20,horizontalalignment='center')
ax.text(0.5+0.25/2,1.05,'NTILE',transform=ax.transAxes,fontsize=20,horizontalalignment='center')
ax.text(0.5+0.25/2+0.25,1.05,'STARDENS',transform=ax.transAxes,fontsize=20,horizontalalignment='center')


# In[ ]:




