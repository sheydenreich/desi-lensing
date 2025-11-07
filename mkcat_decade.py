import numpy as np
import h5py
import healpy as hp
from astropy.table import Table


useful_cols = ['RA', 'DEC', 'MCAL_SEL_1P', 'MCAL_SEL_1M', 'MCAL_SEL_2P', 'MCAL_SEL_2M',
               'MCAL_G_1_1P', 'MCAL_G_1_1M', 'MCAL_G_2_2P', 'MCAL_G_2_2M',
               'MCAL_G_1_2P', 'MCAL_G_1_2M', 'MCAL_G_2_1P', 'MCAL_G_2_1M',
               'MCAL_SEL_NOSHEAR', 'MCAL_G_1_NOSHEAR', 'MCAL_G_2_NOSHEAR',
               'MCAL_W_NOSHEAR', 'MCAL_W_1P', 'MCAL_W_1M', 'MCAL_W_2P', 'MCAL_W_2M',
               'DNF_Z']

inputdir = f"/pscratch/sd/z/zwshao/DECADE/"
outputdir = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensingsurvey_catalogues/decade/"

with h5py.File(f"{inputdir}/shear_catalog_sparse.hdf5", "r") as tfile:
    tab = dict()
    for col in useful_cols:
        tab[col] = tfile[col][()]
    tab = Table(tab)

glon, glat = hp.Rotator(coord=['C','G'], deg=True)(tab['RA'], tab['DEC'], lonlat=True)
NGC_only = (glat > 0)
SGC_only = (glat < 0)

tab.rename_column('DEC','Dec')
tab_ngc = tab[NGC_only]
tab_sgc = tab[SGC_only]

print(np.sum(NGC_only), np.sum(SGC_only))

tab_ngc.write(f"{outputdir}/decade_ngc_cat.hdf5", overwrite=True)
tab_sgc.write(f"{outputdir}/decade_sgc_cat.hdf5", overwrite=True)

nz = dict()
for key in ['NGC', 'SGC']:
    nz[key] = np.load(f"{inputdir}/n_of_z/{key}_n_of_z.npy")

zgrid = np.load(f"{inputdir}/n_of_z/z_grid.npy")

zmin_out = 0
zmax_out = 3
nz_out = 60
zstep_out = (zmax_out - zmin_out) / nz_out
zout = np.linspace(zmin_out + zstep_out/2, zmax_out - zstep_out/2, nz_out)
for galcap in ['NGC', 'SGC']:
    for tombin in range(4):
        nzout = np.interp(zout, zgrid['z_MID'], nz[galcap][tombin])
        with open(f'/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/model_inputs_desiy3/pzwei_sources_decade_{galcap.lower()}_tom{tombin+1}_zmax{zmax_out}.dat', 'w') as f:
            f.write(f'{zmin_out} {zmax_out} {nz_out}\n')
            for i in range(nz_out):
                f.write(f'{zout[i]:.4f} {nzout[i]}\n')