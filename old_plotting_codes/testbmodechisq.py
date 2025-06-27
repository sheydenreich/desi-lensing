import numpy as np
from astropy.io import fits

# Initialisations
lensbin = 1
sourcebin = 4
thmaxfit = 3.39
nzs = 3
ntom = 5
nbin = 15
thmin = 0.005
thmax = 5.
dth = (np.log10(thmax)-np.log10(thmin))/float(nbin)
thbin = np.logspace(np.log10(thmin)+0.5*dth,np.log10(thmax)-0.5*dth,nbin)
# Read in data
stem = '/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/'
datfile = 'lensing_measurements/v0.6/KiDS/bmodes_gammat_BGS_BRIGHT_zmin_0.1_zmax_0.2_lenszbin_3_blindA_boost_True.fits'
hdulist = fits.open(stem+datfile)
table = hdulist[1].data
gxdat = table.field('et')
hdulist.close()
# Read in covariance
covfile = 'model_inputs_desiy1/gxcovcorr_kids1000desiy1bgs.dat'
f = open(stem+covfile,'r')
fields = f.readline().split()
nall,nall1 = int(fields[0]),int(fields[1])
gxcov = np.zeros((nall,nall))
for iall in range(nall):
  for jall in range(nall):
    fields = f.readline().split()
    gxcov[iall,jall] = float(fields[2])
f.close()
# Section of data to use
gxdat = gxdat[thbin < thmaxfit]
# Section of covariance to use
izsall,itomall,ibinall,thbinall = np.empty(nall,dtype='int'),np.empty(nall,dtype='int'),np.empty(nall,dtype='int'),np.empty(nall)
for iall in range(nall):
  izs = iall//(nbin*ntom)
  itom = (iall - izs*nbin*ntom)//nbin
  ibin = iall - izs*nbin*ntom - nbin*itom
  izsall[iall] = izs
  itomall[iall] = itom
  ibinall[iall] = ibin
  thbinall[iall] = thbin[ibin]
cut = (izsall == lensbin-1) & (itomall == sourcebin-1) & (thbinall < thmaxfit)
ibadlst = np.where(np.invert(cut))
gxcov = np.delete(gxcov,ibadlst,axis=0)
gxcov = np.delete(gxcov,ibadlst,axis=1)
# Determine chi-squared
gxcovinv = np.linalg.inv(gxcov)
chisq = np.dot(gxdat,np.dot(gxcovinv,gxdat))
dof = len(gxdat)
print('chisq =',chisq)
print('dof =',dof)
print('data =',gxdat)
print('cov =',gxcov)
print('shape = ',gxcov.shape)
print('thetas = ',thbin)