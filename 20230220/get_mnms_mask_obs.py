'''
Define a map geometry with enough pixels to store the sims
(which are bandlimited to ell = 10_000 and 4_000) and
interpolate the NILC mask onto this geometry.
'''
import os

import matplotlib.pyplot as plt
plt.style.use('rg_friendly')
import numpy as np
from mpi4py import MPI

from pixell import enmap, enplot, utils, curvedsky

opj = os.path.join
comm = MPI.COMM_WORLD

mapdir = '/mnt/home/wcoulton/ceph/ACT_TESTING/NILC/export_sims_july'
odir_base = '/mnt/home/aduivenvoorden/project/actpol/20230622_nilc_july'
releasedir = opj(odir_base, 'release', '20230606')

imgdir = opj(odir_base, 'img')
icovdir = opj(odir_base, 'icov', 'mnms', 'masks')

mask = enmap.read_map(opj(releasedir, 'wide_mask_GAL070_apod_1.50_deg_wExtended.fits'))

for lmax, dg in zip([10_000, 4_000], [2, 4]):    

    dec0, dec1 = enmap.corners(mask.shape, mask.wcs)[:,0]
    ny, nx = lmax + 1, 2 * lmax
    res = [np.pi / (ny - 0), 2 * np.pi / nx]
    shape_dg, wcs_dg = enmap.band_geometry(
        [dec0, dec1], res=res, shape=(ny, nx), variant="fejer1")
    
    mask_dg = enmap.zeros(mask.shape[:-2] + shape_dg[-2:], wcs_dg)
    ainfo = curvedsky.alm_info(lmax)          
    alm = curvedsky.map2alm(mask, ainfo=ainfo)
    curvedsky.alm2map(alm, mask_dg, ainfo=ainfo)            
    
    enmap.write_map(
        opj(icovdir, f'wide_mask_GAL070_apod_1.50_deg_wExtended_mask_obs_apo_dg{dg}.fits'),
        mask_dg.astype(np.float32))

    # Also save a boolean version.
    mask_dg[mask_dg < 1e-4] = 0
    mask_dg[mask_dg >= 1e-4] = 1
    
    enmap.write_map(
        opj(icovdir, f'wide_mask_GAL070_apod_1.50_deg_wExtended_mask_obs_dg{dg}.fits'),
        mask_dg.astype(np.float32))
