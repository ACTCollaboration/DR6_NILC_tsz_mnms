'''
Estimate inverse-variance maps (needed for the tile
mnms models).
'''
import os

import matplotlib.pyplot as plt
plt.style.use('rg_friendly')
import numpy as np
from mpi4py import MPI

import healpy as hp
from pixell import enmap, enplot, utils, curvedsky
from optweight import alm_c_utils, map_utils, mat_utils

comm = MPI.COMM_WORLD
opj = os.path.join

mapdir = '/mnt/home/wcoulton/ceph/ACT_TESTING/NILC/export_sims_july'
odir_base = '/mnt/home/aduivenvoorden/project/actpol/20230622_nilc_july'

noisedir = opj(odir_base, 'noise')
icovdir = opj(odir_base, 'icov')
specdir = opj(odir_base, 'spectra')
maskdir = opj(odir_base, 'icov', 'mnms', 'masks')

nsims = 10

mask_apo_dg2 = enmap.read_map(
    opj(maskdir, 'wide_mask_GAL070_apod_1.50_deg_wExtended_mask_obs_dg2.fits'))
mask_apo_dg4 = enmap.read_map(
    opj(maskdir, 'wide_mask_GAL070_apod_1.50_deg_wExtended_mask_obs_dg4.fits'))

mtypes_itypes = []
for mtype in ['wavelet', 'tiled']:
    for itype in ['wBP_planck', 'wBP_noKspaceCor', 'wBP']:
        mtypes_itypes.append((mtype, itype))
                            
for mtype, itype in mtypes_itypes[comm.rank::comm.size]:
    
    idir = opj(noisedir, 'mnms', mtype)
    odir = opj(icovdir, 'mnms', mtype)    
    idir_spec = opj(specdir, 'mnms', mtype)    
    imgdir = opj(odir_base, 'img', 'mnms', 'icov', mtype)
    
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(odir, exist_ok=True)
    

    mask_apo = mask_apo_dg4 if 'planck' in itype else mask_apo_dg2

    for stype in ['tsz_yy']:

        for sidx in range(nsims):

            print(f'{mtype=}, {sidx=}')

            idir = opj(noisedir, f'sims_{mtype}_{sidx}', 'mnms')

            mapfile = opj(idir, f'ilc_{stype[-2:]}{"_" if itype else ""}{itype}.fits')
            mapname = os.path.splitext(os.path.split(mapfile)[-1])[0]                
            imap = enmap.read_fits(opj(idir, f'{mapname}.fits'))

            # Filter the map by N_ell^{-0.5}.
            lmax_trunc = 4000 if 'planck' in itype else 10_000            
            n_ell = np.load(
                opj(idir_spec, f'{stype}{"_" if itype else ""}{itype}_diff.npy'))
            lmax = n_ell.shape[-1] - 1
            ainfo = curvedsky.alm_info(lmax)
            alm = curvedsky.map2alm(imap, ainfo=ainfo)                

            sqrt_in_ell = n_ell.copy()
            sqrt_in_ell[n_ell != 0] = n_ell[n_ell != 0] ** -0.5
            sqrt_in_ell[lmax_trunc:] = 0            

            if sidx == 0:
                fig, ax = plt.subplots(dpi=300)
                ax.plot(sqrt_in_ell)
                ax.set_yscale('log')
                fig.savefig(opj(imgdir, f'{mapname}_sqrt_in_ell'))
                plt.close(fig)
            
            alm_c_utils.lmul(alm, sqrt_in_ell, ainfo, inplace=True)
            curvedsky.alm2map(alm, imap, ainfo=ainfo)

            if sidx == 0:
                var = imap ** 2 / nsims
            else:
                var += imap ** 2 / nsims

            print(var.shape)

        # Smooth. Planck maps are approx 3 arcmin. ACT+Planck approx 1.
        fwhm = 18 if 'planck' in itype else 9
        b_ell = hp.gauss_beam(np.radians(fwhm / 60), lmax=lmax)        
        alm = curvedsky.map2alm(var, ainfo=ainfo)
        alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
        curvedsky.alm2map(alm, var, ainfo=ainfo)

        plot = enplot.plot(var, ticks=30, colorbar=True, downgrade=4)
        enplot.write(opj(imgdir, f'{mapname}_var'), plot)

        var *= mask_apo
        
        # Threshold small values of variance map            
        minfo = map_utils.match_enmap_minfo(
            var.shape, var.wcs, mtype='fejer1')
        var_1d = map_utils.view_1d(var, minfo)
        ivar_1d = mat_utils.matpow(var_1d, -1)
        ivar_1d = map_utils.threshold_icov(ivar_1d, q_low=0.01, q_high=1)
        ivar = enmap.enmap(map_utils.view_2d(ivar_1d, minfo)[0], wcs=var.wcs)

        plot = enplot.plot(ivar, ticks=30, colorbar=True, downgrade=4)
        enplot.write(opj(imgdir, f'{mapname}_ivar'), plot)
        
        plot = enplot.plot(np.log10(np.abs(ivar)), ticks=30,
                           colorbar=True, downgrade=4)
        enplot.write(opj(imgdir, f'{mapname}_ivar_log'), plot)

        # Save as 32 bit                
        enmap.write_map(opj(odir, f'{mapname}_ivar.fits'), ivar.astype(np.float32))
