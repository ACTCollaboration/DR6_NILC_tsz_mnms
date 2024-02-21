''' Interpolate noise spectra.'''
import os

import matplotlib.pyplot as plt
plt.style.use('rg_friendly')
import numpy as np
from scipy.interpolate import interp1d

from pixell import enmap, enplot, utils, curvedsky

opj = os.path.join

mapdir = '/mnt/home/wcoulton/ceph/ACT_TESTING/NILC/export_sims_july'
odir_base = '/mnt/home/aduivenvoorden/project/actpol/20230622_nilc_july'

specdir = opj(odir_base, 'spectra')

nsims = 10

for mtype in ['wavelet', 'tiled']:

    odir = opj(specdir, 'mnms', mtype)
    imgdir = opj(odir_base, 'img', 'mnms', 'spectra', mtype)

    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(odir, exist_ok=True)

    for itype in ['', 'noKspaceCor', 'planck', 'wBP_planck', 'wBP_noKspaceCor', 'wBP', 'deproj_sz']:
                    
        for stype in ['cmb_TT', 'cmb_EE', 'tsz_yy']:
            
            if itype == 'deproj_sz' and stype in ('cmb_EE', 'tsz_yy'):
                continue
            
            fig, ax = plt.subplots(dpi=300)
            fig2, ax2 = plt.subplots(dpi=300)

            c_ells = []
            for sidx in range(nsims):

                idir = opj(specdir, f'sims_{mtype}_{sidx}')

                bins = np.load(opj(idir, 'ells_diff.npy'))
                dbins = bins * (bins + 1) / 2 / np.pi

                c_ell = np.load(opj(idir, f'{stype}{"_" if itype else ""}{itype}_diff.npy'))[0,0]

                c_ells.append(c_ell)
                
            c_ells = np.asarray(c_ells)
            for sidx in range(nsims):
                ax.plot(bins, c_ells[sidx], color='C0', alpha=0.5)
                ax2.plot(bins, dbins * c_ells[sidx], color='C0', alpha=0.5)

            # Second bin is sometimes negative, just from the MASTER
            # algorithm. It only happens for some of the spectra, so
            # we exclude them from the estimate of the mean in those bins.
            c_ells[c_ells < 0] = np.nan
            
            c_ell_mean = np.nanmean(c_ells, axis=0)
            c_ell_mean[np.isnan(c_ell_mean)] = 0            
            
            ax.plot(bins, c_ell_mean, color='C1', label='mean')
            ax2.plot(bins, dbins * c_ell_mean, color='C1', label='mean')

            # Interpolate.
            lmax = 4000 if 'planck' in itype else 10_000
            binmask = bins < lmax
            bins_trunc = np.zeros(bins[binmask].size + 2)
            bins_trunc[1:-1] = bins[binmask]
            bins_trunc[-1] = lmax

            # Do the interpolation in D_ell. Flatter there.
            c_ell_trunc = c_ell_mean[binmask]
            d_ell_trunc = np.zeros(c_ell_trunc.size + 2)
            d_ell_trunc[1:-1] = dbins[binmask] * c_ell_trunc
            d_ell_trunc[0] = dbins[0] * c_ell_trunc[0]
            d_ell_trunc[-1] = d_ell_trunc[-2]

            ells = np.arange(lmax + 1)
            dells = ells * (ells + 1) / 2 / np.pi
            cs = interp1d(bins_trunc, d_ell_trunc, kind='cubic')

            d_ell_interp = cs(ells)
            c_ell_interp = d_ell_interp.copy()
            c_ell_interp[1:] /= dells[1:]

            # Set small negative values at high ell to zero.
            c_ell_interp[c_ell_interp < 0] = 0
            np.save(opj(odir, f'{stype}{"_" if itype else ""}{itype}_diff'), c_ell_interp)

            
            ax.plot(ells, c_ell_interp, color='C2', label='mean_interp')
            ax2.plot(ells, d_ell_interp, color='C2', label='mean_interp')

            ax.set_xlim(1, lmax)
            ax.set_yscale('log')
            ax.set_xscale('log')

            if 'yy' in stype:            
                ax.set_ylim(1e-18)
                ax2.set_ylim(1e-15)
            else:
                ax.set_ylim(1e-7)
                ax2.set_ylim(1e-2)
                
            ax.legend(frameon=False)
            
            ax2.set_xlim(1, lmax)            
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.legend(frameon=False)
            
            fig.savefig(opj(imgdir, f'{stype}{"_" if itype else ""}{itype}_spectra_diff'))
            fig2.savefig(opj(imgdir, f'{stype}{"_" if itype else ""}{itype}_spectra_diff_dell'))
            plt.close(fig)
            plt.close(fig2)
