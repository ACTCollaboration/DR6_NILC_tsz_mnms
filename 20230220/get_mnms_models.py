'''
Estimate noise models from the 10 simulations.
'''
import os

import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, enplot, curvedsky
from mnms import noise_models, tiled_noise, wav_noise
from optweight import wlm_utils, mat_utils, wavtrans

opj = os.path.join

mapdir = '/mnt/home/wcoulton/ceph/ACT_TESTING/NILC/export_sims_july'
odir_base = '/mnt/home/aduivenvoorden/project/actpol/20230622_nilc_july'

noisedir = opj(odir_base, 'noise')
icovdir = opj(odir_base, 'icov')
specdir = opj(odir_base, 'spectra')
maskdir = opj(odir_base, 'icov', 'mnms', 'masks')
modeldir = opj(odir_base, 'icov', 'mnms', 'models')

os.makedirs(modeldir, exist_ok=True)

nsims = 10

mask_obs_dg2 = enmap.read_map(
    opj(maskdir, 'wide_mask_GAL070_apod_1.50_deg_wExtended_mask_obs_dg2.fits'))
mask_obs_dg4 = enmap.read_map(
    opj(maskdir, 'wide_mask_GAL070_apod_1.50_deg_wExtended_mask_obs_dg4.fits'))

qids = {'wBP' : 'ilc_y',
        'wBP_noKspaceCor' : 'ilc_y_noKspaceCor',
        'wBP_planck': 'ilc_y_planck'}

mtypes_itypes = []
for mtype in ['wavelet', 'tiled']:
    
    for itype in ['wBP_planck', 'wBP_noKspaceCor', 'wBP']:
        mtypes_itypes.append((mtype, itype))

for mtype, itype in mtypes_itypes:

    imgdir = opj(odir_base, 'img', 'mnms', 'icov', mtype)

    for stype in ['tsz_yy']:

        basename = f'ilc_{stype[-2:]}{"_" if itype else ""}{itype}'
        mapname = f'{basename}.fits'
        ivarname = f'{basename}_ivar.fits'
        ivar = enmap.read_map(opj(icovdir, 'mnms', mtype, f'{ivarname}'))
        sqrt_ivar = np.sqrt(ivar)

        mask_obs = mask_obs_dg4 if 'planck' in itype else mask_obs_dg2
        lmax_trunc = 4000 if 'planck' in itype else 10_000

        idir_spec = opj(specdir, 'mnms', mtype)
        n_ell = np.load(
            opj(idir_spec, f'{stype}{"_" if itype else ""}{itype}_diff.npy'))

        n_ell[lmax_trunc:] = 0

        lmax = n_ell.shape[-1] - 1

        filter_kwargs = dict(cov_ell=n_ell, sqrt_ivar=sqrt_ivar, lmax=lmax,
                             post_filt_rel_downgrade=1)

        for sidx in range(nsims):
            print(f'{sidx=}')
            idir = opj(noisedir, f'sims_{mtype}_{sidx}', 'mnms')
            imap = enmap.read_fits(opj(idir, f'{mapname}'))

            if mtype == 'tiled':

                delta_ell_smooth = 400
                width_deg = 4
                height_deg = 4
                
                model = noise_models.TiledNoiseModel.get_model_static(
                    imap, mask_obs=mask_obs, delta_ell_smooth=delta_ell_smooth,
                    width_deg=width_deg, height_deg=height_deg,
                    iso_filt_method='harmonic', ivar_filt_method='basic',
                    filter_kwargs=filter_kwargs)
                if sidx == 0:
                    sqrt_cov_mat_tot = model['sqrt_cov_mat'] ** 2
                    sqrt_cov_ell_tot = model['sqrt_cov_ell']
                else:
                    sqrt_cov_mat_tot += model['sqrt_cov_mat'] ** 2

            else:
                lamb = 1.3
                lmin = 300
                lmax_j = 3100 if 'planck' in itype else 6000
                w_ell, _ = wlm_utils.get_sd_kernels(
                   lamb, lmax, j0=None, lmin=lmin, jmax=None, lmax_j=lmax_j,
                   return_j=False)
                fwhm_fact = 2
                
                model = noise_models.WaveletNoiseModel.get_model_static(
                    imap, w_ell, iso_filt_method='harmonic', fwhm_fact=fwhm_fact,
                    ivar_filt_method=None, filter_kwargs=filter_kwargs)

                if sidx == 0:
                    sqrt_cov_mat_tot = model['sqrt_cov_mat']
                    sqrt_cov_mat_tot.slice_preshape(np.s_[0,:,0,:])
                    sqrt_cov_mat_tot = mat_utils.wavmatpow(
                        sqrt_cov_mat_tot, 2, return_diag=True)
                    sqrt_cov_ell_tot = model['sqrt_cov_ell']
                else:
                    sqrt_cov_mat_tmp = model['sqrt_cov_mat']
                    sqrt_cov_mat_tmp.slice_preshape(np.s_[0,:,0,:])
                    sqrt_cov_mat_tmp = mat_utils.wavmatpow(
                        sqrt_cov_mat_tmp, 2, return_diag=True)
                    for key in sqrt_cov_mat_tot.maps:
                        sqrt_cov_mat_tot.maps[key] += sqrt_cov_mat_tmp.maps[key]


        if mtype == 'tiled':
            model_tot = {'sqrt_cov_mat' : np.sqrt(sqrt_cov_mat_tot / nsims),
                         'sqrt_cov_ell' : sqrt_cov_ell_tot}
            noise_model_name = 'tile_planck' if 'planck' in itype else 'tile'
            filename = f'act_dr6v4_nilc070123_{noise_model_name}_{qids[itype]}_'\
                f'lmax{lmax}_1way_set0_noise_model.hdf5'
            tiled_noise.write_tiled_ndmap(
                opj(modeldir, filename), model_tot['sqrt_cov_mat'],
                extra_datasets={'sqrt_cov_ell' : model_tot['sqrt_cov_ell']})

        else:
            for key in sqrt_cov_mat_tot.maps:
                sqrt_cov_mat_tot.maps[key] /= nsims
            sqrt_cov_mat = mat_utils.wavmatpow(
                sqrt_cov_mat_tot, 0.5, return_diag=True)
            for key in sqrt_cov_mat.maps:
                sqrt_cov_mat.maps[key] = sqrt_cov_mat.maps[key][None,:,None,:]

            model_tot = {'sqrt_cov_mat' : sqrt_cov_mat,
                         'sqrt_cov_ell' : sqrt_cov_ell_tot}
            noise_model_name = 'wav_planck' if 'planck' in itype else 'wav'
            filename = f'act_dr6v4_nilc070123_{noise_model_name}_{qids[itype]}_'\
                f'lmax{lmax}_1way_set0_noise_model.hdf5'
            wavtrans.write_wav(opj(modeldir, filename), model_tot['sqrt_cov_mat'],
                               symm_axes=[[0, 1], [2, 3]],
                               extra={'sqrt_cov_ell' : model_tot['sqrt_cov_ell']})
            
        if mtype == 'tiled':
            seed = 1
        else:
            seed = {1}

        filter_kwargs['sqrt_cov_ell'] = model_tot['sqrt_cov_ell']
        filter_kwargs['shape'] = sqrt_ivar.shape
        filter_kwargs['wcs'] = sqrt_ivar.wcs

        if mtype == 'tiled':
            draw = noise_models.TiledNoiseModel.get_sim_static(
                model_tot['sqrt_cov_mat'], seed, iso_filt_method='harmonic',
                ivar_filt_method='basic', filter_kwargs=filter_kwargs)
        else:
            draw = noise_models.WaveletNoiseModel.get_sim_static(
                model_tot['sqrt_cov_mat'], seed, w_ell, iso_filt_method='harmonic',
                ivar_filt_method=None, filter_kwargs=filter_kwargs)

        draw = draw.reshape(1, *draw.shape[-2:])
        draw *= mask_obs

        plot = enplot.plot(
            draw, colorbar=True, ticks=30, downgrade=4,
            font_size=50)
        enplot.write(opj(imgdir, f'{basename}_draw'), plot)

        # Take pseudo-Cl of input and draw.
        ainfo = curvedsky.alm_info(lmax)
        alm_imap = curvedsky.map2alm(imap * mask_obs, ainfo=ainfo)
        c_ell_imap = ainfo.alm2cl(alm_imap)

        alm_draw = curvedsky.map2alm(draw, ainfo=ainfo)
        c_ell_draw = ainfo.alm2cl(alm_draw)

        fig, ax = plt.subplots(dpi=300)
        ax.plot(c_ell_imap[0], label='imap')
        ax.plot(c_ell_draw[0], label='draw')
        ax.set_yscale('log')
        ax.set_ylim(1e-19)
        ax.legend(frameon=False)
        fig.savefig(opj(imgdir, f'{basename}_spectra'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300)
        ax.plot(c_ell_draw[0] / c_ell_imap[0], label='draw / imap')
        ax.set_ylim(0.8, 1.2)
        ax.legend(frameon=False)
        fig.savefig(opj(imgdir, f'{basename}_spectra_ratio'))
        plt.close(fig)
