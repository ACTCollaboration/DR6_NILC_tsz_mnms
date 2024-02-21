'''
Load the mnms models from disk using the mnms interface
drow one sims to test and compare one of the input sims.
'''
import os

import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, enplot, curvedsky
from mnms import noise_models as nm, tiled_noise, wav_noise
from optweight import wlm_utils, mat_utils, wavtrans

opj = os.path.join

odir_base = '/mnt/home/aduivenvoorden/project/actpol/20230622_nilc_july'
noisedir = opj(odir_base, 'noise')
maskdir = opj(odir_base, 'icov', 'mnms', 'masks')

config_name = 'act_dr6v4_nilc070123'

noise_models = {'wavelet' : 'wav',
                'tiled' : 'tile'}

qids = {'wBP_planck' : 'ilc_y_planck',
        'wBP_noKspaceCor' : 'ilc_y_noKspaceCor',
        'wBP' : 'ilc_y'}

mask_obs_dg2 = enmap.read_map(
    opj(maskdir, 'wide_mask_GAL070_apod_1.50_deg_wExtended_mask_obs_apo_dg2.fits'))
mask_obs_dg4 = enmap.read_map(
    opj(maskdir, 'wide_mask_GAL070_apod_1.50_deg_wExtended_mask_obs_apo_dg4.fits'))

for mtype in ['wavelet', 'tiled']:

    for itype in ['wBP_planck', 'wBP_noKspaceCor', 'wBP']:
        
        imgdir = opj(odir_base, 'img', 'mnms', 'icov', mtype, 'get_mnms_sims')

        os.makedirs(imgdir, exist_ok=True)
        
        noise_model_name = noise_models[mtype]
        lmax = 10_000
        mask = mask_obs_dg2        
        if 'planck' in itype:
            noise_model_name += '_planck'
            lmax = 4_000
            mask = mask_obs_dg4
            
        qid = qids[itype]            

        model = nm.BaseNoiseModel.from_config(config_name, noise_model_name, qid)

        sidx = 0
        midx = 0

        sim = model.get_sim(sidx, midx, lmax, alm=False, write=False, verbose=True)

        # Load up the first of the original sims
        basename = f'ilc_yy_{itype}'
        mapname = f'{basename}.fits'
        
        idir = opj(noisedir, f'sims_{mtype}_{sidx}', 'mnms')
        imap = enmap.read_fits(opj(idir, f'{mapname}'))

        # Apply apodized mask to both, compare spectra.
        imap *= mask
        sim *= mask

        ainfo = curvedsky.alm_info(lmax)
        
        alm_imap = curvedsky.map2alm(imap, ainfo=ainfo)
        c_ell_imap = ainfo.alm2cl(alm_imap)

        alm_sim = curvedsky.map2alm(sim[0,0], ainfo=ainfo)
        c_ell_sim = ainfo.alm2cl(alm_sim)

        fig, ax = plt.subplots(dpi=300)
        ax.plot(c_ell_imap[0], label='input sim')
        ax.plot(c_ell_sim[0], label='emulated')
        ax.set_yscale('log')
        ax.set_ylim(1e-19)
        ax.legend(frameon=False)
        fig.savefig(opj(imgdir, f'{basename}_spectra'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300)
        ax.plot(c_ell_sim[0] / c_ell_imap[0], label='emulated / sim')
        ax.set_ylim(0.8, 1.2)
        ax.legend(frameon=False)
        fig.savefig(opj(imgdir, f'{basename}_spectra_ratio'))
        plt.close(fig)
        

