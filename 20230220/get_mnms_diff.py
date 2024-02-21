'''
Convert the difference maps to a form better suited to mnms.
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

imgdir = opj(odir_base, 'img')
noisedir = opj(odir_base, 'noise')
maskdir = opj(odir_base, 'icov', 'mnms', 'masks')

nsims = 10

mtype_split = []

for mtype in ['wavelet', 'tiled']:
    for sidx in range(nsims):
        mtype_split.append((mtype, sidx))
        
for mtype, sidx in mtype_split[comm.rank::comm.size]:

    print(f'{comm.rank=}, {mtype=}, {sidx=}')

    idir = opj(noisedir, f'sims_{mtype}_{sidx}')
    odir = opj(idir, 'mnms')
    odir_img = opj(imgdir, f'sims_{mtype}_{sidx}', 'diff/mnms')

    os.makedirs(odir, exist_ok=True)
    os.makedirs(odir_img, exist_ok=True)

    # Load diff maps. Just y for now.
    for itype in ['wBP_planck', 'wBP_noKspaceCor', 'wBP']:
    
        for stype in ['tsz_yy']: 
       
            mapfile = opj(idir, f'ilc_{stype[-2:]}{"_" if itype else ""}{itype}.fits')
            mapname = os.path.splitext(os.path.split(mapfile)[-1])[0]                
            imap = enmap.read_fits(opj(idir, f'{mapname}.fits'))
            # Downgrade by factor 2 or 4
            dg = 4 if 'planck' in itype else 2

            lmax = 4000 if 'planck' in itype else 10_000
            ainfo = curvedsky.alm_info(lmax)

            shape_dg, wcs_dg = enmap.read_map_geometry(
                opj(maskdir, f'wide_mask_GAL070_apod_1.50_deg_wExtended_mask_obs_dg{dg}.fits'))
            
            omap = enmap.zeros((1,) + shape_dg[-2:], wcs_dg)
            alm = curvedsky.map2alm(imap, ainfo=ainfo)
            curvedsky.alm2map(alm, omap, ainfo=ainfo)            
            
            enmap.write_fits(opj(odir, f'{mapname}.fits'), omap.astype(np.float32))
