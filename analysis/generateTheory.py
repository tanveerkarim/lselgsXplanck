"""
This script generates theory curves for the fiducial cosmology that is used 
to produce the covariance matrices. 

Author: Tanveer Karim
Last Updated: Nov 18 2022
"""

import os
os.environ["OMP_NUM_THREADS"] = "256"

import sys

import camb
from camb import model, initialpower

import pickle
import numpy as np
import pandas as pd
from time import time 

sys.path.insert(0,'/global/homes/t/tanveerk/SkyLens/') #path to skylens
sys.path.insert(0,'/global/homes/t/tanveerk/SkyLens/skylens') #path to skylens
sys.path.insert(0,'/global/homes/t/tanveerk/lselgsXplanck/src/') #path to helper functions

import skylens
import utilsCross #helper functions

from astropy.cosmology import Planck18_arXiv_v2 as cosmo_planck

cosmo_fid=dict({'h':cosmo_planck.h,
                'Omb':cosmo_planck.Ob0,
                'Omd':cosmo_planck.Om0-cosmo_planck.Ob0,
                's8':0.817,
                'Om':cosmo_planck.Om0,
                'Ase9':2.2,
                'mnu':0,
                'Omk':cosmo_planck.Ok0,
                'tau':0.06,
                'ns':0.965,
                'OmR':cosmo_planck.Ogamma0+cosmo_planck.Onu0,
                'w':-1,
                'wa':0,
                'T_cmb':cosmo_planck.Tcmb0, 
                'Neff':cosmo_planck.Neff,
                'z_max':1090,
                'use_astropy':True})

pk_params={'non_linear':1,
           'kmax':10,
           'kmin':3.e-4,
           'nk':500,
           'scenario':'dmo', 
           'halofit_version':'takahashi',
           'pk_func' :'camb_pk'}

cosmo_params = cosmo_fid

from distributed import LocalCluster
from dask.distributed import Client 

print("start cluster")
#http://distributed.readthedocs.io/en/latest/_modules/distributed/worker.html
c=LocalCluster(n_workers=4,processes=False,memory_limit='150gb',threads_per_worker=1,
               memory_spill_fraction=.99,memory_monitor_interval='2000ms')
client=Client(c)

wigner_files={} 
wigner_files[0]= '/pscratch/sd/t/tanveerk/wig3j_l3072_w6144_0_reorder.zarr/'

test = False

if test:
    NSIDE = 256
    lmax_cl = 3 * NSIDE - 1
    binsize = 10
else:
    NSIDE = 1024
    lmax_cl = 3 * NSIDE - 1
    #lmax_cl = 2*NSIDE
    binsize = 50
    
lmin_cl=0
l0=np.arange(lmin_cl,lmax_cl)

#following defines the ell bins. Using log bins in example, feel free to change.
lmin_cl_Bins=50
lmax_cl_Bins=lmax_cl-10
Nl_bins=20
#l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins))
l_bins = np.arange(lmin_cl_Bins, lmax_cl_Bins, binsize)
#lb=np.sqrt(l_bins[1:]*l_bins[:-1])
lb=0.5*(l_bins[1:]+l_bins[:-1])

l=l0

do_cov=False # if you want to get covariance. Covariance is slow and this should be false if you are calling skylens inside mcmc.
bin_cl=True #bin the theory and covaraince. 

use_window=True #if you want to include the window effect. Code will return pseudo-cl and pseudo-cl covariance
store_win=True # to store window, for some internal reasons. leave it to true.
window_lmax= 3*NSIDE - 1 #smaller value for testing. This should be 2X ell_max in the measurements.
#window_lmax = 2*NSIDE

use_binned_l=False  #FIXME: to speed up computation if using pseudo-cl inside mcmc. Needs to be tested. Leave it false for now.

SSV_cov=False # we donot have good model for super sample and tri-spectrum. We can chat about implementing some approximate analytical forms.
tidal_SSV_cov=False
Tri_cov=False 

bin_xi=True
theta_bins=np.logspace(np.log10(1./60),1,20)

print(f"Calculating NSIDE: {NSIDE}, lmax_cl: {lmax_cl}, window_lmax: {window_lmax}")

def SkyLens_cls(nside, l,  
                dndz_dict, gal_window_dict, gal_maskfile,
                cmb_SN_file, cmb_window_map_arr,
                z_cmb = 1090, zmax_cmb = 1090,
                bg1 = None, bz1 = None, mag_fact = 0, 
                zmin_gal = 0.0, zmax_gal = 1.6, nz = 140, 
                use_window = False,
                Win = None):
    """Returns Skylens object for C_ell calculation based on given maps.
    
    Inputs:
        nside (int) : nside for healpy
        l (array) : multipoles to evaluate 
        dndz_dict (dict) : dictionary containing dndz file location per tomographic bin
        gal_window_dict (dict) : dictionary containing galaxy window function file 
                                location per tomographic bin
        gal_maskfile (str) : Galaxy mask file location
        cmb_SN_file (str) : CMB noise curve file location 
        cmb_window_map_arr (str) : CMB window function file location
        z_cmb (float) : redshift of CMB 
        zmax_cmb (float) : maximum redshift where CMB lensing kernel should be integrated up to
        bg1 (float) : linear bias term for galaxies
        bz1 (dict) : redshift dependent galaxy bias
        mag_fact (float) : magnification bias 
        zmin_gal (float) : min redshift for galaxy sample
        zmax_gal (float) : max redshift for galaxy sample
        nz (int) : number of redshifts where P(k) will be evaluated
        use_window (bool) : whether to evaluate window function
        Win (dict) : optional dict; pass saved window calculated before
        
    Returns:
        kappa0 (dict) : Skylens dict containing Cls, pCls, coupling matrices
    """
    
    results = {}
            
    #tomographed redshift bins for the galaxies
    zl_bin = utilsCross.DESI_elg_bins(l=l, nside = nside, ntomo_bins = len(gal_window_dict), 
                                 bg1 = bg1, bz1 = bz1, mag_fact = mag_fact, 
                                 dndz_arr = dndz_dict, 
                                 gal_maskfile = gal_maskfile, gal_window_arr = gal_window_dict,
                                use_window = use_window)

    np.array([cmb_window_map_file])
    #redshift bins for cmb
    zs_bin = utilsCross.cmb_bins_here(zs = z_cmb, l=l, nside = nside, 
                                 zmax_cmb = zmax_cmb, SN_file = cmb_SN_file, 
                                 cmb_window_map_arr = cmb_window_map_arr,
                                use_window = use_window) # lensing source bin
    
    #names of maps
    corr_kk=('kappa','kappa')
    corr_gg=('galaxy','galaxy')
    corr_gk=('galaxy','kappa')
    corrs=[corr_kk, corr_gg, corr_gk]
    
    #tmpz1 = np.linspace(max(zmin_gal, 1e-4), zmax_gal, nz)
    tmpz1 = np.linspace(0.01, zmax_gal + 0.5, nz)
    #tmpz2 = np.logspace(-4, np.log10(zmax_cmb), nz) #
    #z_PS = np.sort(np.unique(np.around(np.append(tmpz1, tmpz2), 
    #                                   decimals = 3))) #redshifts where P(k) will be evaluated
    z_PS = tmpz1
    print("z_PS: ", len(z_PS))
    
    # if Win is not None:
    #     kappa0 = skylens.Skylens(kappa_zbins=zs_bin,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, 
    #                              galaxy_zbins=zl_bin,
    #                                    use_window=use_window,Tri_cov=Tri_cov, Win = Win,
    #                                    use_binned_l=use_binned_l,wigner_files=wigner_files,
    #                                    SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,
    #                                    store_win=store_win,window_lmax=window_lmax,
    #                                    corrs=corrs, scheduler_info=client.scheduler_info(), log_z_PS=1,
    #                                    cosmo_params = cosmo_params, z_PS=z_PS, pk_params = pk_params)
        
    #else:
    kappa0 = skylens.Skylens(kappa_zbins=zs_bin,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, galaxy_zbins=zl_bin,
                                       use_window=use_window,Tri_cov=Tri_cov, #Win = Win,
                                       use_binned_l=use_binned_l,wigner_files=wigner_files,
                                       SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,
                                       store_win=store_win,window_lmax=window_lmax, wigner_step = 100,
                                       corrs=corrs, scheduler_info=client.scheduler_info(), log_z_PS=1,
                                       cosmo_params = cosmo_params, z_PS=z_PS, pk_params = pk_params)
    
    return kappa0

wtype = 'nnp' #window type 
cmb_SN_file = '/pscratch/sd/t/tanveerk/cmb/lensing/MV/nlkk.dat'
gal_maskfile = '/global/homes/t/tanveerk/lselgsXplanck/finalproducts/mask_bool_dr9.npy'
cmb_window_map_file = np.array(['/global/homes/t/tanveerk/lselgsXplanck/finalproducts/mask_rotated_eq_nside_1024.npy'])
gal_window_map_file = "/global/homes/t/tanveerk/lselgsXplanck/finalproducts/Wg_map_nnp.npy"
gal_window_dict = np.array([gal_window_map_file])

#dndz file
redz_file = pd.read_csv("/global/homes/t/tanveerk/lselgsXplanck/finalproducts/fuji_pz_single_tomo.csv")
dndz = {}
dndz['zrange'] = np.array(redz_file['Redshift_mid'])
dndz['dndz'] = np.array(redz_file['pz'])
zrange = dndz['zrange']

#define cosmology object
tmpcosmo = skylens.cosmology(cosmo_params=cosmo_params)

bg1 = None
bz1 = {}
bz1[0] = 1./tmpcosmo.DZ_int(z=zrange)

mag_fact = 2.621 # set magnification bias ##FIXME: need to change it based on Rongpu's method

print("starting kappa0")
start = time()
kappa0 = SkyLens_cls(nside = NSIDE, l = l,  
                dndz_dict = dndz, gal_window_dict = gal_window_dict, gal_maskfile = gal_maskfile,
                cmb_SN_file = cmb_SN_file, cmb_window_map_arr = cmb_window_map_file,
                z_cmb = 1090, zmax_cmb = 1090,
                bg1 = None, bz1 = bz1, mag_fact = mag_fact, 
                zmin_gal = 0.0, zmax_gal = 3, nz = 140, 
                use_window = use_window,
                Win = None)

print("successful kappa0")

print("Beging cl calcluation")

bi = (0,0)
#calculate C_ells
cl0G = kappa0.cl_tomo() 

#calculate 3x2 C_ells
corr_kk = ('kappa', 'kappa')
corr_gg = ('galaxy', 'galaxy')
corr_kg = ('kappa', 'galaxy')
bi = (0,0)

c_ell_dict = {} #store values

# binned theory C_ells
c_ell_dict['gg_binned'] = cl0G['cl_b'][corr_gg][bi].compute()
c_ell_dict['kg_binned'] = cl0G['cl_b'][corr_kg][bi].compute()
c_ell_dict['kk_binned'] = cl0G['cl_b'][corr_kk][bi].compute()

#unbinned theory C_ells
c_ell_dict['gg'] = cl0G['cl'][corr_gg][bi].compute()
c_ell_dict['kg'] = cl0G['cl'][corr_kg][bi].compute()
c_ell_dict['kk'] = cl0G['cl'][corr_kk][bi].compute()

#unbinned theory pseudo C_ells
c_ell_dict['pgg'] = cl0G['pseudo_cl'][corr_gg][bi].compute()
c_ell_dict['pkg'] = cl0G['pseudo_cl'][corr_kg][bi].compute()
c_ell_dict['pkk'] = cl0G['pseudo_cl'][corr_kk][bi].compute()

#binned theory pseudo C_ells
c_ell_dict['pgg_binned'] = cl0G['pseudo_cl_b'][corr_gg][bi].compute()
c_ell_dict['pkg_binned'] = cl0G['pseudo_cl_b'][corr_kg][bi].compute()
c_ell_dict['pkk_binned'] = cl0G['pseudo_cl_b'][corr_kk][bi].compute()

print("done computing C_ell and D_ell")
print(f"total time: {time() - start}")
pickle.dump(c_ell_dict, open("/pscratch/sd/t/tanveerk/final_data_products/theory_curves.npy", "wb"))
print("done saving")