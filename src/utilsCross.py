"""This file contains all the utility functions"""

import itertools
import numpy as np

def bin_mat(r=[],mat=[],r_bins=[]):
    """Sukhdeep's Code to bins data and covariance arrays

    Input:
    -----
        r  : array which will be used to bin data, e.g. ell values
        mat : array or matrix which will be binned, e.g. Cl values
        bins : array that defines the left edge of the bins,
               bins is the same unit as r

    Output:
    ------
        bin_center : array of mid-point of the bins, e.g. ELL values
        mat_int : binned array or matrix
    """

    bin_center=0.5*(r_bins[1:]+r_bins[:-1])
    n_bins=len(bin_center)
    ndim=len(mat.shape)
    mat_int=np.zeros([n_bins]*ndim,dtype='float64')
    norm_int=np.zeros([n_bins]*ndim,dtype='float64')
    bin_idx=np.digitize(r,r_bins)-1
    r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
    dr=np.gradient(r2)
    r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
    dr=dr[r2_idx]
    r_dr=r*dr

    ls=['i','j','k','l']
    s1=ls[0]
    s2=ls[0]
    r_dr_m=r_dr
    for i in np.arange(ndim-1):
        s1=s2+','+ls[i+1]
        s2+=ls[i+1]
        r_dr_m=np.einsum(s1+'->'+s2,r_dr_m,r_dr)#works ok for 2-d case

    mat_r_dr=mat*r_dr_m
    for indxs in itertools.product(np.arange(min(bin_idx),n_bins),repeat=ndim):
        x={}#np.zeros_like(mat_r_dr,dtype='bool')
        norm_ijk=1
        mat_t=[]
        for nd in np.arange(ndim):
            slc = [slice(None)] * (ndim)
            #x[nd]=bin_idx==indxs[nd]
            slc[nd]=bin_idx==indxs[nd]
            if nd==0:
                mat_t=mat_r_dr[slc]
            else:
                mat_t=mat_t[slc]
            norm_ijk*=np.sum(r_dr[slc[nd]])
        if norm_ijk==0:
            continue
        mat_int[indxs]=np.sum(mat_t)/norm_ijk
        norm_int[indxs]=norm_ijk
    return bin_center,mat_int


import pandas as pd
import numpy as np
from skylens import *

def set_window_here(ztomo_bins_dict={}, nside=1024, mask = None, cmb=False, 
                    window_map_arr = None): #unit_win=False): 
    """
    This function sets the window functions for the datasets. These windows are necessary for converting cl to pseudo-cl.
    
        
    Parameters
    ----------
    ztomo_bins_dict : dictionary processed by source_tomo_bins function that 
                      contains information on the different tomographic bins for Skylens
    nside : NSIDE of the window maps based on healpix formalism
    mask : survey geometry mask in healpix format, same NSIDE as nside
    cmb : flag for using cmb window function
    window_map_arr : numpy array containing list of window map file paths
    
    Returns
    -------
    ztomo_bins_dict : updates the input ztomo_bins_dict with the window and window noise maps
    """
    
    
    #FIXME: make sure nside, etc. are properly matched. if possible, use same nside for cmb and galaxy maps. 
    # Use ud_grade where necessary.
    #assert window_map_file != None, "Provide relevant window maps"
    if window_map_arr is not None:
        pass
    else:
        assert 1 != 1, "Provide relevant window maps"
    
    for i in np.arange(ztomo_bins_dict['n_bins']):
        if cmb:
            window_map=np.load(window_map_arr[i])
            window_map_noise = window_map
            print("processing cmb lensing window")
        else: #this is for galaxies 
            window_map=np.load(window_map_arr[i])
            window_map = window_map.astype(np.float64)
            window_map_noise = np.sqrt(window_map)
        
            if mask is None:
                mask=(window_map != hp.UNSEEN) #FIXME: input proper mask if possible
            window_map[~mask]=hp.UNSEEN
            window_map_noise[~mask]=hp.UNSEEN
        
        ztomo_bins_dict[i]['window']=window_map
        ztomo_bins_dict[i]['window_N']=window_map_noise #window of noise 

    return ztomo_bins_dict

def zbin_pz_norm(ztomo_bins_dict={},tomo_bin_indx=None,zbin_centre=None,p_zspec=None, ndensity=0, bg1 = None,
                 bz1 = None, mag_fact=0):#,k_max=0.3):
    """
        This function prepares the ztomo_bins_dict object that contains necessary information that 
        will be input into Skylens.
        
        Parameters
        ----------
        ztomo_bins_dict : (dict) object that contains necessary information for Skylens  
        tomo_bin_indx : (int) specifies the index of the tomographic bin
        zbin_centre : (np.array) array containing bin centre values 
        ndensity : (float) number density of sample
        bg1 : (float) linear bias of sample
        mag_fact : (float) maginification bias  
        
        ##--DEPRECATED--##
        k_max = kmax to be considered for C_ell calculation 
    """

    dzspec=np.gradient(zbin_centre) if len(zbin_centre)>1 else 1 #spec bin width

    if np.sum(p_zspec*dzspec)!=0:
        p_zspec=p_zspec/np.sum(p_zspec*dzspec) #normalize histogram
    else:
        p_zspec*=0
    nz=dzspec*p_zspec*ndensity

    i=tomo_bin_indx
    x= p_zspec>1.e-10 #1.e-10; incase we have absurd p(z) values

    ztomo_bins_dict[i]['z']=zbin_centre[x]
    ztomo_bins_dict[i]['dz']=np.gradient(zbin_centre[x]) if len(zbin_centre[x])>1 else 1
    ztomo_bins_dict[i]['nz']=nz[x]
    ztomo_bins_dict[i]['ns']=ndensity
    ztomo_bins_dict[i]['W']=1. #redshift dependent weight
    ztomo_bins_dict[i]['pz']=p_zspec[x]*ztomo_bins_dict[i]['W']
    ztomo_bins_dict[i]['pzdz']=ztomo_bins_dict[i]['pz']*ztomo_bins_dict[i]['dz']
    ztomo_bins_dict[i]['Norm']=np.sum(ztomo_bins_dict[i]['pzdz'])
    ztomo_bins_dict[i]['b1'] = bg1 # FIXME: this is the linear galaxy bias. Input proper values. We can also talk about adding other bias models if needed.
    if bz1 is not None:
        ztomo_bins_dict[i]['bz1'] = bz1[x] #array; set b1 to None if passing redz dependent bias 
    else:
        ztomo_bins_dict[i]['bz1'] = None #array; set b1 to None if passing redz dependent bias 
    ztomo_bins_dict[i]['AI']=0. # this will be zero for our project
    ztomo_bins_dict[i]['AI_z']=0. # this will be zero for our project
    ztomo_bins_dict[i]['mag_fact']=mag_fact  #FIXME: You need to figure out the magnification bias prefactor. For example, see appendix B of https://arxiv.org/pdf/1803.08915.pdf
    ztomo_bins_dict[i]['shear_m_bias'] = 1.  #
    
    #convert k to ell
    zm=np.sum(ztomo_bins_dict[i]['z']*ztomo_bins_dict[i]['pzdz'])/ztomo_bins_dict[i]['Norm']
    #ztomo_bins_dict[i]['lm']=k_max*cosmo_h.comoving_transverse_distance(zm).value #not being used at the moment; if needed, talk to Sukhdeep
    return ztomo_bins_dict

def source_tomo_bins(zphoto_bin_centre=None, p_zphoto=None, ntomo_bins=None, 
                     ndensity=2400/3600, ztomo_bins=None, nside=256,
                     use_window=False, bg1=None, bz1 = None, l=None, mag_fact=0,
                     use_shot_noise=True, gal_mask = None,
                     gal_window_arr = None):
    """
        Returns dict object with tomographic information of galaxies as input for Skylens.
        
        Parameters
        ----------
        zphoto_bin_centre (np. array) : bin centres for photometric sample
        p_zphoto (np.array) : redshift distribution function of sample; does not have to be normalized
        ntomo_bins (int) : number of tomographic bins
        ndensity (np.array) : number density of sample/arcmin2, same shape as ntomo_bins
        ztomo_bins (np.array) : edges of the tomographic bins
        nside (int) : NSIDE of the window maps based on healpix formalism
        use_window (bool) : flag to include window function
        bg1 (np.array) : linear bias of the sample, same shape as ntomo_bins
        l (np.array) : array of multipoles
        mag_fact (float) : magnification bias factor
        use_shot_noise (bool) : flag to include shot noise
        gal_mask (str) : file location for galaxy mask 
        gal_window_arr (np.array) : array containing file paths to galaxy window functions
        
        ##--DEPRECATED--##
        n_gal: number density for shot noise calculation
        n_zspec : number of histogram bins in spectroscopic dndz (if zspec_bin_centre is not passed)
        ztomo_bins : edges of tomographic bins in photometric redshift (assign galaxies to tomo bins using photz)
                    e.g. [0.6, 1., 1.6]
        k_max : cut in k-space; CHECK FOR BUG
    """

    ztomo_bins_dict={} #dictionary of tomographic bins

    if ntomo_bins is None:
        ntomo_bins=1

    zmax=max(ztomo_bins)

    #l=[1] if l is None else l
    ztomo_bins_dict['SN']={} #shot noise dict
    ztomo_bins_dict['SN']['galaxy']=np.zeros((len(l),ntomo_bins,ntomo_bins)) # ell X no. of tomo bins X no. of tomo bins 
    ztomo_bins_dict['SN']['kappa']=np.zeros((len(l),ntomo_bins,ntomo_bins))

    for i in np.arange(ntomo_bins):
        ztomo_bins_dict[i]={}
        dzphoto=np.gradient(zphoto_bin_centre[i]) if len(zphoto_bin_centre[i])>1 else [1]
        zbin_centre = np.array(zphoto_bin_centre[i])
        p_zspec=p_zphoto[i] 
        nz=ndensity[i]*p_zspec[i]*dzphoto 
        ns_i=nz.sum()
              
        if bg1 is not None:
            ztomo_bins_dict = zbin_pz_norm(ztomo_bins_dict=ztomo_bins_dict, tomo_bin_indx=i, 
                                       zbin_centre=zbin_centre, 
                                       p_zspec=p_zspec,ndensity=ns_i,bg1=bg1[i],
                                       mag_fact=mag_fact)#,k_max=k_max)
        if bz1 is not None:
            ztomo_bins_dict = zbin_pz_norm(ztomo_bins_dict=ztomo_bins_dict, tomo_bin_indx=i, 
                                       zbin_centre=zbin_centre, 
                                       p_zspec=p_zspec,ndensity=ns_i, bz1 = bz1[i],
                                       mag_fact=mag_fact)#,k_max=k_max)
            
        zmax=max([zmax,max(ztomo_bins_dict[i]['z'])])
        if use_shot_noise:
            ztomo_bins_dict['SN']['galaxy'][:,i,i]=galaxy_shot_noise_calc(zg1=ztomo_bins_dict[i],
                                                                  zg2=ztomo_bins_dict[i])

    ztomo_bins_dict['n_bins']=ntomo_bins #easy to remember the counts
    ztomo_bins_dict['zmax']=zmax
    ztomo_bins_dict['zp']=zphoto_bin_centre
    ztomo_bins_dict['pz']=p_zphoto
    ztomo_bins_dict['z_bins']=ztomo_bins
    
    if use_window:
        ztomo_bins_dict=set_window_here(ztomo_bins_dict=ztomo_bins_dict,nside=nside, 
                                        mask = gal_mask, window_map_arr = gal_window_arr)
    return ztomo_bins_dict

def cmb_bins_here(zs=1090,l=None,use_window=True, nside=1024,zmax_cmb=1090, 
                  SN_file = None, cmb_window_map_arr = None): #unit_win=False):
    """
    This function prepares the cmb lensing map into format required for input into skylens for theory predictions.
    
    Parameters
    ----------
    zs : redshift of surface of last scattering (or source)
    l : numpy array of multipole range
    use_window = Flag to use the CMB window function
    nside = NSIDE of the window maps based on healpix formalism
    zmax_cmb = power spectrum of CMB lensing should be integrated up to this value; default is 1090; for AbacusSummit this is 2.45
    SN_file = file path to the CMB SNR 
    cmb_window_map_arr = numpy array containing file path to the CMB Window Map 
    
    Returns
    -------
    ztomo_bins_dict = Skylens dictionary object containing information on the CMB lensing bin
    """
    
    assert SN_file != None, "Provide CMB SN file in cmb_bins_here"
    #assert cmb_window_map_file != None, "Provide CMB window map in cmb_bins_here"
    if cmb_window_map_arr is not None:
        pass
    else:
        assert 1 != 1, "Provide CMB window map in cmb_bins_here"
    
    ztomo_bins_dict={}
    ztomo_bins_dict[0]={}

    ztomo_bins_dict=zbin_pz_norm(ztomo_bins_dict=ztomo_bins_dict,
                                 tomo_bin_indx=0,zbin_centre=np.atleast_1d(zs),
                                 p_zspec=np.atleast_1d(1), ndensity=np.array([0]), bg1=np.array([1]), bz1 = None)
    ztomo_bins_dict['n_bins']=1 #easy to remember the counts
    ztomo_bins_dict['zmax_cmb']=np.atleast_1d([zmax_cmb])
    ztomo_bins_dict['nz']=1

    SN_read=np.genfromtxt(SN_file, names=('l','nl','nl+cl'))
    SN_intp=interp1d(SN_read['l'], SN_read['nl'],bounds_error=False, fill_value=0)
    SN=SN_intp(l) 
    ztomo_bins_dict['SN']={}
    ztomo_bins_dict['SN']['kappa']=SN.reshape(len(SN),1,1)
    if use_window:
        ztomo_bins_dict=set_window_here(ztomo_bins_dict=ztomo_bins_dict,
                                   nside=nside, cmb=True, window_map_arr = cmb_window_map_arr)
    return ztomo_bins_dict

def DESI_elg_bins(ntomo_bins=1, nside=1024, use_window=True, bg1=None, bz1 = None,
                  l=None, mag_fact=0, ztomo_bins=None, dndz_arr = None, 
                  gal_mask = None, gal_window_arr = None):
    """
    Returns tomographic bin Skylens object for power spectrum measurement
    
    Parameters
    ----------
    ntomo_bins : number of tomographic bins
    nside : NSIDE of the window maps based on healpix formalism
    use_window : flag for using galaxy window function
    bg1 : (np.array) galaxy linear bias, len == ntomo_bins
    bz1 : (dict) redshift dependent galaxy linear bias, bz1.shape == (ntomo_bins, len(zphoto_bin_centre))
    #ndensity : (np.array) sample number density/arcmin^2, len == ntomo_bins 
    l : (np.array) array of multipole range
    mag_fact : (float) magnification bias
    ztomo_bins : (dict) object containing information on tomographic bins 
    dndz_arr : (np.array) array containing file paths to dndz 
    gal_mask (str) : file location for galaxy mask
    gal_window_arr : (np.array) array containing file paths to galaxy window functions
    
    Returns
    -------
    galaxy_bin_info : (dict) object containing tomographic information processed for Skylens input
    """
      
    if dndz_arr is not None:
        pass
    else: 
        assert 1 != 1, "Provide dndz_file argument in DESI_elg_bins"
    if gal_window_arr is not None:
        pass
    else: assert 1 != 1, "Provide galaxy window in DESI_elg_bins"
    
    #assert len(dndz_arr) == ntomo_bins, "Provide same number of dndz files as ntomo_bins"
    assert len(gal_window_arr) == ntomo_bins, "Provide same number of galaxy window files as ntomo_bins"
    
    if bg1 is not None:
        assert len(bg1) == ntomo_bins, "Provide same number of linear bias as tomographic bins"
    
    if((bg1 is not None) | (bz1 is not None)):
        pass
    else:
        print("Provide either bg1 or bz1")
        
    zphoto_bin_centre = {}
    p_zphoto = {}
    ndensity = {}
    
    for i in range(ntomo_bins):
        #print(i)
        #t = pd.read_csv(dndz_arr[i])
        #dz=t['Redshift_mid'][2]-t['Redshift_mid'][1]
        #zmax=max(t['Redshift_mid'])+dz/2
        #zmin=min(t['Redshift_mid'])-dz/2
        #z=t['Redshift_mid']
        #zphoto_bin_centre[i] = t['Redshift_mid']
        #pz=t['dndz']
        #p_zphoto[i] = t['pz']
        
        ##--VERSION WHERE DNDZ IS A DICT--##
        dz = dndz_arr['zrange'][2] - dndz_arr['zrange'][1]
        zmax = max(dndz_arr['zrange']) + dz/2
        zmin = min(dndz_arr['zrange']) - dz/2
        zphoto_bin_centre[i] = dndz_arr['zrange']
        p_zphoto[i] = dndz_arr['dndz']
        
        if bz1[i] is not None:
            assert len(bz1[i]) == len(zphoto_bin_centre[i]), "Provide proper redshift dependent bias"
        
        ns=np.sum(p_zphoto[i])
        d2r = 180/np.pi
        ns/=d2r**2 #convert from deg**2 to rd**2
        ndensity[i] = ns
        print(zmin,zmax,ztomo_bins,ns)
    
    if ztomo_bins is None: #this defines the bin edges if splitting the sample into bins. 
        #Preferably pass it as an argument when using multiple bins.
        ztomo_bins=np.linspace(zmin, min(2,zmax), ntomo_bins+1) #define based on experiment
    
    galaxy_bin_info = source_tomo_bins(zphoto_bin_centre=zphoto_bin_centre, p_zphoto=p_zphoto, 
                                       ndensity=ndensity, 
                                       ntomo_bins = ntomo_bins, mag_fact=mag_fact, 
                                       ztomo_bins=ztomo_bins,nside=nside, 
                                       use_window=use_window,bg1=bg1, bz1 = bz1, l=l, 
                                       gal_mask = gal_mask, gal_window_arr = gal_window_arr)
    return galaxy_bin_info