import healpy as hp 
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


def radec2hpix(nside, ra, dec):
    """ 
    Function transforms RA,DEC to HEALPix index in ring ordering
    
    parameters
    ----------
    nside : int
    
    ra : array_like
        right ascention in deg
    
    dec : array_like
        declination in deg
    
    
    returns
    -------
    hpix : array_like
        HEALPix indices
    
    """
    hpix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    return hpix

def hpixsum(nside, ra, dec, weights=None):
    """
    Aggregates ra and dec onto HEALPix with nside and ring ordering.
    credit: Yu Feng, Ellie Kitanidis, ImagingLSS, UC Berkeley
    parameters
    ----------
    nside: int
    
    ra: array_like
        right ascention in degree.
    dec: array_like
        declination in degree.
    returns
    -------
    weight_hp: array_like
            
    """
    hpix = radec2hpix(nside, ra, dec)
    npix = hp.nside2npix(nside)
    weight_hp = np.bincount(hpix, weights=weights, minlength=npix)
    return weight_hp

# nanomaggie to magnitude converter 
def nmgy2mag(nmgy, ivar=None):
    """
    Name:
        nmgy2mag
    Purpose:
        Convert SDSS nanomaggies to a log10 magnitude.  Also convert
        the inverse variance to mag err if sent.  The basic formulat
        is 
            mag = 22.5-2.5*log_{10}(nanomaggies)
    Calling Sequence:
        mag = nmgy2mag(nmgy)
        mag,err = nmgy2mag(nmgy, ivar=ivar)
    Inputs:
        nmgy: SDSS nanomaggies.  The return value will have the same
            shape as this array.
    Keywords:
        ivar: The inverse variance.  Must have the same shape as nmgy.
            If ivar is sent, then a tuple (mag,err) is returned.
    Outputs:
        The magnitudes.  If ivar= is sent, then a tuple (mag,err)
        is returned.
    Notes:
        The nano-maggie values are clipped to be between 
            [0.001,1.e11]
        which corresponds to a mag range of 30 to -5
    """
    nmgy = np.array(nmgy, ndmin=1, copy=False)

    nmgy_clip = np.clip(nmgy,1e-5,1e11)

    mag = nmgy_clip.copy()
    mag[:] = 22.5-2.5*np.log10(nmgy_clip)

    if ivar is not None:

        ivar = np.array(ivar, ndmin=1, copy=False)
        if ivar.shape != nmgy.shape:
            raise ValueError("ivar must be same shape as input nmgy array")

        err = mag.copy()
        err[:] = np.inf

        w=np.where( ivar > 0 )

        if w[0].size > 0:
            err[w] = np.sqrt(1.0/ivar[w])

            a = 2.5/np.log(10)
            err[w] *= a/nmgy_clip[w]

        return mag, err
    else:
        return mag