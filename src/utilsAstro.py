import healpy as hp 
import numpy as np

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