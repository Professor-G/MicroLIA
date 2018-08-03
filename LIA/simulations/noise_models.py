# -*- coding: utf-8 -*-
"""
Created on Thu July 28 20:30:11 2018

@author: danielgodinez
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from math import log

def create_noise(median, rms, degree=3):
    """Creates a noise model by fitting a one-dimensional smoothing 
    spline of degree k.

    Parameters
    ----------
    median : array
        Baseline magnitudes.
    rms : array
        Corresponding RMS per baseline. 
    k : int
        Degree of the smoothing spline. Default is a 
        cubic spline of degree 3.

    Returns
    -------
    fn : The kth degree spline fit. 
    """
    f = UnivariateSpline(median, rms, w=None, k=degree)

    return f

def add_noise(mag, fn, zp=24):
    """Adds noise to magnitudes given a noise function. 

    Parameters
    ----------
    mag : array
        Magnitude to add noise to. 
    fn : function
        Spline fit, must be defined using the create_noise function. 
    zp : Zeropoint
        Zeropoint of the instrument, default is 24.
        
    Returns
    -------
    mag : array
        The noise-added magnitudes. 
    magerr : array
        The corresponding magnitude errors.
        
    """
    noisy_mag = []
    magerr = []

    for m in mag:
        zp = 23.4
        flux = 10**(-(m-zp)/2.5)
        delta_f = flux*fn(m)*(log(10)/2.5)
        
        f_obs = np.random.normal(flux, delta_f)
        delta_fobs = delta_f
        mag_obs = zp - 2.5*np.log10(f_obs)
        err_mag = (2.5/log(10))*(delta_fobs/f_obs)
        
        noisy_mag.append(mag_obs)
        magerr.append(err_mag)
        
    return np.array(noisy_mag), np.array(magerr)

def add_gaussian_noise(flux):
    """Adds noise to lightcurve given the magnitudes.
    Conversion to flux is performed and Gaussian noise is added.
    If input is in flux set convert to False. 

    Parameters
    ----------
    mag : array
        Mag array to add noise to. 
    zp : zeropoint
        Zeropoint of the instrument, default is 24.
    convert : boolean, optional 
    
    Returns
    -------
    noisy_mag : array
        The noise-added magnitude. 
    magerr : array
        The corresponding magnitude errors.
    """
    noisy_flux=[]
    flux_err = []

    for f in flux:  
        f_obs = np.random.normal(f, np.sqrt(f))
        noisy_flux.append(f_obs)
            
        flux_err.append(abs(f_obs - np.median(flux)))
        
    flux_err=np.array(flux_err)
    noisy_flux=np.array(noisy_flux)

    return np.array(noisy_flux), np.array(flux_err)





