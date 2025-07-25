# -*- coding: utf-8 -*-
"""
Created on Thu July 28 20:30:11 2018

@author: danielgodinez
"""
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Tuple
from scipy.interpolate import UnivariateSpline

def create_noise(median: ArrayLike, rms: ArrayLike, degree: int = 3) -> UnivariateSpline:
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

    if not isinstance(median, np.ndarray):
        median = np.array(median)
    if not isinstance(rms, np.ndarray):
        rms = np.array(rms)

    order = np.array(median).argsort()
    median, rms = median[order], rms[order]

    if len(np.unique(median)) != len(median):
        indices = np.unique(median, return_index=True)[1]
        median, rms = median[indices], rms[indices]
        
    fn = UnivariateSpline(median, rms, k=degree)

    return fn

def add_noise(mag: ArrayLike, fn: Callable[[ArrayLike], np.ndarray], zp: float = 24, exptime: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Adds noise to magnitudes given a noise function. 

    Parameters
    ----------
    mag : array
        Magnitude to add noise to. 
    fn : function
        Spline fit, must be defined using the create_noise function. 
    zp : float
        Zeropoint of the instrument, default is 24.
    exptime : int
        The exposure time of the observations.
        
    Returns
    -------
    mag : array
        The noise-added magnitudes. 
    magerr : array
        The corresponding magnitude errors.
    """  

    flux = 10**(-(mag-zp)/2.5)*exptime

    interp = fn(mag)
    interp[(interp<0)]=0.00001

    delta_fobs = flux*interp*(np.log(10)/2.5)

    f_obs = np.random.normal(flux, delta_fobs)

    mag_obs = zp - 2.5*np.log10(f_obs/exptime)
    magerr = (2.5/np.log(10))*(delta_fobs/f_obs)
        
    return np.array(mag_obs), np.array(magerr)
 
def add_gaussian_noise(mag: ArrayLike, zp: float = 24, exptime: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Adds noise to lightcurve given the magnitudes.

    Parameters
    ----------
    mag : array
        Mag array to add noise to. 
    zp : float
        Zeropoint of the instrument, default is 24.
    exptime : int
        The exposure time of the observations.
    
    Returns
    -------
    noisy_mag : array
        The noise-added magnitude. 
    magerr : array
        The corresponding magnitude errors.
    """
    flux = 10**((mag-zp)/-2.5)*exptime
    
    noisy_flux= np.random.poisson(flux)
    magerr = 2.5/np.log(10)*np.sqrt(noisy_flux)/noisy_flux

    noisy_mag = zp - 2.5*np.log10(noisy_flux/exptime)
    magerr=np.array(magerr)
    mag = np.array(mag)

    return np.array(noisy_mag), np.array(magerr)
