# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 14:30:12 2017
    
    @author: danielgodinez
"""

from __future__ import print_function
from math import log 
from features import *

def extract_all(mag, magerr, convert=True, zp=24):
    """This function will compute the statistics used to train the RF.
    Amplitude dependent features are computed first, after which the
    mag/flux is normalized by the maximum value to compute the remanining
    features. By default a conversion from mag to flux is performed. If input
    is in flux or you wish to work in mag, set convert to False.  
        
    Parameters
    ----------
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    convert : boolean, optional 
        If False the features are computed with the inpute magnitudes,
        defaults to True to convert and compute in flux. 
    zp : float
        Zeropoint of the instrument, defaults to 24.
  
    Returns
    -------
    stats : array
        The 47 features to input into the RF, sorted 
        by alphabetical order. 
    """
    if convert is True:
        flux = 10**(-(mag-zp)/2.5)
        flux_err = (magerr*flux)/(2.5*log(10))
    elif convert is False:
        flux = mag
        flux_err = magerr
    
    amp = amplitude(flux)
    MedBuffRng = median_buffer_range(flux)
    MedBuffRng2 = median_buffer_range2(flux)
    # Normalize by max flux
    norm_flux = flux/np.max(flux)
    norm_fluxerr = flux_err*(norm_flux/flux)

    stetJ = stetsonJ(norm_flux,norm_fluxerr)
    stetK = stetsonK(norm_flux,norm_fluxerr)
    stetL = (stetJ*stetK) / 0.798
    shannon_entr = shannon_entropy(norm_flux,norm_fluxerr)

    stats = np.array((above1(norm_flux), above3(norm_flux), above5(norm_flux), abs_energy(norm_flux), abs_sum_changes(norm_flux), amp, 
        auto_corr(norm_flux), below1(norm_flux), below3(norm_flux), below5(norm_flux), c3(norm_flux), check_for_duplicate(norm_flux), check_for_max_duplicate(norm_flux), 
        check_for_min_duplicate(norm_flux), check_max_last_loc(norm_flux), check_min_last_loc(norm_flux), complexity(norm_flux), con(norm_flux), con2(norm_flux), 
        count_above(norm_flux), count_below(norm_flux), first_loc_max(norm_flux), first_loc_min(norm_flux), integrate(norm_flux), kurtosis(norm_flux), 
        longest_strike_above(norm_flux), longest_strike_below(norm_flux), mean_abs_change(norm_flux), mean_change(norm_flux), mean_second_derivative(norm_flux), 
        medianAbsDev(norm_flux), MedBuffRng, MedBuffRng2, peak_detection(norm_flux), ratio_recurring_points(norm_flux), 
        root_mean_squared(norm_flux), sample_entropy(norm_flux), shannon_entr, skewness(norm_flux), np.std(norm_flux), std_over_mean(norm_flux), stetJ, stetK, stetL, 
        sum_values(norm_flux), time_reversal_asymmetry(norm_flux), vonNeumannRatio(norm_flux)))
    
    stats[np.isinf(stats)] = 0

    return stats


