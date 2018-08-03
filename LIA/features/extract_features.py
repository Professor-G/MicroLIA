# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 14:30:12 2017
    
    @author: danielgodinez
"""

from __future__ import print_function
from math import log 
from features import *


def extract_all(mag, magerr, convert=True):
    """This function will compute the statistics used to train the RF.
    Amplitude dependent features are computed first, after which the
    mag/flux is normalized by the maximum value to compute the remanining
    features. By default a conversion from mag to flux is performed. If input
    is in flux, set convert to False.  
        
    Parameters
    ----------
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    convert : boolean, optional 
        If False the features are computed with the inpute magnitudes,
        defaults to True to convert and compute in flux. 
  
    Returns
    -------
    stats : array
        The 47 features to input into the RF, sorted 
        by alphabetical order. 
    """
    if convert is True:
        flux = 10**(-(mag)/2.5)
        flux_err = (magerr*flux)/(2.5*np.log(10))
    elif convert is False:
        flux = mag
        flux_err = magerr
    
    amp = amplitude(flux)
    MedBuffRng = median_buffer_range(flux)
    MedBuffRng2 = median_buffer_range2(flux)
    # Normalize by max flux
    flux = flux/np.max(flux)

    stetJ = stetsonJ(flux,flux_err)
    stetK = stetsonK(flux,flux_err)
    stetL = (stetJ*stetK) / 0.798
    shannon_entr = shannon_entropy(flux,flux_err)

    stats = np.array((above1(flux), above3(flux), above5(flux), abs_energy(flux), abs_sum_changes(flux), amp, 
        auto_corr(flux), below1(flux), below3(flux), below5(flux), c3(flux), check_for_duplicate(flux), check_for_max_duplicate(flux), 
        check_for_min_duplicate(flux), check_max_last_loc(flux), check_min_last_loc(flux), complexity(flux), con(flux), con2(flux), 
        count_above(flux), count_below(flux), first_loc_max(flux), first_loc_min(flux), integrate(flux), kurtosis(flux), 
        longest_strike_above(flux), longest_strike_below(flux), mean_abs_change(flux), mean_change(flux), mean_second_derivative(flux), 
        medianAbsDev(flux), MedBuffRng, MedBuffRng2, peak_detection(flux), ratio_recurring_points(flux), 
        root_mean_squared(flux), sample_entropy(flux), shannon_entr, skewness(flux), np.std(flux), std_over_mean(flux), stetJ, stetK, stetL, 
        sum_values(flux), time_reversal_asymmetry(flux), vonNeumannRatio(flux)))

    stats[np.isinf(stats)] = 0

    return stats


