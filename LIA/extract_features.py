# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 14:30:12 2017
    
    @author: danielgodinez
"""

from __future__ import print_function
from math import log 
import numpy as np
from LIA.features import *

def extract_all(time, mag, magerr, convert=True, zp=24):
    """This function will compute the statistics used to train the RF.
    Amplitude dependent features are computed first, after which the
    mag/flux is normalized by the maximum value to compute the remanining
    features. By default a conversion from mag to flux is performed. If input
    is in flux or you wish to work in mag, set convert to False.  
        
    Parameters
    ----------
    time : array
        Time of observations
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    convert : boolean, optional 
        If False the features are computed with the input magnitudes,
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
        flux_err = (magerr*flux)/(2.5)*np.log(10)
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

    stats = np.array((above1(norm_flux,norm_fluxerr), above3(norm_flux,norm_fluxerr), above5(norm_flux,norm_fluxerr), 
        abs_energy(norm_flux), abs_sum_changes(norm_flux), amp, auto_corr(norm_flux), below1(norm_flux,norm_fluxerr), 
        below3(norm_flux,norm_fluxerr), below5(norm_flux,norm_fluxerr), c3(norm_flux), check_for_duplicate(norm_flux), 
        check_for_max_duplicate(norm_flux), check_for_min_duplicate(norm_flux), check_max_last_loc(norm_flux), 
        check_min_last_loc(norm_flux), complexity(norm_flux), con(norm_flux,norm_fluxerr), con2(norm_flux,norm_fluxerr), 
        count_above(norm_flux), count_below(norm_flux), first_loc_max(norm_flux), first_loc_min(norm_flux), integrate(norm_flux), 
        kurtosis(norm_flux), longest_strike_above(norm_flux), longest_strike_below(norm_flux), mean_abs_change(norm_flux), 
        mean_change(norm_flux), mean_second_derivative(norm_flux), medianAbsDev(norm_flux), MedBuffRng, MedBuffRng2, 
        peak_detection(norm_flux), ratio_recurring_points(norm_flux), root_mean_squared(norm_flux), sample_entropy(norm_flux), 
        shannon_entropy(norm_flux,norm_fluxerr), skewness(norm_flux), np.std(norm_flux), std_over_mean(norm_flux), stetJ, stetK, stetL, 
        sum_values(norm_flux), time_reversal_asymmetry(norm_flux), vonNeumannRatio(norm_flux), amplitude2(norm_flux), 
        median_distance(time,norm_flux), meanMag(norm_flux,norm_fluxerr), MaxSlope(time,norm_flux), LinearTrend(time,norm_flux), 
        PairSlopeTrend(norm_flux), FluxPercentileRatioMid20(norm_flux), FluxPercentileRatioMid35(norm_flux), FluxPercentileRatioMid50(norm_flux), 
        FluxPercentileRatioMid65(norm_flux), FluxPercentileRatioMid80(norm_flux), PercentAmplitude(norm_flux), PercentDifferenceFluxPercentile(norm_flux), 
        half_mag_amplitude_ratio(norm_flux), cusum(norm_flux), shapiro_wilk(norm_flux), AndersonDarling(norm_flux), Gskew(norm_flux), 
        benford_correlation(norm_flux), mean_n_abs_max(norm_flux), number_of_crossings(norm_flux), number_of_peaks(norm_flux), variance(norm_flux), 
        variance_larger_than_standard_deviation(norm_flux), variation_coefficient(norm_flux), large_standard_deviation(norm_flux), 
        symmetry_looking(norm_flux), index_mass_quantile(norm_flux), number_cwt_peaks(norm_flux), permutation_entropy(norm_flux), quantile(norm_flux))) #78 features


    #The following re-computes the metrics in derivative space

    flux = np.gradient(flux, time) #derivative
    flux_err = np.gradient(flux_err, time) #derivative 

    amp = amplitude(flux)
    MedBuffRng = median_buffer_range(flux)
    MedBuffRng2 = median_buffer_range2(flux)

    # Normalize by max flux
    norm_flux = flux/np.max(flux)
    norm_fluxerr = flux_err*(norm_flux/flux)
    
    stetJ = stetsonJ(norm_flux,norm_fluxerr)
    stetK = stetsonK(norm_flux,norm_fluxerr)
    stetL = (stetJ*stetK) / 0.798

    derivative_stats = np.array((above1(norm_flux,norm_fluxerr), above3(norm_flux,norm_fluxerr), above5(norm_flux,norm_fluxerr), 
        abs_energy(norm_flux), abs_sum_changes(norm_flux), amp, auto_corr(norm_flux), below1(norm_flux,norm_fluxerr), 
        below3(norm_flux,norm_fluxerr), below5(norm_flux,norm_fluxerr), c3(norm_flux), check_for_duplicate(norm_flux), 
        check_for_max_duplicate(norm_flux), check_for_min_duplicate(norm_flux), check_max_last_loc(norm_flux), 
        check_min_last_loc(norm_flux), complexity(norm_flux), con(norm_flux,norm_fluxerr), con2(norm_flux,norm_fluxerr), 
        count_above(norm_flux), count_below(norm_flux), first_loc_max(norm_flux), first_loc_min(norm_flux), integrate(norm_flux), 
        kurtosis(norm_flux), longest_strike_above(norm_flux), longest_strike_below(norm_flux), mean_abs_change(norm_flux), 
        mean_change(norm_flux), mean_second_derivative(norm_flux), medianAbsDev(norm_flux), MedBuffRng, MedBuffRng2, 
        peak_detection(norm_flux), ratio_recurring_points(norm_flux), root_mean_squared(norm_flux), sample_entropy(norm_flux), 
        shannon_entropy(norm_flux,norm_fluxerr), skewness(norm_flux), np.std(norm_flux), std_over_mean(norm_flux), stetJ, stetK, stetL, 
        sum_values(norm_flux), time_reversal_asymmetry(norm_flux), vonNeumannRatio(norm_flux), amplitude2(norm_flux), 
        median_distance(time,norm_flux), meanMag(norm_flux,norm_fluxerr), MaxSlope(time,norm_flux), LinearTrend(time,norm_flux), 
        PairSlopeTrend(norm_flux), FluxPercentileRatioMid20(norm_flux), FluxPercentileRatioMid35(norm_flux), FluxPercentileRatioMid50(norm_flux), 
        FluxPercentileRatioMid65(norm_flux), FluxPercentileRatioMid80(norm_flux), PercentAmplitude(norm_flux), PercentDifferenceFluxPercentile(norm_flux), 
        half_mag_amplitude_ratio(norm_flux), cusum(norm_flux), shapiro_wilk(norm_flux), AndersonDarling(norm_flux), Gskew(norm_flux), 
        benford_correlation(norm_flux), mean_n_abs_max(norm_flux), number_of_crossings(norm_flux), number_of_peaks(norm_flux), variance(norm_flux), 
        variance_larger_than_standard_deviation(norm_flux), variation_coefficient(norm_flux), large_standard_deviation(norm_flux), 
        symmetry_looking(norm_flux), index_mass_quantile(norm_flux), number_cwt_peaks(norm_flux), permutation_entropy(norm_flux), quantile(norm_flux))) #78 features


    stats = np.r_[stats, derivative_stats]
    
    stats[np.isinf(stats)] = 0
    stats[np.isnan(stats)] = 0
    
    return stats


