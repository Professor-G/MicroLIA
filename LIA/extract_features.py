# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 14:30:12 2017
    
    @author: danielgodinez
"""
from __future__ import print_function
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
        The 82 features to use for classification
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

    stats_normal = np.array((above1(norm_flux), above3(norm_flux), above5(norm_flux), abs_energy(norm_flux), abs_sum_changes(norm_flux), amp, auto_corr(norm_flux),
        below1(norm_flux), below3(norm_flux), c3(norm_flux), check_max_last_loc(norm_flux), complexity(norm_flux), count_above(norm_flux), count_below(norm_flux),
        first_loc_max(norm_flux), integrate(norm_flux), kurtosis(norm_flux), longest_strike_above(norm_flux), longest_strike_below(norm_flux), mean_abs_change(norm_flux),
        mean_change(norm_flux), mean_second_derivative(norm_flux), medianAbsDev(norm_flux), MedBuffRng, MedBuffRng2, peak_detection(norm_flux), ratio_recurring_points(norm_flux),
        root_mean_squared(norm_flux), sample_entropy(norm_flux), shannon_entropy(norm_flux,norm_fluxerr), skewness(norm_flux), np.std(norm_flux), std_over_mean(norm_flux),
        stetJ, stetK, stetL, sum_values(norm_flux), time_reversal_asymmetry(norm_flux), vonNeumannRatio(norm_flux), amplitude2(norm_flux), meanMag(norm_flux, norm_fluxerr), 
        MaxSlope(time,norm_flux), LinearTrend(time, norm_flux), PairSlopeTrend(norm_flux), FluxPercentileRatioMid20(norm_flux), FluxPercentileRatioMid35(norm_flux), 
        FluxPercentileRatioMid50(norm_flux), FluxPercentileRatioMid65(norm_flux), FluxPercentileRatioMid80(norm_flux), PercentAmplitude(norm_flux), PercentDifferenceFluxPercentile(norm_flux),
        half_mag_amplitude_ratio(norm_flux), cusum(norm_flux), shapiro_wilk(norm_flux), AndersonDarling(norm_flux), Gskew(norm_flux), benford_correlation(norm_flux),
        number_of_crossings(norm_flux), number_of_peaks(norm_flux), variance(norm_flux), variation_coefficient(norm_flux), index_mass_quantile(norm_flux),
        number_cwt_peaks(norm_flux), permutation_entropy(norm_flux), quantile(norm_flux)))


    #The following re-computes important metrics in derivative space
    flux = np.gradient(flux, time)
    flux_err = np.gradient(flux_err, time) 

    amp = amplitude(flux)
    MedBuffRng = median_buffer_range(flux)
    MedBuffRng2 = median_buffer_range2(flux)

    # Normalize by max flux
    norm_flux = flux/np.max(flux)
    norm_fluxerr = flux_err*(norm_flux/flux)
    
    stetJ = stetsonJ(norm_flux,norm_fluxerr)
    stetK = stetsonK(norm_flux,norm_fluxerr)
    stetL = (stetJ*stetK) / 0.798

    stats_derivative = np.array((amp, longest_strike_above(norm_flux), longest_strike_below(norm_flux), medianAbsDev(norm_flux), MedBuffRng, root_mean_squared(norm_flux),
        sample_entropy(norm_flux), shannon_entropy(norm_flux,norm_fluxerr), stetJ, stetK, stetL, FluxPercentileRatioMid20(norm_flux), FluxPercentileRatioMid35(norm_flux),
        FluxPercentileRatioMid50(norm_flux), FluxPercentileRatioMid65(norm_flux), shapiro_wilk(norm_flux), quantile(norm_flux))) #17

    stats = np.r_[stats_normal, stats_derivative]
    
    stats[np.isinf(stats)] = 0
    stats[np.isnan(stats)] = 0
    
    return stats

"""
stats = np.array((above1(norm_flux), above3(norm_flux), above5(norm_flux), abs_energy(norm_flux), abs_sum_changes(norm_flux), amp, 
        auto_corr(norm_flux), below1(norm_flux), below3(norm_flux), below5(norm_flux), c3(norm_flux), check_for_duplicate(norm_flux), check_for_max_duplicate(norm_flux), 
        check_for_min_duplicate(norm_flux), check_max_last_loc(norm_flux), check_min_last_loc(norm_flux), complexity(norm_flux), con(norm_flux), con2(norm_flux), 
        count_above(norm_flux), count_below(norm_flux), first_loc_max(norm_flux), first_loc_min(norm_flux), integrate(norm_flux), kurtosis(norm_flux), 
        longest_strike_above(norm_flux), longest_strike_below(norm_flux), mean_abs_change(norm_flux), mean_change(norm_flux), mean_second_derivative(norm_flux), 
        medianAbsDev(norm_flux), MedBuffRng, MedBuffRng2, peak_detection(norm_flux), ratio_recurring_points(norm_flux), 
        root_mean_squared(norm_flux), sample_entropy(norm_flux), shannon_entropy(norm_flux,norm_fluxerr), skewness(norm_flux), np.std(norm_flux), std_over_mean(norm_flux), stetJ, stetK, stetL, 
        sum_values(norm_flux), time_reversal_asymmetry(norm_flux), vonNeumannRatio(norm_flux)))

derivative_stats = np.array((above1(norm_flux), above3(norm_flux), above5(norm_flux), 
        abs_energy(norm_flux), abs_sum_changes(norm_flux), amp, auto_corr(norm_flux), below1(norm_flux), 
        below3(norm_flux), below5(norm_flux), c3(norm_flux), check_for_duplicate(norm_flux), 
        check_for_max_duplicate(norm_flux), check_for_min_duplicate(norm_flux), check_max_last_loc(norm_flux), 
        check_min_last_loc(norm_flux), complexity(norm_flux), con(norm_flux), con2(norm_flux), 
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

stat_names = ['above1', 'above3', 'above5', 'abs_energy', 'abs_sum_changes', 'amplitude', 'auto_corr', 'below1', 'below3', 'below5',
        'c3', 'check_for_duplicate', 'check_for_max_duplicate', 'check_for_min_duplicate', 'check_max_last_loc', 'check_min_last_loc',
        'complexity', 'con', 'con2', 'count_above', 'count_below', 'first_loc_max', 'first_loc_min', 'integrate', 'kurtosis',
        'longest_strike_above', 'longest_strike_below', 'mean_abs_change', 'mean_change', 'mean_second_derivative', 'medianAbsDev',
        'median_buffer_range', 'median_buffer_range2', 'peak_detection', 'ratio_recurring_points', 'root_mean_squared', 'sample_entropy',
        'shannon_entropy', 'skewness', 'std', 'std_over_mean', 'stetJ', 'stetK', 'stetL', 'sum_values', 'time_reversal_asymmetry',
        'vonNeumannRatio', 'amplitude2', 'median_distance', 'meanMag', 'MaxSlope', 'LinearTrend', 'PairSlopeTrend', 'FluxPercentileRatioMid20',
        'FluxPercentileRatioMid35', 'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65', 'FluxPercentileRatioMid80', 'PercentAmplitude',
        'PercentDifferenceFluxPercentile', 'half_mag_amplitude_ratio', 'cusum', 'shapiro_wilk', 'AndersonDarling', 'Gskew',
        'benford_correlation', 'mean_n_abs_max', 'number_of_crossings', 'number_of_peaks', 'variance', 'variance_larger_than_standard_deviation',
        'variation_coefficient', 'large_standard_deviation', 'symmetry_looking', 'index_mass_quantile', 'number_cwt_peaks','permutation_entropy','quantile',
        'above1_df', 'above3_df', 'above5_df', 'abs_energy_df', 'abs_sum_changes_df', 'amplitude_df', 'auto_corr_df', 'below1_df', 'below3_df', 'below5_df',
        'c3_df', 'check_for_duplicate_df', 'check_for_max_duplicate_df', 'check_for_min_duplicate_df', 'check_max_last_loc_df', 'check_min_last_loc_df',
        'complexity_df', 'con_df', 'con2_df', 'count_above_df', 'count_below_df', 'first_loc_max_df', 'first_loc_min_df', 'integrate_df', 'kurtosis_df',
        'longest_strike_above_df', 'longest_strike_below_df', 'mean_abs_change_df', 'mean_change_df', 'mean_second_derivative_df', 'medianAbsDev_df',
        'median_buffer_range_df', 'median_buffer_range2_df', 'peak_detection_df', 'ratio_recurring_points_df', 'root_mean_squared_df', 'sample_entropy_df',
        'shannon_entropy_df', 'skewness_df', 'std_df', 'std_over_mean_df', 'stetJ_df', 'stetK_df', 'stetL_df', 'sum_values_df', 'time_reversal_asymmetry_df',
        'vonNeumannRatio_df', 'amplitude2_df', 'median_distance_df', 'meanMag_df', 'MaxSlope_df', 'LinearTrend_df', 'PairSlopeTrend_df', 'FluxPercentileRatioMid20_df',
        'FluxPercentileRatioMid35_df', 'FluxPercentileRatioMid50_df', 'FluxPercentileRatioMid65_df', 'FluxPercentileRatioMid80_df', 'PercentAmplitude_df',
        'PercentDifferenceFluxPercentile_df', 'half_mag_amplitude_ratio_df', 'cusum_df', 'shapiro_wilk_df', 'AndersonDarling_df', 'Gskew_df',
        'benford_correlation_df', 'mean_n_abs_max_df', 'number_of_crossings_df', 'number_of_peaks_df', 'variance_df', 'variance_larger_than_standard_deviation_df',
        'variation_coefficient_df', 'large_standard_deviation_df', 'symmetry_looking_df', 'index_mass_quantile_df', 'number_cwt_peaks_df', 'permutation_entropy_df', 'quantile_df']

"""
