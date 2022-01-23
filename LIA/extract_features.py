# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 14:30:12 2017
    
    @author: danielgodinez
"""
from __future__ import print_function
import numpy as np
from LIA import features
from inspect import getmembers, isfunction
<<<<<<< HEAD
=======

>>>>>>> 4e56799e39cb3591859b33bdf6c8b7c8ebdfe52b

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
        All features to use for classification
    """

    if convert is True:
        flux = 10**(-(mag-zp)/2.5)
        flux_err = (magerr*flux)/(2.5)*np.log(10)
    elif convert is False:
        flux = mag
        flux_err = magerr
<<<<<<< HEAD
=======

>>>>>>> 4e56799e39cb3591859b33bdf6c8b7c8ebdfe52b

    # Normalize by max flux
    norm_flux = flux/np.max(flux)
    norm_fluxerr = flux_err*(norm_flux/flux)



#    amp = amplitude(flux)
#    MedBuffRng = median_buffer_range(flux)
#    MedBuffRng2 = median_buffer_range2(flux)

#    
#    stetJ = stetsonJ(norm_flux,norm_fluxerr)
#    stetK = stetsonK(norm_flux,norm_fluxerr)
#    stetL = (stetJ*stetK) / 0.798

#    stats_normal = np.array((above1(norm_flux,norm_fluxerr), above3(norm_flux,norm_fluxerr), above5(norm_flux,norm_fluxerr), abs_energy(norm_flux), abs_sum_changes(norm_flux), amp, auto_corr(norm_flux),
#        below1(norm_flux,norm_fluxerr), below3(norm_flux,norm_fluxerr), c3(norm_flux), check_max_last_loc(norm_flux), complexity(norm_flux), count_above(norm_flux), count_below(norm_flux),
#        first_loc_max(norm_flux), integrate(norm_flux), kurtosis(norm_flux), longest_strike_above(norm_flux), longest_strike_below(norm_flux), mean_abs_change(norm_flux),
#        mean_change(norm_flux), mean_second_derivative(norm_flux), medianAbsDev(norm_flux), MedBuffRng, MedBuffRng2, peak_detection(norm_flux), ratio_recurring_points(norm_flux),
#        root_mean_squared(norm_flux), sample_entropy(norm_flux), shannon_entropy(norm_flux,norm_fluxerr), skewness(norm_flux), np.std(norm_flux), std_over_mean(norm_flux),
#        stetJ, stetK, stetL, sum_values(norm_flux), time_reversal_asymmetry(norm_flux), vonNeumannRatio(norm_flux), amplitude2(norm_flux), meanMag(norm_flux, norm_fluxerr), 
#        MaxSlope(time,norm_flux), LinearTrend(time, norm_flux), PairSlopeTrend(norm_flux), FluxPercentileRatioMid20(norm_flux), FluxPercentileRatioMid35(norm_flux), 
#        FluxPercentileRatioMid50(norm_flux), FluxPercentileRatioMid65(norm_flux), FluxPercentileRatioMid80(norm_flux), PercentAmplitude(norm_flux), PercentDifferenceFluxPercentile(norm_flux),
#        half_mag_amplitude_ratio(norm_flux), cusum(norm_flux), shapiro_wilk(norm_flux), AndersonDarling(norm_flux), Gskew(norm_flux), benford_correlation(norm_flux),
#        number_of_crossings(norm_flux), number_of_peaks(norm_flux), variance(norm_flux), variation_coefficient(norm_flux), index_mass_quantile(norm_flux),
#        number_cwt_peaks(norm_flux), permutation_entropy(norm_flux), quantile(norm_flux)))


#    #The following re-computes important metrics in derivative space
#    flux = np.gradient(flux, time)
#    flux_err = np.gradient(flux_err, time) 

#    amp = amplitude(flux)
#    MedBuffRng = median_buffer_range(flux)

#    # Normalize by max flux
#    norm_flux = flux/np.max(flux)
#    norm_fluxerr = flux_err*(norm_flux/flux)
#    
#    stetJ = stetsonJ(norm_flux,norm_fluxerr)
#    stetK = stetsonK(norm_flux,norm_fluxerr)
#    stetL = (stetJ*stetK) / 0.798

#    stats_derivative = np.array((amp, longest_strike_above(norm_flux), longest_strike_below(norm_flux), medianAbsDev(norm_flux), MedBuffRng, root_mean_squared(norm_flux),
#        sample_entropy(norm_flux), shannon_entropy(norm_flux,norm_fluxerr), stetJ, stetK, stetL, FluxPercentileRatioMid20(norm_flux), FluxPercentileRatioMid35(norm_flux),
#        FluxPercentileRatioMid50(norm_flux), FluxPercentileRatioMid65(norm_flux), shapiro_wilk(norm_flux), quantile(norm_flux))) #17

#    stats = np.r_[stats_normal, stats_derivative]
#    #import pdb; pdb.set_trace()
#    stats[np.isinf(stats)] = 0
#    stats[np.isnan(stats)] = 0
#    #stats[stats<10**-6] = 10**-6
#    #stats[stats>10**6] = 10**6
    
<<<<<<< HEAD
    all_features_functions = getmembers(features, isfunction)


    stats = []
    #normal space

    for func in all_features_functions:
    
        try:
            if func[0] == 'amplitude' or func[0] == 'median_buffer_range':
                feature = func[1](time, flux, flux_err)  #amplitude dependent features use non-normalized flux
            else:
                feature = func[1](time, norm_flux, norm_fluxerr)

            if isinstance(feature, float) or isinstance(feature, int):
                stats.append(feature)
            else:
                stats.append(np.NaN)

        except:
            stats.append(np.NaN)
=======

    all_features_functions = getmembers(features, isfunction)


    stats = []
    #normal space

    for func in all_features_functions:
    
        try:
            thefeat = func[1](time,norm_flux,norm_fluxerr)
            if isinstance(thefeat, float) or isinstance(thefeat, int):
                stats.append(thefeat)
            else:
                pass
        except:
            #print(func[0]+' bugged')    
            pass
>>>>>>> 4e56799e39cb3591859b33bdf6c8b7c8ebdfe52b
            
    #derivative space
    flux_deriv = np.gradient(flux, time)
    flux_deriv_err = np.gradient(flux_err, time) 
    
<<<<<<< HEAD
    norm_flux_deriv = flux_deriv/np.max(flux_deriv)
    norm_flux_deriv_err = flux_deriv_err*(norm_flux_deriv/flux_deriv)

    for func in all_features_functions:

        try:
            if func[0] == 'amplitude' or func[0] == 'median_buffer_range':
                feature = func[1](time, flux, flux_err)  #amplitude dependent features use non-normalized flux
            else:
                feature = func[1](time, norm_flux, norm_fluxerr)
=======

    norm_flux_deriv = flux_deriv/np.max(flux_deriv)
    norm_flux_deriv_err = flux_deriv_err*(norm_flux_deriv/flux_deriv)

    for func in all_features_functions:
    
        try:
            thefeat = func[1](time,norm_flux_deriv,norm_flux_deriv_err)
            if isinstance(thefeat, float) or isinstance(thefeat, int):
                stats.append(thefeat)
            else:
                pass
        except:
            #print(func[0]+' bugged')    
            pass
    stats = np.array(stats)
    stats[np.isinf(stats)] = 0
    stats[np.isnan(stats)] = 0
    stats[stats<10**-6] = 10**-6
    stats[stats>10**6] = 10**6

    return stats
>>>>>>> 4e56799e39cb3591859b33bdf6c8b7c8ebdfe52b

            if isinstance(feature, float) or isinstance(feature, int):
                stats.append(feature)
            else:
                stats.append(np.NaN)

        except:
            stats.append(np.NaN)
            
    stats = np.array(stats)
    stats[np.isnan(stats)] = 0
    stats[np.isinf(stats)] = 1e6
    stats[stats<1e-6] = 1e-6
    stats[stats>1e6] = 1e6

    return stats


