# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 14:30:12 2017
    
    @author: danielgodinez
"""
from __future__ import print_function
import numpy as np
import itertools
import math
import peakutils
import scipy.integrate as sintegrate
import scipy.signal as ssignal
import scipy.stats as sstats

import warnings; warnings.filterwarnings("ignore")

def shannon_entropy(time, mag, magerr, apply_weights=True):
    """
    Shannon entropy (Shannon et al. 1949) is used as a metric to quantify the amount of
    information carried by a signal. The procedure employed here follows that outlined by
    (D. Mislis et al. 2015). The probability of each point is given by a Cumulative Distribution
    Function (CDF). Following the same procedure as (D. Mislis et al. 2015), this function employs
    both the normal and inversed gaussian CDF, with the total shannon entropy given by a combination of
    the two. See: (SIDRA: a blind algorithm for signal detection in photometric surveys, D. Mislis et al., 2015)
    
    Does not incorporate errors.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------  
    float
        The Shannon Entropy of the lightcurve.
    """
    
    mean = np.median(mag)
    RMS = root_mean_squared(time, mag, magerr, apply_weights=False)
    
    p_list1 = []
    p_list2 = []
    inv_list1 = []
    inv_list2 = []
    
    t = range(0, len(mag))
    d_delta = [i*2.0 for i in magerr]
    
    """Error fn definition: http://mathworld.wolfram.com/Erf.html"""
    def errfn(x):
        def integral(t):
            integrand =  (2./np.sqrt(np.pi))*np.e**(-t**2)
            return integrand
        integ, err = sintegrate.quad(integral, 0, x)
        return integ
    
    """The Gaussian CDF: http://mathworld.wolfram.com/NormalDistribution.html"""
    def normal_gauss(x):
        return 0.5*(1. + errfn(x))
    
    """Inverse Gaussian CDF: http://mathworld.wolfram.com/InverseGaussianDistribution.html"""
    def inv_gauss(x, y):
        return 0.5*(1. + errfn(x)) + (0.5*np.e**((2.*RMS)/mean))*(1. - errfn(y))
    
    for i in t:
        val = normal_gauss((mag[i] + magerr[i] - mean)/(RMS*np.sqrt(2)))
        p_list1.append(val if val >0 else 1)
        
        val2 = normal_gauss((mag[i] - magerr[i] - mean)/(RMS*np.sqrt(2)))
        p_list2.append(val2 if val2 >0 else 1)
        
        val3 = inv_gauss(np.sqrt(RMS/(2.*(mag[i] + magerr[i])))*(((mag[i] + magerr[i])/mean) - 1.), np.sqrt(RMS/(2.*(mag[i] + magerr[i])))*(((mag[i] + magerr[i])/mean) + 1.))
        inv_list1.append(val3 if val3 >0 else 1)
        
        val4 = inv_gauss(np.sqrt(RMS/(2.*(mag[i] - magerr[i])))*(((mag[i] - magerr[i])/mean) - 1.), np.sqrt(RMS/(2.*(mag[i] - magerr[i])))*(((mag[i] - magerr[i])/mean) + 1.))
        inv_list2.append(val4 if val4 >0 else 1)
    
    entropy1 = -sum(np.log2(p_list1)*d_delta + np.log2(p_list2)*d_delta)
    entropy2 = -sum(np.log2(inv_list1)*d_delta + np.log2(inv_list2)*d_delta)

    total_entropy = np.nan_to_num(entropy1 + entropy2)

    return total_entropy

def con(time, mag, magerr, apply_weights=True):
    """
    Con is defined as the number of clusters containing three or more
    consecutive observations with magnitudes brighter than the reference
    magnitude plus 3 standard deviations. For a microlensing event Con = 1,
    assuming a  flat lightcurve prior to the event. The magnitude measurements
    are split into bins such that the reference  magnitude is defined as the mean
    of the measurements in the largest bin.

    In this updated version of the con function, the upper and lower bounds for each 
    measurement are defined as mag[i] + 3*magerr[i] and mag[i] - 3*magerr[i], respectively. 
    These bounds are then used to check if a measurement is within a cluster. 
    If a measurement is outside the bounds and we're in a cluster, the cluster is ended.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of clusters satisfying the conditions to the total number of measurements.     
    """

    if len(mag) < 3:
        return 0

    # Find the median of the magnitudes
    mean = np.median(mag)

    # Initialize variables
    con = 0
    deviating = False

    if apply_weights:
        # Loop over the magnitudes
        for i in range(len(mag)-2):
            
            # Define the upper and lower bounds for each measurement
            upper_bound = mag[i] + 3*magerr[i]
            lower_bound = mag[i] - 3*magerr[i]
            
            # Check if the current measurement is within the bounds
            if (mag[i] <= upper_bound and mag[i] >= lower_bound and
                mag[i+1] <= upper_bound and mag[i+1] >= lower_bound and
                mag[i+2] <= upper_bound and mag[i+2] >= lower_bound):
                
                # If the current measurement is within the bounds and we're not
                # already in a cluster, start a new cluster
                if (not deviating):
                    con += 1
                    deviating = True
                
                # If the current measurement is within the bounds and we're already
                # in a cluster, do nothing
                elif deviating:
                    pass
            
            # If the current measurement is outside the bounds and we're in a
            # cluster, end the cluster
            elif deviating:
                deviating = False
    else:
        for i in range(len(mag)-2):
            first = mag[i]
            second = mag[i+1]
            third = mag[i+2]
            if (first <= mean+3*magerr[i] and
                second <= mean+3*magerr[i+1] and
                third <= mean+3*magerr[i+2]):
                if (not deviating):
                    con += 1
                    deviating = True
                elif deviating:
                    deviating = False

    return con/len(mag)

def kurtosis(time, mag, magerr, apply_weights=True):
    """"
    This function returns the calculated kurtosis of the lightcurve.
    It's a measure of the peakedness (or flatness) of the lightcurve relative
    to a normal distribution. See: www.xycoon.com/peakedness_small_sample_test_1.htm
    
    This updated implementation calculates the weighted mean x_mean and the weighted 
    standard deviation sigma using numpy.average() with the weights parameter set to 1/magerr**2. 
    Then it calculates the weighted kurtosis using the above formula and returns the result.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The calculated kurtosis of the lightcurve.
    """

    if apply_weights:
        x_mean = np.average(mag, weights=1/magerr**2)
        sigma = np.sqrt(np.average((mag-x_mean)**2, weights=1/magerr**2))
        kurtosis = np.sum((mag-x_mean)**4 * 1/magerr**2) / (np.sum(1/magerr**2) * sigma**4) - 3
    else:
        kurtosis = sstats.kurtosis(mag)

    return kurtosis

def skewness(time, mag, magerr, apply_weights=True):
    """
    Skewness measures the asymmetry of a lightcurve, with a positive skewness
    indicating a skew to the right, and a negative skewness indicating a skew to the left.
    
    This function calculates the weighted mean and standard deviation using the photometric 
    errors as weights, and then uses these values to compute the weighted skewness.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The calculated skewness of the lightcurve.
    """
    
    if apply_weights:
        # Calculate the weighted mean and standard deviation
        wmean = np.average(mag, weights=1/magerr**2)
        wstd = np.sqrt(np.sum((mag - wmean)**2 / magerr**2) / np.sum(1/magerr**2))
        skewness = np.sum(((mag - wmean) / wstd)**3 * 1/magerr**2) / np.sum(1/magerr**2)
    else:
        skewness = sstats.skew(mag)

    return skewness

def vonNeumannRatio(time, mag, magerr, apply_weights=True):
    """
    The von Neumann ratio η was defined in 1941 by John von Neumann and serves as the
    mean square successive difference divided by the sample variance. When this ratio is small,
    it is an indication of a strong positive correlation between the successive photometric 
    data points. See: (J. Von Neumann, The Annals of Mathematical Statistics 12, 367 (1941))
    
    In this updated version, np.average() is used to calculate the weighted average of the measurement 
    errors squared as the sample variance. The weights argument in np.average() is used to specify 
    the weights for each element in the input array, with larger weights given to elements with smaller errors. 
    We also modify the calculation of delta to take into account the measurement errors by dividing the 
    differences between successive magnitudes by the corresponding errors.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The calculated von Neumann Ratio of the lightcurve.
    """
    
    if apply_weights:
        n = float(len(mag))
        delta = sum(((mag[1:] - mag[:-1]) / magerr[:-1])**2 / (n-1.))
        sample_variance = np.average(magerr**2, weights=1/magerr**2)
        vNR = delta / sample_variance
    else:
        n = float(len(mag))
        delta = sum((mag[1:] - mag[:-1])**2 / (n-1.))
        sample_variance = np.std(mag)**2
        vNR = delta / sample_variance

    return vNR

def stetsonJ(time, mag, magerr, apply_weights=True):
    """
    The variability index J was first suggested by Peter B. Stetson and serves as a
    measure of the correlation between the data points, tending to 0 for variable stars
    and getting large as the difference between the successive data points increases.
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
    
    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The calculated stetsonJ variability index of the lightcurve.
    """
    
    n = float(len(mag))
    mean = np.median(mag)
    delta_list=[]
    
    for i in range(0, len(mag)-1):
        delta = np.sqrt(n/(n-1.))*((mag[i] - mean)/magerr[i])
        delta2 = np.sqrt(n/(n-1.))*((mag[i+1] - mean)/magerr[i+1])
        delta_list.append(np.nan_to_num(delta*delta2))
    
    stetj = sum(np.sign(delta_list)*np.sqrt(np.abs(delta_list)))

    return stetj

def stetsonK(time, mag, magerr, apply_weights=True):
    """
    The variability index K was first suggested by Peter B. Stetson and serves as a
    measure of the kurtosis of the magnitude distribution.
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The calculated stetsonK variability index of the lightcurve.
    """  
    
    n = float(len(mag))
    mean = np.median(mag)
    delta = np.sqrt((n/(n-1.)))*((mag - mean)/magerr)
    
    stetsonK = ((1./n)*sum(abs(delta)))/(np.sqrt((1./n)*sum(delta**2)))

    return np.nan_to_num(stetsonK)

def stetsonL(time, mag, magerr, apply_weights=True):
    """
    The variability index L was first suggested by Peter B. Stetson and serves as a
    means of distinguishing between different types of variation. When individual random
    errors dominate over the actual variation of the signal, K approaches 0.798 (Gaussian limit).
    Thus, when the nature of the errors is Gaussian, stetsonL = stetsonJ, except it will be amplified
    by a small factor for smoothly varying signals, or suppressed by a large factor when data
    is infrequent or corrupt. 
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
        
    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The calculated stetsonL variability index of the lightcurve.   
    """  
    
    stetL = (stetsonJ(time, mag, magerr, apply_weights=apply_weights)*stetsonK(time, mag, magerr, apply_weights=apply_weights)) / 0.798

    return stetL

def median_buffer_range(time, mag, magerr, apply_weights=True):
    """
    This function returns the ratio of points that are between plus or minus 10% of the
    amplitude value over the mean.

    In this updated version, we compute the weighted mean of the mag array using the 
    corresponding magerr values as weights. Then we use the weighted mean to compute a and b values 
    for the range around the mean that we want to consider. Finally, we compute the ratio of points 
    within this range to the total number of points.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of points that are between plus or minus 10% of the lightcurve's amplitude value over the mean.
    """
    
    amp = amplitude(time, mag, magerr, apply_weights=apply_weights)

    if apply_weights:
        w_mean = np.average(mag, weights=1/magerr**2)
        a = w_mean - amp*0.1
        b = w_mean + amp*0.1
    else:
        #mean = meanMag(mag, magerr)
        mean = np.median(mag)
        a = mean - amp*0.1
        b = mean + amp*0.1
        
    return len(np.argwhere((mag > a) & (mag < b))) / float(len(mag))

def std_over_mean(time, mag, magerr, apply_weights=True):
    """
    A measure of the ratio of standard deviation and mean.
    
    In this version, weights is calculated as the inverse square 
    of magerr. The weighted_mean is calculated as the weighted average 
    of mag, where the weights are given by weights. weighted_var is the 
    weighted variance, and weighted_std is the square root of weighted_var. 
    The final line returns the ratio of weighted_std and weighted_mean.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of standard deviation to the lightcurve's mean.
    """

    if apply_weights:
        weights = 1.0 / (magerr ** 2)
        mean = np.sum(mag * weights) / np.sum(weights)
        weighted_var = np.sum(weights * (mag - mean) ** 2) / np.sum(weights)
        std = np.sqrt(weighted_var)
    else:
        std, mean = np.std(mag), np.median(mag)

    return std / mean

def amplitude(time, mag, magerr, apply_weights=True):
    """
    This amplitude metric is defined as the difference between the maximum magnitude
    measurement and the lowest magnitude measurement, divided by 2. We account for outliers by
    removing the upper and lower 2% of magnitudes.
    
    In this updated implementation we first sort the magnitude and error arrays based on the magnitude values. 
    We then compute the median magnitude value after excluding the upper and lower 2% of magnitudes to account 
    for outliers. We compute both the standard amplitude and the weighted amplitude, where each magnitude measurement 
    is weighed by its corresponding error. The weighted amplitude is then returned by the function.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.
    
    Returns:
    -------
    float
        The calculated amplitude of the lightcurve.
    """
    
    if apply_weights:
        sorted_indices = np.argsort(mag)
        sorted_mag = mag[sorted_indices]
        sorted_magerr = magerr[sorted_indices]
        n = len(mag)
        lower_bound = int(n*0.02)
        upper_bound = int(n*(1-0.02))
        mag_median = np.median(sorted_mag[lower_bound:upper_bound])
        amplitude = (np.max(sorted_mag[lower_bound:upper_bound]) - np.min(sorted_mag[lower_bound:upper_bound])) / 2.0
        amp = np.sum(np.abs(sorted_mag[lower_bound:upper_bound] - mag_median) * sorted_magerr[lower_bound:upper_bound]) / np.sum(sorted_magerr[lower_bound:upper_bound])
    else:
        amp = (np.percentile(mag, 98) - np.percentile(mag, 2)) / 2.0

    return amp

def median_distance(time, mag, magerr, apply_weights=True):
    """
    This function calculates the median Euclidean distance between each photometric
    measurement, a helpful metric for detecting overlapped lightcurves.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The calculated median Euclidean distance between each lightcurve measurement.
    """

    if apply_weights:
        delta_mag = (mag[1:] - mag[:-1])**2
        delta_t = (time[1:] - time[:-1])**2
        delta_magerr = (magerr[1:]**2 + magerr[:-1]**2)
        return np.median(np.sqrt(delta_mag/delta_magerr + delta_t/delta_magerr))
    else:
        delta_mag = (mag[1:] - mag[:-1])**2
        delta_t = (time[1:] - time[:-1])**2      
        return np.median(np.sqrt(delta_mag + delta_t))   
    
def above1(time, mag, magerr, apply_weights=True):
    """
    This function measures the ratio of data points that are above 1 standard deviation
    from the median magnitude, weighted by their errors.
    
    In this updated function, each data point is weighed according to its error. The weighted 
    ratio of points that are above 1 standard deviation from the median magnitude is returned.
    By weighting each data point according to its error, we are taking into account the fact 
    that more weight should be given to data points that have lower measurement uncertainties. 

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the light curve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The number of points above 1 standard deviation from the light curve's median measurement.
    """
    median_mag = np.median(mag)
    std = np.std(mag)
    
    if apply_weights:
        weighted_above1 = np.mean((mag - median_mag) > std)
        weighted_above1 /= np.mean((mag - median_mag + std) / magerr**2)
        return weighted_above1
    else:
        above1 = np.mean((mag - median_mag) > std)
        return above1

def above3(time, mag, magerr, apply_weights=True):
    """
    This function measures the ratio of data points that are above 3 standard deviations
    from the median magnitude, weighted by their errors.
    
    In this updated function, each data point is weighed according to its error. The weighted 
    ratio of points that are above 3 standard deviations from the median magnitude is returned.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the light curve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The number of points above 3 standard deviations from the light curve's median measurement.
    """
    median_mag = np.median(mag)
    std = np.std(mag)
    
    if apply_weights:
        weighted_above3 = np.mean((mag - median_mag) > 3 * std)
        weighted_above3 /= np.mean((mag - median_mag + 3 * std) / magerr**2)
        return weighted_above3
    else:
        above3 = np.mean((mag - median_mag) > 3 * std)
        return above3

def above5(time, mag, magerr, apply_weights=True):
    """
    This function measures the ratio of data points that are above 5 standard deviations
    from the median magnitude, weighted by their errors.
    
    In this updated function, each data point is weighed according to its error. The weighted 
    ratio of points that are above 5 standard deviations from the median magnitude is returned.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the light curve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The number of points above 5 standard deviations from the light curve's median measurement.
    """
    median_mag = np.median(mag)
    std = np.std(mag)
    
    if apply_weights:
        weighted_above5 = np.mean((mag - median_mag) > 5 * std)
        weighted_above5 /= np.mean((mag - median_mag + 5 * std) / magerr**2)
        return weighted_above5
    else:
        above5 = np.mean((mag - median_mag) > 5 * std)
        return above5

def below1(time, mag, magerr, apply_weights=True):
    """
    This function measures the ratio of data points that are below 1 standard deviation
    from the median magnitude, weighted by their errors.
    
    In this updated function, each data point is weighed according to its error. The weighted 
    ratio of points that are below 1 standard deviation from the median magnitude is returned.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the light curve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The number of points below 1 standard deviation from the light curve's median measurement.
    """
    median_mag = np.median(mag)
    std = np.std(mag)
    
    if apply_weights:
        weighted_below1 = np.mean((-mag + median_mag) > std)
        weighted_below1 /= np.mean((-mag + median_mag + std) / magerr**2)
        return weighted_below1
    else:
        below1 = np.mean((-mag + median_mag) > std)
        return below1

def below3(time, mag, magerr, apply_weights=True):
    """
    This function measures the ratio of data points that are below 3 standard deviations
    from the median magnitude, weighted by their errors.
    
    In this updated function, each data point is weighed according to its error. The weighted 
    ratio of points that are below 3 standard deviations from the median magnitude is returned.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the light curve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The number of points below 3 standard deviations from the light curve's median measurement.
    """
    median_mag = np.median(mag)
    std = np.std(mag)
    
    if apply_weights:
        weighted_below3 = np.mean((-mag + median_mag) > 3 * std)
        weighted_below3 /= np.mean((-mag + median_mag + 3 * std) / magerr**2)
        return weighted_below3
    else:
        below3 = np.mean((-mag + median_mag) > 3 * std)
        return below3

def below5(time, mag, magerr, apply_weights=True):
    """
    This function measures the ratio of data points that are below 5 standard deviations
    from the median magnitude, weighted by their errors.
    
    In this updated function, each data point is weighed according to its error. The weighted 
    ratio of points that are below 5 standard deviations from the median magnitude is returned.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the light curve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The number of points below 5 standard deviations from the light curve's median measurement.
    """
    median_mag = np.median(mag)
    std = np.std(mag)
    
    if apply_weights:
        weighted_below5 = np.mean((-mag + median_mag) > 5 * std)
        weighted_below5 /= np.mean((-mag + median_mag + 5 * std) / magerr**2)
        return weighted_below5
    else:
        below5 = np.mean((-mag + median_mag) > 5 * std)
        return below5

def medianAbsDev(time, mag, magerr, apply_weights=True):
    """
    A measure of the mean average distance between each magnitude value
    and the mean magnitude. https://en.wikipedia.org/wiki/Median_absolute_deviation 
    
    This updated function first calculates the median of the magnitude array, 
    then calculates the absolute deviation from the median, divided by the corresponding error value. 
    The median of these absolute deviations is returned as the MAD value.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The median absolute deviation of the lightcurve.
    """

    if apply_weights:
        med = np.median(mag)
        absdev = np.abs(mag - med) / magerr
        mad = np.median(absdev)
        return mad
    else:
        array = np.ma.array(mag).compressed() 
        med = np.median(array)
        return np.median(np.abs(array - med))

def root_mean_squared(time, mag, magerr, apply_weights=True):
    """
    A measure of the root mean square deviation that takes into account the photometric errors.
    
    In this new version, the magnitudes are weighted by their corresponding errors, which takes
    into account the uncertainty in the measurements. The weighted mean of the magnitudes is 
    subtracted from each magnitude to calculate the weighted deviations, which are then squared 
    and averaged to get the weighted mean of the squared deviations. Finally, the square root of 
    this quantity gives the root mean square deviation that takes into account the photometric errors.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The root-mean-square of the lightcurve.
    """

    if apply_weights:
        weighted_mean = np.sum(mag / magerr**2) / np.sum(1 / magerr**2)
        weights = 1 / magerr**2
        deviations = (mag - weighted_mean)
        weighted_deviations = deviations * weights
        weighted_dev_squared = np.sum(weighted_deviations**2) / np.sum(weights)
        rms = np.sqrt(weighted_dev_squared) #Root mean square deviation
    else:
        rms = np.sqrt(np.median(mag)**2)

    return rms

def meanMag(time,mag, magerr, apply_weights=True):
    """
    Calculates mean magnitude, weighted by the errors.
        
    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The weighted mean measurement of the lightcurve.
    """
            
    return sum(mag/magerr**2)/sum(1./magerr**2)

def integrate(time, mag, magerr, apply_weights=True):
    """
    Integrate magnitude using the trapezoidal rule.
    See: http://en.wikipedia.org/wiki/Trapezoidal_rule
    
    In the case of integrating the magnitude using the trapezoidal rule, 
    it is not necessary to incorporate the error since the error in magnitude 
    will affect each individual data point, but not the overall integration. 
    The trapezoidal rule uses the values of the magnitudes and their timestamps 
    to compute the area under the curve, without considering the individual errors at each point.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The integral value of the lightcurve.
    """
    
    integrated_mag = np.trapz(mag, time)
    
    return integrated_mag

def auto_corr(time, mag, magerr, apply_weights=True):
    """
    Similarity between observations as a function of a time lag between them.
    
    This version of the function first calculates the mean and standard deviation 
    of the magnitudes, and then uses these values to normalize the data before 
    computing the autocovariance function. The weights for each data point are 
    also calculated based on their measurement uncertainties, and are used to compute 
    the weighted autocovariance. Finally, the autocovariance function is normalized by 
    its value at zero lag to obtain the autocorrelation function.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The auto-correlation measurement of the lightcurve.
    """

    if apply_weights:
        # Calculate the mean and standard deviation of the magnitudes
        mag_mean = np.mean(mag)
        mag_std = np.std(mag)
        
        # Calculate the normalized magnitudes
        normalized_mag = (mag - mag_mean) / mag_std
        
        # Calculate the autocovariance function
        autocov = np.correlate(normalized_mag, normalized_mag, mode='full')
        autocov = autocov[autocov.size // 2:]
        
        # Calculate the weights for each data point based on their measurement uncertainties
        weights = 1. / magerr**2
        
        # Compute the weighted autocovariance function
        weighted_autocov = np.sum(weights * autocov)
        
        # Normalize by the autocovariance at zero lag to obtain the autocorrelation function
        auto_corr = weighted_autocov / (np.sum(weights) * autocov[0])
    else:
        # If weights are not applied, use the regular correlation function
        auto_corr = np.corrcoef(mag[:-1], mag[1:])[1, 0]

    return auto_corr

def peak_detection(time, mag, magerr, apply_weights=True):
    """
    Function to detect number of peaks.
    
    Does not need to incorporate error since it is simply detecting 
    the number of peaks in the lightcurve, which is based on the 
    magnitude values alone.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The number of peaks detected in the lightcurve.
    """
    
    mag = abs(mag - np.median(mag))

    try:
        indices = peakutils.indexes(mag, thres=.5, min_dist=10)
    except ValueError:
        indices = []

    return len(indices)/len(mag)

"""
#Below stats used by Richards et al (2011)
"""

def MaxSlope(time, mag, magerr, apply_weights=True):
    """
    Examining successive (time-sorted) magnitudes, the maximal first difference
    (value of delta magnitude over delta time)
    
    In this updated version of the function, the slope between successive magnitudes 
    is calculated using the errors as weights, and the weighted slope is returned as 
    a single value, not the max slopes as is the case when apply_weights=False.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The maximum-slope detected within the lightcurve.
    """

    if apply_weights:
        #Calculate the slope using the errors as weights
        weights = 1 / magerr[:-1]**2 + 1 / magerr[1:]**2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope = np.abs((mag[1:] - mag[:-1]) / (time[1:] - time[:-1]))
        weighted_slope = np.sum(weights[np.isfinite(slope)] * slope[np.isfinite(slope)]) / np.sum(weights[np.isfinite(slope)])
        return weighted_slope
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope = np.abs(mag[1:] - mag[:-1]) / (time[1:] - time[:-1])
        return np.max(slope[np.isfinite(slope)])

def LinearTrend(time, mag, magerr, apply_weights=True):
    """
    Slope of a weighted linear fit to the light-curve.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The slope of a weighted linear fit to the lightcurve.
    """

    if apply_weights:
        # Perform weighted linear regression
        weights = 1.0 / magerr**2
        regression_slope = np.polyfit(time, mag, deg=1, w=weights)[0]
    else:
        regression_slope = sstats.linregress(time, mag)[0]

    return regression_slope

def PairSlopeTrend(time, mag, magerr, apply_weights=True):
    """
    This is the percentage of all pairs of consecutive flux measurements that have positive slope,
    considering only the last 30 (time-sorted) magnitude measurements.

    This updated function incorporates error by calculating the weighted first differences 
    and then taking the weighted mean of the positive differences and negative differences separately.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The percentage of all pairs of consecutive lightcurve measurements that have positive a slope.
    """

    if apply_weights:
        data_last = mag[-30:]
        err_last = magerr[-30:]
        # Calculate the weighted first differences
        weights = 1.0 / (err_last[:-1]**2 + err_last[1:]**2)
        diff = data_last[1:] - data_last[:-1]
        diff_weighted = diff * weights
        pos_diff_weighted = diff_weighted[diff > 0]
        neg_diff_weighted = diff_weighted[diff < 0]
        # Calculate the weighted mean of positive and negative differences
        pos_mean = np.mean(pos_diff_weighted) if len(pos_diff_weighted) > 0 else 0
        neg_mean = np.mean(neg_diff_weighted) if len(neg_diff_weighted) > 0 else 0
        # Calculate the percentage of pairs with a positive slope
        PST = len(pos_diff_weighted) / (len(pos_diff_weighted) + len(neg_diff_weighted))
    else:
        data_last = mag[-30:]
        #PST = (len(np.where(np.diff(data_last) > 0)[0]) - len(np.where(np.diff(data_last) <= 0)[0])) / 30.0
        PST = len(np.where(np.diff(data_last) > 0)[0]) / 30.0

    return PST

def FluxPercentileRatioMid20(time, mag, magerr, apply_weights=True):
    """
    In order to characterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (60th - 40th) over (95th - 5th)
    
    In this updated version of the function, we first sort the magnitude data and associated uncertainties. 
    We then calculate the weighted percentiles of the magnitude data using the cumulative sum of the weights. 
    We calculate the percentiles using np.interp with the cumulative sum of the weights, and then calculate the 
    flux percentile ratios using the weighted percentiles.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of the (60th - 40th) flux percentile over the (95th - 5th) percentile.
    """

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)
    
    if apply_weights:
        weights = 1.0 / (magerr ** 2)
        cumulative_weights = np.cumsum(weights)
        percentiles = np.interp([0.05, 0.40, 0.60, 0.95], cumulative_weights / cumulative_weights[-1], sorted_data)
    else:
        percentiles = np.percentile(sorted_data, [5, 40, 60, 95])
    
    F_40_60 = percentiles[2] - percentiles[1]
    F_5_95 = percentiles[3] - percentiles[0]
    F_mid20 = F_40_60 / F_5_95

    return F_mid20

def FluxPercentileRatioMid35(time, mag, magerr, apply_weights=True):
    """
    In order to characterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (67.5th - 32.5th) over (95th - 5th)
    
    In this updated version of the function, we first sort the magnitude data and associated uncertainties. 
    We then calculate the weighted percentiles of the magnitude data using np.interp with the cumulative sum of 
    the weights, and then calculate the flux percentile ratios using the weighted percentiles.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of the (67.5th - 32.5th) flux percentile over the (95th - 5th) percentile.
    """

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)

    if apply_weights:
        weights = 1.0 / (magerr ** 2)
        cumulative_weights = np.cumsum(weights)
        percentiles = np.interp([0.05, 0.325, 0.675, 0.95], cumulative_weights / cumulative_weights[-1], sorted_data)
    else:
        percentiles = np.percentile(sorted_data, [5, 32.5, 67.5, 95])

    F_325_675 = percentiles[2] - percentiles[1]
    F_5_95 = percentiles[3] - percentiles[0]
    F_mid35 = F_325_675 / F_5_95

    return F_mid35

def FluxPercentileRatioMid50(time, mag, magerr, apply_weights=True):
    """
    In order to characterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (75th - 25th) over (95th - 5th)
    
    In this updated version of the function, we first sort the magnitude data and associated uncertainties. 
    We then calculate the weighted percentiles of the magnitude data using np.interp with the cumulative sum of 
    the weights, and then calculate the flux percentile ratios using the weighted percentiles.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of the (75th - 25th) flux percentile over the (95th - 5th) percentile.
    """

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)

    if apply_weights:
        weights = 1.0 / (magerr ** 2)
        cumulative_weights = np.cumsum(weights)
        percentiles = np.interp([0.05, 0.25, 0.75, 0.95], cumulative_weights / cumulative_weights[-1], sorted_data)
    else:
        percentiles = np.percentile(sorted_data, [5, 25, 75, 95])

    F_25_75 = percentiles[2] - percentiles[1]
    F_5_95 = percentiles[3] - percentiles[0]
    F_mid50 = F_25_75 / F_5_95

    return F_mid50

def FluxPercentileRatioMid65(time, mag, magerr, apply_weights=True):
    """
    In order to characterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (82.5th - 17.5th) over (95th - 5th)

    In this updated version of the function, we first sort the magnitude data and associated uncertainties. 
    We then calculate the weighted percentiles of the magnitude data using np.interp with the cumulative sum of 
    the weights, and then calculate the flux percentile ratios using the weighted percentiles.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of the (82.5th - 17.5th) flux percentile over the (95th - 5th) percentile.
    """

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)

    if apply_weights:
        weights = 1.0 / (magerr ** 2)
        cumulative_weights = np.cumsum(weights)
        percentiles = np.interp([0.05, 0.175, 0.825, 0.95], cumulative_weights / cumulative_weights[-1], sorted_data)
    else:
        percentiles = np.percentile(sorted_data, [5, 17.5, 82.5, 95])

    F_175_825 = percentiles[2] - percentiles[1]
    F_5_95 = percentiles[3] - percentiles[0]
    F_mid65 = F_175_825 / F_5_95

    return F_mid65

def FluxPercentileRatioMid80(time, mag, magerr, apply_weights=True):
    """
    In order to characterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (90th - 10th) over (95th - 5th)

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of the (90th - 10th) flux percentile over the (95th - 5th) percentile.
    """
 
    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)

    if apply_weights:
        weights = 1.0 / (magerr ** 2)
        cumulative_weights = np.cumsum(weights)
        percentiles = np.interp([0.05, 0.10, 0.90, 0.95], cumulative_weights / cumulative_weights[-1], sorted_data)
    else:
        percentiles = np.percentile(sorted_data, [5, 10, 90, 95])

    F_10_90 = percentiles[2] - percentiles[1]
    F_5_95 = percentiles[3] - percentiles[0]
    F_mid80 = F_10_90 / F_5_95

    return F_mid80

def PercentAmplitude(time, mag, magerr, apply_weights=True):
    """
    The largest absolute departure from the median flux, divided by the median flux
    Largest percentage difference between either the max or min magnitude and the median.
    
    This function calculates both the regular median and a weighted median that takes into 
    account the photometric errors. It then calculates the largest absolute departure from 
    each of these medians and returns the largest percentage difference between either the 
    max or min magnitude and the weighted median.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The largest percentage difference present in the lightcurve with respect to the median.
    """

    if apply_weights:
        weights = 1.0 / magerr**2
        median = np.median(mag)
        w_median = np.ma.average(mag, weights=weights)
        distance_median = np.abs(mag - median)
        w_distance_median = np.abs(mag - w_median)
        max_distance = np.max(distance_median)
        w_max_distance = np.max(w_distance_median)
        percent_amplitude = max_distance / median
        w_percent_amplitude = w_max_distance / w_median
        return w_percent_amplitude
    else:
        median = np.median(mag)
        distance_median = np.abs(mag - median)
        max_distance = np.max(distance_median)
        return max_distance / median

def PercentDifferenceFluxPercentile(time, mag, magerr, apply_weights=True):
    """
    Calculates the ratio of the flux difference between the 5th and 95th percentiles of 
    the data (F5,95) to the median flux.

    If apply_weights is set to True, the function first sorts the mag array and the corresponding magerr
    based on the magnitude values. It then calculates the weighted percentiles by interpolating the values 
    at the desired percentiles (2%, 5%, 95%, and 98%) using cumulative weights. The difference between 
    the 95th and 5th percentiles (F5,95) is calculated based on the interpolated values.

    If apply_weights is set to False, the function directly operates on the mag array. It sorts the array and 
    calculates the indices corresponding to the 5th and 95th percentiles. The flux difference between these 
    percentiles (F5,95) is then calculated.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of the flux difference between the 5th and 95th percentiles to the median lightcurve measurement.
    """

    if apply_weights:
        sorted_indices = np.argsort(mag)
        sorted_mag = mag[sorted_indices]
        sorted_magerr = magerr[sorted_indices]
        weights = 1 / sorted_magerr ** 2
        total_weights = np.sum(weights)
        cum_weights = np.cumsum(weights) / total_weights
        percentile_95 = np.interp(0.95, cum_weights, sorted_mag)
        percentile_5 = np.interp(0.05, cum_weights, sorted_mag)
        F_5_95 = percentile_95 - percentile_5
    else:
        sorted_data = np.sort(mag)
        lc_length = len(sorted_data)
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]

    return F_5_95 / np.median(mag)

#Below stats from Kim (2015), used in Upsilon
#https://arxiv.org/pdf/1512.01611.pdf

def half_mag_amplitude_ratio(time, mag, magerr, apply_weights=True):
    """
    The ratio of the squared sum of residuals of magnitudes that are either brighter 
    than or fainter than the mean magnitude. For EB-like variability, having sharp 
    flux gradients around its eclipses, this value is larger than 1.

    In this modified version, the weighted standard deviation of each set of 
    magnitudes (i.e., those above and those below the median) is used if apply_weights is set to True.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The ratio of squared residuals for measurements above and below the median.    
    """

    if apply_weights:
        #For fainter magnitude than average.
        avg = np.median(mag)
        index = np.argwhere(mag > avg)
        lower_mag = mag[index]
        lower_magerr = magerr[index]
        lower_weighted_std = np.sum((lower_mag - avg)**2 / lower_magerr**2) / np.sum(1. / lower_magerr**2)
        #For brighter magnitude than average.
        index = np.argwhere(mag <= avg)
        higher_mag = mag[index]
        higher_magerr = magerr[index]
        higher_weighted_std = np.sum((higher_mag - avg)**2 / higher_magerr**2) / np.sum(1. / higher_magerr**2)
        ratio = np.sqrt(lower_weighted_std / higher_weighted_std)
    else:
        avg = np.median(mag)
        index = np.argwhere(mag > avg)
        lower_mag = mag[index]
        lower_weighted_std = (1./len(index))*np.sum((lower_mag - avg)**2)
        # For brighter magnitude than average.
        index = np.argwhere(mag <= avg)
        higher_mag = mag[index]
        higher_weighted_std = (1./len(index))*np.sum((higher_mag - avg)**2)
        ratio = np.sqrt(lower_weighted_std / higher_weighted_std)
            
    return ratio

def cusum(time, mag, magerr, apply_weights=True):
    """
    Range of cumulative sum.

    In this updated version, we first calculate the weighted standard deviation of 
    the magnitude using the formula wstd = np.sqrt(np.sum((mag - np.median(mag))**2 / magerr**2) / np.sum(1. / magerr**2)). 
    Then we use this value instead of np.std(mag) to normalize the cumulative sum. This takes into account the error in 
    the measurements of the magnitude.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The range of the cumulative sums of the lightcurve measurements.
    """

    if apply_weights:
        #Calculate the weighted standard deviation
        wstd = np.sqrt(np.sum((mag - np.median(mag))**2 / magerr**2) / np.sum(1. / magerr**2))
        c = np.cumsum(mag - np.median(mag)) * 1./(len(mag)*wstd)
    else:
        c = np.cumsum(mag - np.median(mag)) * 1./(len(mag)*np.std(mag))

    return np.max(c) - np.min(c)

def shapiro_wilk(time, mag, magerr, apply_weights=True):
    """
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
    
    If this statistic is close to 1, then it suggests that the null hypothesis cannot be rejected, 
    which means the data is likely to follow a normal distribution. Note that there is no error incorporation, 
    as the Shapiro-Wilk test implemented in scipy.stats does not provide an option to incorporate measurement error.
    
    Note that the Shapiro-Wilk test implemented in scipy.stats does not provide the option to incorporate measurement errors.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------
    float
        The Shapiro-Wilk statistic.
    """

    shapiro_w = sstats.shapiro(mag)[0]

    return shapiro_w

#Following stats pulled from FEETS
#https://feets.readthedocs.io/en/latest/tutorial.html

def AndersonDarling(time, mag, magerr, apply_weights=True):
    """
    The Anderson-Darling test is a statistical test of whether a given 
    sample of data is drawn from a given probability distribution. 
    When applied to testing if a normal distribution adequately describes a set of data, 
    it is one of the most powerful statistical tools for detecting most departures from normality.
    
    It is a measure of how well the data fits a normal distribution. The AndersonDarling_Weighted() function applies the same test, but with weights based on the input errors magerr. 
    Both functions return a value between 0 and 1, with values closer to 1 indicating a better fit to a normal distribution. In short, values closer to 1 indicate a higher confidence 
    that the data follow a normal distribution.
    
    The weighted Anderson-Darling test is a statistical test of whether a given 
    sample of data is drawn from a given probability distribution. 
    When applied to testing if a normal distribution adequately describes a set of data, 
    it is one of the most powerful statistical tools for detecting most departures from normality.
    
    From Kim et al. 2009: "To test normality, we use the Anderson–Darling test (Anderson & Darling 1952; Stephens 1974) 
    which tests the null hypothesis that a data set comes from the normal distribution."
    (Doi:10.1111/j.1365-2966.2009.14967.x.)

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    float
        The Anderson-Darling test statistic.
    """
    
    if apply_weights:
        weights = 1.0 / np.square(magerr)
        wmag = np.average(mag, weights=weights)
        wmagsq = np.average(np.square(mag), weights=weights)
        wvar = wmagsq - wmag**2
        z = (mag - wmag) / np.sqrt(wvar)
        z_sorted = np.sort(z)
        n = len(mag)
        s = np.zeros(n)
        for i in range(n):
            s[i] = (2*i + 1) * np.log(sstats.norm.cdf(z_sorted[i])) + (2*(n-i)-1) * np.log(1 - sstats.norm.cdf(z_sorted[i]))
        ander = -n - np.sum(s) / n
    else:   
        ander = sstats.anderson(mag)[0]

    return 1 / (1.0 + np.exp(-10 * (ander - 0.3)))

def Gskew(time, mag, magerr, apply_weights=True):
    """
    Gskew is a measure of the skewness of a distribution of magnitudes. It is defined as the 
    sum of the medians of the magnitudes below and above the 3rd and 97th percentiles, respectively, 
    minus twice the median magnitude. In other words, Gskew is a measure of the asymmetry of the 
    distribution of magnitudes. A positive Gskew value indicates a distribution that is skewed to the 
    right (has a long tail on the right side), while a negative Gskew value indicates a distribution 
    that is skewed to the left (has a long tail on the left side).

    It is essentially a median-based measure of the skewness. See: Lopez et al. 2016: "A machine learned classifier for RR Lyrae in the VVV survey" 

    Gskew = mq3 + mq97 − 2m
    mq3 is the median of magnitudes lesser or equal than the quantile 3.
    mq97 is the median of magnitudes greater or equal than the quantile 97.
    2m is 2 times the median magnitude.

    If apply_weights=True a modified version will be used that incorporates the photometric 
    errors of the data points. It calculates a weighted median for the magnitudes that 
    fall below the 3rd percentile and above the 97th percentile, using the inverse 
    square of the photometric errors as weights. The resulting weighted medians and 
    the median magnitude are then used to calculate the Gskew value.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------  
    float   
        The median-based skewness of the lightcurve.
    """

    if apply_weights:
        #Sort the magnitude and error arrays by magnitude
        sorted_indices = np.argsort(mag)
        sorted_mag = mag[sorted_indices]
        sorted_magerr = magerr[sorted_indices]

        # Calculate the cumulative weights
        weights = 1.0 / np.square(sorted_magerr)
        cum_weights = np.cumsum(weights)

        # Calculate the indices of the median and quantiles
        median_index = np.searchsorted(cum_weights, 0.5 * cum_weights[-1])
        q3_index = np.searchsorted(cum_weights, 0.03 * cum_weights[-1])
        q97_index = np.searchsorted(cum_weights, 0.97 * cum_weights[-1])

        # Calculate the median and quantiles
        median_mag = sorted_mag[median_index]
        F_3_value = sorted_mag[q3_index]
        F_97_value = sorted_mag[q97_index]

        # Calculate the weighted median of magnitudes <= F_3_value
        cum_weights_3 = cum_weights[:q3_index]
        weights_3 = weights[:q3_index]
        cum_weights_3 -= cum_weights_3[0]
        cum_weights_3 /= cum_weights_3[-1]
        mq3 = np.interp(0.5, cum_weights_3[::-1], sorted_mag[:q3_index][::-1])

        # Calculate the weighted median of magnitudes >= F_97_value
        cum_weights_97 = cum_weights[q97_index-1:]
        weights_97 = weights[q97_index-1:]
        cum_weights_97 -= cum_weights_97[0]
        cum_weights_97 /= cum_weights_97[-1]
        mq97 = np.interp(0.5, cum_weights_97, sorted_mag[q97_index-1:])

        gs = mq3 + mq97 - 2 * median_mag

    else:
        median_mag = np.median(mag)
        F_3_value = np.percentile(mag, 3)
        F_97_value = np.percentile(mag, 97)
        gs = (np.median(mag[mag <= F_3_value]) + np.median(mag[mag >= F_97_value]) - 2*median_mag)

    return gs

def abs_energy(time, mag, magerr, apply_weights=True):
    """
    Calculates the absolute energy of the time series, defined to be the sum over the squared
    values of the time-series.

    If apply_weights is set to True, we calculate the inverse square of the photometric errors 
    and use them as weights to calculate the weighted sum of squares of the magnitudes.
    In the case where weights are applied, the magnitudes are multiplied by their corresponding 
    weights (inverse square of photometric errors) before squaring and summing them. This results in 
    a larger contribution from data points with smaller errors, as they receive higher weights. As a result, 
    the absolute energy value increases. On the other hand, when weights are not applied (apply_weights=False), 
    all magnitudes are squared and summed without considering their errors. This leads to a smaller 
    absolute energy value as the contributions from individual data points are not scaled by their respective errors.



    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------   
    float  
        The sum over the squared values of the lightcurve measurements.
    """

    if apply_weights:
        weights = 1.0 / np.square(magerr)
        abs_energy = np.sum(weights * np.square(mag))
    else:
        abs_energy = np.dot(mag, mag)

    return abs_energy

def abs_sum_changes(time, mag, magerr, apply_weights=True):
    """
    Calculates the sum over the abs value of consecutive changes in mag, weighted by the errors.
    
    In this updated version we incorporate photometric errors by dividing the absolute value 
    of the difference between consecutive magnitudes by the square root of the sum of their squared errors. 
    Therefore larger errors will result in smaller weight for the corresponding changes in magnitude.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------    
    float 
        The sum over the absolute value of consecutive changes in the lightcurve measurements.
    """

    if apply_weights:
        delta_mag = np.abs(np.diff(mag))
        delta_err = np.sqrt(np.square(magerr[:-1]) + np.square(magerr[1:]))
        weighted_delta = delta_mag / delta_err
        return np.sum(weighted_delta)
    else:
        return np.sum(np.abs(np.diff(mag)))

def benford_correlation(time, mag, magerr, apply_weights=True):
    """
    Useful for anomaly detection applications. Calculates the 
    correlation from first digit distribution when compared to 
    the Newcomb-Benford’s Law distribution, weighted by the inverse variance of the magnitudes.
    
    In this updated version, we calculate the weights as the inverse 
    variance of the magnitudes (i.e., the inverse of the squared photometric errors), and use 
    these weights to calculate the weighted distribution of the data. We then normalize this 
    weighted distribution and compute the weighted correlation between the Benford distribution 
    and the weighted data distribution.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    float
        The Benford Correlation statistic.
    """

    if apply_weights:
        #Retrieve first digit from data
        x = np.array([int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(mag))])
        # benford distribution
        benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
        #Calculate weights as inverse variance of magnitudes
        weights = 1 / np.square(np.nan_to_num(magerr))
        #Calculate weighted distribution of data
        weighted_data_distribution = np.zeros(9)
        for i in range(1, 10):
            mask = (x == i)
            weighted_data_distribution[i-1] = np.sum(weights[mask])
        #Normalize weighted distribution
        weighted_data_distribution /= np.sum(weights)
        #Weighted correlation
        benford_corr = np.corrcoef(benford_distribution, weighted_data_distribution)[0, 1]
    else:
        #Retrieve first digit from data
        x = np.array([int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(mag))])
        #Benford distribution
        benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
        data_distribution = np.array([(x == n).mean() for n in range(1, 10)])
        #np.corrcoef outputs the normalized covariance (correlation) between benford_distribution and data_distribution.
        #In this case returns a 2x2 matrix, the  [0, 1] and [1, 1] are the values between the two arrays
        benford_corr = np.corrcoef(benford_distribution, data_distribution)[0, 1]

    return benford_corr

def c3(time, mag, magerr, lag=1, apply_weights=True):
    """
    The C3 measure is a way to estimate the non-linearity of a time series by measuring the third-order 
    correlation between the values of the time series. It is based on the idea that a truly linear time 
    series will have a third-order correlation of zero, while a non-linear time series will have a 
    non-zero third-order correlation. The lag parameter controls the distance between the three values 
    of the time series that are used to calculate the third-order correlation. A larger lag value will 
    capture longer-term correlations in the data, while a smaller lag value will capture shorter-term correlations.
    
    In this updated version, we first calculate the terms that make up the third-order correlation using the 
    mag array and its two rolled versions. We then use the error propagation formula to calculate the errors associated 
    with these terms. Finally, we calculate the third-order correlation as the weighted average of the terms, where the 
    weights are given by the inverse squared errors. Note that this version of the function assumes Gaussian errors in the magerr array.

    See: Measure of non-linearity in time series: [1] Schreiber, T. and Schmitz, A. (1997).
    Discrimination power of measures for nonlinearity in a time series
    PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        lag: int
            The lag to use. Defaults to 1.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------  
    float   
        The C3 non-linearity statistic.
    """

    n = len(mag)
    if 2 * lag >= n:
        return 0
    else:
        if apply_weights:
            # Calculate third-order correlation with error propagation
            roll1 = np.roll(mag, -lag)
            roll2 = np.roll(mag, -2 * lag)
            term1 = mag * roll1 * roll2
            term2 = magerr**2 * (roll1 * roll2 + mag * roll2 + mag * roll1)
            third_corr = np.sum(term1 / term2) / np.sum(1 / term2)
            return third_corr
        else:
            return np.mean((np.roll(mag, 2 * -lag) * np.roll(mag, -lag) * mag)[0 : (n - 2 * lag)])

def complexity(time, mag, magerr, apply_weights=True):
    """
    This function calculator is an estimate for a time series complexity.
    A higher value represents more complexity (more peaks,valleys,etc.)
    See: Batista, Gustavo EAPA, et al (2014). CID: an efficient complexity-invariant 
    distance for time series. Data Mining and Knowledge Difscovery 28.3 (2014): 634-669.
    
    To incorporate errors into the complexity function, we apply the weighted standard deviation formula. 
    We exclude the last element of magerr since np.diff reduces the size of the mag array 
    by one. Also, if the sum of the weights is zero, we return 0 to avoid division by zero errors.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    float
        The complexity measurement of the lightcurve.
    """

    if apply_weights:
        dmag = np.diff(mag)
        w = 1 / magerr[:-1]**2  #weights based on magerr
        w_sum = np.sum(w)
        if w_sum == 0:
            return 0
        else:
            #weighted standard deviation
            sd = np.sqrt(np.average((dmag - np.average(dmag, weights=w))**2, weights=w))
            return sd
    else:
        mag = np.diff(mag)
        return np.sqrt(np.dot(mag, mag))

def count_above(time, mag, magerr, apply_weights=True):
    """
    Number of values higher than the weighted median.

    This function calculates the weighted median of the mag array using the photometric errors in magerr, 
    and then counts the number of values in mag that are above the weighted median. The fraction of values 
    above the weighted median is then calculated using the weights from magerr. If magerr is zero for all values, 
    the function returns zero.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    float
        The number of lightcurve measurements above the median value.
    """ 

    if apply_weights:
        #Calculate the weighted median
        weights = 1.0 / (magerr ** 2)
        w_median = np.median(np.insert(mag, 0, -np.inf))
        
        #Calculate the number of values above the weighted median
        above = np.where(mag > w_median)[0].size
        total = len(mag)
        
        #Calculate the weighted fraction of values above the median
        w_above = np.sum(weights[mag > w_median])
        w_total = np.sum(weights)
        if w_total == 0:
            return 0
        else:
            return w_above / w_total
    else:
        return (np.where(mag > np.median(mag))[0].size)/len(mag)

def count_below(time, mag, magerr, apply_weights=True):
    """
    Number of values below the median.

    To incorporate errors, we use the weighted median instead of the regular median. 
    The weighted median takes into account the uncertainties associated with each data point.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    float
        The number of lightcurve measurements below the median value.
    """

    if apply_weights:
        #Compute the weighted median
        weights = 1/magerr**2
        median = np.average(mag, weights=weights)
        #Count the number of values below the weighted median
        below_median = mag < median
        return np.sum(below_median * weights) / np.sum(weights)
    else:
        return (np.where(mag < np.median(mag))[0].size)/len(mag)

def first_loc_max(time, mag, magerr, apply_weights=True):
    """
    Calculates the location of maximum mag relative to the length of mag array, 
    weighted by inverse square of magerr if apply_weights is True.
    
    In this modified version, we first calculate the inverse square 
    of magerr and set it to 0 where magerr2 is 0 to avoid division by zero. 
    Then we multiply each value of mag by the corresponding inverse square of magerr, 
    giving more weight to values with smaller magerr. Finally, we find the index of the 
    maximum value in the weighted mag array using np.argmax, and return the location of 
    the maximum relative to the length of mag.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    float
        The relative position of the maximum detected lightcurve measurement.
    """

    if len(mag) == 0:
        return np.NaN

    if apply_weights:
        #Calculate inverse square of magerr
        magerr2 = magerr ** 2
        inv_magerr2 = np.where(magerr2 > 0, 1 / magerr2, 0)
        #Weight the maximum value by inverse square of magerr
        weighted_max = np.argmax(mag * inv_magerr2)
        #Return location of maximum mag relative to the length of mag array
        return weighted_max / len(mag)
    else:
        return np.argmax(mag) / len(mag)

def first_loc_min(time, mag, magerr, apply_weights=True):
    """
    Calculates the location of minimum mag relative to the 
    length of mag array.
    
    This updated implementation first computes the weights for each measurement 
    using the provided photometric errors in magerr. It then replaces all zero 
    weights with 1 to avoid division by zero. Finally, it computes the location 
    of the minimum mag by taking the weighted minimum of mag using the computed weights.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------    
    float 
        The relative position of the minimum detected lightcurve measurement.
    """

    if len(mag) == 0:
        return np.NaN

    if apply_weights:
        # Compute weights
        weights = 1.0 / (magerr ** 2)
        # Replace all zero weights with 1 to avoid division by zero
        weights[weights == 0] = 1
        # Compute location of minimum mag
        w_argmin = np.argmin(mag * weights)
        loc_min = w_argmin / len(mag) 
        return loc_min
    else:
        return np.argmin(mag) / len(mag)

def check_for_duplicate(time, mag, magerr, apply_weights=True):
    """
    Checks if any value in mag repeats, taking into account photometric errors if apply_weights
    is enabled so as to identify measurements that are within the error bars.
    
    To incorporate error, we use np.isclose to check if two values of mag are close to each other, 
    taking into account their respective errors. The tolerance is set using the atol argument, which is 
    set to the sum of the errors in quadrature, to account for the fact that two measurements with similar 
    values but different errors may still be considered duplicates. If a duplicate is found, the function 
    returns 1, otherwise it returns 0.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    int
        Whether or not a duplicate lightcurve measurement is found, 1 for True, 0 for False.
    """

    if apply_weights:
        # Check for duplicates with photometric error tolerance
        for i in range(len(mag)):
            for j in range(i+1, len(mag)):
                if np.isclose(mag[i], mag[j], rtol=0, atol=2*np.sqrt(magerr[i]**2 + magerr[j]**2)):
                    return 1
        return 0
    else:
        if mag.size != np.unique(mag).size:
            return 1
        else:     
            return 0

def check_for_max_duplicate(time, mag, magerr, apply_weights=True):
    """
    Checks if the maximum value in mag repeats, taking into account photometric errors if apply_weights
    is enabled so as to identify measurements that are within the error bars.

    To incorporate error, we use np.isclose to check if the maximum value in mag is close to any other
    value in mag, taking into account their respective errors. The tolerance is set using the atol argument, 
    which is calculated as the maximum of the maximum error and the error of the closest value to the maximum, 
    to account for the fact that two measurements with similar values but different errors may still be 
    considered duplicates. If a duplicate is found, the function returns 1, otherwise it returns 0.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------    
    int 
        Whether or not the maximum lightcurve measurement is found more than once, 1 for True, 0 for False.
    """

    if apply_weights:
        max_mag = np.max(mag)
        #Calculate the atol value based on the maximum error and the error of the closest value to the maximum
        closest_mag_err = np.min(np.abs(mag - max_mag))  #error of the closest value to the maximum
        atol = np.maximum(magerr, closest_mag_err)  # max of the maximum error and the error of the closest value to the maximum
        #Check for duplicates with photometric error tolerance
        for i in range(len(mag)):
            if np.isclose(mag[i], max_mag, rtol=0, atol=atol[i]):
                return 1
        return 0
    else:
        if np.sum(mag == np.max(mag)) >= 2:
             return 1
        else:     
             return 0

def check_for_min_duplicate(time, mag, magerr, apply_weights=True):
    """
    Checks if the minimum value in mag repeats, taking into account photometric errors if apply_weights
    is enabled so as to identify measurements that are within the error bars.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    int
        Whether or not the minimum lightcurve measurement is found more than once, 1 for True, 0 for False.
    """

    if apply_weights:
        min_mag = np.min(mag)
        #Find the indices of the minimum values in mag
        min_idx = np.where(np.isclose(mag, min_mag, rtol=0, atol=magerr))[0]
        #Check for duplicates with photometric error tolerance
        num_duplicates = len(min_idx)
        if num_duplicates > 1:
            return 1
        return 0
    else:
        if np.sum(mag == np.min(mag)) >= 2:
            return 1
        else:     
            return 0

def check_max_last_loc(time, mag, magerr, apply_weights=True):
    """
    Calculates the position of last maximum mag relative to
    the length of mag array, taking into account photometric errors.
    
    In this implementation, we first find the maximum value in mag and calculate the tolerance 
    value atol as the maximum photometric error in magerr. We then use np.isclose() with atol 
    as the atol argument to find the indices of all values in mag that are within tolerance of max_mag. 
    We select the last index in the resulting array (which corresponds to the last maximum value in mag) and 
    calculate its position relative to the length of mag. If there are no values within tolerance (3sigma), we return np.NaN.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    float
        The relative position where the maximum lightcurve measurement was last found.
    """

    if apply_weights:
        #Find the maximum value in mag
        max_mag = np.max(mag)
        #Calculate the tolerance value based on the photometric errors
        atol = np.max(magerr) * 3
        #Find the index of the last maximum value within tolerance
        idx = np.where(np.isclose(mag, max_mag, rtol=0, atol=atol))[0]
        if len(idx) > 0:
            return 1.0 - idx[-1] / len(mag)
        else:
            return np.NaN
    else:
        return 1.0 - np.argmax(mag[::-1]) / len(mag) if len(mag) > 0 else np.NaN

def check_min_last_loc(time, mag, magerr, apply_weights=True):
    """
    Calculates the position of last minimum mag relative to
    the length of mag array, taking into account photometric errors.
    
    To incorporate errors, this implementation finds the minimum value in mag, 
    then calculates the atol value based on the error of the minimum value and 
    the error of the closest value to the minimum. It then uses np.isclose() with 
    atol as the atol argument to find the indices of all values in mag that are within 
    tolerance of min_mag. It selects the last index in the resulting array (which corresponds 
    to the last minimum value in mag) and calculates its position relative to the length of mag. 
    If there are no values within tolerance, it returns np.NaN.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The relative position where the minimum lightcurve measurement was last found.
    """

    if apply_weights:
        #Find the minimum value in mag
        min_mag = np.min(mag)
        #Find the error of the closest value to the minimum in magerr
        closest_mag_err = magerr[np.argmin(mag)]
        #Calculate the atol value based on the minimum error and the error of the closest value to the minimum
        atol = magerr + closest_mag_err
        #Find the index of the last minimum value within tolerance
        idx = np.where(np.isclose(mag, min_mag, rtol=0, atol=atol))[0]
        if len(idx) > 0:
            return 1.0 - idx[-1] / len(mag)
        else:
            return np.NaN
    else:
        return 1.0 - np.argmin(mag[::-1]) / len(mag) if len(mag) > 0 else np.NaN

def longest_strike_above(time, mag, magerr, apply_weights=True):
    """
    Calculates the length of the longest consecutive subsequence in 
    mag that is bigger than the median. 
    
    This updated implementation first calculates the median of the mag 
    array and creates a boolean mask of True for elements greater than the 
    median plus their errors and False for elements less than or equal to the 
    median plus their errors. It then splits the mask into groups of consecutive 
    True values, and returns the length of the longest group as a fraction of 
    the length of the mag array. If there are no values greater than the median plus 
    their errors, the function returns 0.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The length of the longest consecutive subsequence in the lightcurve measurements which are larger than the median.
    """

    if apply_weights:
        median = np.median(mag)
        mask = mag > median + magerr
        if np.sum(mask) == 0:
            return 0
        else:
            groups = np.split(mask, np.where(np.diff(mask.astype(int)) != 0)[0]+1)
            return np.max([len(group) for group in groups if np.all(group)]) / len(mag)
    else:
        val = np.max([len(list(group)) for value, group in itertools.groupby(mag) if value == 1]) if mag.size > 0 else 0
        return val / len(mag)

def longest_strike_below(time, mag, magerr, apply_weights=True):
    """
    Calculates the length of the longest consecutive subsequence in mag 
    that is smaller than the median.
    
    To incorporate errors, first we calculate the median of mag and create 
    a boolean mask of True for elements smaller than the median minus their errors 
    and False for elements greater than or equal to the median minus their errors. 
    Then we split the mask into groups of consecutive True values, and return the 
    length of the longest group as a fraction of the length of the mag array. If 
    there are no values smaller than the median minus their errors, the function returns 0.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The length of the longest consecutive subsequence in the lightcurve measurements which are smaller than the median.
    """

    if apply_weights:
        median = np.median(mag)
        mask = mag < median - magerr
        if np.sum(mask) == 0:
            return 0
        else:
            groups = np.split(mask, np.where(np.diff(mask.astype(int)) != 0)[0]+1)
            return np.max([len(group) for group in groups if np.all(group)]) / len(mag)
    else:
        val = np.max([len(list(group)) for value, group in itertools.groupby(mag) if value == 1]) if mag.size > 0 else 0
        return val / len(mag)

def mean_change(time, mag, magerr, apply_weights=True):
    """
    Calculates mean over the differences between subsequent observations,
    weighted by the inverse square of their errors if apply_weights is True.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------  
    float   
        The mean over the differences between all subsequent lightcurve measurements.
    """

    if apply_weights:
        if len(mag) < 2:
            return np.NaN
        else:
            diffs = np.diff(mag)
            weights = 1.0 / (magerr[1:]**2 + magerr[:-1]**2)
            return np.average(diffs, weights=weights)
    else:
        return (mag[-1] - mag[0]) / (len(mag) - 1) if len(mag) > 1 else np.NaN

def mean_abs_change(time, mag, magerr, apply_weights=True):
    """
    Calculates the mean absolute change in the magnitude per unit of error.
    
    To incorporate error we weight each absolute difference by the corresponding error, 
    and then take the mean of the weighted differences. This would give a measure of the 
    average absolute change in units of the error.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------  
    float   
        The mean absolute change in the lightcurve measurements.
    """

    if apply_weights:
        diffs = np.abs(np.diff(mag))
        weights = magerr[:-1] + magerr[1:]
        return np.average(diffs, weights=weights)
    else:
        return np.mean(np.abs(np.diff(mag)))

def mean_n_abs_max(time, mag, magerr, number_of_maxima=10, apply_weights=True):
    """
    Calculates the weighted arithmetic mean of the n absolute maximum values of the time series, n=10 by design.
    
    We incorporate errors in the calculation by sorting the absolute values of the magnitude and corresponding 
    errors, and then taking the arithmetic mean of the top n maximum values weighted by their errors.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        number_of_maxima : int
            The number of maxima to consider. Defaults to 10.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------     
    float
        The mean of the 10 absolute maximum values of the lightcurve. 
    """
    
    if number_of_maxima >= len(mag):
        return np.NaN

    if apply_weights:
        sort_idx = np.argpartition(np.abs(mag), -number_of_maxima)[-number_of_maxima:]
        mag_sorted = mag[sort_idx]
        magerr_sorted = magerr[sort_idx]
        weights = 1 / np.square(magerr_sorted)
        weighted_mean = np.sum(np.abs(mag_sorted) * weights) / np.sum(weights)
        return weighted_mean
    else:
        n_absolute_maximum_values = np.sort(np.abs(mag))[-number_of_maxima:]
        return np.mean(n_absolute_maximum_values)

def mean_second_derivative(time, mag, magerr, apply_weights=True):
    """
    Calculates the weighted mean value of a central approximation of the second derivative,
    where weights are the inverse square of the errors. Note that the first and last values 
    of the second derivative are not included in the calculation, as they cannot be approximated 
    using a central difference.
    
    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    -------  
    float   
        The mean value of a central approximation of the second derivative.
    """

    if len(mag) < 3:
        return np.NaN
    
    if apply_weights:
        diffs = np.diff(mag)
        times = np.diff(time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            errors = np.abs(diffs / times ** 2) * np.sqrt((magerr[:-1] / diffs) ** 2 + (magerr[1:] / diffs) ** 2)
        mask = np.isfinite(errors)
        diffs, times, errors = diffs[mask], times[mask], errors[mask]
        weights = 1 / errors ** 2
        weighted_diffs = diffs[1:-1] * weights[1:-1]
        return np.sum(weighted_diffs) / np.sum(weights[1:-1])
    else:   
        return (mag[-1] - mag[-2] - mag[1] + mag[0]) / (2 * (len(mag) - 2)) if len(mag) > 2 else np.NaN

def number_of_crossings(time, mag, magerr, apply_weights=True):
    """
    Calculates the number of crossings of x on the median, m. A crossing is defined as two 
    sequential values where the first value is lower than m and the next is greater, 
    or vice-versa. If you set m to zero, you will get the number of zero crossings.
    
    We incorporate errors by calculating the differences between consecutive values of the positive array and store it in 
    the crossings variable. Finally, we multiply the crossings array with a Boolean array that checks if the difference 
    between consecutive values of mag is greater than the corresponding error in magerr. The resulting array will have a 
    value of 1 for each crossing that is greater than the corresponding error, and 0 for each crossing that is smaller 
    than or equal to the error. We then sum this array to get the total number of crossings.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The number of crossings about the lightcurve's median.
    """

    positive = mag > np.median(mag)

    if apply_weights:
        crossings = np.abs(np.diff(positive))
        # check if the difference is greater than the corresponding error
        crossings = crossings * (np.abs(np.diff(mag)) > magerr[:-1])
        return np.sum(crossings) / len(mag)
    else:
        return (np.where(np.diff(positive))[0].size) / len(mag)

def number_of_peaks(time, mag, magerr, n=7, apply_weights=True):
    """
    Calculates the number of peaks of at least support n in the time series x. 
    A peak of support n is defined as a subsequence of x where a value occurs, 
    which is bigger than its n neighbors to the left and to the right.
    n = 7
    
    Hence in the sequence:

    >>> x = [3, 0, 0, 4, 0, 0, 13]

    4 is a peak of support 1 and 2 because in the subsequences

    >>> [0, 4, 0]
    >>> [0, 0, 4, 0, 0]

    4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
    and its bigger than 4.
    
    To incorporate the error bars we first reduce the mag and magerr arrays by n elements from both ends to ensure that 
    we can check for peaks of support n. We then iterate over the mag array and calculate the differences between 
    the values of the mag array and its i-th neighbor to the left and to the right. We also calculate the corresponding 
    errors for the differences using the error arrays magerr and xerr_reduced. We then check if the absolute value of 
    the difference is greater than the corresponding error to determine if we have a peak of support n. Finally, we combine 
    the results using logical AND to get the total number of peaks.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        n : int
            The support of the peak. Defaults to 7.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The number of peaks of at least support 7 within the lightcurve, normalized according to the size of the lightcurve.
    """

    if apply_weights:
        x_reduced = mag[n:-n]
        xerr_reduced = magerr[n:-n]
        res = None
        for i in range(1, n + 1):
            # calculate differences with the i-th neighbor to the left and to the right
            diff_left = x_reduced - np.roll(mag, i)[n:-n]
            diff_right = x_reduced - np.roll(mag, -i)[n:-n]

            # calculate the corresponding errors for the differences
            err_left = np.sqrt(xerr_reduced ** 2 + np.roll(magerr, i)[n:-n] ** 2)
            err_right = np.sqrt(xerr_reduced ** 2 + np.roll(magerr, -i)[n:-n] ** 2)

            # check if the difference is greater than the corresponding error
            result_first = np.abs(diff_left) > err_left
            result_second = np.abs(diff_right) > err_right
            # combine the results with logical AND
            if res is None:
                res = result_first & result_second
            else:
                res &= result_first & result_second
        return float(np.sum(res)/len(mag))
    else:
        x_reduced = mag[n:-n]
        res = None
        for i in range(1, n + 1):
            result_first = x_reduced > np.roll(mag, i)[n:-n]
            if res is None:
                res = result_first
            else:
                res &= result_first
            res &= x_reduced > np.roll(mag, -i)[n:-n]
        return float(np.sum(res)/len(mag))

def ratio_recurring_points(time, mag, magerr, apply_weights=True):
    """
    Calculates the ratio of unique values, that are present in the time 
    series more than once, normalized to the number of data points. 
    
    If apply weights is set to True, the photometric errors will be 
    used by looping over the unique values and checking if the number of values 
    that are close to it (using the np.isclose function) is greater than 1. If so, 
    the value is counted as a recurring point.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The number of unique measurements normalized according to the size of the lightcurve.
    """

    unique, counts = np.unique(mag, return_counts=True)

    if counts.shape[0] == 0:
        return 0
    
    if apply_weights:
        recurring_count = 0
        for i in range(len(unique)):
            if np.sum(np.isclose(mag, unique[i], atol=magerr)) > 1:
                recurring_count += 1
        return recurring_count / float(counts.shape[0])
    else:
        return np.sum(counts > 1) / float(counts.shape[0])

def sample_entropy(time, mag, magerr, apply_weights=True):
    """
    Calculates sample entropy: http://en.wikipedia.org/wiki/Sample_Entropy
    
    One approach to incorporate error is to modify the distance metric used in the algorithm 
    to account for measurement error. "Modified Sample Entropy Method in the Presence of Noise" by Zhang et al. 
    proposes a modified version of sample entropy that uses a weighted distance metric based on both the difference 
    in magnitudes and the difference in measurement errors between pairs of data points, but
    MicroLIA does not support this as the noise level may alter the value range significantly. 

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The sample entropy of the lightcurve.
    """

    m = 2  # common value for m, according to wikipedia...
    tolerance = 0.2 * np.std(mag)  # 0.2 is a common value for r, according to wikipedia...
    every_n = 1

    num_shifts = (len(mag) - m) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(m)
    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)

    xm = np.asarray(mag)[indexer]
    B = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm])

    m += 1
    num_shifts = (len(mag) - m) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(m)
    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)

    xmp1 = np.asarray(mag)[indexer]
    A = np.sum([np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1])

    SampEn = -np.log(A / B)

    return SampEn

def sum_values(time, mag, magerr, apply_weights=True):
    """
    Sums over all mag values.
    
    If apply_weights=True, the formula for weighted mean is used to calculate the sum of the magnitudes. 
    The weights are given by the inverse square of the magnitudes' errors.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The sum of all the measurements, normalized to the size of the lightcurve.
    """

    if apply_weights:
        return np.sum(mag/magerr**2)/np.sum(1/magerr**2)
    else:
        return np.sum(mag)/len(mag)

def time_reversal_asymmetry(time, mag, magerr, lag=1, apply_weights=True):
    """
    Derives the time reversal asymmetric statistic introduced by Fulcher.

    See: (Fulcher, B.D., Jones, N.S. (2014). Highly comparative feature-based time-series classification. Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.)
    
    We incorporate errors by dividing each term by the square of its corresponding magerr, 
    which effectively gives more weight to terms with smaller errors. Note that this modification 
    assumes that the errors are Gaussian and uncorrelated, which may not always be true in practice.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        lag : int
            The lag to apply. Defaults to 1.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The time reversal symmetry statistic.
    """

    n = len(mag)

    if 2 * lag >= n:
        return 0
    else:
        one_lag = np.roll(mag, -lag)
        two_lag = np.roll(mag, 2 * -lag)
        if apply_weights:
            weights = 1.0 / (magerr ** 2)
            weighted_mean = np.sum(mag * weights) / np.sum(weights ** 2)
            result = ((two_lag * two_lag * one_lag - one_lag * mag * mag) / (magerr * magerr)) / weights
            return np.mean(result[:n - 2 * lag])
        else:
            result = (two_lag * two_lag * one_lag - one_lag * mag * mag)
            return np.mean(result[:n - 2 * lag])

def variance(time, mag, magerr, apply_weights=True):
    """
    Calculates the variance, or the weighted variance of the light curve if apply_weights=True.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The variance of the lightcurve measurements.
    """

    if apply_weights:
        return np.sum((mag - np.mean(mag))**2 / magerr**2) / np.sum(1/magerr**2)
    else:
        return np.var(mag)

def variance_larger_than_standard_deviation(time, mag, magerr, apply_weights=True):
    """
    This feature denotes if the variance of x is greater than its standard deviation. 
    Is equal to variance of x being larger than 1. 1 is True, 0 is False. 

    If apply_weights=True a weighting factor to the magnitude values when computing the variance and standard deviation will be used. 
    This factor gives more weight to the more precise measurements and less weight to the less precise measurements.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    int    
        Whether or not the lightcurve's variance is greater than the standard deviation, 1 for True, 0 for False.
    """

    if apply_weights:
        weight = 1.0 / (magerr * magerr)
        weighted_mean = np.sum(mag * weight) / np.sum(weight)
        weighted_var = np.sum(weight * (mag - weighted_mean) ** 2) / np.sum(weight)
        weighted_std = np.sqrt(weighted_var)
        if weighted_var > weighted_std:
            return 1
        else:
            return 0
    else:
        var = np.var(mag)
        if var > np.sqrt(var):
            return 1
        else:
            return 0

def variation_coefficient(time, mag, magerr, apply_weights=True):
    """
    Calculates the variation coefficient (standard error / mean, gives the relative value of variation around mean) of x.
    
    We incorporate errors by using the weighted standard deviation and weighted mean.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The lightcurve standard deviation over the mean.
    """

    if apply_weights:
        weighted_mean = np.sum(mag / magerr**2) / np.sum(1 / magerr**2)
        weighted_std = np.sqrt(np.sum((mag - weighted_mean)**2 / magerr**2) / np.sum(1 / magerr**2))
        if weighted_mean != 0:
            return weighted_std / weighted_mean
        else:
            return np.nan
    else:
        mean = np.mean(mag)
        if mean != 0:
            return np.std(mag) / mean
        else:
            return np.nan

def large_standard_deviation(time, mag, magerr, r=.3, apply_weights=True):
    """
    Does time series have "large" standard deviation?

    Boolean variable denoting if the standard dev of x is higher than 'r' times the range = difference between max and
    min of x. To incorporate errors we use the weighted standard deviation instead of the regular standard deviation.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        r : float
            The percentage of the range to compare with. Must be between 0 and 1. Defaults to 0.3.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    int    
        Whether or not the lightcurve has a large standard deviation, 1 for True, 0 for False.
    """

    if apply_weights:   
        weights = 1/magerr**2  # calculate weights from magerr
        weighted_std = np.sqrt(np.sum(weights * (mag - np.average(mag, weights=weights))**2) / np.sum(weights))
        if weighted_std > (r * (np.max(mag) - np.min(mag))):
            return 1
        else:
            return 0
    else:
        if np.std(mag) > (r * (np.max(mag) - np.min(mag))):
            return 1
        else:
            return 0

def symmetry_looking(time, mag, magerr, r=0.5, apply_weights=True):
    """
    Check to see if the distribution of the mag "looks symmetric". This is the case if:

    | mean(X)-median(X)| < r * (max(X)-min(X))

    where r is the percentage of the range to compare with.
    
    If apply_weights=True, the weighted mean and the weighted median are used instead of the regular mean and median.
    
    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        r : float
            The percentage of the range to compare with. Must be between 0 and 1. Deaults to 0.5.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    int    
        Whether or not the lightcurve appears symmetric, 1 for True, 0 for False.
    """

    if apply_weights:
        weights = 1 / magerr ** 2
        w_mean = np.sum(mag * weights) / np.sum(weights)
        sorted_indices = np.argsort(mag)
        cum_weights = np.cumsum(weights[sorted_indices])
        median_index = np.searchsorted(cum_weights, 0.5 * np.sum(weights))
        w_median = mag[sorted_indices[median_index]]
        max_min_difference = np.max(mag) - np.min(mag)
        if np.abs(w_mean - w_median) < (r * max_min_difference):
            return 1
        else:
            return 0
    else:
        mean_median_difference = np.abs(np.mean(mag) - np.median(mag))
        max_min_difference = np.max(mag) - np.min(mag)
        if mean_median_difference < ( r * max_min_difference):
            return 1
        else:
            return 0
  
def index_mass_quantile(time, mag, magerr, r=0.5, apply_weights=True):
    """
    Calculates the relative index i of time series x where r% of the mass of x lies left of i.
    For example for r = 50% this feature will return the mass center of the time series.
    
    Errors are incorporated into this function by weighing the contributions of each data point with its inverse variance.
    
    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        r : float 
            The percentage of the range to compare with. Must be between 0 and 1. Defaults to 0.5.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The relative index of the lightcurve measurement that is to the right of the 50% of the cumulative sum.
    """

    if apply_weights:
        abs_mag = np.abs(mag)
        inv_var = 1.0 / (magerr ** 2)
        weighted_abs_mag = np.sum(abs_mag * inv_var)
        #Calculate the cumulative sum of the weighted absolute values of mag
        cum_weighted_abs_mag = np.cumsum(abs_mag * inv_var) / weighted_abs_mag
        #Find the index i where r% of the mass of x lies left of i
        i = np.argmax(cum_weighted_abs_mag >= r) + 1
        #Return the relative index 
        return i / len(mag)
    else:
        abs_x = np.abs(mag)
        s = np.sum(abs_x)
        mass_centralized = np.cumsum(abs_x) / s
        return (np.argmax(mass_centralized >= r) + 1) / len(mag)

def number_cwt_peaks(time, mag, magerr, n=30, apply_weights=True):
    """
    Number of different peaks in the magnitude array.

    To estimate the numbers of peaks, x is smoothed by a ricker wavelet for widths ranging from 1 to n. This feature
    calculator returns the number of peaks that occur at enough width scales and with sufficiently high
    Signal-to-Noise-Ratio (SNR). Weights are not meant to be applied in this case.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        n : int 
            The maximum time width to consider. Defaults to 30.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The number of peaks in the lightcurve after smoothing with a wavelet transformation.
    """

    val = len(ssignal.find_peaks_cwt(vector=mag, widths=np.array(list(range(1, n + 1))), wavelet=ssignal.ricker))
    return val / len(mag)

def permutation_entropy(time, mag, magerr, tau=1, dimension=3, apply_weights=True):
    """
    Calculate the permutation entropy.

    Ref: https://www.aptech.com/blog/permutation-entropy/
         Bandt, Christoph and Bernd Pompe.
         “Permutation entropy: a natural complexity measure for time series.”
         Physical review letters 88 17 (2002): 174102 
    
    In this modified version, if apply_weights=True, we compute the weights as the reciprocal of the magnitude errors 
    and divide each count by the corresponding weight. Then we compute the weighted average probabilities and return
    the negative sum of the probabilities times their logarithms.

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        tau : int
            The embedded time delay that determines the time separation between the measurements.
            Defaults to 1.
        dimension : int
            The embedding dimension to use. Defaults to 3.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The permutation entropy of the lightcurve.
    """

    num_shifts = (len(mag) - dimension) // tau + 1
    shift_starts = tau * np.arange(num_shifts)
    indices = np.arange(dimension)
    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)

    X = np.asarray(mag)[indexer]
    permutations = np.argsort(np.argsort(X))
    counts = np.unique(permutations, axis=0, return_counts=True)[1]

    if apply_weights:
        weights = 1 / np.asarray(magerr)[indexer]
        weighted_counts = counts / np.expand_dims(weights.reshape(-1), axis=1)
        probs = np.sum(weighted_counts, axis=1) / np.sum(weighted_counts)
    else:
        probs = counts / len(permutations)

    return -np.sum(probs * np.log(probs))

def quantile(time, mag, magerr, r=0.75, apply_weights=True):
    """
    Calculates the r quantile of the mag. This is the value of mag greater than r% of the ordered values.

    Errors are not incorporated in this function. 

    Parameters:
    ----------
        time : array
            The timestamps of the corresponding mag and magerr measurements. Must be an array.
        mag : array
            The time-varying intensity of the lightcurve. Must be an array.
        magerr : array
            Photometric error for the intensity. Must be an array.
        r : float
            The percentage of the range to compare with. Must be between 0 and 1. Defaults to 0.75.
        apply_weights : bool, optional
            Whether to apply weights based on the magnitude errors. Defaults to True.

    Returns:
    ------- 
    float    
        The 75th quantile of the ordered lightcurve measurements.
    """

    return np.quantile(mag, r)

