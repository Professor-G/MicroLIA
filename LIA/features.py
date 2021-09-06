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
from scipy.integrate import quad
from scipy.signal import find_peaks_cwt, ricker
from scipy.stats import shapiro, linregress, anderson
import warnings

warnings.filterwarnings("ignore")

def shannon_entropy(mag, magerr):
    """
    Shannon entropy (Shannon et al. 1949) is used as a metric to quantify the amount of
    information carried by a signal. The procedure employed here follows that outlined by
    (D. Mislis et al. 2015). The probability of each point is given by a Cumulative Distribution
    Function (CDF). Following the same procedure as (D. Mislis et al. 2015), this function employs
    both the normal and inversed gaussian CDF, with the total shannon entropy given by a combination of
    the two. See: (SIDRA: a blind algorithm for signal detection in photometric surveys, D. Mislis et al., 2015)
     
    Parameters
    ----------   
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------  
    rtype: float
    """
    
    mean = np.median(mag)
    RMS = root_mean_squared(mag)
    
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
        integ, err = quad(integral, 0, x)
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

def con(mag,magerr):
    """
    Con is defined as the number of clusters containing three or more
    consecutive observations with magnitudes brighter than the reference
    magnitude plus 3 standard deviations. For a microlensing event Con = 1,
    assuming a  flat lightcurve prior to the event. The magnitude measurements
    are split into bins such that the reference  magnitude is defined as the mean
    of the measurements in the largest bin.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.
   
    Returns
    -------  
    rtype: float       
    """

    mean = np.median(mag)
    std = np.std(mag)
    deviatingThreshold = mean - 3*std
    con = 0
    deviating = False

    a = np.argwhere(mag < deviatingThreshold)
    if len(a) < 3:
        return 0
    else:
        for i in range(len(mag)-2):
            first = mag[i]
            second = mag[i+1]
            third = mag[i+2]
            if (first <= deviatingThreshold and
                second <= deviatingThreshold and
                third <= deviatingThreshold):
                if (not deviating):
                    con += 1
                    deviating = True
                elif deviating:
                    deviating = False

    return con

    
def con2(mag,magerr):
    """
    Con is defined as the number of clusters containing three or more
    consecutive observations with magnitudes brighter than the mean plus
    2 standard deviations. For a microlensing event Con = 1, assuming a
    flat lightcurve prior to the event.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------  
    :rtype: float
    """
    mean = np.median(mag)
    std = np.std(mag)
    deviatingThreshold = mean - 2*std
    con = 0
    deviating = False
    
    a = np.argwhere(mag < deviatingThreshold)
    if len(a) < 3:
        return 0
    else:
        for i in range(len(mag)-2):
            first = mag[i]
            second = mag[i+1]
            third = mag[i+2]
            if (first <= deviatingThreshold and
                second <= deviatingThreshold and
                third <= deviatingThreshold):
                if (not deviating):
                    con += 1
                    deviating = True
                elif deviating:
                    deviating = False

    return con

def kurtosis(mag):
    """"
    Kurtosis function returns the calculated kurtosis of the lightcurve.
    It's a measure of the peakedness (or flatness) of the lightcurve relative
    to a normal distribution. See: www.xycoon.com/peakedness_small_sample_test_1.htm
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    
    Returns
    -------     
    rtype: float
    """

    mean = np.median(mag)
    std = np.std(mag)
    n = np.float(len(mag))
    
    try:
        kurtosis = (n*(n+1.)/((n-1.)*(n-2.)*(n-3.))*sum(((mag - mean)/std)**4)) - \
            (3.*((n-1.)**2)/((n-2.)*(n-3.)))
    except ZeroDivisionError:
        kurtosis = 0.0

    return kurtosis

def skewness(mag):
    """
    Skewness measures the assymetry of a lightcurve, with a positive skewness
    indicating a skew to the right, and a negative skewness indicating a skew to the left.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------        
    :rtype: float
    """
    
    mean = np.median(mag)
    std = np.std(mag)
    n = np.float(len(mag))
    try:
        skewness = (1./n)*sum((mag - mean)**3/std**3)
    except ZeroDivisionError:
        skewness = 0.0
    return skewness

def vonNeumannRatio(mag):
    """
    The von Neumann ratio η was defined in 1941 by John von Neumann and serves as the
    mean square successive difference divided by the sample variance. When this ratio is small,
    it is an indication of a strong positive correlation between the successive photometric 
    data points. See: (J. Von Neumann, The Annals of Mathematical Statistics 12, 367 (1941))
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    
    Returns
    -------  
    rtype: float
    """
    
    n = np.float(len(mag))
    delta = sum((mag[1:] - mag[:-1])**2 / (n-1.))
    sample_variance = np.std(mag)**2
    try:
        vonNeumannRatio = delta / sample_variance
    except ZeroDivisionError:
        vonNeumannRatio = 0.0
    
    return vonNeumannRatio


def stetsonJ(mag, magerr):
    """
    The variability index J was first suggested by Peter B. Stetson and serves as a
    measure of the correlation between the data points, tending to 0 for variable stars
    and getting large as the difference between the successive data points increases.
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    ------- 
    rtype: float
    """
    
    n = np.float(len(mag))
    mean = np.median(mag)
    delta_list=[]
    
    for i in range(0, len(mag)-1):
        delta = np.sqrt(n/(n-1.))*((mag[i] - mean)/magerr[i])
        delta2 = np.sqrt(n/(n-1.))*((mag[i+1] - mean)/magerr[i+1])

        
        delta_list.append(np.nan_to_num(delta*delta2))
    
    stetj = sum(np.sign(delta_list)*np.sqrt(np.abs(delta_list)))

    #stetj blows up in derivative space!
    if stetj > 10**6:
        stetj = 10**6
    elif stetj < 10**-6:
        stetj =  10**-6

    return stetj


def stetsonK(mag, magerr):
    """
    The variability index K was first suggested by Peter B. Stetson and serves as a
    measure of the kurtosis of the magnitude distribution.
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).

    Parameters
    ----------   
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------    
    rtype: float
    """  
    
    n = np.float(len(mag))
    mean = np.median(mag)
    try:
        delta = np.sqrt((n/(n-1.)))*((mag - mean)/magerr)
    except ZeroDivisionError:
        delta = 0.0

    try:
        stetsonK = ((1./n)*sum(abs(delta)))/(np.sqrt((1./n)*sum(delta**2)))
    except ZeroDivisionError:
        stetsonK = 0.0
            
    return np.nan_to_num(stetsonK)

def stetsonL(mag,magerr):
    """
    The variability index L was first suggested by Peter B. Stetson and serves as a
    means of distinguishing between different types of variation. When individual random
    errors dominate over the actual variation of the signal, K approaches 0.798 (Gaussian limit).
    Thus, when the nature of the errors is Gaussian, stetsonL = stetsonJ, except it will be amplified
    by a small factor for smoothly varying signals, or suppressed by a large factor when data
    is infrequent or corrupt. 
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
        
    Parameters
    ----------   
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------    
    rtype: float    
    """  
    
    stetL = (stetsonJ(mag,magerr)*stetsonK(mag,magerr)) / 0.798
    return stetL

def median_buffer_range(mag):
    """
    This function returns the ratio of points that are between plus or minus 10% of the
    amplitude value over the mean.
        
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    
    Returns
    -------     
    rtype: float
    """
    
    n = np.float(len(mag))
    amp = amplitude(mag)
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    a = mean - amp/10.
    b = mean + amp/10.
    
    median_buffer_range = len(np.argwhere((mag > a) & (mag < b))) / n
    return median_buffer_range

def median_buffer_range2(mag):
    """
    This function returns the ratio of points that are more than 20% of the amplitude
    value over the mean.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    
    Returns
    ------- 
    rtype: float
    """
    
    n = np.float(len(mag))
    amp = amplitude(mag)
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    a = mean - amp/5.
    
    median_buffer_range = len(np.argwhere((mag < a))) / n
    return median_buffer_range

def std_over_mean(mag):
    """
    A measure of the ratio of standard deviation and mean.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    
    Returns
    -------     
    rtype: float
    """

    std = np.std(mag)
    mean = np.median(mag)
    
    std_over_mean = std/mean
    return std_over_mean

def amplitude2(mag):
    """
    This amplitude metric is defined as the difference between the maximum magnitude
    measurement and the lowest magnitude measurement. We account for outliers by
    removing the upper and lower 2% of magnitudes.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    
    Returns
    ------- 
    rtype: float
    """

    lower = np.percentile(mag, 2)
    upper = np.percentile(mag, 98)
    amplitude2 = upper - lower
    
    return amplitude2


def amplitude(mag):
    """
    This amplitude metric is defined as the difference between the maximum magnitude
    measurement and the lowest magnitude measurement, divided by 2. We account for 
    outliers by removing the upper and lower 2% of magnitudes.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    
    Returns
    ------- 
    rtype: float
    """

    mag_range = amplitude2(mag)
    amplitude = mag_range / 2.0

    return amplitude

def median_distance(time, mag):
    """
    This function calculates the median eucledian distance between each photometric 
    measurement, helpful metric for detecting overlapped lightcurves.
    
    Parameters
    ----------
    time : time of observations.
    mag: the time-varying intensity of the lightcurve. Must be an array.
    
    Returns
    -------     
    rtype: float    
    """

    delta_mag = (mag[1:] - mag[:-1])**2
    delta_t = (time[1:] - time[:-1])**2
    
    distance = np.median(np.sqrt(delta_mag + delta_t))
    return distance

def above1(mag):
    """
    This function measures the ratio of data points that are above 1 standard deviation
    from the median magnitude.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------     
    rtype: float
    """
    
    a = np.median(mag) + np.std(mag)
    above1 = len(np.argwhere(mag > a) )
    
    return above1

def above3(mag):
    """
    This function measures the ratio of data points that are above 3 standard deviations
    from the median magnitude.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------     
    rtype: float
    """
    
    a = np.median(mag) + 3*np.std(mag)
    above3 = len(np.argwhere(mag > a) )
    
    return above3

def above5(mag):
    """
    This function measures the ratio of data points that are above 5 standard deviations
    from the median magnitude.
        
    Parameters
    ----------   
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------    
    rtype: float   
    """
    
    a = np.median(mag) + 5*np.std(mag)
    above5 = len(np.argwhere(mag > a))
    
    return above5

def below1(mag):
    """
    This function measures the ratio of data points that are below 1 standard deviations
    from the median magnitude.
        
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------     
    rtype: float
    """
    
    a = np.median(mag) - np.std(mag)
    below1 = len(np.argwhere(mag < a))
    
    return below1

def below3(mag):
    """
    This function measures the ratio of data points that are below 3 standard deviations
    from the median magnitude.
        
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------     
    rtype: float
    """
    
    a = np.median(mag) - 3*np.std(mag)
    below3 = len(np.argwhere(mag < a))
    
    return below3

def below5(mag):
    """
    This function measures the ratio of data points that are below 5 standard deviations
    from the median magnitude.
        
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------     
    rtype: float
    """
    
    a = np.median(mag) - 5*np.std(mag)
    below5 = len(np.argwhere(mag < a))
    
    return below5

def medianAbsDev(mag):
    """"
    A measure of the mean average distance between each magnitude value
    and the mean magnitude. https://en.wikipedia.org/wiki/Median_absolute_deviation 
        
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """
    
    array = np.ma.array(mag).compressed() 
    med = np.median(array)
    medianAbsDev = np.median(np.abs(array - med))
    
    return medianAbsDev

def root_mean_squared(mag):
    """
    A measure of the root mean square deviation.
        
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    mean = np.median(mag)
    rms = np.sqrt(mean**2)
    
    return rms

def meanMag(mag, magerr):
    """
    Calculates mean magnitude, weighted by the errors.
        
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------     
    rtype: float
    """
        
    mean = sum(mag/magerr**2)/sum(1./magerr**2)
    
    return mean

def integrate(mag):
    """
    Integrate magnitude using the trapezoidal rule.
    See: http://en.wikipedia.org/wiki/Trapezoidal_rule

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    integral = np.trapz(mag)

    return integral

def auto_corr(mag):
    """
    Similarity between observations as a function of a time lag between them.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------     
    rtype: float
    """

    auto_corr = np.corrcoef(mag[:-1],mag[1:])[1,0]

    return auto_corr


def peak_detection(mag):
    """
    Function to detect number of peaks.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """
    
    mag = abs(mag - np.median(mag))

    try:
        indices = peakutils.indexes(mag, thres=.5, min_dist=10)
    except ValueError:
        indices = []

    return len(indices)


def normalize(mag, magerr):
    """
    Normalizes the magnitude to range from 0 to 1, scaling
    magerr in the proccess.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------     
    rtype: array
    """
    
    mag2 = (mag - np.min(mag)) / np.ptp(mag)
    magerr = magerr*(mag2/mag)
    
    return np.array(mag2), np.array(magerr)


#Below stats from Richards et al (2011)

def MaxSlope(time, mag):
    """
    Examining successive (time-sorted) magnitudes, the maximal first difference
    (value of delta magnitude over delta time)

    Parameters
    ----------
    time: time of observations. Must be an array.
    mag: the time-varying intensity of the lightcurve. Must be an array.
    magerr: photometric error for the intensity. Must be an array.

    Returns
    -------     
    rtype: float
    """

    slope = np.abs(mag[1:] - mag[:-1]) / (time[1:] - time[:-1])

    return np.max(slope)

def LinearTrend(time, mag):
    """
    Slope of a linear fit to the light-curve.
    """

    regression_slope = linregress(time, mag)[0]

    return regression_slope

def PairSlopeTrend(mag):
    """
    Considering the last 30 (time-sorted) measurements of source magnitude,
    the fraction of increasing first differences minus the fraction of
    decreasing first differences.
    Percentage of all pairs of consecutive flux measurements that have positive slope

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    data_last = mag[-30:]

    PST = (len(np.where(np.diff(data_last) > 0)[0]) - len(np.where(np.diff(data_last) <= 0)[0])) / 30.0

    return PST

def FluxPercentileRatioMid20(mag):
    """
    In order to caracterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (60th - 40th) over (95th - 5th)

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)

    F_60_index = int(math.ceil(0.60 * lc_length))
    F_40_index = int(math.ceil(0.40 * lc_length))
    F_5_index = int(math.ceil(0.05 * lc_length))
    F_95_index = int(math.ceil(0.95 * lc_length))

    F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid20 = F_40_60 / F_5_95

    return F_mid20

def FluxPercentileRatioMid35(mag):
    """
    In order to caracterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (67.5th - 32.5th) over (95th - 5th)

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)

    F_325_index = int(math.ceil(0.325 * lc_length))
    F_675_index = int(math.ceil(0.675 * lc_length))
    F_5_index = int(math.ceil(0.05 * lc_length))
    F_95_index = int(math.ceil(0.95 * lc_length))

    F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid35 = F_325_675 / F_5_95

    return F_mid35

def FluxPercentileRatioMid50(mag):
    """
    In order to caracterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (75th - 25th) over (95th - 5th)

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)

    F_25_index = int(math.ceil(0.25 * lc_length))
    F_75_index = int(math.ceil(0.75 * lc_length))
    F_5_index = int(math.ceil(0.05 * lc_length))
    F_95_index = int(math.ceil(0.95 * lc_length))

    F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid50 = F_25_75 / F_5_95

    return F_mid50

def FluxPercentileRatioMid65(mag):
    """
    In order to caracterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (82.5th - 17.5th) over (95th - 5th)

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)

    F_175_index = int(math.ceil(0.175 * lc_length))
    F_825_index = int(math.ceil(0.825 * lc_length))
    F_5_index = int(math.ceil(0.05 * lc_length))
    F_95_index = int(math.ceil(0.95 * lc_length))

    F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid65 = F_175_825 / F_5_95

    return F_mid65

def FluxPercentileRatioMid80(mag):
    """
    In order to caracterize the sorted magnitudes distribution we use percentiles. 
    If F5,95 is the difference between 95% and 5% magnitude values, we calculate the following:
    Ratio of flux percentiles (90th - 10th) over (95th - 5th)

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)

    F_10_index = int(math.ceil(0.10 * lc_length))
    F_90_index = int(math.ceil(0.90 * lc_length))
    F_5_index = int(math.ceil(0.05 * lc_length))
    F_95_index = int(math.ceil(0.95 * lc_length))

    F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid80 = F_10_90 / F_5_95

    return F_mid80

def PercentAmplitude(mag):
    """
    The largest absolute departure from the median flux, divided by the median flux
    Largest percentage difference between either the max or min magnitude and the median.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    median = np.median(mag)
    distance_median = np.abs(mag - median)
    max_distance = np.max(distance_median)

    percent_amplitude = max_distance / median

    return percent_amplitude

def PercentDifferenceFluxPercentile(mag):
    """
    Ratio of F5,95 over the median flux.
    Difference between the 2nd & 98th flux percentiles.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    median = np.median(mag)

    sorted_data = np.sort(mag)
    lc_length = len(sorted_data)
    F_5_index = int(math.ceil(0.05 * lc_length))
    F_95_index = int(math.ceil(0.95 * lc_length))
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]

    percent_difference = F_5_95 / median

    return percent_difference


#Below stats from Kim (2015), used in Upsilon
#https://arxiv.org/pdf/1512.01611.pdf

def half_mag_amplitude_ratio(mag):
    """
    The ratio of the squared sum of residuals of magnitudes
    that are either brighter than or fainter than the mean
    magnitude. For EB-like variability, having sharp flux gradients around its eclipses, A is larger
    than 1

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    # For fainter magnitude than average.
    avg = np.median(mag)

    index = np.argwhere(mag > avg)
    lower_mag = mag[index]

    try:
        lower_weighted_std = (1./len(index))*np.sum((lower_mag - avg)**2)
    except ZeroDivisionError:
        lower_weighted_std = 0

    # For brighter magnitude than average.
    index = np.argwhere(mag <= avg)
    higher_mag = mag[index]
    try:
        higher_weighted_std = (1./len(index))*np.sum((higher_mag - avg)**2)
    except ZeroDivisionError:
        higher_weighted_std = 0

    # Return ratio.
    try:
        ratio = np.sqrt(lower_weighted_std / higher_weighted_std)
    except ZeroDivisionError:
        ratio = 0
            
    return ratio


def cusum(mag):
    """
    Range of cumulative sum

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    c = np.cumsum(mag - np.median(mag)) * 1./(len(mag)*np.std(mag))

    return np.max(c) - np.min(c)

def shapiro_wilk(mag):
    """
    Normalization-test.
    The Shapiro-Wilk test tests the null hypothesis that the 
    data was drawn from a normal distribution.
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """
    shapiro_w = shapiro(mag)[0]

    return shapiro_w


#following stats pulled from FEETS
#https://feets.readthedocs.io/en/latest/tutorial.html

def AndersonDarling(mag):
    """
    The Anderson-Darling test is a statistical test of whether a given 
    sample of data is drawn from a given probability distribution. 
    When applied to testing if a normal distribution adequately describes a set of data, 
    it is one of the most powerful statistical tools for detecting most departures from normality.
    
    From Kim et al. 2009: "To test normality, we use the Anderson–Darling test (Anderson & Darling 1952; Stephens 1974) 
    which tests the null hypothesis that a data set comes from the normal distribution."
    (Doi:10.1111/j.1365-2966.2009.14967.x.)

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    ander = anderson(mag)[0]

    return 1 / (1.0 + np.exp(-10 * (ander - 0.3)))



def Gskew(mag):
    """
    Median-based measure of the skew
    Gskew = mq3 + mq97 − 2m
    mq3  is the median of magnitudes lesser or equal than the quantile 3.
    mq97 is the median of magnitudes greater or equal than the quantile 97.
    2m is 2 times the median magnitude.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    median_mag = np.median(mag)
    F_3_value = np.percentile(mag, 3)
    F_97_value = np.percentile(mag, 97)

    gs = (np.median(mag[mag <= F_3_value]) + np.median(mag[mag >= F_97_value]) - 2*median_mag)

    return gs


# The following features are derived using the Python package tsfresh.
# Please see: http://tsfresh.readthedocs.io/en/latest/

def abs_energy(mag):
    """
    Returns the absolute energy of the time series, defined to be the sum over the squared
    values of the time-series.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    energy = np.dot(mag, mag)

    return energy

def abs_sum_changes(mag):
    """
    Returns sum over the abs value of consecutive changes in mag.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    val = np.sum(np.abs(np.diff(mag)))

    return val

def benford_correlation(mag):
    """
    Useful for anomaly detection applications. Returns the 
    correlation from first digit distribution when compared to 
    the Newcomb-Benford’s Law distribution

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    # retrieve first digit from data
    x = np.array([int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(mag))])

    # benford distribution
    benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
    data_distribution = np.array([(x == n).mean() for n in range(1, 10)])

    # np.corrcoef outputs the normalized covariance (correlation) between benford_distribution and data_distribution.
    # In this case returns a 2x2 matrix, the  [0, 1] and [1, 1] are the values between the two arrays
    benford_corr = np.corrcoef(benford_distribution, data_distribution)[0, 1]

    return benford_corr

def c3(mag, lag=1):
    """
    A measure of non-linearity.
    See: Measure of non-linearity in time series: [1] Schreiber, T. and Schmitz, A. (1997).
    Discrimination power of measures for nonlinearity in a time series
    PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    lag: the lag that should be used in the calculation of the feature.

    Returns
    -------     
    rtype: float
    """

    n = len(mag)
    if 2 * lag >= n:
        val = 0
    else:
        val = np.mean((np.roll(mag, 2 * -lag) * np.roll(mag, -lag) * mag)[0 : (n - 2 * lag)])

    return val

def complexity(mag):
    """
    This function calculator is an estimate for a time series complexity.
    A higher value represents more complexity (more peaks,valleys,etc.)
    See: Batista, Gustavo EAPA, et al (2014). CID: an efficient complexity-invariant 
    distance for time series. Data Mining and Knowledge Difscovery 28.3 (2014): 634-669.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    mag = np.diff(mag)
    val = np.sqrt(np.dot(mag, mag))

    return val

def count_above(mag):
    """
    Number of values higher than the median

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    m = np.median(mag)
    val = np.where(mag > m)[0].size

    return val

def count_below(mag):
    """
    Number of values below the median

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    m = np.median(mag)
    val = np.where(mag < m)[0].size

    return val

def first_loc_max(mag):
    """
    Returns location of maximum mag relative to the 
    lenght of mag array.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    val = np.argmax(mag) / len(mag) if len(mag) > 0 else np.NaN

    return val

def first_loc_min(mag):
    """
    Returns location of minimum mag relative to the 
    lenght of mag array.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    val = np.argmin(mag) / len(mag) if len(mag) > 0 else np.NaN
    
    return val

def check_for_duplicate(mag):
    """
    Checks if any val in mag repeats.
    1 if True, 0 if False

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    if mag.size != np.unique(mag).size:
        val = 1
    else:     
        val = 0

    return val

def check_for_max_duplicate(mag):
    """
    Checks if the maximum value in mag repeats.
    1 if True, 0 if False

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    if np.sum(mag == np.max(mag)) >= 2:
        val = 1
    else:     
        val = 0

    return val

def check_for_min_duplicate(mag):
    """
    Checks if the minimum value in mag repeats.
    1 if True, 0 if False.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """
    
    if np.sum(mag == np.min(mag)) >= 2:
        val = 1
    else:     
        val = 0

    return val

def check_max_last_loc(mag):
    """
    Returns position of last maximum mag relative to
    the length of mag array.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    val = 1.0 - np.argmax(mag[::-1]) / len(mag) if len(mag) > 0 else np.NaN

    return val

def check_min_last_loc(mag):
    """
    Returns position of last minimum mag relative to
    the length of mag array.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    val = 1.0 - np.argmin(mag[::-1]) / len(mag) if len(mag) > 0 else np.NaN

    return val

def get_length_sequences_where(x):
    """
    This method calculates the length of all sub-sequences where the array is either True or 1.

    Parameters
    ---------   
    x: An iterable containing only 1, True, 0 and False values

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    lag: the lag that should be used in the calculation of the feature.

    Returns
    -------     
    rtype: list
        A list with the length of all sub-sequences where the array is either True or False.
    """
 
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]

def longest_strike_above(mag):
    """
    Returns the length of the longest consecutive subsequence in 
    mag that is bigger than the median. 

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    val = np.max(get_length_sequences_where(mag > np.median(mag))) if mag.size > 0 else 0

    return val

def longest_strike_below(mag):
    """
    Returns the length of the longest consecutive subsequence in mag 
    that is smaller than the median.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    val = np.max(get_length_sequences_where(mag < np.median(mag))) if mag.size > 0 else 0

    return val

def mean_change(mag):
    """
    Returns mean over the differences between subsequent observations.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    val = (mag[-1] - mag[0]) / (len(mag) - 1) if len(mag) > 1 else np.NaN

    return val

def mean_abs_change(mag):
    """
    Returns mean over the abs differences between subsequent observations.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    val = np.mean(np.abs(np.diff(mag)))

    return val

def mean_n_abs_max(mag, number_of_maxima=1):
    """
    Calculates the arithmetic mean of the n absolute maximum values of the time series, n = 1.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    number_of_maxima: the number of maxima to be considered

    Returns
    -------     
    rtype: float
    """

    n_absolute_maximum_values = np.sort(np.absolute(mag))[-number_of_maxima:]
    val = np.mean(n_absolute_maximum_values) if len(mag) > number_of_maxima else np.NaN

    return val

def mean_second_derivative(mag):
    """
    Returns the mean value of a central approximation of the second derivative.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    val = (mag[-1] - mag[-2] - mag[1] + mag[0]) / (2 * (len(mag) - 2)) if len(mag) > 2 else np.NaN

    return val

def number_of_crossings(mag):
    """
    Calculates the number of crossings of x on the median, m. A crossing is defined as two 
    sequential values where the first value is lower than m and the next is greater, 
    or vice-versa. If you set m to zero, you will get the number of zero crossings.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    positive = mag > np.median(mag)
    val = np.where(np.diff(positive))[0].size

    return val

def number_of_peaks(mag, n=7):
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

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    n: the support of the peak

    Returns
    -------     
    rtype: int
    """

    x_reduced = mag[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = x_reduced > np.roll(mag, i)[n:-n]

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= x_reduced > np.roll(mag, -i)[n:-n]

    val = np.sum(res)

    return val

def ratio_recurring_points(mag):
    """
    Returns the ratio of unique values, that are present in the time 
    series more than once, normalized to the number of data points. 
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    unique, counts = np.unique(mag, return_counts=True)

    if counts.shape[0] == 0:
        return 0

    val = np.sum(counts > 1) / float(counts.shape[0])

    return val

def into_subchunks(mag, subchunk_length, every_n=1):
    """
    Split the time series x into subwindows of length "subchunk_length", starting every "every_n".
    This is NOT a standalone feature, it is required for the computation of a few metrics.

    For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix

        0  2  4
        1  3  5
        2  4  6

    with the settings subchunk_length = 3 and every_n = 2
    """

    num_shifts = (len(mag) - subchunk_length) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(subchunk_length)

    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)

    return np.asarray(mag)[indexer]

def sample_entropy(mag):
    """
    Returns sample entropy: http://en.wikipedia.org/wiki/Sample_Entropy
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    m = 2  # common value for m, according to wikipedia...
    tolerance = 0.2 * np.std(mag)  # 0.2 is a common value for r, according to wikipedia...

    xm = into_subchunks(mag, m)
    B = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm])

    xmp1 = into_subchunks(mag, m + 1)
    A = np.sum([np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1])

    SampEn = -np.log(A / B)

    return SampEn

def sum_values(mag):
    """
    Sums over all mag values.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    val = np.sum(mag)

    return val

def time_reversal_asymmetry(mag, lag=1):
    """
    Derives a feature introduced by Fulcher.
    See: (Fulcher, B.D., Jones, N.S. (2014). Highly comparative 
    feature-based time-series classification. Knowledge and Data Engineering, 
    IEEE Transactions on 26, 3026–3037.)
    
    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    lag: the lag that should be used in the calculation.

    Returns
    -------     
    rtype: float
    """

    n = len(mag)

    if 2 * lag >= n:
        val = 0
    else:
        one_lag = np.roll(mag, -lag)
        two_lag = np.roll(mag, 2 * -lag)
        val = np.mean((two_lag * two_lag * one_lag - one_lag * mag * mag)[0 : (n - 2 * lag)])

    return val

def variance(mag):
    """
    Returns the variance.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    val = np.var(mag)

    return val

def variance_larger_than_standard_deviation(mag):
    """
    This feature denotes if the variance of x is greater than its standard deviation. 
    Is equal to variance of x being larger than 1. 

    1 is True, 0 is False.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: int
    """

    var = np.var(mag)

    if var > np.sqrt(var):
        val = 1
    else:
        val = 0

    return val

def variation_coefficient(mag):
    """
    Returns the variation coefficient (standard error / mean, give relative value of variation around mean) of x.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.

    Returns
    -------     
    rtype: float
    """

    mean = np.mean(mag)

    if mean != 0:
        val = np.std(mag) / mean
    else:
        val = np.nan

    return val


def large_standard_deviation(mag, r=.3):
    """
    Does time series have "large" standard deviation?

    Boolean variable denoting if the standard dev of x is higher than 'r' times the range = difference between max and
    min of x.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    r: the percentage of the range to compare with.

    Returns
    -------     
    rtype: float
    """

    if np.std(mag) > (r * (np.max(mag) - np.min(mag))):
        val = 1
    else:
        val = 0

    return val

def symmetry_looking(mag, r=0.5):
    """
    Check to see if the distribution of the mag "looks symmetric". This is the case if:

    | mean(X)-median(X)| < r * (max(X)-min(X))

    where r is the percentage of the range to compare with.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    r: the percentage of the range to compare with.

    Returns
    -------     
    rtype: int
    """

    mean_median_difference = np.abs(np.mean(mag) - np.median(mag))
    max_min_difference = np.max(mag) - np.min(mag)

    if mean_median_difference < ( r * max_min_difference):
        val = 1
    else:
        val = 0

    return val
  
def index_mass_quantile(mag, r=0.5):
    """
    Calculates the relative index i of time series x where r% of the mass of x lies left of i.
    For example for r = 50% this feature will return the mass center of the time series.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    r: the percentage of the range to compare with.

    Returns
    -------     
    rtype: float
    """

    abs_x = np.abs(mag)
    s = np.sum(abs_x)

    mass_centralized = np.cumsum(abs_x) / s
    val = (np.argmax(mass_centralized >= r) + 1) / len(mag)

    return val

def number_cwt_peaks(mag, n=30):
    """
    Number of different peaks in the magnitude array.

    To estimamte the numbers of peaks, x is smoothed by a ricker wavelet for widths ranging from 1 to n. This feature
    calculator returns the number of peaks that occur at enough width scales and with sufficiently high
    Signal-to-Noise-Ratio (SNR)

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array. 
    param n : maximum time width to consider

    Returns
    -------   
    rtype: int
    """

    val = len(find_peaks_cwt(vector=mag, widths=np.array(list(range(1, n + 1))), wavelet=ricker))

    return val


def permutation_entropy(mag, tau=1, dimension=3):
    """
    Calculate the permutation entropy.

    Ref: https://www.aptech.com/blog/permutation-entropy/
         Bandt, Christoph and Bernd Pompe.
         “Permutation entropy: a natural complexity measure for time series.”
         Physical review letters 88 17 (2002): 174102 

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array. 
    tau: the embedded time delay that determines the time separation between the mag values.
    dimension: the embedding dimension.

    Returns
    -------   
    rtype: float
    """

    X = into_subchunks(mag, dimension, tau)
    permutations = np.argsort(np.argsort(X))
    counts = np.unique(permutations, axis=0, return_counts=True)[1]

    probs = counts / len(permutations)
    val = -np.sum(probs * np.log(probs))

    return val

def quantile(mag, r=0.75):
    """
    Calculates the r quantile of the mag. This is the value of mag greater than r% of the ordered values.

    Parameters
    ----------
    mag: the time-varying intensity of the lightcurve. Must be an array.
    r: the percentage of the range to compare with.

    Returns
    -------     
    rtype: float
    """

    val = np.quantile(mag, r)

    return val 

def remove_allbad(mjd, mag, magerr):
    """
    Function to remove bad photometric measurements.

    rtype : array
    """
    
    bad = np.where(np.isnan(magerr) == True)
    magerr = np.delete(magerr, bad)
    mjd = np.delete(mjd, bad)
    mag = np.delete(mag, bad)
    
    bad = np.where(magerr == 0)
    magerr = np.delete(magerr, bad)
    mjd = np.delete(mjd, bad)
    mag = np.delete(mag, bad)
    
    bad = np.where(mag == 0)
    magerr = np.delete(magerr, bad)
    mjd = np.delete(mjd, bad)
    mag = np.delete(mag, bad)
    
    return mjd, mag, magerr

#The following code was done to exclude outliers -- not currently employed#    
#mag, magerr = remove_bad(mag, magerr)
#diff = mag - meanMag(mag, magerr)
#hist, edges = np.histogram(diff, bins = 10)
#val = np.where(hist == max(hist))
#bin_range = np.where((diff > edges[val[0][0]]) & (diff < edges[val[0][0]+1]))
#mean = meanMag(mag[bin_range], magerr[bin_range])


