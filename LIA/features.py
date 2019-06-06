# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 14:30:12 2017
    
    @author: danielgodinez
"""

from __future__ import print_function
import numpy as np
from scipy.integrate import quad
import peakutils
import tsfresh.feature_extraction.feature_calculators as ts

def shannon_entropy(mag, magerr):
    """Shannon entropy (Shannon et al. 1949) is used as a metric to quantify the amount of
        information carried by a signal. The procedure employed here follows that outlined by
        (D. Mislis et al. 2015). The probability of each point is given by a Cumulative Distribution
        Function (CDF). Following the same procedure as (D. Mislis et al. 2015), this function employs
        both the normal and inversed gaussian CDF, with the total shannon entropy given by a combination of
        the two. See: (SIDRA: a blind algorithm for signal detection in photometric surveys, D. Mislis et al., 2015)
        
        :param mag: the time-varying intensity of the lightcurve. Must be an array.
        :param magerr: photometric error for the intensity. Must be an array.
        
        :rtype: float
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

def con(mag):
    """Con is defined as the number of clusters containing three or more
        consecutive observations with magnitudes brighter than the reference
        magnitude plus 3 standard deviations. For a microlensing event Con = 1,
        assuming a  flat lightcurve prior to the event. The magnitude measurements
        are split into bins such that the reference  magnitude is defined as the mean
        of the measurements in the largest bin.
        
        
        #The following code was done to exclude outliers -- not currently employed#
        
        mag, magerr = remove_bad(mag, magerr)
        diff = mag - meanMag(mag, magerr)
        hist, edges = np.histogram(diff, bins = 10)
        val = np.where(hist == max(hist))
        bin_range = np.where((diff > edges[val[0][0]]) & (diff < edges[val[0][0]+1]))
        mean = meanMag(mag[bin_range], magerr[bin_range])
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

def con2(mag):
    """Con is defined as the number of clusters containing three or more
        consecutive observations with magnitudes brighter than the mean plus
        2 standard deviations. For a microlensing event Con = 1, assuming a
        flat lightcurve prior to the event.
        
    
        #The following code was done to exclude outliers -- not currently employed#
        
        mag, magerr = remove_bad(mag, magerr)
        diff = mag - meanMag(mag, magerr)
        hist, edges = np.histogram(diff, bins = 10)
        val = np.where(hist == max(hist))
        bin_range = np.where((diff > edges[val[0][0]]) & (diff < edges[val[0][0]+1]))
        mean = meanMag(mag[bin_range], magerr[bin_range])
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
    """"Kurtosis function returns the calculated kurtosis of the lightcurve.
        It's a measure of the peakedness (or flatness) of the lightcurve relative
        to a normal distribution. See: www.xycoon.com/peakedness_small_sample_test_1.htm
        
        :rtype: float
    
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
    """Skewness measures the assymetry of a lightcurve, with a positive skewness
        indicating a skew to the right, and a negative skewness indicating a skew to the left.
        
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
    """The von Neumann ratio η was defined in 1941 by John von Neumann and serves as the
        mean square successive difference divided by the sample variance. When this ratio is small,
        it is an indication of a strong positive correlation between the successive photometric data points.
        See: (J. Von Neumann, The Annals of Mathematical Statistics 12, 367 (1941))
        
        :rtype: float
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
    """The variability index J was first suggested by Peter B. Stetson and serves as a
        measure of the correlation between the data points, tending to 0 for variable stars
        and getting large as the difference between the successive data points increases.
        See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
        
        :rtype: float
    """
    
    n = np.float(len(mag))
    mean = np.median(mag)
    delta_list=[]
    
    for i in range(0, len(mag)-1):
        delta = np.sqrt(n/(n-1.))*((mag[i] - mean)/magerr[i])
        delta2 = np.sqrt(n/(n-1.))*((mag[i+1] - mean)/magerr[i+1])

        
        delta_list.append(np.nan_to_num(delta*delta2))
    
    stetj = sum(np.sign(delta_list)*np.sqrt(np.abs(delta_list)))
    return stetj


def stetsonK(mag, magerr):
    """The variability index K was first suggested by Peter B. Stetson and serves as a
        measure of the kurtosis of the magnitude distribution.
        See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
        
        :rtype: float
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
    """The variability index L was first suggested by Peter B. Stetson and serves as a
        means of distinguishing between different types of variation. When individual random
        errors dominate over the actual variation of the signal, K approaches 0.798 (Gaussian limit).
        Thus, when the nature of the errors is Gaussian, stetsonL = stetsonJ, except it will be amplified
        by a small factor for smoothly varying signals, or suppressed by a large factor when data
        is infrequent or corrupt.
        See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
        
        :rtype: float
    """  
    
    stetL = (stetsonJ(mag,magerr)*stetsonK(mag,magerr)) / 0.798
    return stetL


def median_buffer_range(mag):
    """This function returns the ratio of points that are between plus or minus 10% of the
        amplitude value over the mean
        
        :rtype: float
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
    """This function returns the ratio of points that are more than 20% of the amplitude
        value over the mean
        
        :rtype: float
    """
    
    n = np.float(len(mag))
    amp = amplitude(mag)
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    a = mean - amp/5.
    
    median_buffer_range = len(np.argwhere((mag < a))) / n
    return median_buffer_range

def std_over_mean(mag):
    """A measure of the ratio of standard deviation and mean, both weighted by the errors.
        
        :rtype: float
    """
    
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    std = np.std(mag)
    
    std_over_mean = std/mean
    return std_over_mean

def amplitude(mag):
    """The amplitude of the lightcurve defined as the difference between the maximum magnitude
        measurement and the lowest magnitude measurement. To account for outliers, an array of the
        absolute value of the magnitude minus weighted mean is created. From this array, a 5%
        threshold is applied such that top 5% of points are ommitted as outliers and the amplitude
        is left to be defined as the maximun magnitude minus the minimum magnitude of the remaining points.
        
        :rtype: float
    """
    lower = np.percentile(mag, 2)
    upper = np.percentile(mag, 98)
    amplitude = upper - lower
    
    return amplitude

def median_distance(mjd, mag):
    """This function calcualtes the median eucledian distance between each photometric measurement,
        helpful metric for detecting overlapped lightcurves.
        
        type: float
        
    """
    
    delta_mag = (mag[1:] - mag[:-1])**2
    delta_t = (mjd[1:] - mjd[:-1])**2
    
    distance = np.median(np.sqrt(delta_mag + delta_t))
    return distance

def above1(mag):
    """This function measures the ratio of data points that are above 1 standard deviation
        from the median magnitude.
        
        :rtype: float
    """
    
    a = np.median(mag) + np.std(mag)
    
    above1 = len(np.argwhere(mag > a) )
    
    return above1

def above3(mag):
    """This function measures the ratio of data points that are above 3 standard deviations
        from the median magnitude.
        
        :rtype: float
    """
    
    a = np.median(mag) + 3*np.std(mag)
    
    above3 = len(np.argwhere(mag > a) )
    
    return above3

def above5(mag):
    """This function measures the ratio of data points that are above 5 standard deviations
        from the median magnitude.
        
        :rtype: float
    """
    
    a = np.median(mag) + 5*np.std(mag)
    
    above5 = len(np.argwhere(mag > a))
    
    return above5

def below1(mag):
    """This function measures the ratio of data points that are below 1 standard deviations
        from the median magnitude.
        
        :rtype: float
        """
    
    a = np.median(mag) - np.std(mag)
    below1 = len(np.argwhere(mag < a))
    
    return below1

def below3(mag):
    """This function measures the ratio of data points that are below 3 standard deviations
        from the median magnitude.
        
        :rtype: float
        """
    
    a = np.median(mag) - 3*np.std(mag)
    below3 = len(np.argwhere(mag < a))
    
    return below3

def below5(mag):
    """This function measures the ratio of data points that are below 5 standard deviations
        from the median magnitude.
        
        :rtype: float
        """
    
    a = np.median(mag) - 5*np.std(mag)
    below5 = len(np.argwhere(mag < a))
    
    return below5

def medianAbsDev(mag):
    """"A measure of the mean average distance between each magnitude value
        and the mean magnitude. https://en.wikipedia.org/wiki/Median_absolute_deviation 
        
        :rtype: float
        """
    
    array = np.ma.array(mag).compressed() 
    med = np.median(array)
    medianAbsDev = np.median(np.abs(array - med))
    
    return medianAbsDev

def root_mean_squared(mag):
    """A measure of the root mean square deviation.
        
        :rtype: float
    """
    
    #mean = meanMag(mag, magerr)
    #rms = np.sqrt(sum(((mag - mean)/magerr)**2)/sum(1./magerr**2))
    mean = np.median(mag)
    rms = np.sqrt(mean**2)

    
    return rms

def meanMag(mag, magerr):
    """Calculates mean magnitude, weighted by the errors.
        
        rtype: float
    """
    
    mean = sum(mag/magerr**2)/sum(1./magerr**2)
    
    return mean

def integrate(mag):
    """Integrate magnitude using the trapezoidal rule.
        See: http://en.wikipedia.org/wiki/Trapezoidal_rule
    """
    integral = np.trapz(mag)
    return integral


def remove_allbad(mjd, mag, magerr):
    """Function to remove bad photometric points.
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

def peak_detection(mag):
    """Function to detect number of peaks.
    
        rtype: int
    """
    
    mag = abs(mag - np.median(mag))
    try:
        indices = peakutils.indexes(mag, thres=.5, min_dist=10)#, thres_abs=False)
    except ValueError:
        indices = []
    #  if len(indices) == 0:
    #   indices = peakutils.indexes(mag, thres=0.1, min_dist=10, thres_abs=False)
    
    return len(indices)

# The following features are derived using the Python package tsfresh.
# Please see: http://tsfresh.readthedocs.io/en/latest/

def abs_energy(mag):
    """Returns the absolute energy of the time series, defined to be the sum over the squared
    values of the time-series.

        rtype: float
    """
    energy = ts.abs_energy(mag)
    return energy

def abs_sum_changes(mag):
    """Returns sum over the abs value of consecutive changes in mag.

    rtype: float
    """
    val = ts.absolute_sum_of_changes(mag)
    return val

def auto_corr(mag):
    """Similarity between observations as a function of a time lag between them.

    rtype:float
    """
    auto_corr = ts.autocorrelation(mag, 1)
    return auto_corr

def c3(mag):
    """A measure of non-linearity.
    See: Measure of non-linearity in time series: [1] Schreiber, T. and Schmitz, A. (1997).
    Discrimination power of measures for nonlinearity in a time series
    PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

    rtype: float
    """
    c3 = ts.c3(mag, 1)
    return c3

def complexity(mag):
    """This function calculator is an estimate for a time series complexity.
    A higher value represents more complexity (more peaks,valleys,etc.)
    See: Batista, Gustavo EAPA, et al (2014). CID: an efficient complexity-invariant 
    distance for time series. Data Mining and Knowledge Difscovery 28.3 (2014): 634-669.

    rtype: float
    """
    c = ts.cid_ce(mag, True)
    return c

def count_above(mag):
    """Number of values higher than mean(mag)

    rtype: int
    """
    num = ts.count_above_mean(mag)
    return num

def count_below(mag):
    """Number of values below the mean(mag)

    rtype: int
    """
    num = ts.count_below_mean(mag)
    return num

def first_loc_max(mag):
    """Returns location of maximum mag relative to the 
    lenght of mag array.

    rtype: float
    """
    loc = ts.first_location_of_maximum(mag)
    return loc

def first_loc_min(mag):
    """Returns location of minimum mag relative to the 
    lenght of mag array.

    rtype: float
    """

    loc = ts.first_location_of_minimum(mag)
    return loc

def check_for_duplicate(mag):
    """Checks if any val in mag repeats.
    1 if True, 0 if False

    rtype: int
    """
    check = str(ts.has_duplicate(mag))
    if check == 'False':
        val = 0.0
    else:
        val = 1.0
    return val

def check_for_max_duplicate(mag):
    """Checks if the maximum value in mag repeats.
    1 if True, 0 if False

    rtype: float
    """
    check = str(ts.has_duplicate_max(mag))
    if check == 'False':
        val = 0.0
    else:
        val = 1.0
    return val

def check_for_min_duplicate(mag):
    """Checks if the minimum value in mag repeats.
    1 if True, 0 if False

    rtype: float
    """
    check = str(ts.has_duplicate_min(mag))
    if check == 'False':
        val = 0.0
    else:
        val = 1.0
    return val

def check_max_last_loc(mag):
    """Returns position of last maximum mag relative to
    the length of mag array.

    rtype: float
    """
    val = ts.last_location_of_maximum(mag)
    return val

def check_min_last_loc(mag):
    """Returns position of last minimum mag relative to
    the length of mag array.

    rtype: float
    """
    val = ts.last_location_of_minimum(mag)
    return val

def longest_strike_above(mag):
    """Returns the length of the longest consecutive subsequence in 
    mag that is bigger than the mean. 

    rtype: int
    """
    val = ts.longest_strike_above_mean(mag)
    return val

def longest_strike_below(mag):
    """Returns the length of the longest consecutive subsequence in mag 
    that is smaller than mean

    rtype: int
    """
    val = ts.longest_strike_below_mean(mag)
    return val

def mean_change(mag):
    """Returns mean over the differences between subsequent observations/

    rtype: float
    """
    val = ts.mean_change(mag)
    return val

def mean_abs_change(mag):
    """Returns mean over the abs differences between subsequent observations.

    rtype: float
    """
    val = ts.mean_abs_change(mag)
    return val

def mean_second_derivative(mag):
    """Returns the mean value of a central approximation of the second derivative.

    rtype: float
    """
    val = ts.mean_second_derivative_central(mag)
    return val

def ratio_recurring_points(mag):
    """Returns the ratio of unique values, that are present in the time 
    series more than once, normalized to the number of data points. 
    
    rtype: float
    """
    val = ts.percentage_of_reoccurring_values_to_all_values(mag)
    return val

def sample_entropy(mag):
    """Returns sample entropy: http://en.wikipedia.org/wiki/Sample_Entropy
    
    rtype: float
    """
    entropy = ts.sample_entropy(mag)
    return entropy

def sum_values(mag):
    """Sums over all mag values.

    rtype: float
    """
    val = ts.sum_values(mag)
    return val

def time_reversal_asymmetry(mag):
    """Derives a feature introduced by Fulcher.
        See: Fulcher, B.D., Jones, N.S. (2014). Highly comparative 
        feature-based time-series classification.
        Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.

    rtype: float
    """
    val = ts.time_reversal_asymmetry_statistic(mag, 1)
    return val

def normalize(mag, magerr):
    """Normalizes the magnitude to range from 0 to 1, scaling
    magerr in the proccess.

    rtype: array
    """
    
    mag2 = (mag - np.min(mag)) / np.ptp(mag)
    magerr = magerr*(mag2/mag)
    
    return np.array(mag2), np.array(magerr)



