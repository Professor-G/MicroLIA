# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:30:12 2017

@author: danielgodinez
"""
import numpy as np
from scipy.stats import skew
from astropy.stats import median_absolute_deviation
from scipy.stats import tvar
from scipy.integrate import quad

def shannon_entropy(mag, magerr=None):
    """Shannon entropy (Shannon et al. 1949) is used as a metric to quantify the amount of
    information carried by a signal. The procedure employed here follows that outlined by
    (D. Mislis et al. 2015). The probability of each point is given by a Cumulative Distribution 
    Function (CDF). Following the same procedure as (D. Mislis et al. 2015), this function employs
    both the normal and inversed gaussian CDF, with the total shannon entropy given by a combination of
    the two. See: (SIDRA: a blind algorithm for signal detection in photometric surveys, D. Mislis et al., 2015)
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :param magerr: photometric error for the intensity. Must be an array.
    If magerr = None the default is 0 for every photometric point.
    
    :rtype: float
    """
    if magerr is None:
        magerr = np.array([0.001] * len(mag))
        
    RMS = np.sqrt((np.mean(mag)**2))
    mean = np.mean(mag)
    
    p_list1 = []
    p_list2 = [] 
    inv_list1 = []
    inv_list2 = []
    inv_list3 = []
    inv_list4 = []
    
    t = range(0, len(mag))
    
    def normal_gauss(t):
        return 0.5*((1 + 2/np.sqrt(np.pi))*np.e**(-t**2))
        
    def inverse_gauss1(t):
        return 0.5*((1 + 2/np.sqrt(np.pi))*np.e**(-t**2)) 
    
    def inverse_gauss2(t):
        return (0.5*np.e**((2*RMS)/mean))*(1 - (2/np.sqrt(np.pi))*np.e**(-t**2))
        
    def shannon_entropy1(mag, magerr):
        """
        This function utilizes the normal Gaussian CDF to set the probability of 
        each point in the lightcurve and computes the Shannon Entropy given this distribution.
        """
        
        for i in t:
            prob, err = quad(normal_gauss, 0, (mag[i] + magerr[i] - mean)/(RMS*np.sqrt(2)))
            p_list1.append(prob)
        
            prob2, err = quad(normal_gauss, 0, (mag[i] - magerr[i] - mean)/(RMS*np.sqrt(2)))
            p_list2.append(prob2)
        
        d_delta = [i * 2.0 for i in magerr]    
      
        entropy1 = -sum(np.nan_to_num(np.log2(p_list1))*d_delta + np.nan_to_num(np.log2(p_list2))*d_delta)
        return np.array(entropy1)
        
    def shannon_entropy2(mag, magerr):
        """
        This function utilizes the inverse Gaussian CDF to set the probability of each point
        in the lightcurve and computes the Shannon Entropy given this distribution.
        """
        
        for i in t:
            prob, err = quad(inverse_gauss1, 0, np.sqrt(RMS/(2*mag[i] + magerr[i]))*(((mag[i] + magerr[i])/mean) - 1))
            inv_list1.append(prob)
            
            prob2, err = quad(inverse_gauss2, 0, np.sqrt(RMS/(2*mag[i] + magerr[i]))*(((mag[i] + magerr[i])/mean) + 1))
            inv_list2.append(prob2)
            
        p2_list1 = [inv_list1[i] + inv_list2[i] for i in range(len(inv_list1))]
        
        for i in t:
            prob, err = quad(inverse_gauss1, 0, np.sqrt(RMS/(2*mag[i] - magerr[i]))*(((mag[i] - magerr[i])/mean) - 1))
            inv_list3.append(prob)
            
            prob2, err = quad(inverse_gauss2, 0, np.sqrt(RMS/(2*mag[i] - magerr[i]))*(((mag[i] - magerr[i])/mean) + 1))
            inv_list4.append(prob2)
            
        p2_list2 = [inv_list3[i] + inv_list4[i] for i in range(len(inv_list3))]
        
        d_delta = [i * 2 for i in magerr]
        entropy2 = -sum(np.nan_to_num(np.log2(p2_list1))*d_delta + np.nan_to_num(np.log2(p2_list2))*d_delta)
        return np.array(entropy2)
    
    """The total Shannon Entropy is calculated by adding the values calculated using both the normal
    and inverse Gaussian CDF
    """    
    total_entropy = shannon_entropy1(mag, magerr) + shannon_entropy2(mag, magerr)
    return total_entropy

def auto_correlation(mag):
    """The autocorrelation integral calculates the correlation of a given signal as a function of 
    the time delay of each measurement. Has been employed in previous research as a metric to 
    differentitate between lightcurve classes. See: (SIDRA: a blind algorithm for signal
    detection in photometric surveys, D. Mislis et al., 2015)
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
        
    :rtype: float
    """
    n = len(mag)
    mean = np.mean(mag)
    RMS = np.sqrt((np.mean(mag)**2))
    
    sum_list = []
    val_list = []
    
    t = range(1, len(mag))
            
    for i in t:
        sum1 = np.array(sum((mag[0:n-i] - mean)*(mag[i:n] - mean)))
        sum_list.append(sum1)
        
        val = np.array(1/((n-i)*RMS**2))
        val_list.append(val)
        
    auto_corr = abs(sum([x*y for x,y in zip(sum_list, val_list)]))        
   
    #auto_corr = abs(sum((1/((n-t)*RMS**2))*sum((mag[0:n-t] - mean)*(mag[t:n] - mean))))
    return auto_corr
    
def kurtosis(mag):
    """"Kurtosis function returns the calculated kurtosis of the lightcurve. 
    It's a measure of the peakedness (or flatness) of the lightcurve relative 
    to a normal distribution. See http://www.xycoon.com/peakedness_small_sample_test_1.htm
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :return: the function will return a float.
    :rtype: float
    """""
   
    t = float(len(mag))
    kurtosis = (t*(t+1.)/((t-1.)*(t-2.)*(t-3.))*sum(((mag - np.mean(mag))/np.std(mag))**4)) - \
    (3.*((t-1.)**2.)/((t-2.)*(t-3.)))
    return kurtosis
    
def skewness(mag):
    """Skewness measures the assymetry of a lightcurve, with a positive skewness
    indicating a skew to the right, and a negative skewness indicating a skew to the left. 
    This is calculated using the skew function in the scipy.stats package.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :return: the function will return a float.
    :rtype: float
    """
    
    skewness = skew(mag, axis = 0, bias = True)
    return skewness

def vonNeumannRatio(mag):
    """The von Neumann ratio η was defined in 1941 by John von Neumann and serves as the 
    mean square successive difference divided by the sample variance. When this ratio is small, 
    it is an indication of a strong positive correlation between the successive photometric data points. 
    See: (J. Von Neumann, The Annals of Mathematical Statistics 12, 367 (1941))
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """

    t = np.float(len(mag))
    delta = sum((mag[1:] - mag[:-1])**2 / (t-1.))
    sample_variance = tvar(mag, limits=None)
    
    vonNeumannRatio = delta / sample_variance
    return vonNeumannRatio
    
def stetsonJ(time, mag, magerr):
    """The variability index J was first suggested by Peter B. Stetson and is defined 
    as the measure of the correlation between the data points. J tends to 0 for variable 
    sources and gets large as the difference between the successive data points increases. 
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
    
    :param time: the time-varying data of the lightcurve. Must be an array.
    
    :param mag: the time-varying intensity of the object. Must be an array.
    
    :param magerr: photometric error for the intensity. Must be an array.
    If magerr = None the default is 0 for every photometric point. 
    
    :rtype: float
    """
    
    if magerr is None:
        magerr = np.array([0.001] * len(time))
    
    t = np.float(len(mag))
    range1 = range(0, len(time)-1)
    range2 = range(1, len(time))
    delta = np.sqrt((t/(t-1.)))*((mag - np.mean(mag))/magerr)

    sign = np.sign(((delta[range1]*delta[range2]))*(np.sqrt(np.absolute(delta[range1]*delta[range2]))))
    stetsonJ = sum(((sign*delta[range1]*delta[range2]))*(np.sqrt(np.absolute(delta[range1]*delta[range2]))))
    return np.nan_to_num(stetsonJ)
    
def stetsonK(time, mag, magerr):
    """The variability index K was first suggested by Peter B. Stetson and serves as a 
    measure of the kurtosis of the magnitude distribution. 
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
    
    :param time: the time-varying data of the lightcurve. Must be an array.
    
    :param mag: the time-varying intensity of the object. Must be an array.
    
    :param magerr: photometric error for the intensity. Must be an array.
    If magerr = None the default is 0 for every photometric point. 
    
    :rtype: float
    """
    
    if magerr is None:
        magerr = np.array([0.001] * len(time))
        
    t = np.float(len(mag))
    delta = np.sqrt((t/(t-1.)))*((mag - np.mean(mag))/magerr)
        
    stetsonK = ((1./t)*sum(abs(delta)))/(np.sqrt((1./t)*sum((delta)**2)))
    return np.nan_to_num(stetsonK)

def median_buffer_range(mag):
    """This function returns the ratio of points that are between plus or minus 10% of the
    amplitude value.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    t = np.float(len(mag))
    amplitude = np.max(mag) - np.min(mag)   
    
    median_buffer_range = np.float(len((np.where(((amplitude/10) - np.median(mag) < mag) & 
    ((amplitude/10) + np.median(mag) > mag))[0]))) / t
    
    return median_buffer_range
    
def std_over_mean(mag):
    """A measure of the ratio of standard deviation and sample mean.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    std_over_mean = np.std(mag)/np.mean(mag)
    return std_over_mean
      
def amplitude(mag):
    """The amplitude of the lightcurve defined as the difference between the maximum magnitude
    measurement and the lowest magnitude measurement.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    amplitude = np.max(mag) - np.min(mag)
    return amplitude
    
def above1(mag):
    """This function measures the ratio of data points that are above 1 standard deviation 
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    t = np.float(len(mag))
    AboveMeanBySTD_1 = np.float(len((np.where(mag > np.std(mag)+np.mean(mag)))[0])) / t
    return AboveMeanBySTD_1

def above3(mag):
    """This function measures the ratio of data points that are above 3 standard deviations 
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    t = np.float(len(mag))
    AboveMeanBySTD_3 = np.float(len((np.where(mag > 3*np.std(mag)+np.mean(mag)))[0])) / t
    return AboveMeanBySTD_3

def above5(mag):
    """This function measures the ratio of data points that are above 5 standard deviations
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    t = np.float(len(mag))
    AboveMeanBySTD_5 = np.float(len((np.where(mag > 5*np.std(mag)+np.mean(mag)))[0])) / t
    return AboveMeanBySTD_5
        
def below1(mag):
    """This function measures the ratio of data points that are below 1 standard deviations 
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    t = np.float(len(mag))
    BelowMeanBySTD_1 = np.float(len((np.where(mag < np.std(mag)+np.mean(mag))[0]))) / t
    return BelowMeanBySTD_1
      
def below3(mag):
    """This function measures the ratio of data points that are below 3 standard deviations
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    t = np.float(len(mag))
    BelowMeanBySTD_3 = np.float(len((np.where(mag < 3*np.std(mag)+np.mean(mag))[0]))) / t
    return BelowMeanBySTD_3
      
def below5(mag):
    """This function measures the ratio of data points that are below 5 standard deviations
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    t = np.float(len(mag))
    BelowMeanBySTD_5 = np.float(len((np.where(mag < 5*np.std(mag)+np.mean(mag))[0]))) / t
    return BelowMeanBySTD_5
    
def medianAbsDev(mag):
    """"A measure of the mean average distance between each magnitude value 
    and the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    medianAbsDev = median_absolute_deviation(mag, axis = None)
    return medianAbsDev
    
def RMS(mag):
    """A measure of the quadratic mean
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    RMS = np.sqrt((np.mean(mag)**2))
    return RMS
    
def medianMag(mag):
    """Calculates median magnitude in the distribution.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
    
    medianMag = np.median(mag)
    return medianMag
    
def meanMag(mag):
    """Calculates mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float"""
   
    mean = np.mean(mag)
    return mean

def minMag(mag):
    """Calculates the minimum value for the magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
   
    min_mag = np.amin(mag)
    return min_mag
    
def maxMag(mag):
    """Calulates the maximum value for the magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :rtype: float
    """
   
    max_mag = np.amax(mag)
    return max_mag
        
def compute_statistics(time, mag, magerr):
    """This function will compute all the statistics and return them in an array in the 
    following order: shannon_entropy, auto_correlation, kurtosis, skewness, vonNeumannRatio,
    stetsonJ, stetsonK, median_buffer_Rance, std_over_mean, amplitude, AboveMeanBySTD_1,
    AboveMeanBySTD_3, AboveMeanBySTD_5, BelowMeanBySTD_1, BelowMeanBySTD_3, BelowMeanBySTD_5, 
    medianAbdsDev, RMS
        
    :param time: the time-varying data of the lightcurve. Must be an array.

    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    
    :param magerr: photometric error for the intensity. Must be an array.
    If magerr = None the default is 0 for every photometric point. 
    
    :return: the function will return an array with the statistics.
    :rtype: array, float
    """
    
    if magerr is None:
        magerr = np.array([0.001] * len(time))
        
    stat_array = (shannon_entropy(mag, magerr), auto_correlation(mag), kurtosis(mag), skewness(mag), 
                  vonNeumannRatio(mag), stetsonJ(time, mag, magerr), stetsonK(time, mag, magerr),
                  median_buffer_range(mag), std_over_mean(mag), amplitude(mag), above1(mag), above3(mag),
                  above5(mag), below1(mag), below3(mag), below5(mag), medianAbsDev(mag), RMS(mag))

    return stat_array
