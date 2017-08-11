# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:30:12 2017

@author: danielgodinez
"""
import numpy as np
from astropy.stats import median_absolute_deviation
from scipy.integrate import quad
from scipy.cluster.hierarchy import fclusterdata

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
    
    
    mag, magerr = remove_bad(mag, magerr)
    
    mean =  meanMag(mag, magerr)   
    RMS = RootMS(mag, magerr)
    
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
        
    def shannon_entropy1(mag, magerr):
        """
        This function utilizes the normal Gaussian CDF to set the probability of 
        each point in the lightcurve and computes the Shannon Entropy given this distribution.
        """
        
        for i in t:
            val = normal_gauss((mag[i] + magerr[i] - mean)/(RMS*np.sqrt(2)))
            p_list1.append(val)
        
            val2 = normal_gauss((mag[i] - magerr[i] - mean)/(RMS*np.sqrt(2)))
            p_list2.append(val2)
                                    
        p_list3 = [1 if i <= 0 else i for i in p_list1]
        p_list4 = [1 if i <= 0 else i for i in p_list2]
        
        entropy = -sum(np.log2(p_list3)*d_delta + np.log2(p_list4)*d_delta)
        return entropy
            
    def shannon_entropy2(mag, magerr):
        """
        This function utilizes the inverse Gaussian CDF to set the probability of each point
        in the lightcurve and computes the Shannon Entropy given this distribution.
        """
        
        for i in t:
            val = inv_gauss(np.sqrt(RMS/(2.*(mag[i] + magerr[i])))*(((mag[i] + magerr[i])/mean) - 1.), 
                            np.sqrt(RMS/(2.*(mag[i] + magerr[i])))*(((mag[i] + magerr[i])/mean) + 1.))
            inv_list1.append(val)
            
            val2 = inv_gauss(np.sqrt(RMS/(2.*(mag[i] - magerr[i])))*(((mag[i] - magerr[i])/mean) - 1.), 
                            np.sqrt(RMS/(2.*(mag[i] - magerr[i])))*(((mag[i] - magerr[i])/mean) + 1.))
            inv_list2.append(val2)         
            
        inv_list3 = [1 if i <= 0 else i for i in inv_list1]
        inv_list4 = [1 if i <= 0 else i for i in inv_list2]
        
        entropy = -sum(np.log2(inv_list3)*d_delta + np.log2(inv_list4)*d_delta)
        return entropy
        
    """The total Shannon Entropy is calculated by adding the values calculated using both the normal
    and inverse Gaussian CDF
    """      
    total_entropy = np.nan_to_num(shannon_entropy1(mag, magerr) + shannon_entropy2(mag, magerr))
    return total_entropy

   
def auto_correlation(mag, magerr):
    """The autocorrelation integral calculates the correlation of a given signal as a function of 
    the time delay of each measurement. Has been employed in previous research as a metric to 
    differentitate between lightcurve classes. See: (SIDRA: a blind algorithm for signal
    detection in photometric surveys, D. Mislis et al., 2015)
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.  
    :param mag: the time-varying intensity of the lightcurve. Must be an array.

    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)    
    n = np.float(len(mag))
    mean = np.median(mag)
    RMS = RootMS(mag, magerr)
    t = range(1, len(mag))
        
    sum_list = []
    val_list = []
    
            
    for i in t:
        sum1 = np.array(sum((mag[0:int(n)-i] - mean)*(mag[i:int(n)] - mean)))
        sum_list.append(sum1)
        
        val = np.array(1/((n-i)*RMS**2))
        val_list.append(val)
        
    auto_corr = abs(sum([x*y for x,y in zip(sum_list, val_list)]))        
   
    return auto_corr
"""
def con(mag, magerr):



    
    
    std = deviation(mag, magerr)
    
    
    deviatingThreshold = median - magerr
    con = 0  
    deviating = False
    
    a = np.argwhere(magerr < deviatingThreshold)
    
    if len(a) < 3:
        return 0
    else:
        for i in xrange(len(magerr)-2):
            first = magerr[i]
            second = magerr[i+1]
            third = magerr[i+2]
            if (first <= deviatingThreshold and
                second <= deviatingThreshold and
                third <= deviatingThreshold):
                    if (not deviating):
                        con += 1
                        deviating = True
                    elif deviating:
                        deviating = False


    con2 = 0
    deviating = False
    a = np.argwhere(magerr < deviatingThreshold)
    deviatingThreshold = median + magerr

    if len(a) < 3:
        return 0
    else:
        for i in xrange(len(mag)-2):
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

    return con + con2


def con2(mag, magerr):

    
    mag, magerr = remove_bad(mag, magerr)
    diff = mag - meanMag(mag, magerr)
    hist, edges = np.histogram(diff, bins = 10)
    val = np.where(hist == max(hist))
    bin_range = np.where((diff > edges[val[0][0]]) & (diff < edges[val[0][0]+1]))
    
    mean = meanMag(mag[bin_range], magerr[bin_range])
    std = deviation(mag, magerr)    
    deviatingThreshold = mean - 2*std    
    con = 0  
    deviating = False
    
    a = np.argwhere(magerr < deviatingThreshold)
    
    if len(a) < 3:
        return 0
    else:
        for i in xrange(len(mag)-2):
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
"""

def con(mag, magerr):
    """Con is defined as the number of clusters containing three or more
        consecutive observations with magnitudes brighter than the median
        magnitude plus 3 standard deviations. For a microlensing event Con = 1,
        assuming a  flat lightcurve prior to the event.
        :param mag: the time-varying intensity of the lightcurve. Must be an array.
        :param magerr: photometric error for the intensity. Must be an array.
        
        :rtype: float
    """
    
    #diff = mag - meanMag(mag, magerr)
    #hist, edges = np.histogram(diff, bins = 10)
    #val = np.where(hist == max(hist))
    #bin_range = np.where((diff > edges[val[0][0]]) & (diff < edges[val[0][0]+1]))
    
    mag, magerr = remove_bad(mag, magerr)
    median = np.median(mag)
    
    deviatingThreshold = np.array(median - magerr)
    con = 0
    deviating = False
    
    a = np.argwhere(magerr < deviatingThreshold)
    
    if len(a) < 3:
        return 0
    else:
        for i in xrange(len(magerr)-2):
            first = mag[i]
            second = mag[i+1]
            third = mag[i+2]
            if ((first <= deviatingThreshold[i]).all() == True and
                (second <= deviatingThreshold[i+1]).all() == True and
                (third <= deviatingThreshold[i+2]).all() == True):
                if (not deviating):
                    con += 1
                    deviating = True
                elif deviating:
                    deviating = False

    con2 = 0
    deviating = False

    deviatingThreshold = np.array(median + magerr)
    a = np.argwhere(magerr > deviatingThreshold)
    
    if len(a) < 3:
        return 0
    else:
        for i in xrange(len(mag)-2):
            first = mag[i]
            second = mag[i+1]
            third = mag[i+2]
            if ((first >= deviatingThreshold).all() == True and
                (second >= deviatingThreshold).all() == True and
                (third >= deviatingThreshold).all() == True):
                if (not deviating):
                    con2 += 1
                    deviating = True
                elif deviating:
                    deviating = False

    return con + con2


def con2(mag, magerr):
    """Only looks at bin below the reference magnitude
    """

    mag, magerr = remove_bad(mag, magerr)
    median = np.median(mag)
    
    deviatingThreshold = np.array(median + magerr)
    con = 0
    deviating = False
    
    a = np.argwhere(magerr > deviatingThreshold)
    
    if len(a) < 3:
        return 0
    else:
        for i in xrange(len(mag)-2):
            first = mag[i]
            second = mag[i+1]
            third = mag[i+2]
            if ((first >= deviatingThreshold).all() == True and
                (second >= deviatingThreshold).all() == True and
                (third >= deviatingThreshold).all() == True):
                if (not deviating):
                    con += 1
                    deviating = True
                elif deviating:
                    deviating = False

    return con


def kurtosis(mag, magerr):
    """"Kurtosis function returns the calculated kurtosis of the lightcurve. 
    It's a measure of the peakedness (or flatness) of the lightcurve relative 
    to a normal distribution. See: www.xycoon.com/peakedness_small_sample_test_1.htm
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.

    :rtype: float
    """""
    
    mag, magerr = remove_bad(mag, magerr)
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    std = deviation(mag, magerr)
   
    n = np.float(len(mag))
    kurtosis = (n*(n+1.)/((n-1.)*(n-2.)*(n-3.))*sum(((mag - mean)/std)**4)) - \
    (3.*((n-1.)**2)/((n-2.)*(n-3.)))
    return kurtosis
        

def skewness(mag, magerr):
    """Skewness measures the assymetry of a lightcurve, with a positive skewness
    indicating a skew to the right, and a negative skewness indicating a skew to the left. 
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.

    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    std = deviation(mag, magerr)
    n = np.float(len(mag))
    
    skewness = (1./n)*sum((mag - mean)**3/std**3)
    #skewness = skew(mag, axis = 0, bias = True)
    return skewness

def vonNeumannRatio(mag, magerr):
    """The von Neumann ratio η was defined in 1941 by John von Neumann and serves as the 
    mean square successive difference divided by the sample variance. When this ratio is small, 
    it is an indication of a strong positive correlation between the successive photometric data points. 
    See: (J. Von Neumann, The Annals of Mathematical Statistics 12, 367 (1941))
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.

    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    n = np.float(len(mag))
    delta = sum((mag[1:] - mag[:-1])**2 / (n-1.))
    sample_variance = deviation(mag, magerr)**2
    
    vonNeumannRatio = delta / sample_variance
    return vonNeumannRatio
    
def stetsonJ(mag, magerr):
    """The variability index K was first suggested by Peter B. Stetson and serves as a 
    measure of the kurtosis of the magnitude distribution. 
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
        
    :param mag: the time-varying intensity of the object. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :rtype: float
    """

    mag, magerr = remove_bad(mag, magerr)
    n = np.float(len(mag))
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    delta_list = []
    
    for i in range(0, len(mag)):
        delta = np.sqrt(n/(n-1.))*((mag[i] - mean)/magerr[i])
        delta_list.append(delta)
    
    val = np.nan_to_num([x*y for x,y in zip(delta_list[0:int(n)-1], delta_list[1:int(n)])])    
    sign = np.sign(val)
    stetj = sum(sign*np.sqrt(np.abs(val)))
    return stetj
  
def stetsonK(mag, magerr):
    """The variability index K was first suggested by Peter B. Stetson and serves as a 
    measure of the kurtosis of the magnitude distribution. 
    See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996)).
        
    :param mag: the time-varying intensity of the object. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)            
    n = np.float(len(mag))
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    
    delta = np.sqrt((n/(n-1.)))*((mag - mean)/magerr)
        
    stetsonK = ((1./n)*sum(abs(delta)))/(np.sqrt((1./n)*sum(delta**2)))
    return np.nan_to_num(stetsonK)


    
def median_buffer_range(mag, magerr):
    """This function returns the ratio of points that are between plus or minus 10% of the
    amplitude value over the mean
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    n = np.float(len(mag))
    amp = amplitude(mag, magerr) 
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    a = mean - amp/10. 
    b = mean + amp/10. 
    
    median_buffer_range = len(np.argwhere((mag > a) & (mag < b))) / n
    
    return median_buffer_range

def median_buffer_range2(mag, magerr):
    """This function returns the ratio of points that are more than 20% of the amplitude
    value over the mean
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.

    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    n = np.float(len(mag))
    amp = amplitude(mag, magerr) 
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    a = mean - amp/5. 
 
    
    median_buffer_range = len(np.argwhere((mag < a))) / n
    
    return median_buffer_range
    
    
def std_over_mean(mag, magerr):
    """A measure of the ratio of standard deviation and mean, both weighted by the errors.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.

    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    mean = meanMag(mag, magerr)
    std = deviation(mag, magerr)
    
    std_over_mean = std/mean
    return std_over_mean
      
    
def amplitude(mag, magerr):
    """The amplitude of the lightcurve defined as the difference between the maximum magnitude
    measurement and the lowest magnitude measurement. To account for outliers, an array of the
    absolute value of the magnitude minus weighted mean is created. From this array, a 5% 
    threshold is applied such that top 5% of points are ommitted as outliers and the amplitude
    is left to be defined as the maximun magnitude minus the minimum magnitude of the remaining points. 
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    array_list = [] 
    
    for i in range(0, len(mag)):
        array = np.abs(mag[i] -np.median(mag))
        array_list.append(array)
        
    index = np.argsort(array_list)
    index2 = len(mag) - np.int(len(mag)*0.95)
    threshold = index[-index2]
    
    mag = np.delete(mag, threshold)
    amplitude = np.max(mag) - np.min(mag)
    
    return amplitude

def median_distance(mjd, mag, magerr):

    bad = np.where(mag == 0)
    magerr = np.delete(magerr, bad)
    mag = np.delete(mag, bad)
    mjd = np.delete(mjd, bad)
    
    bad = np.where(np.isnan(magerr) == True)
    magerr = np.delete(magerr, bad)
    mag = np.delete(mag, bad)
    mjd = np.delete(mjd, bad)
    
    bad = np.where(np.isfinite == False)
    magerr = np.delete(magerr, bad)
    mag = np.delete(mag, bad)
    mjd = np.delete(mjd, bad)
    
    mag1 = (mag[1:] - mag[:-1])**2
    time1 = (mjd[1:] - mjd[:-1])**2

    distance = np.median(np.sqrt(mag1 + time1))
    return distance

def clusters(mag, magerr):
    
    mag, magerr = remove_bad(mag, magerr)
    new_mag = mag.reshape(len(mag), 1)

    cluster = fclusterdata(new_mag, 0.1, criterion='distance')
    num_clusters =  len(np.unique(cluster))
    return num_clusters

def above1(mag, magerr):
    """This function measures the ratio of data points that are above 1 standard deviation 
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.

    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    a = meanMag(mag, magerr) - deviation(mag, magerr)
    
    above1 = len(np.argwhere(mag < a) )  
    
    return above1

def above3(mag, magerr):
    """This function measures the ratio of data points that are above 3 standard deviations 
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    a = meanMag(mag, magerr) - 3*deviation(mag, magerr)
    
    above3 = len(np.argwhere(mag < a) )  
    
    return above3

def above5(mag, magerr):
    """This function measures the ratio of data points that are above 5 standard deviations
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    a = meanMag(mag, magerr) - 5*deviation(mag, magerr)
    
    above5 = len(np.argwhere(mag < a))    
    
    return above5
        
def below1(mag, magerr):
    """This function measures the ratio of data points that are below 1 standard deviations 
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.

    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    a = meanMag(mag, magerr) + deviation(mag, magerr)
    
    below1 = len(np.argwhere(mag > a)) 
    
    return below1
    
def below3(mag, magerr):
    """This function measures the ratio of data points that are below 3 standard deviations
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    a = meanMag(mag, magerr) + 3*deviation(mag, magerr)
    
    below3 = len(np.argwhere(mag > a))
    
    return below3
    
def below5(mag, magerr):
    """This function measures the ratio of data points that are below 5 standard deviations
    from the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    a = meanMag(mag, magerr) + 5*deviation(mag, magerr)
    
    below5 = len(np.argwhere(mag > a))  
    
    return below5
    
def medianAbsDev(mag, magerr):
    """"A measure of the mean average distance between each magnitude value 
    and the mean magnitude.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.

    :rtype: float
    """
    
    mag = remove_bad(mag, magerr)[0]
    medianAbsDev = median_absolute_deviation(mag)
    
    return medianAbsDev
    
def RootMS(mag, magerr):
    """A measure of the root mean square deviation, weighted by the errors.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.

    :rtype: float
    """
    
    mag, magerr = remove_bad(mag, magerr)
    #mean = meanMag(mag, magerr)
    mean = np.median(mag)
    rms = np.sqrt(sum(((mag - mean)/magerr)**2)/sum(1./magerr**2))
    
    return rms

    
def meanMag(mag, magerr):
    """Calculates mean magnitude, weighted by the errors.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    rtype: float
    """
   
    mag, magerr = remove_bad(mag, magerr)
    mean = sum(mag/magerr**2)/sum(1./magerr**2)
    
    return mean
    
def deviation(mag, magerr):
     """Calculates the standard deviation, weighted by the errors.
    
     :param mag: the time-varying intensity of the lightcurve. Must be an array.  
     :param magerr: photometric error for the intensity. Must be an array.
    
     rtype: float
     """

     mag, magerr = remove_bad(mag, magerr)
     m = np.float(len(np.argwhere(magerr != 0)))
     std = np.sqrt(sum((1./magerr**2)*(mag - meanMag(mag, magerr))**2)/(((m - 1.)/m)*sum(1./magerr**2)))
     
     return std
        
def remove_bad(mag, magerr):
    """This function removes all points in the lightcurve that do not contain
    a calculation for photometric error.
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    rtype: float
    """
    
    bad = np.where(magerr == 0)
    magerr = np.delete(magerr, bad)
    mag = np.delete(mag, bad)
    
    bad = np.where(mag == 0)
    magerr = np.delete(magerr, bad)
    mag = np.delete(mag, bad)
    
    bad = np.where(np.isnan(magerr) == True)
    magerr = np.delete(magerr, bad)
    mag = np.delete(mag, bad)
    
    bad = np.where(np.isfinite(magerr) == False)
    magerr = np.delete(magerr, bad)
    mag = np.delete(mag, bad)
    
    return mag, magerr

def beyond1std(mag, magerr):
    
    num = above1(mag, magerr)
    above1_std = np.float(num/len(mag))
    
    num2 = below1(mag, magerr)
    below1_std = np.float(num2/len(mag))
    tot_beyond = below1_std + above1_std
    
    return tot_beyond
        
def compute_statistics(mjd, mag, magerr):
    """This function will compute all the statistics and return them in an array in the 
    following order: shannon_entropy, auto_correlation, kurtosis, skewness, vonNeumannRatio,
    stetsonJ, stetsonK, median_buffer_Rance, std_over_mean, std_over_mean, below1, medianAbdsDev, RMS
    
    :param mag: the time-varying intensity of the lightcurve. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :return: the function will return an array with the statistics.
    :rtype: array, float
    """
    
    stat_array = (shannon_entropy(mag, magerr), auto_correlation(mag, magerr), kurtosis(mag, magerr),
             skewness(mag, magerr), vonNeumannRatio(mag, magerr), stetsonJ(mag, magerr), stetsonK(mag, magerr), con(mag, magerr),
  median_buffer_range(mag, magerr), median_buffer_range2(mag, magerr), std_over_mean(mag, magerr),
                medianAbsDev(mag, magerr), RootMS(mag, magerr), amplitude(mag, magerr), median_distance(mjd, mag, magerr), clusters(mag, magerr), deviation(mag, magerr), beyond1std(mag, magerr), con(mag, magerr), con2(mag, magerr))
    
    #stat_array = (auto_correlation(mag, magerr), kurtosis(mag, magerr), skewness(mag, magerr), vonNeumannRatio(mag, magerr),
    #               stetsonJ(mag, magerr), stetsonK(mag, magerr), median_buffer_range(mag, magerr),
    #              median_buffer_range2(mag, magerr), std_over_mean(mag, magerr), medianAbsDev(mag, magerr), RootMS(mag, magerr))
  
    return stat_array
