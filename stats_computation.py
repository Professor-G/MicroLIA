# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:30:12 2017

@author: danielgodinez
"""
import numpy as np
from scipy.stats import skew
from astropy.stats import median_absolute_deviation
from scipy.stats import tvar
import unittest

def RMS(mag):
    """A measure of the quadratic mean"""
    RMS = np.sqrt((np.mean(mag)**2))
    return RMS

def medianMag(mag):
    """Calculates median magnitude in the distribution"""
    medianMag = np.median(mag)
    return medianMag
    
def meanMag(mag):
    """Calculates mean magnitude"""
    mean = np.mean(mag)
    return mean

def minMag(mag):
    """Calculates the minimum value for the magnitude"""
    min_mag = np.amin(mag)
    return min_mag
    
def maxMag(mag):
    """Calulates the maximum value for the magnitude"""
    max_mag = np.amax(mag)
    return max_mag
    
def medianAbsDev(mag):
    """"A measure of the mean average distance between each magnitude value and the mean magnitude"""
    medianAbsDev = median_absolute_deviation(mag, axis = None)
    return medianAbsDev
    
def kurtosis(mjd, mag):
    """"Kurtosis function returns the calculated kurtosis of the lightcurve. It's a measure of the peakedness (or flatness) of the lightcurve relative to a normal distribution. See http://www.xycoon.com/peakedness_small_sample_test_1.htm"""""
    t = float(len(mag))
    kurtosis = (t*(t+1.)/((t-1.)*(t-2.)*(t-3.))*sum(((mag - np.mean(mag))/np.std(mag))**4)) - (3.*((t-1.)**2.)/((t-2.)*(t-3.)))
    return kurtosis
    
    
def skewness(mag):
    """Skewness measures the assymetry of a lightcurve, with a positive skewness indicating a skew to the right, and a negative skewness indicating a skew to the left. This is calculated using the skew function in the scipy.stats package."""
    skewness = skew(mag, axis = 0, bias = True)
    return skewness

def stetsonJ(mjd, mag, magerr):
    """The variability index J was first suggested by Peter B. Stetson and is defined as the measure of the correlation between the data points. J tends to 0 for variable sources and gets large as the difference between the successive data points increases. See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996))."""
    t = np.float(len(mag))
    range1 = range(0, len(mjd)-1)
    range2 = range(1, len(mjd))
    delta = np.sqrt((t/(t-1.)))*((mag - np.mean(mag))/magerr)

    sign = np.sign(((delta[range1]*delta[range2]))*(np.sqrt(np.absolute(delta[range1]*delta[range2]))))
    stetsonJ = sum(((sign*delta[range1]*delta[range2]))*(np.sqrt(np.absolute(delta[range1]*delta[range2]))))
    return stetsonJ
    
def stetsonK(mjd, mag, magerr):
    """The variability index K was first suggested by Peter B. Stetson and serves as a measure of the kurtosis of the magnitude distribution. See: (P. B. Stetson, Publications of the Astronomical Society of the Pacific 108, 851 (1996))."""
    t = np.float(len(mag))
    delta = np.sqrt((t/(t-1.)))*((mag - np.mean(mag))/magerr)
        
    stetsonK = ((1./t)*sum(abs(delta)))/(np.sqrt((1./t)*sum((delta)**2)))
    return stetsonK

def vonNeumannRatio(mag):
    """The von Neumann ratio η was defined in 1941 by John von Neumann and serves as the mean square successive difference divided by the sample variance. When this ratio is small, it is an indication of a strong positive correlation between the successive photometric data points. See: (J. Von Neumann, The Annals of Mathematical Statistics 12, 367 (1941))"""
    t = np.float(len(mag))
    delta = sum((mag[1:] - mag[:-1])**2 / (t-1.))
    sample_variance = tvar(mag, limits=None)
    
    vonNeumannRatio = delta / sample_variance
    return vonNeumannRatio
        
def above1(mag):
    """This function measures the ratio of data points that are above 1 standard deviation from the mean magnitude."""
    t = np.float(len(mag))
    AboveMeanBySTD_1 = np.float(len((np.where(mag > np.std(mag)+np.mean(mag)))[0])) / t
    return AboveMeanBySTD_1

def above3(mag):
    """This function measures the ratio of data points that are above 3 standard deviations from the mean magnitude."""
    t = np.float(len(mag))
    AboveMeanBySTD_3 = np.float(len((np.where(mag > 3*np.std(mag)+np.mean(mag)))[0])) / t
    return AboveMeanBySTD_3

def above5(mag):
    """This function measures the ratio of data points that are above 5 standard deviations from the mean magnitude."""
    t = np.float(len(mag))
    AboveMeanBySTD_5 = np.float(len((np.where(mag > 5*np.std(mag)+np.mean(mag)))[0])) / t
    return AboveMeanBySTD_5
        
def below1(mag):
    """This function measures the ratio of data points that are below 1 standard deviations from the mean magnitude."""
    t = np.float(len(mag))
    BelowMeanBySTD_1 = np.float(len((np.where(mag < np.std(mag)+np.mean(mag))[0]))) / t
    return BelowMeanBySTD_1
      
def below3(mag):
    """This function measures the ratio of data points that are below 3 standard deviations from the mean magnitude."""
    t = np.float(len(mag))
    BelowMeanBySTD_3 = np.float(len((np.where(mag < 3*np.std(mag)+np.mean(mag))[0]))) / t
    return BelowMeanBySTD_3
      
def below5(mag):
    """This function measures the ratio of data points that are below 5 standard deviations from the mean magnitude."""
    t = np.float(len(mag))
    BelowMeanBySTD_5 = np.float(len((np.where(mag < 5*np.std(mag)+np.mean(mag))[0]))) / t
    return BelowMeanBySTD_5
      

def compute_statistics(mjd, mag, magerr):
      """This function will compute all the statistics and return them in an array in the following order: Root Mean Square, Median, Mean, Min, Max, Median Absolute Deviation, Kurtosis, Skewness, StetsonJ, StetsonK, vonNeumannRatio, Above 1, Above 3, Above 5, Below 1, Below 3, Below 5."""
    
      t = float(len(mjd))
    
      RMS = np.sqrt((np.mean(mag)**2))    
      median = np.median(mag)       
      mean = np.mean(mag)    
      maxMag = np.amax(mag)    
      minMag = np.amin(mag)    
      medianAbsDev = median_absolute_deviation(mag, axis = None)
    
      skewness = skew(mag, axis = 0, bias = True)
      kurtosis = (t*(t+1.)/((t-1.)*(t-2.)*(t-3.))*sum(((mag - np.mean(mag))/np.std(mag))**4)) - (3.*((t-1.)**2.)/((t-2.)*(t-3.)))

      range1 = range(0, len(mjd)-1)
      range2 = range(1, len(mjd))

      delta3 = np.sqrt((t/(t-1.)))*((mag - np.mean(mag))/magerr)
      sign = np.sign(((delta3[range1]*delta3[range2]))*(np.sqrt(np.absolute(delta3[range1]*delta3[range2]))))
      
      stetsonJ = sum(((sign*delta3[range1]*delta3[range2]))*(np.sqrt(np.absolute(delta3[range1]*delta3[range2]))))
      stetsonK = ((1./t)*sum(abs(delta3)))/(np.sqrt((1./t)*sum((delta3)**2)))
     
      delta2 = sum((mag[1:] - mag[:-1])**2 / (t-1.))
      sample_variance = tvar(mag, limits=None)
    
      vonNeumannRatio = delta2 / sample_variance
          
      AboveMeanBySTD_1 = np.float(len((np.where(mag > np.std(mag)+np.mean(mag)))[0])) / t
      AboveMeanBySTD_3 = np.float(len((np.where(mag > 3*np.std(mag)+np.mean(mag)))[0])) / t
      AboveMeanBySTD_5 = np.float(len((np.where(mag > 5*np.std(mag)+np.mean(mag)))[0])) / t

      BelowMeanBySTD_1 = np.float(len((np.where(mag < np.std(mag)+np.mean(mag))[0]))) / t
      BelowMeanBySTD_3 = np.float(len((np.where(mag < 3*np.std(mag)+np.mean(mag))[0]))) / t
      BelowMeanBySTD_5 = np.float(len((np.where(mag < 5*np.std(mag)+np.mean(mag))[0]))) / t
 
      return [RMS, median, mean, maxMag, minMag, medianAbsDev, kurtosis, skewness, stetsonJ, stetsonK, vonNeumannRatio, AboveMeanBySTD_1, AboveMeanBySTD_3, AboveMeanBySTD_5, BelowMeanBySTD_1, BelowMeanBySTD_3, BelowMeanBySTD_5]

mag = np.array([18, 18.3, 18.1, 18, 18.4, 18.9, 19.2, 19.3, 19.5, 19.2, 18.8, 18.3, 18.6])
magerr = np.array([0.01, 0.01, 0.03, 0.09, 0.04, 0.1, 0.03, 0.13, 0.04, 0.06, 0.09, 0.1, 0.35])
mjd = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
"""
class MyTest(unittest.TestCase):
    def test_rms(value):
        value.assertEqual( RootMeanSquare(mag), 18.661538461538463 )

    def test_mean(value):
        value.assertEqual( meanMag(mag), 18.661538461538463 )

    def test_median(value):
        value.assertEqual( medianMag(mag), 18.600000000000001)
    
    def test_min(value):
        value.assertEqual( minMag(mag), 18.0)

    def test_max(value):
        value.assertEqual( maxMag(mag), 19.5)
    
    def test_medianAbsDev(value):
        value.assertEqual( medianAbsDev(mag), 0.5)
    
    def test_kurtosis(value):
        value.assertEqual( kurtosis(mjd, mag), -1.0149598629254664 )
    
    def test_skewness(value):
        value.assertEqual( skewness(mag), 0.1868991393928264 )

    def test_stetsonJ(value):
        value.assertEqual( stetsonJ(mjd, mag, magerr), 159412.78061393721 )

    def test_stetsonK(value):
        value.assertEqual( stetsonK(mjd, mag, magerr), 0.64699834923516031 )

    def test_vonNeumannRatio(value):
        value.assertEqual( vonNeumannRatio(mag), 0.38896680691912117 )
    
    def test_above1(value):
        value.assertEqual( above1(mag), 0.3076923076923077 )
    
    def test_above3(value):
        value.assertEqual( above3(mag), 0.0 )
    
    def test_above5(value):
        value.assertEqual( above5(mag), 0.0)
    
    def test_below1(value):
        value.assertEqual( below1(mag), 0.6923076923076923 )

    def test_below3(value):
        value.assertEqual( below3(mag), 1.0 )

    def test_below5(value):
        value.assertEqual( below5(mag), 1.0 )

    def test_compute_statistics(value):
        value.assertEqual( compute_statistics(mjd, mag, magerr), [18.661538461538463, 18.600000000000001, 18.661538461538463, 19.5, 18.0, 0.5, -1.0149598629254664, 0.1868991393928264, 159412.78061393721, 0.64699834923516031, 0.38896680691912117, 0.3076923076923077, 0.0, 0.0, 0.6923076923076923, 1.0, 1.0] )
    
unittest.main()
"""