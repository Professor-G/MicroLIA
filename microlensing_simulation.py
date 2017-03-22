# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:30:15 2016

@author: danielgodinez
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from stats_computation import compute_statistics


def simulate_microlensing(time, mag, magerr = None):
    """Simulates a microlensing event given the inserted flat lightcurve. The angular 
    impact parameter is chosen from a random distribution between 0.0 and 1.0. 
    Likewise the time of maximum amplification t_0 is chosen from a normal 
    distribution with a mean of 20 days and a standard deviation of 5 days and 
    the timescale t_e from a uniform distribution between 0.0 and 30 days. 
    These parameter spaces were determined given an analysis of the OGLE III
    microlensing survey. See: The OGLE-III planet detection efficiency from six 
    years of microlensing observations (2003 to 2008), (Y. Tsapras et al (2016)).
    
    :param time: the time-varying data of the lightcurve. Must be an array.
    
    :param mag: the time-varying intensity of the object. Must be an array.
    
    :param magerr: photometric error for the intensity. If magerr = None the 
    default is 0 for every photometric point.
    
    :return: the function will return the simulated intensity of the simulated
    lightcurve as well as the following simulation parameters: u_0, t_0, t_e.
    :rtype: array, float
    """
    if magerr is None:
        magerr = np.array([0] * len(time))
    
    u_0 = np.random.uniform(low = 0, high = 1.0, size = 1)
    t_0 = np.random.choice(time)
    t_e = np.random.normal(loc = 30, scale = 10.0, size = 1)
    
    zp = 25.0
    g = np.random.uniform(0,10)
    
    u_t = np.sqrt(u_0**2 + ((time - t_0) / t_e)**2)
    magnification = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
    
    flux = 10**((mag - zp) / -2.5)
    baseline = np.median(flux)
    f_s = baseline / (1 + g)
    f_b = g * f_s
    
    normalised_flux = flux/baseline
    amplification_obs = normalised_flux*magnification
    
    microlensing_flux = (f_s*amplification_obs + f_b)
    microlensing_mag = zp - 2.5*np.log10(microlensing_flux)
    
    return microlensing_mag  
    print ('u_0 :',u_0, 't_0 :',t_0, 't_e :', t_e)
    
def plot_microlensing(time, mag, magerr = None):
    """Plots a simulated microlensing event from an inserted flat lightcurve.
    
    :param time: the time-varying data of the lightcurve. Must be an array.
    
    :param mag: the time-varying intensity of the object. Must be an array.
    
    :param magerr: photometric error for the intensity. If magerr = None the 
    default is 0 for every photometric point.
    
    :return: the function will return a plot of the microlensing lightcurve.
    :rtype: plot
    """
    
    if magerr is None:
        magerr = np.array([0] * len(time))
        
    intensity = simulate_microlensing(time, mag, magerr)
    
    plt.plot(time, intensity, 'ro')
    plt.gca().invert_yaxis
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title('Simulated Microlensing')
    
def microlensing_statistics(time, mag, magerr = None):
    """Simulates a microlensing event given an inserted lightcurve, and calculates
    various lightcurve statistics. Returns them in an array in the following order: 
    skewness, kurtosis, stetsonJ, stetsonK, vonNeumannRatio, std_over_mean, median_buffer_range, 
    amplitude, Below1, Below3, Below5, Above1, Above3, Above5, magRMS, medianAbsDev, meanMag, 
    medianMag, minMag, maxMag.

    :param time: the time-varying data of the lightcurve. Must be an array.
    
    :param mag: the time-varying intensity of the object. Must be an array.
    
    :param magerr: photometric error for the intensity. If magerr = None the 
    default is 0 for every photometric point.
    
    :return: the function will return the lightcurve statistics in the order listed above.
    :rtype: array
    
    """
    
    if magerr is None:
        magerr = np.array([0] * len(time))
        
    microlensing_mag = simulate_microlensing(time, mag, magerr)
    stats = compute_statistics(time, microlensing_mag, magerr)
    
    return stats
   
