# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:30:15 2016

@author: danielgodinez
"""
from __future__ import print_function
import numpy as np
from stats_computation import compute_statistics


def simulate_microlensing(time, mag, magerr):
    """Simulates a microlensing event given the inserted lightcurve. The angular impact parameter is chosen 
    from a random distribution between 0.0 and 1.0. Likewise the time of maximum amplification t_0 is chosen
    from a normal distribution with a mean of 20 days and a standard deviation of 5 days and the timescale t_e 
    from a uniform distribution between 0.0 and 30 days. These parameter spaces were determined given an analysis of
    the OGLE III microlensing survey. See: The OGLE-III planet detection efficiency from six years of microlensing observations (2003 to 2008), (Y. Tsapras et al (2016)).
    """
    
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
   
   # print ('u_0 :',u_0, 't_0 :',t_0, 't_e :', t_e)
    
def microlensing_statistics(time, mag, magerr):
    """Simulates a microlensing event given an inserted lightcurve, and calculates
    various lightcurve statistics. Returns them in an array in the following order: 
    skewness, kurtosis, stetsonJ, stetsonK, vonNeumannRatio, std_over_mean, median_buffer_range, amplitude, Below1, Below3, Below5, Above1, Above3, Above5, magRMS, medianAbsDev, meanMag, medianMag, minMag, maxMag. """
   
    microlensing_mag = simulate_microlensing(time, mag, magerr)
    stats = compute_statistics(time, microlensing_mag, magerr)
    
    return stats
   
