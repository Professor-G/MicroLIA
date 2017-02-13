# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:30:15 2016

@author: danielgodinez
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from astropy.stats import median_absolute_deviation
from scipy.stats import tvar


def simulate_microlensing(time, mag, magerr):
    """Simulates a microlensing event given the inserted lightcurve. The angular impact parameter is chosen 
    from a random distribution between 0.0 and 1.0. Likewise the time of maximum amplification t_0 is chosen f
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
    
    plt.gca().invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('Simulated Microlensing')
    print ('u_0 :',u_0, 't_0 :',t_0, 't_e :', t_e)
    return plt.errorbar(time, microlensing_mag, yerr = magerr, fmt = 'ro')
    
def microlensing_statistics(time, mag, magerr):
    """Simulates a microlensing event given an inserted lightcurve, and calculates
    various lightcurve statistics. Returns them in an array in the following order: 
    skewness, kurtosis, stetsonJ, stetsonK, vonNeumannRatio, std_over_mean, median_buffer_range, amplitude, Below1, Below3, Below5, Above1, Above3, Above5, magRMS, medianAbsDev, meanMag, medianMag, minMag, maxMag. """
   
    t = np.float(len(time))
    
    
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
    
    skewness = skew(microlensing_mag, axis = 0, bias = True)
    kurtosis = (t*(t+1.)/((t-1.)*(t-2.)*(t-3.))*sum(((microlensing_mag - np.mean(microlensing_mag))/np.std(microlensing_mag))**4)) - (3.*((t-1.)**2.)/((t-2.)*(t-3.)))

    magRMS = np.std(microlensing_mag)       
    medianMag = np.median(microlensing_mag)       
    meanMag = np.mean(microlensing_mag)    
    maxMag = np.amax(microlensing_mag)
    minMag = np.amin(microlensing_mag)
    
    medianAbsDev = median_absolute_deviation(microlensing_mag, axis = None) 
    std_over_mean = np.std(microlensing_mag)/np.mean(microlensing_mag)
    
    range1 = range(0, len(time)-1)
    range2 = range(1, len(time))
    delta3 = np.sqrt((t/(t-1.)))*((microlensing_mag - np.mean(microlensing_mag))/magerr)

    sign = np.sign(((delta3[range1]*delta3[range2]))*(np.sqrt(np.absolute(delta3[range1]*delta3[range2]))))
    stetJ = sum(((sign*delta3[range1]*delta3[range2]))*(np.sqrt(np.absolute(delta3[range1]*delta3[range2]))))
    stetsonJ = np.nan_to_num(stetJ)
    
    stetK = ((1./t)*sum(abs(delta3)))/(np.sqrt((1./t)*sum((delta3)**2)))
    stetsonK = np.nan_to_num(stetK)
        
    delta2 = sum((microlensing_mag[1:] - microlensing_mag[:-1])**2 / (t-1.))
    sample_variance = tvar(microlensing_mag, limits=None)
    vonNeumannRatio = delta2 / sample_variance
          
    amplitude = maxMag - minMag
    median_buffer_range = np.float(len((np.where(((amplitude/10) - medianMag < microlensing_mag) & ((amplitude/10) + medianMag > microlensing_mag))[0]))) / t
     
    Above1 = np.float(len((np.where(microlensing_mag > np.std(microlensing_mag)+np.mean(microlensing_mag)))[0])) / t
    Above3 = np.float(len((np.where(microlensing_mag > 3*np.std(microlensing_mag)+np.mean(microlensing_mag)))[0])) / t
    Above5 = np.float(len((np.where(microlensing_mag > 5*np.std(microlensing_mag)+np.mean(microlensing_mag)))[0])) / t

    Below1 = np.float(len((np.where(microlensing_mag < np.std(microlensing_mag)+np.mean(microlensing_mag))[0]))) / t
    Below3 = np.float(len((np.where(microlensing_mag < 3*np.std(microlensing_mag)+np.mean(microlensing_mag))[0]))) / t
    Below5 = np.float(len((np.where(microlensing_mag < 5*np.std(microlensing_mag)+np.mean(microlensing_mag))[0]))) / t
 
    return np.array([skewness, kurtosis, stetsonJ, stetsonK, vonNeumannRatio, std_over_mean, median_buffer_range, amplitude, Below1, Below3, Below5, Above1, Above3, Above5, magRMS, medianAbsDev, meanMag, medianMag, minMag, maxMag])
      
   
