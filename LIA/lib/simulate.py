# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""
from __future__ import division
import numpy as np
from gatspy import datasets, periodic

def microlensing(timestamps, baseline):
    """Simulates a microlensing event.  
    The microlensing parameter space is determined using data from an 
    analysis of the OGLE III microlensing survey from Y. Tsapras et al (2016).
    See: The OGLE-III planet detection efficiency from six years of microlensing observations (2003 to 2008).
    (https://arxiv.org/abs/1602.02519)

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline magnitude at which to simulate the lightcurve.

    Returns
    -------
    mag : array
        Simulated magnitude given the timestamps.
    u_0 : float
        The source minimum impact parameter.
    t_0 : float
        The time of maximum magnification.
    t_E : float
        The timescale of the event in days.
    blend_ratio : float
        The blending coefficient chosen between 0 and 10.     
    """   

    mag = constant(timestamps, baseline)
    # Set bounds to ensure enough measurements are available near t_0 
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    
    t_0 = np.random.uniform(lower_bound, upper_bound)        
    u_0 = np.random.uniform(0, 1.5)
    t_e = np.random.normal(30, 10.0)
    blend_ratio = np.random.uniform(0,10)

    u_t = np.sqrt(u_0**2 + ((timestamps - t_0) / t_e)**2)
    magnification = (u_t**2 + 2.) / (u_t * np.sqrt(u_t**2 + 4.))
 
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    
    flux_obs = f_s*magnification + f_b+flux_noise
    microlensing_mag = -2.5*np.log10(flux_obs)
    
    return np.array(microlensing_mag), baseline, u_0, t_0, t_e, blend_ratio
    
def cv(timestamps, baseline):
    """Simulates Cataclysmic Variable event.
    The outburst can be reasonably well represented as three linear phases: a steeply 
    positive gradient in the rising phase, a flat phase at maximum brightness followed by a declining 
    phase of somewhat shallower negative gradient. The period is selected from a normal distribution
    centered about 100 days with a standard deviation of 200 days. The outburtst amplitude ranges from
    0.5 to 5.0 mag, selected from a uniform random function. 

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline magnitude at which to simulate the lightcurve.

    Returns
    -------
    mag : array
        Simulated magnitudes given the timestamps and baseline. 
    outburst_start_times : array
        The start time of each outburst.
    outburst_end_times : array
        The end time of each outburst.
    end_rise_times : array
        The end time of each rise (start time of max amplitude).
    end_high_times : array
        The end time of each peak (end time of max amplitude).
    """

    period = abs(np.random.normal(100, 200))
    amplitude = np.random.uniform(0.5, 5.0)
    lc = np.zeros(len(timestamps))
    # First generate the times when outbursts start. Note that the
    # duration of outbursts can vary for a single object, so the t_end_outbursts will be added later.
    start_times = []

    min_start = min(timestamps)
    max_start = min((min(timestamps)+period),max(timestamps))

    first_outburst_time = np.random.uniform(min_start, max_start)

    start_times.append(first_outburst_time)
    t_start = first_outburst_time + period

    for t in np.arange(t_start,max(timestamps),period):
        start_times.append(t)

    outburst_end_times = []
    duration_times = []
    end_rise_times = []
    end_high_times = []    
    
    for t_start_outburst in start_times:
    # Since each outburst can be a different shape,
    # generate the lightcurve morphology parameters for each outburst:
        duration = np.random.uniform(3.0, (period/10.0))
        duration_times.append(duration)
        t_end_outburst = t_start_outburst + duration
        outburst_end_times.append(t_end_outburst)        
        rise_time = np.random.uniform(0.5,1.0)
        high_state_time = np.random.normal(0.4*duration, 0.2*duration)
        drop_time = duration - rise_time - high_state_time
        t_end_rise = t_start_outburst + rise_time
        t_end_high = t_start_outburst + rise_time + high_state_time
        end_rise_times.append(t_end_rise)
        end_high_times.append(t_end_high)  
        # Rise and drop is modeled as a straight lines with differing gradients
        rise_gradient = -1.0 * amplitude / rise_time

        drop_gradient = (amplitude / drop_time)

        for i in range(0,len(timestamps),1):
                if timestamps[i] >= t_start_outburst and timestamps[i] <= t_end_rise:
                        lc[i] = rise_gradient * (timestamps[i] -t_start_outburst)
                elif timestamps[i] >= t_end_rise and timestamps[i] <= t_end_high:
                        lc[i] = -1.0 * amplitude
                elif timestamps[i] > t_end_high and timestamps[i] <= t_end_outburst:
                        lc[i] = -amplitude + ( drop_gradient * (timestamps[i] - t_end_high))

    lc = lc+baseline 
    return np.array(lc), np.array(start_times), np.array(outburst_end_times), np.array(end_rise_times), np.array(end_high_times)

def variable(timestamps, baseline, bailey=None):
    """Simulates a variable source. 
    This simulation is done using the gatspy module provided by AstroML. 
    This module implements a template-based model using RR Lyrae
    lightcurves from Sesar et al. 2010, which contains observations of
    483 RR Lyrae in Stripe 82 (from SDSS) measured approximately over a 
    decade. For the purpose of simulating variable stars we apply a 
    single band ('r') template model and modify only the period. We 
    currently only provide simulated RR Lyrae (periods < 1 day) or 
    Cepheid Variables which have an average period of 10 days.
    
    See:
    Template-based Period Fitting: https://www.astroml.org/gatspy/periodic/template.html
    Period distribution for RR Lyrae from Sesar et al. 2010 (https://arxiv.org/abs/0910.4611).
    Period distribution for Cepheids from Becker et al. 1977 (http://adsabs.harvard.edu/abs/1977ApJ...218..633B)

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline at which to simulate the lightcurve.
    bailey : int, optional 
        The type of variable to simulate. A bailey
        value of 1 simulaes RR Lyrae type ab, a value
        of 2 simulates RR Lyrae type c, and a value of 
        3 simulates a Cepheid variable. If not provided
        it defaults to a random choice between the three. 

    Returns
    -------
    mag : array
        Simulated magnitudes given the timestamps.
    amplitude : float
        Amplitude of the signal in mag. 
    period : float
        Period of the signal in days.
    """
    if bailey is None:
        bailey = np.random.randint(1,4)
    if bailey < 0 or bailey > 3:
        raise RuntimeError("Bailey out of range, must be between 1 and 3.")

    if bailey == 1:
        period = np.random.normal(0.6, 0.15)
    elif bailey == 2:
        period = np.random.normal(0.33, 0.1)
    elif bailey == 3:
        period = np.random.lognormal(0., 0.2)
        period = 10**period
    
    period=np.abs(period) #on rare occassions period is negative which breaks the code
    inx = np.random.randint(0, 482) #index to choose RR Lyrae lc
    rrlyrae = datasets.fetch_rrlyrae()
    lcid = rrlyrae.ids[inx]
    t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
    mask = (filts == 'r')
    t_r, mag_r, dmag_r = t[mask], mag[mask], dmag[mask]

    model = periodic.RRLyraeTemplateModeler('r')
    model.fit(t_r, mag_r, dmag_r)
    mag_fit = model.predict(timestamps, period = period)
    #amplitude = datasets.fetch_rrlyrae_lc_params()[inx][3]

    #The following fit is only done to extract amplitude.
    mag_fit_amp = model.predict(np.arange(0, period, 0.01), period = period)
    amplitude = np.ptp(mag_fit_amp) / 2.0
    #Bring lc down to 0 and add input baseline
    mag_fit = mag_fit - np.mean(mag_fit_amp)
    mag_fit = mag_fit+baseline

    return np.array(mag_fit), amplitude, period

def constant(timestamps, baseline):
    """Simulates a constant source displaying no variability.  

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline magnitude of the lightcurve.

    Returns
    -------
    mag : array
        Simulated magnitudes given the timestamps.
    """
    mag = [baseline] * len(timestamps)

    return np.array(mag)
   

def alternate_CV(timestamps, baseline):
  
    