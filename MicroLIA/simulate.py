# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""
from __future__ import division
import numpy as np
import os

def microlensing(timestamps, baseline, t0_dist=None, u0_dist=None, tE_dist=None):
    """Simulates a microlensing event.  
    The microlensing parameter space is determined using data from an 
    analysis of the OGLE III microlensing survey from Y. Tsapras et al (2016).
    See: The OGLE-III planet detection efficiency from six years of microlensing observations (2003 to 2008).
    (https://arxiv.org/abs/1602.02519)
    Parameters
    ----------
    timestamps : array
        Timestamps of the lightcurve.
    baseline : float
        Baseline magnitude of the lightcurve.
    t0_dist: array, optional
        An array containing the minumum and maximum t0 value to be 
        considered during the microlensing simulations. The indivial
        t0 per simulation will be selected from a uniform distribution
        between these two values.
    u0_dist: array, optional
        An array containing the minumum and maximum u0 value to be 
        considered during the microlensing simulations. The indivial
        u0 per simulation will be selected from a uniform distribution
        between these two values.
    te_dist: array, optional
        An array containing the minumum and maximum tE value to be 
        considered during the microlensing simulations. The indivial
        tE per simulation will be selected from a uniform distribution
        between these two values.
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

    if u0_dist:
       lower_bound = u0_dist[0]
       upper_bound = u0_dist[1]
       u_0 = np.random.uniform(lower_bound, upper_bound) 
    else:
       u_0 = np.random.uniform(0, 1.0)

    if tE_dist:
       lower_bound = tE_dist[0]
       upper_bound = tE_dist[1]
       t_e = np.random.normal(lower_bound, upper_bound) 
    else:
       t_e = np.random.normal(30, 10.0)

    if t0_dist:
        lower_bound = t0_dist[0]
        upper_bound = t0_dist[1]
    else:
        # Set bounds to ensure enough measurements are available near t_0 
        lower_bound = np.percentile(timestamps, 1)-0.5*t_e
        upper_bound = np.percentile(timestamps, 99)+0.5*t_e

    t_0 = np.random.uniform(lower_bound, upper_bound)  

    blend_ratio = np.random.uniform(0,1)

    u_t = np.sqrt(u_0**2 + ((timestamps - t_0) / t_e)**2)
    magnification = (u_t**2 + 2.) / (u_t * np.sqrt(u_t**2 + 4.))

    flux = 10**((baseline) / -2.5)

    flux_base = np.median(flux)

    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s

    flux_obs = f_s*magnification + f_b
    microlensing_mag = -2.5*np.log10(flux_obs)

    return np.array(microlensing_mag), [baseline]*len(flux_obs), u_0, t_0, t_e, blend_ratio
    
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

def variable(timestamps, baseline, bailey=None):       #theory, McGill et al. (2018)
    """Simulates a variable star.  

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline magnitude at which to simulate the lightcurve.
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
    time, ampl_k, phase_k, period = setup_parameters(timestamps, bailey)
    lightcurve = np.array(baseline)

    for idx in range(len(ampl_k)):
        lightcurve = lightcurve + ampl_k[idx] * np.cos(((2*np.pi*(idx+1))/period)*time+phase_k[idx])
    amplitude = np.ptp(lightcurve) / 2.0

    return np.array(lightcurve), amplitude, period 

def simulate_mira_lightcurve(timestamps, baseline, primary_period, amplitude_pp, secondary_period, amplitude_sp, tertiary_period, amplitude_tp):
    """Simulates LPV - Miras  
    Miras data from OGLE III: http://www.astrouw.edu.pl/ogle/ogle3/OIII-CVS/blg/lpv/pap.pdf

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline magnitude at which to simulate the lightcurve.

    Returns
    -------
    mag : array
        Simulated magnitudes given the timestamps.
    """
    amplitudes, periods = random_mira_parameters(primary_period, amplitude_pp, secondary_period, amplitude_sp, tertiary_period, amplitude_tp)
    lc = np.array(baseline)

    for idx in range(len(amplitudes)):
        lc = lc + amplitudes[idx]* np.cos((2*np.pi*(idx+1))/periods[idx]*timestamps)

    return np.array(lc)

def parametersRR0():            
    """
    McGill et al. (2018): Microlens mass determination for Gaia’s predicted photometric events.
    """
    a1=  0.31932222222222223
    ratio12 = 0.4231184105222867 
    ratio13 = 0.3079439089738683 
    ratio14 = 0.19454399944326523
    f1 =  3.9621766666666667
    f2 =  8.201326666666667
    f3 =  6.259693777777778
    return a1, ratio12, ratio13, ratio14, f1, f2, f3

def parametersRR1():
    a1 =  0.24711999999999998
    ratio12 = 0.1740045322110716 
    ratio13 = 0.08066256609474477 
    ratio14 = 0.033964605589727
    f1 =  4.597792666666666
    f2 =  2.881016
    f3 =  1.9828297333333336
    return a1, ratio12, ratio13, ratio14, f1, f2, f3

def uncertainties(time, curve, uncertain_factor):       
    """
    optional, add random uncertainties, controlled by the uncertain_factor
    """
    N = len(time)
    uncertainty = np.random.normal(0, uncertain_factor/100, N)
    realcurve = []   
                                       #curve with uncertainties
    for idx in range(N):
        realcurve.append(curve[idx]+uncertainty[idx])

    return realcurve

def setup_parameters(timestamps, bailey=None):   
    """
    Setup of random physical parameters
    """
    time = np.array(timestamps) 
    if bailey is None:
        bailey = np.random.randint(1,4)
    if bailey < 0 or bailey > 3:
        raise RuntimeError("Bailey out of range, must be between 1 and 3.")

    a1, ratio12, ratio13, ratio14, f1, f2, f3  = parametersRR1()

    if bailey == 1:
        period = np.random.normal(0.6, 0.15)
        a1, ratio12, ratio13, ratio14, f1, f2, f3  = parametersRR0()
    elif bailey == 2:
        period = np.random.normal(0.33, 0.1)
    elif bailey == 3:
        period = np.random.lognormal(0., 0.2)
        period = 10**period

    s = 20
    period=np.abs(period)  
    n1 = np.random.normal(a1, 2*a1/s)
    ampl_k = [n1, np.random.normal(n1*ratio12, n1*ratio12/s), np.random.normal(n1*ratio13, n1*ratio13/s), np.random.normal(n1*ratio14, n1*ratio14/s)]
    phase_k = [0, np.random.normal(f1, f1/s), np.random.normal(f2, f2/s), np.random.normal(f3, f3/s)]
    
    return time, ampl_k, phase_k, period

def random_mira_parameters(primary_period, amplitude_pp, secondary_period, amplitude_sp, tertiary_period, amplitude_tp):
    """
    Setup of random physical parameters
    """
    len_miras = len(primary_period)
    rand_idx = np.random.randint(0,len_miras,1)
    amplitudes = [amplitude_pp[rand_idx], amplitude_sp[rand_idx], amplitude_tp[rand_idx]]
    periods = [primary_period[rand_idx], secondary_period[rand_idx], tertiary_period[rand_idx]]

    return amplitudes, periods
