# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""
from __future__ import division
from gatspy import datasets, periodic
import pkg_resources
import importlib.util
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

    lc = lc + baseline 

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

def rrlyr_variable(timestamps, baseline, bailey=None):
    """Simulates a variable source. 
    This simulation is done using the gatspy module provided by AstroML. 
    This module implements a template-based model using RR Lyrae
    lightcurves from Sesar et al. 2010, which contains observations of
    RR Lyrae in Stripe 82 (from SDSS) measured approximately over a 
    decade. For the purpose of simulating variable stars we apply a 
    single band ('g') template model and modify only the period. We 
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
        3 simulates a Cepheid variable period, but note the amplitude
        is still derived from the RRLyrae template. If not provided
        it defaults to a random choice between the 1 and 2. 

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
        bailey = np.random.randint(1,3) #Removing Bailey=3 option
    if bailey < 0 or bailey > 2:
        raise RuntimeError("Bailey out of range, must be between 1 and 2.")

    if bailey == 1:
        period = np.random.normal(0.6, 0.15)
    elif bailey == 2:
        period = np.random.normal(0.33, 0.1)
    elif bailey == 3:
        period = np.random.lognormal(0., 0.2)
        period = 10**period
    
    period=np.abs(period) #on rare occassions period is negative which breaks the code

    #Fetch random RRLyrae template 
    rrlyrae = datasets.fetch_rrlyrae_templates(data_home=get_rrlyr_data_path())
    unique_id = np.unique([i[:-1] for i in rrlyrae.ids])
    inx = np.random.randint(len(unique_id)) #Random index to choose RR Lyrae lc

    filt = 'g' #All 23 RRLyr templates have g data! 
    lcid = unique_id[inx] + filt #rrlyrae.ids[inx]
    t, mag = rrlyrae.get_template(lcid)

    #Fit to our simulated cadence
    model = periodic.RRLyraeTemplateModeler(filt)
    model.fit(t, mag)
    mag_fit = model.predict(timestamps, period = period)

    #The following fit is only done to extract amplitude.
    mag_fit_amp = model.predict(np.arange(0, period, 0.01), period = period)
    amplitude = np.ptp(mag_fit_amp) / 2.0

    #Bring lc down to 0 and add input baseline
    mag_fit = mag_fit - np.mean(mag_fit_amp)
    mag_fit = mag_fit+baseline

    return np.array(mag_fit), amplitude, period

def variable(timestamps, baseline, bailey=None):       
    """Simulates a variable star. Theory from McGill et al. (2018).
    
    This function is outdated! Replaced with rrlyr_variable which employs the use
    of templates, thus more representative of true RRLyrae stars.

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
    bailey : float
        Period of the signal in days.   
    """

    time, ampl_k, phase_k, period = setup_parameters(timestamps, bailey)
    lightcurve = np.array(baseline)

    for idx in range(len(ampl_k)):
        lightcurve = lightcurve + ampl_k[idx] * np.cos(((2*np.pi*(idx+1))/period)*time+phase_k[idx])

    amplitude = np.ptp(lightcurve) / 2.0

    return np.array(lightcurve), amplitude, period 

def simulate_mira_lightcurve(timestamps, baseline, primary_period, amplitude_pp, secondary_period, amplitude_sp, tertiary_period, amplitude_tp):
    """
    Simulates a Mira long-period variable (LPV) lightcurve.

    Parameters
    ----------
    timestamps : array-like
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline magnitude at which to simulate the lightcurve.
    primary_period : float
        Primary period of the Mira.
    amplitude_pp : float
        Amplitude of the primary period.
    secondary_period : float
        Secondary period of the Mira.
    amplitude_sp : float
        Amplitude of the secondary period.
    tertiary_period : float
        Tertiary period of the Mira.
    amplitude_tp : float
        Amplitude of the tertiary period.

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


#McGill et al. (2018): Microlens mass determination for Gaia’s predicted photometric events.

def parametersRR0():            
    """
    Returns the physical parameters for RR0 type variable stars.
    
    The f1, f2, and f3 parameters in the Mira lightcurve simulations represent the factors that control 
    the amplitude of the secondary, tertiary, and higher-order periodic variations in the Mira lightcurve. 
    These parameters allow for the modulation of the amplitudes of the secondary and tertiary periods relative 
    to the primary period.

    In the context of the Mira lightcurve simulation, these parameters determine the contribution of the secondary 
    and tertiary periods to the overall variability of the lightcurve. By adjusting the values of f1, f2, and f3, you 
    can control the relative strengths of these additional periodic variations.

    These parameters are specific to the simulation model and are chosen based on the desired characteristics of the 
    simulated Mira lightcurve. They are typically determined based on observational data or theoretical considerations.

    Returns
    -------
    a1 : float
        Parameter a1.
    ratio12 : float
        Ratio between parameter 1 and parameter 2.
    ratio13 : float
        Ratio between parameter 1 and parameter 3.
    ratio14 : float
        Ratio between parameter 1 and parameter 4.
    f1 : float
        Parameter f1.
    f2 : float
        Parameter f2.
    f3 : float
        Parameter f3.
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
    """
    Returns the physical parameters for RR1 type variable stars.

    Returns
    -------
    a1 : float
        Parameter a1.
    ratio12 : float
        Ratio between parameter 1 and parameter 2.
    ratio13 : float
        Ratio between parameter 1 and parameter 3.
    ratio14 : float
        Ratio between parameter 1 and parameter 4.
    f1 : float
        Parameter f1.
    f2 : float
        Parameter f2.
    f3 : float
        Parameter f3.
    """

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
    Adds random uncertainties to a given curve.

    Parameters
    ----------
    time : array-like
        Times at which the curve is defined.
    curve : array-like
        Curve values.
    uncertain_factor : float
        Uncertainty factor in percentage.

    Returns
    -------
    realcurve : array
        Curve with added uncertainties.
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
    
    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    bailey : int, optional 
        The type of variable to simulate. A bailey
        value of 1 simulaes RR Lyrae type ab, a value
        of 2 simulates RR Lyrae type c, and a value of 
        3 simulates a Cepheid variable. If not provided
        it defaults to a random choice between the three. 
    
    Returns
    -------
    time, amp, phase, period : array
        Time, amplitude, phase, period.
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
    Sets up random physical parameters for simulating Mira lightcurves.

    Parameters
    ----------
    primary_period : array-like
        Array of primary periods.
    amplitude_pp : array-like
        Array of amplitudes for the primary periods.
    secondary_period : array-like
        Array of secondary periods.
    amplitude_sp : array-like
        Array of amplitudes for the secondary periods.
    tertiary_period : array-like
        Array of tertiary periods.
    amplitude_tp : array-like
        Array of amplitudes for the tertiary periods.

    Returns
    -------
    amplitudes : list
        List of amplitudes for the simulated lightcurve.
    periods : list
        List of periods for the simulated lightcurve.
    """

    len_miras = len(primary_period)
    rand_idx = np.random.randint(0,len_miras,1)
    amplitudes = [amplitude_pp[rand_idx], amplitude_sp[rand_idx], amplitude_tp[rand_idx]]
    periods = [primary_period[rand_idx], secondary_period[rand_idx], tertiary_period[rand_idx]]

    return amplitudes, periods

def get_rrlyr_data_path():
    """
    Retrieves the path to the RRLyrae template data directory within the MicroLIA package.

    Args:
        None

    Returns:
        data_path (str): Path to the data directory.
    """

    resource_package = __name__
    resource_path = 'data/Sesar2010'
    data_path = pkg_resources.resource_filename(resource_package, resource_path)
    
    return data_path

   