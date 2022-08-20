# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""
from __future__ import division
import numpy as np
from MicroLIA import simulate
from MicroLIA import training_set

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report

def test_microlensing(timestamps, microlensing_mag, magerr, baseline, u_0, t_0, t_e, blend_ratio, n=7):
    """Test to ensure proper microlensing signal.
    This requires 7 measurements with a magnification of at least 1.34, imposing
    additional magnification thresholds to ensure the microlensing signal doesn't 
    mimic a noisy constant.

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    microlensing_mag : array
        Microlensing simulated magnitudes given the timestamps. 
    magerr : array
        Photometric error for each mag measurement.
    baseline : float
        Baseline magnitude of the event. 
    u_0 : float
        The source minimum impact parameter.
    t_0 : float
        The time of maximum magnification.
    t_E : float
        The timescale of the event in days.
    blend_ratio : float
        The blending coefficient.
    n : int, optional
        The mininum number of measurements that should be within the 
        microlensing signal when simulating the lightcurves. 
    Returns
    -------
    condition : boolean
        Returns True if microlensing passes the quality test. 
    """
    mag = simulate.constant(timestamps, baseline)
    condition = False
    signal_indices = np.argwhere((timestamps >= (t_0 - t_e)) & (timestamps <= (t_0 + t_e))) 
    if len(signal_indices) >= n:
        mean1 = np.mean(mag[signal_indices])
        mean2 = np.mean(microlensing_mag[signal_indices])
                
        signal_measurements = []
        for inx in signal_indices:
           value = (mag[inx] - microlensing_mag[inx]) / magerr[inx]
           signal_measurements.append(value)

        signal_measurements = np.array(signal_measurements)
        if (len(np.argwhere(signal_measurements >= 3)) > 0 and 
           mean2 < (mean1 - 0.05) and 
           len(np.argwhere(signal_measurements > 3)) >= 0.33*len(signal_indices) and 
           (1.0/u_0) > blend_ratio):
               condition = True
               
    return condition  

def test_cv(timestamps, outburst_start_times, outburst_end_times, end_rise_times, end_high_times, n1=7, n2=1):
    """Test to ensure proper CV signal.
    This requires 7 measurements within ANY outburst, with at least one 
    occurring within the rise or fall.

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    outburst_start_times : array
        The start time of each outburst.
    outburst_end_times : array
        The end time of each outburst.
    end_rise_times : array
        The end time of each rise (start time of max amplitude).
    end_high_times : array
        The end time of each peak (end time of max amplitude).
    n1 : int, optional
        The mininum number of measurements that should be within 
        at least one outburst, defaults to 7.
    n2 : int, optional
        The mininum number of measurements that should be within the 
        rise or drop of at least one outburst, defaults to 1.
        
    Returns
    -------
    condition : boolean
        Returns True if CV passes the quality test. 
    """
    signal_measurements = []
    rise_measurements = []
    fall_measurements = []
    condition = False
    for k in range(len(outburst_start_times)):
        inx = len(np.argwhere((timestamps >= outburst_start_times[k])&(timestamps <= outburst_end_times[k])))
        signal_measurements.append(inx)

        inx = len(np.argwhere((timestamps >= outburst_start_times[k])&(timestamps <= end_rise_times[k])))
        rise_measurements.append(inx)  

        inx = len(np.argwhere((timestamps >= end_high_times[k])&(timestamps <= outburst_end_times[k])))
        fall_measurements.append(inx) 

    for k in range(len(signal_measurements)):
        if signal_measurements[k] >= n1:
            if rise_measurements[k] or fall_measurements[k] >= n2:
                condition = True
                break 

    return condition 
