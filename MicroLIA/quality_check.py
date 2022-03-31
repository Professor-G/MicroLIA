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

def test_classifier(train_feats, train_pca_feats, test_feats, test_pca_feats):
    """This function will test the Random Forest
    and Neural Network classifier, both with PCA
    and non-PCA transformation.

    Can only run this if a training set has already
    been created, as this function requires the two
    output files 'all_features' and 'pca_features'.

    Parameters
    ----------
    all_feats : str
        Name of text file containing all features 
        This is output after running training_set.create()
    pca_feats : str
        Name of text file containing PCA features 
        This is output after running training_set.create()

    Returns
    -------
    Classification reports and mean accuracy
    after 10-fold cross-validation for both
    RF and NN classifiers, tested with PCA transformation
    and without.
    """
    try:
        np.loadtxt(train_feats, dtype=str)
    except IOError:
        raise ValueError("Could not find features file, please check directory and try again.")

    print("")
    print("------------------------------")
    print("Testing classifier without PCA...")
    print("------------------------------")

    X_train = np.loadtxt(train_feats, dtype=str)[:,2:].astype(float)
    y_train = np.loadtxt(train_feats, dtype=str)[:,0]

    X_test = np.loadtxt(test_feats, dtype=str)[:,2:].astype(float)
    y_test = np.loadtxt(test_feats, dtype=str)[:,0]

    RF=RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    RF_pred_test = RF.predict(X_test)
    RF_cross_validation = cross_validate(RF, X_test, y_test, cv=10)

    NN = MLPClassifier(hidden_layer_sizes=(1000,), max_iter=5000, activation='relu', solver='adam', tol=1e-4, learning_rate_init=.0001).fit(X_train, y_train)
    NN_pred_test = NN.predict(X_test)
    NN_cross_validation = cross_validate(NN, X_test, y_test, cv=10)

    print(" --- Random Forest Classification Report ---")
    print(classification_report(y_test, RF_pred_test))
    print("Mean Accuracy After 10-fold Cross-Validation: "+ str(round(np.mean(RF_cross_validation['test_score'])*100, 2))+'%')
    print("---------------------------------------------")
    print("")
    print(" --- Neural Network Classification Report ---")
    print(classification_report(y_test, NN_pred_test))
    print("Mean Accuracy After 10-fold Cross-Validation: "+ str(round(np.mean(NN_cross_validation['test_score'])*100, 2))+'%')
    print("---------------------------------------------")

    print("")
    print("------------------------------")
    print("Testing classifier with PCA...")
    print("------------------------------")

    X_train = np.loadtxt(train_pca_feats, dtype=str)[:,2:].astype(float)
    y_train = np.loadtxt(train_pca_feats, dtype=str)[:,0]

    X_test = np.loadtxt(test_pca_feats, dtype=str)[:,2:].astype(float)
    y_test = np.loadtxt(test_pca_feats, dtype=str)[:,0]

    RF=RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    RF_pred_test = RF.predict(X_test)
    RF_cross_validation = cross_validate(RF, X_test, y_test, cv=10)

    NN = MLPClassifier(hidden_layer_sizes=(1000,), max_iter=5000, activation='relu', solver='adam', tol=1e-4, learning_rate_init=.0001).fit(X_train, y_train)
    NN_pred_test = NN.predict(X_test)
    NN_cross_validation = cross_validate(NN, X_test, y_test, cv=10)

    print(" --- Random Forest Classification Report ---")
    print(classification_report(y_test, RF_pred_test))
    print("Mean Accuracy After 10-fold Cross-Validation: "+ str(round(np.mean(RF_cross_validation['test_score'])*100, 2))+'%')
    print("---------------------------------------------")
    print("")
    print(" --- Neural Network Classification Report ---")
    print(classification_report(y_test, NN_pred_test))
    print("Mean Accuracy After 10-fold Cross-Validation: "+ str(round(np.mean(NN_cross_validation['test_score'])*100, 2))+'%')
    print("---------------------------------------------")

def create_test(timestamps, min_mag, max_mag, noise, zp, exptime, n_class, ml_n1, cv_n1, cv_n2, t0_dist, u0_dist, tE_dist, train_feats, train_pca_feats):
    """This function will create a test and then it will output the classification 
    accuracy after 10-fold Cross-Validation.

    """
    
    training_set.create(timestamps=timestamps, min_mag=min_mag, max_mag=max_mag, noise=None, zp=zp, exptime=exptime, n_class=n_class, ml_n1=ml_n1, cv_n1=cv_n1, cv_n2=cv_n2, 
    t0_dist=t0_dist, u0_dist=u0_dist, tE_dist=tE_dist, filename='_TEST', test=False)

    test_classifier(train_feats, train_pca_feats, 'all_features_TEST.txt', 'pca_features_TEST.txt')










