# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import numpy as np
from warnings import warn
from LIA import extract_features

def predict(time, mag, magerr, rf_model, pca_model):
    """This function uses machine learning to classify any given lightcurve as either
        a cataclysmic variable (CV), a variable source, microlensing, or a constant star 
        displaying no variability.
        
    Parameters
    ----------
    time : array
        Time of observations.
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    rf_model: fn
        The random forest model created using the function in the
        create_models module.
    pca_model: fn
        The PCA transformation devised using the function in the
        create_models module. 

    Returns
    -------
    prediction : str
        Predicted class.
    ml_pred : float
        Probability source is microlensing
    cons_pred : float
        Probability source is constant
    cv_pred : float
        Probability source is cataclysmic variable
    var_pred : float
        Probability source is variable
    lpv_pred : float
        Probabily source is a long period variable
    """
    if len(mag) < 30:
        warn('The number of data points is low -- results may be unstable')

    classes = ['CONSTANT', 'CV', 'LPV', 'ML', 'VARIABLE']
    array=[]
    array.append(extract_features.extract_all(time, mag, magerr, convert=True))

    stat_array = pca_model.transform(array)
    stat_array = array #checking to see if using only stats works
    pred = rf_model.predict_proba(stat_array)
    cons_pred, cv_pred, lpv_pred, ml_pred, var_pred = pred[:,0],pred[:,1],pred[:,2],pred[:,3],pred[:,4]
    prediction = classes[np.argmax(pred)]

    return prediction, ml_pred, cons_pred, cv_pred, var_pred, lpv_pred

