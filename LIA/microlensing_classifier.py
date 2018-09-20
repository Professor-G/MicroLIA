# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from warnings import warn
from LIA.lib import extract_features

def predict(mag, magerr, rf_model, pca_model):
    """This function uses machine learning to classify any given lightcurve as either
        a cataclysmic variable (CV), a variable source, microlensing, or a constant star 
        displaying no variability.
        
    Parameters
    ----------
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
        Probability source is CV
    var_pred : float
        Probability source is variable
    """
    if len(mag) < 30:
        warn('The number of data points is low -- results may be unstable')

    array=[]
    stat_array = array.append(extract_features.extract_all(mag, magerr, convert=True))
    array=np.array([i for i in array])
    stat_array = pca_model.transform(array)
    #stat_array=stat_array[:,np.arange(0,46)]
    
    prediction =rf_model.predict(stat_array)
    cons_pred = rf_model.predict_proba(stat_array)[:,0] #CONSTANT
    cv_pred = rf_model.predict_proba(stat_array)[:,1] #CV
    ml_pred = rf_model.predict_proba(stat_array)[:,2] #ML
    var_pred = rf_model.predict_proba(stat_array)[:,3] #VARIABLE
    
    return prediction, ml_pred, cons_pred, cv_pred, var_pred

