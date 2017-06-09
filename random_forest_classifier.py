# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:59:14 2017

@author: danielgodinez
"""
from stats_computation import compute_statistics
import numpy as np
from sklearn.ensemble import RandomForestClassifier

training_set = np.loadtxt('training_set_all.txt', dtype = str)
rf = RandomForestClassifier(n_estimators=100, max_features = 3) 
rf.fit(training_set[:,1:18].astype(float),training_set[:,0])

def predict_class(time, mag, magerr = None):
    """This function uses machine learning to classify any given lightcurve as either
    a Cataclysmic Variable (CV), a Lyrae Variable, Microlensing, or a constant source.
    
    :param time: the time-varying data of the lightcurve. Must be an array.
    
    :param mag: the time-varying intensity of the object. Must be an array.
    
    :param magerr: photometric error for the intensity. Must be an array.
    If magerr = None the default is 0 for every photometric point. 
    
    :return: the function will return the predicted class.
    :rtype: string
    """
    
    if magerr is None:
        magerr = np.array([0.0001] * len(time))
        
    stat_array = compute_statistics(time, mag, magerr)
      
    prediction = rf.predict(stat_array[0:17])#.astype(float)
    probability_prediction = rf.predict_proba(stat_array[0:17])
    
    print(probability_prediction)     
    return prediction
    




      
    
    
