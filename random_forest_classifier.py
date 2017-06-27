# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:59:14 2017

@author: danielgodinez
"""
from stats_computation import compute_statistics
import numpy as np
from sklearn.ensemble import RandomForestClassifier

training_set = np.loadtxt('training_set.txt', dtype = str)
rf = RandomForestClassifier(n_estimators=300, max_features = 3) 
rf.fit(training_set[:,[2,3,5,6,7,8,11,13,14,17,20,21]].astype(float),training_set[:,0])

def predict_class(mag, magerr):
    """This function uses machine learning to classify any given lightcurve as either
    a Cataclysmic Variable (CV), a Lyrae Variable, Microlensing, or a constant source.
    
    :param time: the time-varying data of the lightcurve. Must be an array.
    :param mag: the time-varying intensity of the object. Must be an array.
    
    :return: the function will return the predicted class along with the probability that it's microlensing.
    :rtype: string, float
    """
        
    stat_array = compute_statistics(mag, magerr)
      
    prediction =rf.predict(stat_array[0:12])#.astype(float)
    probability_prediction = rf.predict_proba(stat_array[0:12])
    
    return prediction, probability_prediction[:,3]
    




      
    
    
