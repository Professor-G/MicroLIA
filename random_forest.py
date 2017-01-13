# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:41:05 2017

@author: danielgodinez
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def classifier(test_statistics):
    """This classifier is trained using the contents in "training_set.txt" to differentiate between Cataclysmic Variables, RR Lyrae Variables, constant stars, and Microlensing. This function outputs the predicted class of your source, as well as the prediction probability for each of the four classes."""
    training_set = np.loadtxt('training_set.txt', dtype = str)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(training_set[:,1:9].astype(float), training_set[:,0])
    
    results = rf.predict(test_statistics[1:9])
    return results
    return rf.predict_proba(test_statistics[1:9])
    
