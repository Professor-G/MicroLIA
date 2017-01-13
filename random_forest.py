# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:41:05 2017

@author: danielgodinez
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def classifier(test_statistics):
    training_set = np.loadtxt('training_set.txt', dtype = str)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(training_set[:,1:9].astype(float), training_set[:,0])
    
    results = rf.predict(test_statistics[1:9])
    return results
    return rf.predict_proba(test_statistics[1:9])
    
