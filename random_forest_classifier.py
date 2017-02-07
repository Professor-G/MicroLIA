# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:59:14 2017

@author: danielgodinez
"""
from stats_computation import compute_statistics
import numpy as np
from sklearn.ensemble import RandomForestClassifier

training_set = np.loadtxt('training_set_ALL.txt', dtype = str)
rf = RandomForestClassifier(n_estimators=100) 
rf.fit(training_set[:,7:16].astype(float),training_set[:,0])

def predict_class(time, mag, magerr):
    
      """This function uses machine learning to classify any given lightcurve as either a Cataclysmic Variable (CV), a Lyrae Variable, Microlensing, or a constant source."""
      stat_array = compute_statistics(time, mag, magerr)
      
      prediction = rf.predict(stat_array[6:15])#.astype(float))
      probability_prediction = rf.predict_proba(stat_array[6:15])
     
      return prediction
      return probability_prediction
      print prediction
      print probability_prediction
      
    
    
