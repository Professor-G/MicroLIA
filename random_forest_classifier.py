# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:59:14 2017

@author: danielgodinez
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew
from astropy.stats import median_absolute_deviation
from scipy.stats import tvar

data = np.loadtxt('flat_7.dat')
time = data[:,4]
mag = data[:,5]
magerr = data[:,6]

def predict_class(time, mag, magerr):
      """This function uses machine learning to classify any given lightcurve as either a Cataclysmic Variable (CV), a Lyrae Variable, Microlensing, or a constant source."""
      t = float(len(time))
    
      RMS = np.sqrt((np.mean(mag)**2))     
      medianAbsDev = median_absolute_deviation(mag, axis = None)      
      std_over_mean = np.std(mag)/np.mean(mag)
      amplitude = np.max(mag) - np.min(mag)
      median_buffer_range = np.float(len((np.where(((amplitude/10) - np.median(mag) < mag) & ((amplitude/10) + np.median(mag) > mag))[0]))) / t

      skewness = skew(mag, axis = 0, bias = True)
      kurtosis = (t*(t+1.)/((t-1.)*(t-2.)*(t-3.))*sum(((mag - np.mean(mag))/np.std(mag))**4)) - (3.*((t-1.)**2.)/((t-2.)*(t-3.)))

      range1 = range(0, len(time)-1)
      range2 = range(1, len(time))
      delta3 = np.sqrt((t/(t-1.)))*((mag - np.mean(mag))/magerr)
      sign = np.sign(((delta3[range1]*delta3[range2]))*(np.sqrt(np.absolute(delta3[range1]*delta3[range2]))))
      
      stetsonJ = sum(((sign*delta3[range1]*delta3[range2]))*(np.sqrt(np.absolute(delta3[range1]*delta3[range2]))))
      stetsonK = ((1./t)*sum(abs(delta3)))/(np.sqrt((1./t)*sum((delta3)**2)))
     
      delta2 = sum((mag[1:] - mag[:-1])**2 / (t-1.))
      sample_variance = tvar(mag, limits=None)
    
      vonNeumannRatio = delta2 / sample_variance
          
      AboveMeanBySTD_1 = np.float(len((np.where(mag > np.std(mag)+np.mean(mag)))[0])) / t
      AboveMeanBySTD_3 = np.float(len((np.where(mag > 3*np.std(mag)+np.mean(mag)))[0])) / t
      AboveMeanBySTD_5 = np.float(len((np.where(mag > 5*np.std(mag)+np.mean(mag)))[0])) / t

      BelowMeanBySTD_1 = np.float(len((np.where(mag < np.std(mag)+np.mean(mag))[0]))) / t
      BelowMeanBySTD_3 = np.float(len((np.where(mag < 3*np.std(mag)+np.mean(mag))[0]))) / t
      BelowMeanBySTD_5 = np.float(len((np.where(mag < 5*np.std(mag)+np.mean(mag))[0]))) / t

      stat_array = [BelowMeanBySTD_1, BelowMeanBySTD_3, BelowMeanBySTD_5, AboveMeanBySTD_1, AboveMeanBySTD_3, AboveMeanBySTD_5, skewness, stetsonK, vonNeumannRatio, kurtosis, medianAbsDev, stetsonJ, amplitude, std_over_mean, median_buffer_range, RMS]

      training_set = np.loadtxt('training_set_ALL.txt', dtype = str)

      rf = RandomForestClassifier(n_estimators=100) 
      rf.fit(training_set[:,7:16].astype(float),training_set[:,0])
      prediction = rf.predict(stat_array[6:15])#.astype(float))
      probability_prediction = rf.predict_proba(stat_array[6:15])
     
      return prediction
      return probability_prediction
      print prediction
      print probability_prediction
      
predict_class(time, mag, magerr)  
    
    