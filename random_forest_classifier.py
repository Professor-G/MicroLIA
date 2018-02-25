# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA import microlmodels
from stats_computation2 import compute_statistics
import csv
import os

training_set = np.loadtxt('Training_dataset.txt', dtype = str)
rf = RandomForestClassifier(n_estimators=300, max_features = 3)
rf.fit(training_set[:,[2,3,5,6,7,8,9,10,11,12,13,14,21]].astype(float),training_set[:,0])


def predict_class(mjd, mag, magerr):
    """This function uses machine learning to classify any given lightcurve as either
        a Cataclysmic Variable (CV), a Lyrae Variable, Microlensing, a constant source, or
        some other type of transient/variable event.
        
        :param time: the time-varying data of the lightcurve. Must be an array.
        :param mag: the time-varying intensity of the object. Must be an array.
        :param accuracy: when accuracy is 2 the algorithm repeats the RF prediction
        three times for a more accurate, averaged result. Default is 1 with no repetition.
        
        :return: the function will return the predicted class along with the probability that it's microlensing.
        :rtype: string, float
        """
    
    stat_array = compute_statistics(mjd, mag, magerr)
    prediction = rf.predict(stat_array[0:13])#.astype(float)
    probability_prediction = rf.predict_proba(stat_array[0:13])[:,3]
    
    if prediction == 'ML':
        print 'Event detected with', np.float(probability_prediction)*100,'% confidence. Now fitting with pyLIMA...'
        #creates temporary file to avoid error when inputting data to pylima#
        with open('temporary.txt', 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            mjd = [np.float(x) for x in mjd]
            mag = [np.float(x) for x in mag]
            magerr = [np.float(x) for x in magerr]
            writer.writerows(zip(mjd, mag, magerr))
        
        data = np.loadtxt('temporary.txt')
        
        your_event = event.Event()
        your_event.name = 'Detection'
        
        telescope_1 = telescopes.Telescope(name='Detection', camera_filter='i', light_curve_magnitude=data)
        your_event.telescopes.append(telescope_1)
        model_1 = microlmodels.create_model('PSPL', your_event)
        your_event.fit(model_1,'LM')
        your_event.fits[0].produce_outputs()
        plt.close()
        os.remove('temporary.txt')
        
        Chi2 = your_event.fits[0].outputs.fit_parameters.chichi
        uo = your_event.fits[0].outputs.fit_parameters.uo
        to = your_event.fits[0].outputs.fit_parameters.to
        tE = your_event.fits[0].outputs.fit_parameters.tE
        reduced_chi = Chi2/(len(mjd)-4.0)
        
        if tE >= 1 and uo != 0 and uo < 2.0 and reduced_chi <= 3.0 and len(np.argwhere(((to - np.abs(tE)) < mjd) & ((to + np.abs(tE)) > mjd))) >= 2:
            prediction = prediction
            print 'Mirolensing candidate detected with parameters { uo:', uo,'|tE:', tE,'|to:', to,'}'
        else:
            prediction = 'BAD'
            print 'False alert -- fitted PSPL parameters not within range'

    return prediction, np.float(probability_prediction)

