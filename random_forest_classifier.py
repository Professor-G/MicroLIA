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
from stats_computation import compute_statistics
import csv
import os

training_set = np.loadtxt('Training_dataset.txt', dtype = str)
rf = RandomForestClassifier(n_estimators=300, max_features = None)
rf.fit(training_set[:,[1,3,4,5,6,7,10,12,13,19,20,21,22]].astype(float),training_set[:,0])


def predict_class(mjd, mag, magerr):
    """This function uses machine learning to classify any given lightcurve as either
        a Cataclysmic Variable (CV), a Lyrae Variable, Microlensing, a constant source, or
        some other type of transient/variable event.
        
        :param time: the time-varying data of the lightcurve. Must be an array.
        :param mag: the time-varying intensity of the object. Must be an array.
        :param magerr: photometric error for the intensity. Must be an array.
        
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
        
        to_err = your_event.fits[0].outputs.fit_errors.err_to
        tE_err = your_event.fits[0].outputs.fit_errors.err_tE
        uo_err = your_event.fits[0].outputs.fit_errors.err_uo
        
        Chi2 = your_event.fits[0].outputs.fit_parameters.chichi
        uo = your_event.fits[0].outputs.fit_parameters.uo
        to = your_event.fits[0].outputs.fit_parameters.to
        tE = your_event.fits[0].outputs.fit_parameters.tE
        reduced_chi = Chi2/(len(mjd)-4.0)
        
        if to_err*tE_err*uo_err == 0.0:
            
            print 'Fitting now...'
            
            your_event.fit(model_1,'DE')
            your_event.fits[-1].produce_outputs()
            plt.close()
            Chi2 = your_event.fits[-1].outputs.fit_parameters.chichi
            uo = your_event.fits[-1].outputs.fit_parameters.uo
            to = your_event.fits[-1].outputs.fit_parameters.to
            tE = your_event.fits[-1].outputs.fit_parameters.tE
            reduced_chi = 3.0
                
        os.remove('temporary.txt')
        
        if tE >= 1 and uo != 0 and uo < 2.0 and reduced_chi <= 3.0 and len(np.argwhere(((to - np.abs(tE)) < mjd) & ((to + np.abs(tE)) > mjd))) >= 2:
            prediction = prediction
            print 'Mirolensing candidate detected with parameters { uo:', uo,'|tE:', tE,'|to:', to,'}'
        else:
            prediction = 'OTHER'
            print 'False alert -- fitted PSPL parameters not within range'

    return prediction, np.float(probability_prediction)

data=np.loadtxt('ml_event.txt')
print predict_class(data[:,0], data[:,1], data[:,2])



