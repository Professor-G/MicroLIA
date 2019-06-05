# -*- coding: utf-8 -*-
"""
    Created on Fri Jan 13 14:46:19 2017
    @author: danielgodinez
"""
import numpy as np
import unittest

import sys
sys.path.append('../')
from LIA import microlensing_classifier
from LIA import models

data=np.loadtxt('ml_event.txt')
mag=data[:,1]
magerr=data[:,2]

rf,pca = models.create_models('all_features.txt', 'pca_features.txt')

class Test(unittest.TestCase):
    """
    Unittest to ensure the classifier is working correctly. 
    """
    def test_predict(value):
        value.assertEqual( microlensing_classifier.predict(mag,magerr,rf,pca)[0], 'ML', "Classifier failed, predicted class is not correct.")
    def test_probability_prediction(value):
        pred = microlensing_classifier.predict(mag,magerr, rf,pca)[1]
        value.assertTrue(pred >= 0.4 and pred <= 0.6, "Classifier failed, probability prediction not within range.")

unittest.main()
    
