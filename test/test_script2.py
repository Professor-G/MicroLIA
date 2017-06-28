# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:20:25 2017

@author: danielgodinez
"""
import sys
sys.path.append('../')
import stats_computation as sc
import random_forest_classifier as rf
import numpy as np
import unittest

data = np.loadtxt('TestML_2')

mag = data[:,1]
magerr = data[:,2]

class Test(unittest.TestCase):
    def test_compute_statistics(value):
        value.assertEqual( sc.compute_statistics(mag, magerr), (0.87383493175485616, 45.801672292354191,
                          -6.4480083062854563, 1.5740162737474053, 517.25538087579378, 0.55438404839089117, 0.0, 1.0,
                           0.9071235347159603, 0.027051397655545536, 0.0053277971807380031, 82, 0.09804122669179631), "Statistics array incorrect" )
         
    def test_predict_class(value):
        value.assertEqual( rf.predict_class(mag, magerr)[0], ['ML'], "Incorrect classification")
        
    def test_prob_prediction(value):
        value.assertGreater(rf.predict_class(mag, magerr)[1], 0.5, msg = "Incorrect probability prediction")
    
unittest.main()

