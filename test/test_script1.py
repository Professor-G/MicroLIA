# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:48:32 2017

@author: danielgodinez
"""
import sys
sys.path.append('../')
import stats_computation as sc
import random_forest_classifier as rf
import numpy as np
import unittest

data = np.loadtxt('TestML_1')

mag = data[:,1]
magerr = data[:,2]

class Test(unittest.TestCase):
    def test_compute_statistics(value):
        value.assertEqual( sc.compute_statistics(mag, magerr), (24.594547983246159, 31.900833388628115, -5.5455545901966845, 1.080251522752754, 834.21666444573793, 0.26063945940595257, 0.952513966480447, 0.0086303068969277325, 0, 0.0097154255645506282, 0.13189839781219653, 1.0119460116981802, 158.81274000968659), "Statistics array incorrect" )
         
    def test_predict_class(value):
        value.assertEqual( rf.predict_class(mag, magerr)[0], ['ML'], "Incorrect classification")
        
    def test_prob_prediction(value):
        value.assertGreater(rf.predict_class(mag, magerr)[1], 0.75, msg = "Incorrect probability prediction")

    
unittest.main()

