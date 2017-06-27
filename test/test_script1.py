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
        value.assertEqual( sc.compute_statistics(mag, magerr), (0.70346613225989629, 27.905462817947186,
                          -5.0100150397808205, 1.080251522752754, 1423.3576640580793, 0.36144175147524155, 0.952513966480447, 
                          0.0086303068969277325, 0, 17.0, 0.0097154255645506282, 0.12967701108376908), "Statistics array incorrect" )
         
    def test_predict_class(value):
        value.assertEqual( rf.predict_class(mag, magerr)[0], ['ML'], "Incorrect classification")
        
    def test_prob_prediction(value):
        value.assertGreater(rf.predict_class(mag, magerr)[1], 0.75, msg = "Incorrect probability prediction")

    
unittest.main()

