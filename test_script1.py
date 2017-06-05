# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:48:32 2017

@author: danielgodinez
"""

import stats_computation as sc
import random_forest_classifier as rf
import numpy as np
import unittest

data = np.loadtxt('TestML_1')

mjd = data[:,0]
mag = data[:,1]
magerr = data[:,2]

class Test(unittest.TestCase):
    def test_compute_statistics(value):
        value.assertEqual( sc.compute_statistics(mjd, mag, magerr), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -5.4354180599375495,
                          0.34793189299316019, 1.130441843074534, 31.401821018066236, 0.0097154255645506282, 6798930.9057225743,
                          1.0119460116981802, 0.0084235779847892222, 1.0, 15.048898825155755), "Statistics array 1 incorrect" )
         
    def test_predict_class(value):
        value.assertEqual( rf.predict_class(mjd, mag, magerr), ['ML'], "Incorrect classification 1")
    
unittest.main()

