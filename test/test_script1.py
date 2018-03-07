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

data = np.loadtxt('ML__0.42333333__268.32281086___-30.18382827_.txt')

hjd = data[:,0]
mag = data[:,1]
magerr = data[:,2]

class Test(unittest.TestCase):
    def test_compute_statistics(value):
        value.assertEqual( sc.compute_statistics(hjd, mag, magerr), (31.271665935767356, -1.7721541664549127, -0.4382519061248828, 0.16293295827678067, 499.02838314983762, 0.82943815920860375, 0.2, 0.0053153654680141126, 7, 0.094594999999999985, 0.10457274940901241, 0.34906999999999755, 0.06168402183645659), "Statistics array incorrect" )
         
    def test_predict_class(value):
        value.assertEqual( rf.predict_class(hjd, mag, magerr)[0], ['ML'], "Incorrect classification")
        

    
unittest.main()

