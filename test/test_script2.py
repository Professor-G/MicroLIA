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

data = np.loadtxt('ML__0.55__268.33051201___-30.21089307_.txt')

hjd = data[:,0]
mag = data[:,1]
magerr = data[:,2]

class Test(unittest.TestCase):
    def test_compute_statistics(value):
        value.assertEqual( sc.compute_statistics(mag, magerr), (10.639980200569692, -1.3951623050335764, -0.88075925533940425, 0.0293252276553836, 4191.4315058255061, 0.58676955737970438, 0.2366412213740458, 0.031868709124813495, 50, 0.26666000000000167, 0.76975206077278036, 1.5553899999999992, 0.059940895316137464), "Statistics array incorrect" )
         
    def test_predict_class(value):
        value.assertEqual( rf.predict_class(mag, magerr)[0], ['ML'], "Incorrect classification")
        
unittest.main()

