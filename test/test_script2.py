# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:20:25 2017

@author: danielgodinez
"""
import stats_computation as sc
import random_forest_classifier as rf
import numpy as np
import unittest

data = np.loadtxt('TestML_2')

mjd = data[:,0]
mag = data[:,1]
magerr = data[:,2]

class Test(unittest.TestCase):
    def test_compute_statistics(value):
        value.assertEqual( sc.compute_statistics(mjd, mag, magerr), (0.957619477006312, 1.0, 1.0, 0.04238052299368801, 0.0, 0.0, 
                 -4.116707163123203, 0.54847590773118993, 1.1497516942965478, 23.550972031666717, 0.039397067254302698, 
                  14788.588739719389, 1.1423033579932351, 0.0062298419394635051, 0.9684400360685302, 18.413412754867913), "Statistics array incorrect" )
         
    def test_predict_class(value):
        value.assertEqual( rf.predict_class(mjd, mag, magerr), ['ML'], "Incorrect classification")
    
unittest.main()
