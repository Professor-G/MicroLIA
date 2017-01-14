# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:46:19 2017

@author: danielgodinez
"""

import stats_computation as sc
import numpy as np
import unittest

mag = np.array([18, 18.3, 18.1, 18, 18.4, 18.9, 19.2, 19.3, 19.5, 19.2, 18.8, 18.3, 18.6])
magerr = np.array([0.01, 0.01, 0.03, 0.09, 0.04, 0.1, 0.03, 0.13, 0.04, 0.06, 0.09, 0.1, 0.35])
mjd = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

class Test(unittest.TestCase):
    def test_rms(value):
        value.assertEqual( sc.RMS(mag), 18.661538461538463, "RMS function failed" )

    def test_mean(value):
        value.assertEqual( sc.meanMag(mag), 18.661538461538463, "Mean function failed" )

    def test_median(value):
        value.assertEqual( sc.medianMag(mag), 18.600000000000001, "Median function failed" )
    
    def test_min(value):
        value.assertEqual( sc.minMag(mag), 18.0, "Min function failed" )

    def test_max(value):
        value.assertEqual( sc.maxMag(mag), 19.5, "Max function failed" )
    
    def test_medianAbsDev(value):
        value.assertEqual( sc.medianAbsDev(mag), 0.5, "MedAbsDev function failed" )
    
    def test_kurtosis(value):
        value.assertEqual( sc.kurtosis(mjd, mag), -1.0149598629254664, "Kurtosis function failed" )
    
    def test_skewness(value):
        value.assertEqual( sc.skewness(mag), 0.1868991393928264, "Skewness function failed" )
        
    def test_stetsonJ(value):
        value.assertEqual( sc.stetsonJ(mjd, mag, magerr), 159412.78061393721, "stetsonJ function failed" )

    def test_stetsonK(value):
        value.assertEqual( sc.stetsonK(mjd, mag, magerr), 0.64699834923516031, "stetsonK function failed" )

    def test_vonNeumannRatio(value):
        value.assertEqual( sc.vonNeumannRatio(mag), 0.38896680691912117, "vonNeumannRatio function failed" )
    
    def test_above1(value):
        value.assertEqual( sc.above1(mag), 0.3076923076923077, "Above 1 incorrect" )
    
    def test_above3(value):
        value.assertEqual( sc.above3(mag), 0.0, "Above 3 incorrect" )
    
    def test_above5(value):
        value.assertEqual( sc.above5(mag), 0.0, "Above 5 incorrect" )
    
    def test_below1(value):
        value.assertEqual( sc.below1(mag), 0.6923076923076923, "Below 1 incorrect" )

    def test_below3(value):
        value.assertEqual( sc.below3(mag), 1.0, "Below 3 incorrect" )

    def test_below5(value):
        value.assertEqual( sc.below5(mag), 1.0, "Below 5 incorrect" )

    def test_compute_statistics(value):
        value.assertEqual( sc.compute_statistics(mjd, mag, magerr), [18.661538461538463, 18.600000000000001, 18.661538461538463, 19.5, 18.0, 0.5, -1.0149598629254664, 0.1868991393928264, 159412.78061393721, 0.64699834923516031, 0.38896680691912117, 0.3076923076923077, 0.0, 0.0, 0.6923076923076923, 1.0, 1.0], "Statistics array incorrect" )
    
unittest.main()
