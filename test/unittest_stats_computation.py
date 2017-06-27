# -*- coding: utf-8 -*-
"""
    Created on Fri Jan 13 14:46:19 2017
    @author: danielgodinez
"""
import sys
sys.path.append('../')
import stats_computation as sc
import numpy as np
import unittest

mag = np.array([18, 18.3, 18.1, 18, 18.4, 18.9, 19.2, 19.3, 19.5, 19.2, 18.8, 18.3, 18.6])
magerr = np.array([0.01, 0.01, 0.03, 0.09, 0.04, 0.1, 0.03, 0.13, 0.04, 0.06, 0.09, 0.1, 0.35])

class Test(unittest.TestCase):
    def test_shannon_entropy(value):
        value.assertEqual( sc.shannon_entropy(mag, magerr), 2.9615690385194195, "Shannon Entropy function failed" )
    
    def test_auto_correlation(value):
        value.assertEqual( sc.auto_correlation(mag, magerr), 6.1866744592471097, "Auto Correlation function failed" )
    
    def test_con(value):
        value.assertEqual( sc.con(mag, magerr), 0.0, "Con function failed" )
    
    def test_con2(value):
        value.assertEqual( sc.con2(mag, magerr), 0.0, "Con2 function failed" )
    
    def test_kurtosis(value):
        value.assertEqual( sc.kurtosis(mag, magerr), 35.73018979880348, "Kurtosis function failed" )
    
    def test_kurtosis2(value):
        value.assertEqual( sc.kurtosis2(mag, magerr), 287.62555238692636, "Kurtosis2 function failed" )
    
    def test_skewness(value):
        value.assertEqual( sc.skewness(mag, magerr), 7.7924269178998333, "Skewness function failed" )
        
    def test_vonNeumannRatio(value):
        value.assertEqual( sc.vonNeumannRatio(mag, magerr), 0.77789008863456521, "vonNeumannRatio function failed" )
    
    def test_stetsonJ(value):
        value.assertEqual( sc.stetsonJ(mag, magerr), 73.105694334803772, "stetsonJ function failed" )
    
    def test_stetsonK(value):
        value.assertEqual( sc.stetsonK(mag, magerr), 0.70898031447518206, "stetsonK function failed" )
    
    def test_median_buffer_range(value):
        value.assertEqual( sc.median_buffer_range(mag, magerr), 0.23076923076923078, "Median Buffer Range function failed")
    
    def test_median_buffer_range(value):
        value.assertEqual( sc.median_buffer_range2(mag, magerr), 0.0, "Median Buffer Range2 function failed")
    
    def test_std_over_mean(value):
        value.assertEqual( sc.std_over_mean(mag, magerr), 0.020281331511023799, "STDoverMean function failed")
    
    def test_amplitude(value):
        value.assertEqual( sc.amplitude(mag, magerr), 1.3000000000000007, "Amplitude function failed")
    
    def test_above1(value):
        value.assertEqual( sc.above1(mag, magerr), 0.0, "Above 1 incorrect" )
    
    def test_above3(value):
        value.assertEqual( sc.above3(mag, magerr), 0.0, "Above 3 incorrect" )
    
    def test_above5(value):
        value.assertEqual( sc.above5(mag, magerr), 0.0, "Above 5 incorrect" )
    
    def test_below1(value):
        value.assertEqual( sc.below1(mag, magerr), 6.0, "Below 1 incorrect" )
    
    def test_below3(value):
        value.assertEqual( sc.below3(mag, magerr), 1.0, "Below 3 incorrect" )
    
    def test_below5(value):
        value.assertEqual( sc.below5(mag, magerr), 0.0, "Below 5 incorrect" )
    
    def test_median_absolute_deviation(value):
        value.assertEqual( sc.medianAbsDev(mag, magerr), 0.5, "Median Absolute Deviation function failed")
    
    def test_RootMS(value):
        value.assertEqual( sc.RootMS(mag, magerr), 0.35577398610092409, "RMS function failed")
    
    def test_meanMag(value):
        value.assertEqual( sc.meanMag(mag, magerr), 18.258234425049459, "Mean function failed")
    
    def test_deviation(value):
        value.assertEqual( sc.deviation(mag, magerr), 0.3703013051804151, "Standard Deviation function failed")
    
    def test_compute_statistics(value):
        value.assertEqual( sc.compute_statistics(mag, magerr), (6.1866744592471097, 35.73018979880348, 7.7924269178998333, 0.77789008863456521, 73.105694334803772, 0.70898031447518206, 0.23076923076923078, 0.020281331511023799, 6.0, 0.0, 0.5, 0.35577398610092409), "Compute Statistics function failed")

unittest.main()
    
