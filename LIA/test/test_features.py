# -*- coding: utf-8 -*-
"""
    Created on Fri Jan 13 14:46:19 2017
    @author: danielgodinez
"""
import numpy as np
from math import log
import unittest

import sys
sys.path.append('../lib')
from features import *
from extract_features import extract_all

data=np.loadtxt('ml_event.txt')
mag=data[:,1]
magerr=data[:,2]

flux = 10**(-(mag-24)/2.5)
flux_err = (magerr*flux)/(2.5*log(10))

norm_flux = flux/np.max(flux)
norm_fluxerr = flux_err*(norm_flux/flux)

class Test(unittest.TestCase):
    """
    Unittest to ensure all individual features including 
    feature extraction works. 
    """
    def test_above1(value):
        value.assertEqual( above1(norm_flux), 25, "Above 1 function failed")
    def test_above3(value):
        value.assertEqual( above3(norm_flux), 9, "Above 3 function failed")
    def test_above5(value):
        value.assertEqual( above5(norm_flux), 0, "Above 5 function failed")
    def test_abs_energy(value):
        value.assertEqual( abs_energy(norm_flux), 27.956794796425804, "Absolute Energy function failed")
    def test_abs_sum_changes(value):
        value.assertEqual( abs_sum_changes(norm_flux), 2.5587282410138945, "Absolute Sum Changes function failed")
    def test_amplitude(value):
        value.assertEqual( amplitude(flux), 165.98223230178843, "Amplitude function failed")
    def test_auto_corr(value):
        value.assertEqual( auto_corr(norm_flux), 0.974462638675433, "Auto Correlation function failed")
    def test_below1(value):
        value.assertEqual( below1(norm_flux), 0, "Below 1 function failed")
    def test_below3(value):
        value.assertEqual( below3(norm_flux), 0, "Below 3 function failed")    
    def test_below5(value):
        value.assertEqual( below5(norm_flux), 0, "Below 5 function failed")
    def test_c3(value):
        value.assertEqual( c3(norm_flux), 0.1348222168019175, "C3 function failed")
    def test_check_for_duplicate(value):
        value.assertEqual( check_for_duplicate(norm_flux), 0.0, "Check for duplicate function failed")
    def test_check_for_max_duplicate(value):
        value.assertEqual( check_for_max_duplicate(norm_flux), 0.0, "Check for max duplicated function failed")
    def test_check_for_min_duplicate(value):
        value.assertEqual( check_for_min_duplicate(norm_flux), 0.0, "Check for min duplicate function failed")
    def test_check_max_last_loc(value):
        value.assertEqual( check_max_last_loc(norm_flux), 0.16793893129770987, "Check max last loc function failed")
    def test_check_min_last_loc(value):
        value.assertEqual( check_min_last_loc(norm_flux), 0.9923664122137404, "Check min last loc function failed")
    def test_complexity(value):
        value.assertEqual( complexity(norm_flux), 2.8269963070564583, "Complexity function failed")
    def test_con(value):
        value.assertEqual( con(norm_flux), 0, "Con function failed")
    def test_con2(value):
        value.assertEqual( con2(norm_flux), 0, "Con2 function failed")   
    def test_count_above(value):
        value.assertEqual( count_above(norm_flux), 52, "Count above function failed")
    def test_count_below(value):
        value.assertEqual( count_below(norm_flux), 79, "Count below function failed")
    def test_first_loc_max(value):
        value.assertEqual( first_loc_max(norm_flux), 0.16030534351145037, "First loc max function failed")
    def test_first_loc_min(value):
        value.assertEqual( first_loc_min(norm_flux), 0.9847328244274809, "First loc min function failed")
    def test_integrate(value):
        value.assertEqual( integrate(norm_flux), 53.911696534216034, "Integrate function failed")
    def test_kurtosis(value):
        value.assertEqual( kurtosis(norm_flux), 6.279982429980464, "Kurtosis function failed")
    def test_longest_strike_above(value):
        value.assertEqual( longest_strike_above(norm_flux), 40, "Longest strike above function failed")
    def test_longest_strike_below(value):
        value.assertEqual( longest_strike_below(norm_flux), 75, "Longest strike below function failed")
    def test_mean_abs_change(value):
        value.assertEqual( mean_abs_change(norm_flux), 0.019682524930876112, "Mean absolute change function failed")
    def test_mean_change(value):
        value.assertEqual( mean_change(norm_flux), -0.0010180985141054354, "Mean change function failed")
    def test_mean_second_derivative(value):
        value.assertEqual( mean_second_derivative(norm_flux), 6.901739551149018e-05, "Mean second derivative function failed")
    def test_medianAbsDev(value):
        value.assertEqual( medianAbsDev(norm_flux), 0.07121593471836243, "Median absolute deviation function failed")
    def test_median_buffer_range(value):
        value.assertEqual( median_buffer_range(flux), 0.5801526717557252, "Median Buffer Range function failed")
    def test_median_buffer_range2(value):
        value.assertEqual( median_buffer_range2(flux), 0.0, "Median Buffer Range2 function failed")
    def test_peak_detection(value):
        value.assertEqual( peak_detection(norm_flux), 1, "Peak detection function failed")
    def test_ratio_recurring_points(value):
        value.assertEqual( ratio_recurring_points(norm_flux), 0.0, "Ratio of recurring points function failed")
    def test_root_mean_squared(value):
        value.assertEqual( root_mean_squared(norm_flux), 0.31997791191994446, "Root mean squared function failed")
    def test_sample_entropy(value):
        value.assertEqual( sample_entropy(norm_flux), 1.3731109467076321, "Sample entropy function failed")
    def test_shannon_entropy(value):
        value.assertEqual( shannon_entropy(norm_flux, norm_fluxerr), 1.0452976003780887, "Shannon entropy function failed")
    def test_skewness(value):
        value.assertEqual( skewness(norm_flux), 3.0170914134956104, "Skewness function failed")
    def test_std_over_mean(value):
        value.assertEqual( std_over_mean(norm_flux), 0.6408048944267444, "STD over mean function failed")
    def test_stetsonJ(value):
        value.assertEqual( stetsonJ(norm_flux, norm_fluxerr), 17167.09927270387, "stetsonJ function failed")
    def test_stetsonK(value):
        value.assertEqual( stetsonK(norm_flux, norm_fluxerr), 0.6858140506027742, "stetsonK function failed")
    def test_stetsonL(value):
        value.assertEqual( stetsonL(norm_flux, norm_fluxerr), 14753.68156555511, "stetsonL function failed")
    def test_sum_values(value):
        value.assertEqual( sum_values(norm_flux), 54.22954557484559, "Sum values function failed")
    def test_time_reversal_asymmetry(value):
        value.assertEqual( time_reversal_asymmetry(norm_flux), -0.000377481662001788, "Time reversal symmetry function failed")
    def test_vonNeumannRatio(value):
        value.assertEqual( vonNeumannRatio(norm_flux), 0.061476216308545044, "von Neumann Ratio function failed")
    def test_extract_all(value):
        arr1 = extract_all(mag,magerr)
        arr2 = [ 2.50000000e+01,  9.00000000e+00,  0.00000000e+00,  2.79567948e+01,
            2.55872824e+00,  1.65982232e+02,  9.74462639e-01,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  1.34822217e-01,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  1.67938931e-01,  9.92366412e-01,
            2.82699631e+00,  0.00000000e+00,  0.00000000e+00,  5.20000000e+01,
            7.90000000e+01,  1.60305344e-01,  9.84732824e-01,  5.39116965e+01,
            6.27998243e+00,  4.00000000e+01,  7.50000000e+01,  1.96825249e-02,
            -1.01809851e-03,  6.90173955e-05,  7.12159347e-02,  5.80152672e-01,
            0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  3.19977912e-01,
            1.37311095e+00,  1.04529760e+00,  3.01709141e+00,  2.05043412e-01,
            6.40804894e-01,  1.71670993e+04,  6.85814051e-01,  1.47536816e+04,
            5.42295456e+01, -3.77481662e-04,  6.14762163e-02]
        arr1 = np.round(arr1, 4)
        arr2 = np.round(arr2, 4)
        value.assertEqual( len(np.argwhere((arr1 == arr2) == True)), 47, "Extract all features function failed")

unittest.main()
    
