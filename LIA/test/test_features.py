# -*- coding: utf-8 -*-
"""
    Created on Fri Jan 13 14:46:19 2017
    @author: danielgodinez
"""
import numpy as np
from math import log
import unittest

import sys
sys.path.append('../')
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
    
