# -*- coding: utf-8 -*-
"""
    Created on Fri Jan 13 14:46:19 2017
    @author: danielgodinez
"""
import unittest
import numpy as np
import pandas as pd 

import sys
sys.path.append('../../')
from MicroLIA import ensemble_model
from MicroLIA.optimization import impute_missing_values

from sklearn.model_selection import cross_validate

csv = pd.read_csv('MicroLIA_Training_Set.csv')

model = ensemble_model.Classifier(clf='xgb', csv_file=csv)
model.load('test_model')

test_lc = np.loadtxt('test_ogle_lc.dat')
time, mag, magerr = test_lc[:,0], test_lc[:,1], test_lc[:,2]

class Test(unittest.TestCase):
    """
    Unittest to ensure the classifier is working correctly. 
    """

    def test_predict(self):

        predicted_value = model.predict(d[:, 0], d[:, 1], d[:, 2], convert=True, zp=22, apply_weights=True)[:,1][3]
        expected_value = 0.96119

        self.assertAlmostEqual(predicted_value, expected_value, delta=0.01, msg="Classifier failed, probability prediction is not within the limits.")

    def test_cv_score(self):

        cv = cross_validate(model.model, model.data_x, model.data_y, cv=3) 
        accuracy = np.mean(cv['test_score'])
        expected_value = 0.95

        self.assertAlmostEqual(accuracy, expected_value, delta=0.025, msg='Classifier failed, the mean 10-fold cross-validation accuracy is not within the limits.')

    def test_base_rf_model(self):
        new_model = ensemble_model.Classifier(model.data_x, model.data_y, clf='rf', impute=True, optimize=False)
        new_model.create()

        cv = cross_validate(new_model.model, new_model.data_x, new_model.data_y, cv=3) 
        accuracy = np.mean(cv['test_score'])
        expected_value = 0.9

        self.assertAlmostEqual(accuracy, expected_value, delta=0.01, msg='Classifier failed, the mean 10-fold cross-validation accuracy is not within the limits.')

    def test_base_xgb_model(self):
        new_model = ensemble_model.Classifier(model.data_x, model.data_y, clf='xgb', impute=True, optimize=False)
        new_model.create()

        cv = cross_validate(new_model.model, new_model.data_x, new_model.data_y, cv=3) 
        accuracy = np.mean(cv['test_score'])
        expected_value = 0.9

        self.assertAlmostEqual(accuracy, expected_value, delta=0.01, msg='Classifier failed, the mean 10-fold cross-validation accuracy is not within the limits.')

    def test_base_nn_model(self):
        new_model = ensemble_model.Classifier(model.data_x, model.data_y, clf='nn', impute=True, optimize=False)
        new_model.create()

        cv = cross_validate(new_model.model, new_model.data_x, new_model.data_y, cv=3) 
        accuracy = np.mean(cv['test_score'])
        expected_value = 0.9

        self.assertAlmostEqual(accuracy, expected_value, delta=0.01, msg='Classifier failed, the mean 10-fold cross-validation accuracy is not within the limits.')

    def test_knn_imputer(self):
        model.data_x[[10,10]] = np.nan 
        model.data_x = impute_missing_values(model.data_x, imputer=None, strategy='knn', k=3)
        expected_value = 0.1
        self.assertAlmostEqual(model.data_x[[10,10]], expected_value, delta=0.01, msg='knn imputation failed!')

    def test_median_imputer(self):
        model.data_x[[10,10]] = np.nan 
        model.data_x = impute_missing_values(model.data_x, imputer=None, strategy='median')
        expected_value = 0.1
        self.assertAlmostEqual(model.data_x[[10,10]], expected_value, delta=0.01, msg='median imputation failed!')

    def test_mean_imputer(self):
        model.data_x[[10,10]] = np.nan 
        model.data_x = impute_missing_values(model.data_x, imputer=None, strategy='mean')
        expected_value = 0.1
        self.assertAlmostEqual(model.data_x[[10,10]], expected_value, delta=0.01, msg='mean imputation failed!')

    def test_mode_imputer(self):
        model.data_x[[10,10]] = np.nan 
        model.data_x = impute_missing_values(model.data_x, imputer=None, strategy='mode')
        expected_value = 0.1
        self.assertAlmostEqual(model.data_x[[10,10]], expected_value, delta=0.01, msg='mode imputation failed!')

    def test_constant_imputer(self):
        model.data_x[[10,10]] = np.nan 
        model.data_x = impute_missing_values(model.data_x, imputer=None, strategy='constant', constant_value=0)
        expected_value = 0.1
        self.assertAlmostEqual(model.data_x[[10,10]], expected_value, delta=0.01, msg='mode imputation failed!')




unittest.main()
    

