# -*- coding: utf-8 -*-
"""
    Created on Fri Jan 13 14:46:19 2017
    @author: danielgodinez
"""
import unittest
import numpy as np
import pandas as pd 
import pkg_resources

import sys
sys.path.append('../../')
from MicroLIA import ensemble_model
from MicroLIA.optimization import impute_missing_values

from sklearn.model_selection import cross_validate

resource_package = __name__
file = pkg_resources.resource_filename(resource_package, 'MicroLIA_Training_Set_OGLE_IV.csv')
csv = pd.read_csv(file)

resource_package = __name__
folder = pkg_resources.resource_filename(resource_package, 'test_model_xgb')

model = ensemble_model.Classifier(clf='xgb', csv_file=csv)
model.load(folder)

resource_package = __name__
file = pkg_resources.resource_filename(resource_package, 'test_ogle_lc.dat')

test_lc = np.loadtxt(file)
time, mag, magerr = test_lc[:,0], test_lc[:,1], test_lc[:,2]

class Test(unittest.TestCase):
    """Unittest to ensure the classifier is working correctly. 
    """

    def test_predict(self):
        value = model.predict(time, mag, magerr, convert=True, zp=22, apply_weights=True)[:,1][3]
        expected_value = 0.88243955
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg="Classifier failed, probability prediction is not within the limits.")

    def test_base_rf_model(self):
        new_model = ensemble_model.Classifier(model.data_x, model.data_y, clf='rf', impute=True, optimize=False)
        new_model.create(overwrite_training=True)

        cv = cross_validate(new_model.model, new_model.data_x, new_model.data_y, cv=3) 
        value = np.mean(cv['test_score'])
        expected_value = 0.9857501
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg='Classifier failed (rf), the mean 3-fold cross-validation accuracy is not within the limits.')

        value = new_model.predict(time, mag, magerr, convert=True, zp=22, apply_weights=True)[:,1][3]
        expected_value = 0.07
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg="Classifier failed (rf), probability prediction is not within the limits.")

    def test_base_xgb_model(self):
        new_model = ensemble_model.Classifier(model.data_x, model.data_y, clf='xgb', impute=True, optimize=False)
        new_model.create(overwrite_training=True)

        cv = cross_validate(new_model.model, new_model.data_x, new_model.data_y, cv=3) 
        value = np.mean(cv['test_score'])
        expected_value = 0.9892495
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg='Classifier failed (xgb), the mean 3-fold cross-validation accuracy is not within the limits.')

        value = new_model.predict(time, mag, magerr, convert=True, zp=22, apply_weights=True)[:,1][3]
        expected_value = 0.0013825
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg="Classifier failed (rf), probability prediction is not within the limits.")

    def test_base_nn_model(self):
        new_model = ensemble_model.Classifier(model.data_x, model.data_y, clf='nn', impute=True, optimize=False)
        new_model.create(overwrite_training=True)

        cv = cross_validate(new_model.model, new_model.data_x, new_model.data_y, cv=3) 
        value = np.mean(cv['test_score'])
        expected_value = 0.6969853
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg='Classifier failed (nn), the mean 3-fold cross-validation accuracy is not within the limits.')

        value = new_model.predict(time, mag, magerr, convert=True, zp=22, apply_weights=True)[:,1][3]
        expected_value = 0.9999999
        self.assertAlmostEqual(np.round(value), expected_value, delta=0.01, msg="Classifier failed (nn), probability prediction is not within the limits.")

    def test_knn_imputer(self):
        model.data_x[10,10] = np.nan 
        data_x, imputer = impute_missing_values(model.data_x, imputer=None, strategy='knn', k=3)
        
        value = data_x[10,10]
        expected_value = 5.8444108
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg='knn imputation failed!')

    def test_median_imputer(self):
        model.data_x[10,10] = np.nan 
        data_x, imputer= impute_missing_values(model.data_x, imputer=None, strategy='median')

        value = data_x[10,10]
        expected_value = 1.4765445
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg='median imputation failed!')

    def test_mean_imputer(self):
        model.data_x[10,10] = np.nan 
        data_x, imputer = impute_missing_values(model.data_x, imputer=None, strategy='mean')

        value = data_x[10,10]
        expected_value = 4.1045127
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg='mean imputation failed!')

    def test_mode_imputer(self):
        model.data_x[10,10] = np.nan 
        data_x, imputer = impute_missing_values(model.data_x, imputer=None, strategy='mode')

        value = data_x[10,10]
        expected_value = 0.03297212
        self.assertAlmostEqual(value, expected_value, delta=0.01, msg='mode imputation failed!')

    def test_constant_imputer(self):
        model.data_x[10,10] = np.nan 
        data_x, imputer = impute_missing_values(model.data_x, imputer=None, strategy='constant', constant_value=100)
        
        value = data_x[10,10]
        expected_value = 100
        self.assertEqual(value, expected_value, msg='constant imputation failed!')

unittest.main()
    

