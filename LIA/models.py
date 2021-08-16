# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
import unittest
from LIA import microlensing_classifier

data=np.loadtxt('./test/ml_event.txt')
mag=data[:,1]
magerr=data[:,2]

def create_models(all_feats, pca_feats):
    """Creates the Random Forest model and PCA transformation used for classification.
    
    Parameters
    ----------
    all_feats : str
        Name of text file containing all features and class label.
    pca_stats : str
        Name of text file containing PCA features and class label.  

    Returns
    -------
    rf_model : fn
        Trained random forest ensemble. 
    pca_model : fn
        PCA transformation.
    """
    coeffs = np.loadtxt(all_feats,usecols=np.arange(2,49))
    pca = decomposition.PCA(n_components=44, whiten=True, svd_solver='auto')
    pca.fit(coeffs)
    #feat_strengths = pca.explained_variance_ratio_
    training_set = np.loadtxt(pca_feats, dtype = str,usecols=np.arange(0,49))
    rf=RandomForestClassifier(n_estimators=1000, max_depth = 4, max_features=2, min_samples_leaf = 4, min_samples_split=2)
    #rf.fit(training_set[:,np.arange(2,46)].astype(float),training_set[:,0])
    rf.fit(coeffs,training_set[:,0])
    #import pdb; pdb.set_trace()
    return rf, pca

rf,pca = create_models('./test/all_features.txt', './test/pca_features.txt')

class Test(unittest.TestCase):
    """
    Unittest to ensure the classifier is working correctly. 
    """
    def test_predict(value):
        value.assertEqual( microlensing_classifier.predict(mag,magerr,rf,pca)[0], 'ML', "Classifier failed, predicted class is not correct.")
    def test_probability_prediction(value):
        pred = microlensing_classifier.predict(mag,magerr, rf,pca)[1]
        value.assertTrue(pred >= 0.4 and pred <= 0.6, "Classifier failed, probability prediction not within range.")

unittest.main()
    