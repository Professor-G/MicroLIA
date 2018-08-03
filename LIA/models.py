# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition

def create_models(all_feats, pca_feats):
    """Creates the Random Forest model and PCA transformation used for classificaition.
    
    Parameters
    ----------
    all_feats : name of txt_file
        Text file containing all features and class label.
    pca_stats : name of txt_file
        Text file containing PCA features and class label.  

    Returns
    -------
    rf_model : fn
        Trained random forest ensemble. 
    pca_model : fn
        PCA transformation.
    """
    coeffs = np.loadtxt(all_feats,usecols=np.arange(2,49))
    pca = decomposition.PCA(n_components=int(np.ceil(min(coeffs.shape)/2.))+2, whiten=True, svd_solver='auto')
    pca.fit(coeffs)
    #feat_strengths = pca.explained_variance_ratio_
    training_set = np.loadtxt(pca_feats, dtype = str)
    rf=RandomForestClassifier(n_estimators=1500, max_depth = 16, max_features=5, min_samples_split=16)
    rf.fit(training_set[:,np.arange(1,27)].astype(float),training_set[:,0])

    return rf, pca

