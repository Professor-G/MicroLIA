# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import numpy as np
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def create_models(all_feats, pca_feats, model='rf'):
    """Creates the Random Forest model and PCA transformation used for classification.
    
    Parameters
    ----------
    all_feats : str
        Name of text file containing all features and class label.
    pca_stats : str
        Name of text file containing PCA features and class label.  
    model : str
        Model to use for classification. 
        'rf': Random Forest 
        'nn': Neural Network

    Returns
    -------
    model : fn
        Trained random forest ensemble. 
    pca_model : fn
        PCA transformation.
    """
    coeffs = np.loadtxt(all_feats,usecols=np.arange(2,84))
    pca = decomposition.PCA(n_components=70, whiten=True, svd_solver='auto')
    pca.fit(coeffs)
    #feat_strengths = pca.explained_variance_ratio_
    training_set = np.loadtxt(pca_feats, dtype = str)
    #training_set = np.loadtxt(all_feats, dtype = str) #testing to see if using only feats works
    if model == 'rf':
        model = RandomForestClassifier(n_estimators=100)#, max_depth = 4, max_features=2, min_samples_leaf = 4, min_samples_split=2)
    elif model == 'nn':
        model = MLPClassifier(hidden_layer_sizes=(1000,), max_iter=5000, activation='relu', solver='adam', verbose=10, tol=1e-4, learning_rate_init=.0001)

    model.fit(training_set[:,np.arange(2,72)].astype(float),training_set[:,0])

    return model, pca