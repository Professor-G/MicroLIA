# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import numpy as np
from warnings import warn
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def create_models(all_feats, pca_feats=None, model='rf'):
    """Creates the Random Forest model and PCA transformation used for classification.
    
    Parameters
    ----------
    all_feats : str
        Name of text file containing all features and class label.
    pca_feats : optional, str
        Name of text file containing PCA features and class label if
        you wish to apply PCA transformation. Default is None, which means
        the original features are used for classification. We recommend
        PCA transformation if using a Neural Network! 
    model : str
        Model to use for classification. 
        'rf': Random Forest 
        'nn': Neural Network

    Returns
    -------
    model : fn
        Trained classifier.
    pca_model : fn
        PCA transformation if pca_feats is not None.
    """
    coeffs = np.loadtxt(all_feats,dtype=str)[:,2:].astype(float)
    
    if model == 'rf':
        model = RandomForestClassifier(n_estimators=1000)#, max_depth = 4, max_features=2, min_samples_leaf = 4, min_samples_split=2)
    elif model == 'nn':
        print("Training NN...")
        model = MLPClassifier(hidden_layer_sizes=(1000,750,500,250,125), max_iter=5000, activation='relu', solver='adam', verbose=10, tol=1e-4, learning_rate_init=.0001)
<<<<<<< HEAD
=======

>>>>>>> 4e56799e39cb3591859b33bdf6c8b7c8ebdfe52b
        if pca_feats is None:
            warn('Neural network classifier may works best with PCA transformation!')
    else:
        raise ValueError("Model must be 'rf' or 'nn'")

    if pca_feats:
        training_set = np.loadtxt(pca_feats, dtype = str)
        pca = decomposition.PCA(n_components= training_set.shape[1]-2, whiten=True, svd_solver='auto')
        pca.fit(coeffs)
        model.fit(training_set[:,2:].astype(float),training_set[:,0])

        return model, pca
    else:
<<<<<<< HEAD

        training_set = np.loadtxt(all_feats, dtype = str)
        model.fit(training_set[:,2:].astype(float),training_set[:,0])
        return model 
=======

        training_set = np.loadtxt(all_feats, dtype = str)
        model.fit(training_set[:,2:].astype(float),training_set[:,0])
        return model 

    
>>>>>>> 4e56799e39cb3591859b33bdf6c8b7c8ebdfe52b
