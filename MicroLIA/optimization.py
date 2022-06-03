#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  11 12:04:23 2021

@author: daniel
"""
import sys 
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)
import numpy as np
import random

from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from boruta import BorutaPy

from skopt import BayesSearchCV, plots, gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer, Categorical 
from skopt.utils import use_named_args

import optuna
optuna.logging.set_verbosity(optuna.logging.WARN)

def objective_nn(trial):
    """
    The Optuna objective function for the scikit-learn implementatin of the
    MLP classifier. The Optuna software was for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    See software documentation: https://github.com/optuna/optuna-examples 
    """

    learning_rate_init= trial.suggest_float('learning_rate_init', .0001, 0.1)
    solver = trial.suggest_categorical("solver", ["sgd", "adam"]) #"lbfgs"
    activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
    learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
    alpha = trial.suggest_float("alpha", 0.00001, 0.001)
    batch_size = trial.suggest_int('batch_size', 1, 1000)
    
    n_layers = trial.suggest_int('hidden_layer_sizes', 1, 10)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 1, 2500))

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)

    clf = MLPClassifier(hidden_layer_sizes=tuple(layers),learning_rate_init=learning_rate_init, solver=solver, 
        activation=activation, alpha=alpha, batch_size=batch_size)
    clf.fit(x_train, y_train)

    return clf.score(x_test, y_test)

def objective_rf(trial):
    """
    The Optuna objective function for the scikit-learn implementatin of the
    Random Forest classifier. The Optuna software was for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    See software documentation: https://github.com/optuna/optuna-examples 
    """

    n_estimators = trial.suggest_int('n_estimators', 100, 3000)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 1, 100)
    min_samples_split = trial.suggest_float('min_samples_split', 0,1)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_int('max_features', 1, data_x.shape[1])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)

    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                max_features=max_features, bootstrap=bootstrap)
    
    clf.fit(x_train, y_train)

    return clf.score(x_test, y_test)

def hyper_opt(data_x, data_y, clf='rf', n_iter=25):
    """
    Optimizes hyperparameters using a k-fold cross validation splitting strategy.
    This function uses Bayesian Optimizattion and should only be used for
    optimizing the scikit-learn implementation of the Random Forest Classifier.

    Example:
        The function will create the classification engine and optimize the hyperparameters
        using an iterative approach:

        >>> model, params = hyper_opt(data_x, data_y, clf='rf') 
        
        The first output is our optimal classifier, and will be used to make predictions:
        
        >>> prediction = model.predict(new_data)
        
        The second output of the optimize function is the dictionary containing
        the hyperparameter combination that yielded the highest mean accuracy.
        
    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        clf (str): The machine learning classifier to optimize. Can either be
            'rf' for Random Forest or 'nn' for Neural Network. Defaults to 'rf'.
        n_iter (int, optional): The maximum number of iterations to perform during 
            the hyperparameter search. Defaults to 25.
        
    Returns:
        The first output is the Random Forest classifier with the optimal hyperparameters.
        Second output is a dictionary containing the optimal hyperparameters.
    """

    def logging_callback(study, trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            print("Highest accuracy: {} (Trial # {}). ".format(trial.value, trial.number))

    sampler = optuna.samplers.TPESampler(seed=10) 
    study = optuna.create_study(direction='maximize', sampler=sampler)

    if clf == 'rf':
        print('Beginning optimization procedure, this will take a while...')
        try:
            study.optimize(objective_rf, n_trials=n_iter, callbacks=[logging_callback])
            params = study.best_trial.params
            clf = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], 
                max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], 
                min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], 
                bootstrap=params['bootstrap'])
            return clf, params
        except:
            print('Failed to optimize with Optuna, switching over to BayesSearchCV...')
    elif clf == 'nn':
        try:
            study.optimize(objective_nn, n_trials=n_iter, callbacks=[logging_callback])
            params = study.best_trial.params
            layers = [param for param in params if 'n_units_' in param]
            layers = tuple(params[i] for i in layers)
            clf = MLPClassifier(hidden_layer_sizes=tuple(layers), learning_rate_init=params['learning_rate_init'], 
                activation=params['activation'], learning_rate=params['learning_rate'], alpha=params['alpha'], 
                batch_size=params['batch_size'], solver=params['solver'])
            return clf, params
        except:
            raise ValueError('Error occurred when optimizing neural network with Optuna.')
    else:
        raise ValueError('clf argument must either be "rf"" or "nn".')


    params = {
        'criterion': ["gini", "entropy"],
        'n_estimators': [int(x) for x in np.linspace(50,1000, num=20)], 
        'max_features': [len(data_x[0]), "sqrt", "log2"],
        'max_depth': [int(x) for x in np.linspace(5,50,num=5)],
        'min_samples_split': [3,4,6,7,8,9,10],
        'min_samples_leaf': [1,3,5,7,9,10],
        'max_leaf_nodes': [int(x) for x in np.linspace(2,200)],
        'bootstrap': [True,False]   
    }
        
    gs = BayesSearchCV(n_iter=n_iter, estimator=RandomForestClassifier(), search_spaces=params, 
        optimizer_kwargs={'base_estimator': 'RF'}, cv=10)
    print('Performing Bayesian optimization...')
    gs.fit(data_x, data_y)
    best_est = gs.best_estimator_
    best_score = np.round(gs.best_score_, 3)
    print(f"Highest mean accuracy: {best_score}")
    #plot_convergence(gs.optimizer_results_[0])
    #plot_objective(gs.optimizer_results_[0])
    #plots.plot_evaluations(gs.optimizer_results_[0])
    
    return gs.best_estimator_, gs.best_params_

def boruta_opt(data_x, data_y):
    """
    Applies the Boruta algorithm (Kursa & Rudnicki 2011) to identify features
    that perform worse than random chance.
    See: https://arxiv.org/pdf/1106.5112.pdf

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
            
    Returns:
        1D array containing the indices of the selected features. This can then
        be used to index the columns in the data_x array.
    """

    classifier = RandomForestClassifier()

    feat_selector = BorutaPy(classifier, n_estimators='auto', verbose=0, random_state=1)
    print('Running feature optimization...')
    feat_selector.fit(data_x, data_y)

    feat_selector.support = np.array([str(feat) for feat in feat_selector.support_])
    index = np.where(feat_selector.support == 'True')[0]
    print('Feature selection complete, {} selected out of {}! Calculating classification accuracy...'.format(len(index),len(feat_selector.support)))
    
    cv = cross_validate(classifier, data_x[:,index], data_y, cv=10)
    print('Accuracy without feature optimization: {}'.format(np.round(np.mean(cv['test_score']),3)))
    print('-------------------------------------')
    cv = cross_validate(classifier, data_x[:,index], data_y, cv=10)   
    print('Accuracy with feature optimization: {}'.format(np.round(np.mean(cv['test_score']),3)))
    
    return index

def Strawman_imputation(data):
    """
    Perform Strawman imputation, a time-efficient algorithm
    in which missing data values are replaced with the median
    value of the entire, non-NaN sample. If the data is a hot-encoded
    boolean (as the RF does not allow True or False), then the 
    instance that is used the most will be computed as the median. 

    This is the baseline algorithm used by (Tang & Ishwaran 2017).
    See: https://arxiv.org/pdf/1701.05305.pdf

    Note:
        This function assumes each row corresponds to one sample, and 
        that missing values are masked as either NaN or inf. 

    Args:
        data (ndarray): 1D array if single parameter is input. If
            data is 2-dimensional, the medians will be calculated
            using the non-missing values in each corresponding column.

    Returns:
        The data array with the missing values filled in. 
    """

    data[data>1e6] = 1e6
    data[(data>0) * (data<1e-6)] = 1e-6
    data_x[data_x<-1e6] = -1e6

    if np.all(np.isfinite(data)):
        print('No missing values in data, returning original array.')
        return data 

    if len(data.shape) == 1:
        mask = np.where(np.isfinite(data))[0]
        median = np.median(data[mask])
        data[np.isnan(data)] = median 

        return data

    Ny, Nx = data.shape
    imputed_data = np.zeros((Ny,Nx))

    for i in range(Nx):
        mask = np.where(np.isfinite(data[:,i]))[0]
        median = np.median(data[:,i][mask])

        for j in range(Ny):
            if np.isnan(data[j,i]) == True or np.isinf(data[j,i]) == True:
                imputed_data[j,i] = median
            else:
                imputed_data[j,i] = data[j,i]

    return imputed_data 

def KNN_imputation(data, imputer=None, k=3):
    """
    Performs k-Nearest Neighbor imputation and transformation.
    By default the imputer will be created and returned, unless
    the imputer argument is set, in which case only the transformed
    data is output. 

    As this bundles neighbors according to their eucledian distance,
    it is sensitive to outliers. Can also yield weak predictions if the
    training features are heaviliy correlated.
    
    Args:
        imputer (optional): A KNNImputer class instance, configured using sklearn.impute.KNNImputer.
            Defaults to None, in which case the transformation is created using
            the data itself. 
        data (ndarray): 1D array if single parameter is input. If
            data is 2-dimensional, the medians will be calculated
            using the non-missing values in each corresponding column.
        k (int, optional): If imputer is None, this is the number
            of nearest neighbors to consider when computing the imputation.
            Defaults to 3. If imputer argument is set, this variable is ignored.

    Note:
        Tang & Ishwaran 2017 reported that if there is low to medium
        correlation in the dataset, RF imputation algorithms perform 
        better than KNN imputation

    Example:
        If we have our training data in an array called training_set, we 
        can create the imputer so that we can call it to transform new data
        when making on-the-field predictions.

        >>> imputed_data, knn_imputer = KNN_imputation(data=training_set, imputer=None)
        
        Now we can use the imputed data to create our machine learning model.
        Afterwards, when new data is input for prediction, we will insert our 
        imputer into the pipelinen by calling this function again, but this time
        with the imputer argument set:

        >>> new_data = knn_imputation(new_data, imputer=knn_imputer)

    Returns:
        The first output is the data array with with the missing values filled in. 
        The second output is the KNN Imputer that should be used to transform
        new data, prior to predictions. 
    """

    data[data>1e6] = 1e6
    data[(data>0) * (data<1e-6)] = 1e-6
    data_x[data_x<-1e6] = -1e6

    if np.all(np.isfinite(data)) and imputer is None:
        raise ValueError('No missing values in training dataset, do not apply imputation algorithms!')

    if imputer is None:
        imputer = KNNImputer(n_neighbors=k)
        imputer.fit(data)
        imputed_data = imputer.transform(data)
        return imputed_data, imputer

    return imputer.transform(data) 

def MissForest_imputation(data, imputer=None):
    """
    Imputation algorithm created by Stekhoven and Buhlmann (2012).
    See: https://academic.oup.com/bioinformatics/article/28/1/112/219101

    By default the imputer will be created and returned, unless
    the imputer argument is set, in which case only the transformed
    data is output. 

    Note:
        The RF imputation procedures improve performance if the features are heavily correlated.
        Correlation is important for RF imputation, see: https://arxiv.org/pdf/1701.05305.pdf
    
    Args: 
        data (ndarray): 1D array if single parameter is input. If
            data is 2-dimensional, the medians will be calculated
            using the non-missing values in each corresponding column.
        imputer (optional): A MissForest class instance, configured using 
            the missingpy API. Defaults to None, in which case the transformation 
            is created using the data itself.

    Example:
        If we have our training data in an array called training_set, we 
        can create the imputer so that we can call it to transform new data
        when making on-the-field predictions.

        >>> imputed_data, rf_imputer = MissForest_imputation(data=training_set, imputer=None)
        
        Now we can use the imputed data to create our machine learning model.
        Afterwards, when new data is input for prediction, we will insert our 
        imputer into the pipelinen by calling this function again, but this time
        with the imputer argument set:

        >>> new_data = MissForest_imputer(new_data, imputer=rf_imputer)

    Returns:
        The first output is the data array with with the missing values filled in. 
        The second output is the Miss Forest Imputer that should be used to transform
        new data, prior to predictions. 
    """
    data[data>1e6] = 1e6
    data[(data>0) * (data<1e-6)] = 1e-6
    data_x[data_x<-1e6] = -1e6
    
    if np.all(np.isfinite(data)) and imputer is None:
        raise ValueError('No missing values in training dataset, do not apply imputation algorithms!')

    if imputer is None:
        imputer = MissForest(verbose=0)
        imputer.fit(data)
        imputed_data = imputer.transform(data)
        return imputed_data, imputer

    return imputer.transform(data) 

