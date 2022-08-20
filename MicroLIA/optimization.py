#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  11 12:04:23 2021

@author: daniel
"""
import sys 
import random
import numpy as np
from pandas import DataFrame
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)
from collections import Counter 
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

from skopt import BayesSearchCV, plots, gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer, Categorical 
from skopt.utils import use_named_args

import optuna
from boruta import BorutaPy
from BorutaShap import BorutaShap
optuna.logging.set_verbosity(optuna.logging.WARNING)
from xgboost import XGBClassifier

class objective_xgb(object):
    """
    The Optuna objective function for the tree-based XGBoost classifier. 
    The Optuna software for hyperparameter optimization was published in 
    2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902
    """
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __call__(self, trial):
        booster = trial.suggest_categorical('booster', ['gbtree', 'dart'])
        reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 1)
        reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 1)
        max_depth = trial.suggest_int('max_depth', 2, 25)
        eta = trial.suggest_loguniform('eta', 1e-8, 1)
        gamma = trial.suggest_loguniform('gamma', 1e-8, 1)
        grow_policy = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

        if booster == "dart":
            sample_type = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            normalize_type = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            rate_drop = trial.suggest_loguniform('rate_drop', 1e-8, 1)
            skip_drop = trial.suggest_loguniform('skip_drop', 1e-8, 1)

            clf = XGBClassifier(booster=booster, reg_lambda=reg_lambda, reg_alpha=reg_alpha, max_depth=max_depth, eta=eta, 
                gamma=gamma, grow_policy=grow_policy, sample_type=sample_type, normalize_type=normalize_type,rate_drop=rate_drop, 
                skip_drop=skip_drop, tree_method='hist')
        
        elif booster == 'gbtree':
            subsample = trial.suggest_loguniform('subsample', 1e-6, 1.0)
            clf = XGBClassifier(booster=booster, reg_lambda=reg_lambda, reg_alpha=reg_alpha, max_depth=max_depth, eta=eta, 
                gamma=gamma, grow_policy=grow_policy, subsample=subsample, tree_method='hist')

        cv = cross_validate(clf, self.data_x, self.data_y, cv=3)
        final_score = np.round(np.mean(cv['test_score']), 6)

        return final_score
        
class objective_nn(object):
    """
    The Optuna objective function for the scikit-learn implementatin of the
    MLP classifier. The Optuna software for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    See Optuna examples: https://github.com/optuna/optuna-examples 
    """

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __call__(self, trial):
        learning_rate_init= trial.suggest_float('learning_rate_init', 1e-6, 1)
        solver = trial.suggest_categorical("solver", ["sgd", "adam"]) #"lbfgs"
        activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        alpha = trial.suggest_float("alpha", 1e-6, 1)
        batch_size = trial.suggest_int('batch_size', 1, 1000)
        
        n_layers = trial.suggest_int('hidden_layer_sizes', 1, 10)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_{i}', 100, 1000))

        clf = MLPClassifier(hidden_layer_sizes=tuple(layers),learning_rate_init=learning_rate_init, 
            solver=solver, activation=activation, alpha=alpha, batch_size=batch_size, max_iter=2500)
        
        cv = cross_validate(clf, self.data_x, self.data_y, cv=3)
        final_score = np.round(np.mean(cv['test_score']), 6)

        return final_score

class objective_rf(object):
    """
    The Optuna objective function for the scikit-learn implementatin of the
    Random Forest classifier. The Optuna software for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902
    """

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __call__(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 3000)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 2, 25)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
        max_features = trial.suggest_int('max_features', 1, self.data_x.shape[1])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                    max_features=max_features, bootstrap=bootstrap)
        
        cv = cross_validate(clf, self.data_x, self.data_y, cv=3)
        final_score = np.round(np.mean(cv['test_score']), 6)

        return final_score

def hyper_opt(data_x, data_y, clf='rf', n_iter=25, return_study=True, balance=True):
    """
    Optimizes hyperparameters using a k-fold cross validation splitting strategy.
    This function uses Bayesian Optimizattion and should only be used for
    optimizing the scikit-learn implementation of the Random Forest Classifier.
    
    Note:
        If save_study=True, the Optuna study object will be the third output. This
        object can be used for various analysis, including optimization visualization.

        See: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html

    Example:
        The function will create the classification engine and optimize the hyperparameters
        using an iterative approach:

        >>> model, params = hyper_opt(data_x, data_y, clf='rf') 
        
        The first output is our optimal classifier, and will be used to make predictions:
        
        >>> prediction = model.predict(new_data)
        
        The second output of the optimize function is the dictionary containing
        the hyperparameter combination that yielded the highest mean accuracy.

        If save_study = True, the Optuna study object will also be returned as the third output.
        This can be used to plot the optimization results, see: https://optuna.readthedocs.io/en/latest/tutorial/10_key_features/005_visualization.html#sphx-glr-tutorial-10-key-features-005-visualization-py

        >>> from optuna.visualization.matplotlib import plot_contour
        >>> 
        >>> model, params, study = hyper_opt(data_x, data_y, clf='rf', save_study=True) 
        >>> plot_contour(study)
        
    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        clf (str): The machine learning classifier to optimize. Can either be
            'rf' for Random Forest, 'nn' for Neural Network, or 'xgb' for eXtreme Gradient Boosting. 
            Defaults to 'rf'.
        n_iter (int, optional): The maximum number of iterations to perform during 
            the hyperparameter search. Defaults to 25.
        return_study (bool, optional): If True the Optuna study object will be returned. This
            can be used to review the method attributes, such as optimization plots. Defaults to False.
        balance (bool, optional): If True, a weights array will be calculated and used
            when fitting the classifier. This can improve classification when classes
            are imbalanced. This is only applied if the classification is a binary task. 
            Defaults to True.        
            
    Returns:
        The first output is the classifier with the optimal hyperparameters.
        Second output is a dictionary containing the optimal hyperparameters.
        If save_study=True, the Optuna study object will be the third output.
    """

    if clf == 'rf':
        model_0 = RandomForestClassifier()
    elif clf == 'nn':
        model_0 = MLPClassifier()
    elif clf == 'xgb':
        model_0 = XGBClassifier()
        if all(isinstance(val, (int, str)) for val in data_y):
            print('XGBoost classifier requires numerical class labels! Converting class labels as follows:')
            print('____________________________________')
            y = np.zeros(len(data_y))
            for i in range(len(np.unique(data_y))):
                print(str(np.unique(data_y)[i]).ljust(10)+'  ------------->     '+str(i))
                index = np.where(data_y == np.unique(data_y)[i])[0]
                y[index] = i
            data_y = y 
            print('------------------------------------')
    else:
        raise ValueError('clf argument must either be "rf", "nn", or "xgb".')

    cv = cross_validate(model_0, data_x, data_y, cv=3)
    initial_score = np.round(np.mean(cv['test_score']), 6)

    sampler = optuna.samplers.TPESampler(seed=1909) 
    study = optuna.create_study(direction='maximize', sampler=sampler)
    print('Starting hyperparameter optimization, this will take a while...')
    #If binary classification task, can deal with imbalance classes with weights hyperparameter
    if len(np.unique(data_y)) == 2:
        counter = Counter(data_y)
        if counter[np.unique(data_y)[0]] != counter[np.unique(data_y)[1]]:
            if balance:
                print('Unbalanced dataset detected, will train classifier with weights! To disable, set balance=False')
                if clf == 'xgb':
                    total_negative = len(np.where(data_y == counter.most_common(1)[0][0])[0])
                    total_positive = len(data_y) - total_negative
                    sample_weight = total_negative / total_positive
                elif clf == 'rf':
                    sample_weight = np.zeros(len(data_y))
                    for i,label in enumerate(np.unique(data_y)):
                        index = np.where(data_y == label)[0]
                        sample_weight[index] = len(index) 
                elif clf == 'nn':
                    print('Unbalanced dataset detected but MLPClassifier() does not support sample weights.')
            else:
                sample_weight = None
        else:
            sample_weight = None

    if clf == 'rf':
        try:
            objective = objective_rf(data_x, data_y)
            study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
            params = study.best_trial.params
            model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], 
                max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], 
                min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], 
                bootstrap=params['bootstrap'], class_weight=sample_weight)
        except:
            print('Failed to optimize with Optuna, switching over to BayesSearchCV...')
            params = {
                'criterion': ["gini", "entropy"],
                'n_estimators': [int(x) for x in np.linspace(50,1000, num=20)], 
                'max_features': [data_x.shape[1], "sqrt", "log2"],
                'max_depth': [int(x) for x in np.linspace(5,50,num=5)],
                'min_samples_split': [3,4,6,7,8,9,10],
                'min_samples_leaf': [1,3,5,7,9,10],
                'max_leaf_nodes': [int(x) for x in np.linspace(2,200)],
                'bootstrap': [True,False]   
            }
            gs = BayesSearchCV(n_iter=n_iter, estimator=RandomForestClassifier(), search_spaces=params, 
                optimizer_kwargs={'base_estimator': 'RF'}, cv=3)
            gs.fit(data_x, data_y)
            best_est, best_score = gs.best_estimator_, np.round(gs.best_score_, 4)
            print('Highest mean accuracy: {}'.format(best_score))
            return gs.best_estimator_, gs.best_params_

    elif clf == 'nn':
        try:
            objective = objective_nn(data_x, data_y)
            study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
            params = study.best_trial.params
            layers = [param for param in params if 'n_units_' in param]
            layers = tuple(params[layer] for layer in layers)
            model = MLPClassifier(hidden_layer_sizes=tuple(layers), learning_rate_init=params['learning_rate_init'], 
                activation=params['activation'], learning_rate=params['learning_rate'], alpha=params['alpha'], 
                batch_size=params['batch_size'], solver=params['solver'], max_iter=2500)
        except:
            print('Failed to optimize with Optuna, switching over to BayesSearchCV...')
            params = {
                'hidden_layer_sizes': [(100,),(50,100,50),(75,50,20),(150,100,50),(120,80,40),(100,50,30)],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'alpha': [0.00001, 0.5], 
                'activation': ['tanh', 'logistic', 'relu'],
                'solver': ['sgd', 'adam'],
                'max_iter': [100, 150, 200] 
            }
            gs = BayesSearchCV(n_iter=n_iter, estimator=MLPClassifier(), search_spaces=params, cv=3)
            gs.fit(data_x, data_y)
            best_est, best_score = gs.best_estimator_, np.round(gs.best_score_, 4)
            print('Highest mean accuracy: {}'.format(best_score))
            return gs.best_estimator_, gs.best_params_
          
    elif clf == 'xgb':
        objective = objective_xgb(data_x, data_y)
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
        params = study.best_trial.params
        if params['booster'] == 'dart':
            model = XGBClassifier(booster=params['booster'], reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], 
                max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], grow_policy=params['grow_policy'],
                sample_type=params['sample_type'], normalize_type=params['normalize_type'],rate_drop=params['rate_drop'], 
                skip_drop=params['skip_drop'], scale_pos_weight=sample_weight)
        elif params['booster'] == 'gbtree':
            model = XGBClassifier(booster=params['booster'], reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], 
                max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], grow_policy=params['grow_policy'],
                subsample=params['subsample'], scale_pos_weight=sample_weight)
       
    final_score = study.best_value

    if initial_score > final_score:
        print('Hyperparameter optimization complete! 3-fold CV accuracy of {} is lower than the base accuracy of {}, try increasing the value of n_iter and run again.'.format(np.round(final_score, 4), np.round(initial_score, 4)))
    else:
        print('Hyperparameter optimization complete! 3-fold CV accuracy of {} is higher than the base accuracy of {}.'.format(np.round(final_score, 4), np.round(initial_score, 4)))

    if return_study:
        return model, params, study
    return model, params

def borutashap_opt(data_x, data_y, model='rf', boruta_trials=50):
    """
    Applies a combination of the Boruta algorithm and
    Shapley values, a method developed by Eoghan Keany (2020).

    See: https://doi.org/10.5281/zenodo.4247618

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        model (str): The ensemble method to use when fitting and calculating
            the feature importance metric. Only two options are currently
            supported, 'rf' for Random Forest and 'xgb' for Extreme Gradient Boosting.
            Defaults to 'rf'.
        boruta_trials (int): The number of trials to run. A larger number is
            better as the distribution will be more robust to random fluctuations. 
            Defaults to 50.
    Returns:
        First output is a 1D array containing the indices of the selected features. 
        These indices can then be used to select the columns in the data_x array.
        Second output is the feature selection object, which contains feature selection
        history information and visualization options.
    """
    
    if boruta_trials == 0:
        return np.arange(data_x.shape[1])

    if boruta_trials < 20:
        print('Results are unstable if boruta_trials is too low!')
    if np.any(np.isnan(data_x)):
        print('NaN values detected, applying Strawman imputation...')
        data_x = Strawman_imputation(data_x)

    if model == 'rf':
        classifier = RandomForestClassifier()
    elif model == 'xgb':
        classifier = XGBClassifier()
    else:
        raise ValueError('Model argument must either be "rf" or "xgb".')
    
    try:
        #BorutaShap program requires input to have the columns attribute
        #Converting to Pandas dataframe
        cols = [str(i) for i in np.arange(data_x.shape[1])]
        X = DataFrame(data_x, columns=cols)
        y = np.zeros(len(data_y))

        #Below is to convert categorical labels to numerical, as per BorutaShap requirements
        for i, label in enumerate(np.unique(data_y)):
            mask = np.where(data_y == label)[0]
            y[mask] = i

        feat_selector = BorutaShap(model=classifier, importance_measure='shap', classification=True)
        print('Running feature selection...')
        feat_selector.fit(X=X, y=y, n_trials=boruta_trials, verbose=False, random_state=1909)

        index = np.array([int(feat) for feat in feat_selector.accepted])
        index.sort()
        print('Feature selection complete, {} selected out of {}!'.format(len(index), data_x.shape[1]))
    except:
        print('Boruta with Shapley values failed, switching to original Boruta algorithm...')
        index = boruta_opt(data_x, data_y)

    return index, feat_selector

def boruta_opt(data_x, data_y):
    """
    Applies the Boruta algorithm (Kursa & Rudnicki 2011) to identify features
    that perform worse than random.

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

    feat_selector = BorutaPy(classifier, n_estimators='auto', random_state=1909)
    print('Running feature selection...')
    feat_selector.fit(data_x, data_y)

    feats = np.array([str(feat) for feat in feat_selector.support_])
    index = np.where(feats == 'True')[0]
    print('Feature selection complete, {} selected out of {}!'.format(len(index),len(feat_selector.support)))
    
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

    if imputer is None:
        imputer = KNNImputer(n_neighbors=k)
        imputer.fit(data)
        imputed_data = imputer.transform(data)
        return imputed_data, imputer

    return imputer.transform(data) 

def MissForest_imputation(data):
    """
    !!! THIS ALGORITHM REFITS EVERY TIME, THEREFORE NOT HELPFUL
    FOR IMPUTING NEW, UNSEEN DATA. USE KNN_IMPUTATION INSTEAD !!!

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

    if np.all(np.isfinite(data)):
        raise ValueError('No missing values in training dataset, do not apply MissForest imputation!')
    
    imputer = MissForest(verbose=0)
    imputer.fit(data)
    imputed_data = imputer.transform(data)

    return imputer.transform(data) 

