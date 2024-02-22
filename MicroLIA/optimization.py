#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  25 10:39:23 2023

@author: daniel
"""
import os, sys
os.environ['PYTHONHASHSEED'], os.environ["TF_DETERMINISTIC_OPS"] = '0', '1'

import joblib   
from pandas import DataFrame
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)
from collections import Counter 
import numpy as np
import random as python_random
np.random.seed(1909), python_random.seed(1909)

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate

import optuna
from boruta import BorutaPy
from BorutaShap import BorutaShap
from skopt import BayesSearchCV
from xgboost import XGBClassifier, DMatrix, train
optuna.logging.set_verbosity(optuna.logging.WARNING)


class objective_xgb(object):
    """
    Optimization objective function for the tree-based XGBoost classifier. 
    The Optuna software for hyperparameter optimization was published in 
    2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902
    
    Note:
        If opt_cv is between 0 and 1, a pruning procedure will be initiliazed (a procedure incompatible with cross-validation),
        so as to speed up the XGB optimization. A random validation data will be generated according to this ratio,
        which will replace the cross-validation method used by default. It will prune according
        to the f1-score of the validation data, which would be 10% of the training data if opt_cv=0.1, for example. 
        Need more testing to make this more cross-validation routine more robust. Recommend to set opt_cv > 1. 
        
    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels. 
        limit_search (bool): If True, the search space for the parameters will be expanded,
            as there are some hyperparameters that can range from 0 to inf. Defaults to False.
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. If this is
            between 0 and 1, this value would be used as the ratio of the validation to training data.
            Defaults to 3. Can be set to None in which case 5-fold cross-validation is employed. Cannot be 1. 
        eval_metric (str): The evaluation metric when evaluating the validation data, used when
            opt_cv is less than 1. Defaults to "f1". For all options see eval_metric from: https://xgboost.readthedocs.io/en/latest/parameter.html#metrics
        min_gamma (float): Controls the optimization of the gamma Tree Booster hyperparameter. The gamma parameter is the 
            lowest loss reduction needed on a tree leaf node in order to partition again. The algorithm's level of conservatism 
            increases with gamma, therefore it acts as a regularizer. By default, during the optimization routine will consider a miminum value 
            of 0 when tuning the gamma parameter, unless this min_gamma input is set. This parameter determines the lowest gamma value the 
            optimizer should consider. Must be less than 5. Defaults to 0. Consider increasing this to ~1 if the optimized models
            are overfitting.

    Returns:
        The cross-validation accuracy (if opt_cv is greater than 1, 1 would be single instance accuracy)
        or, if opt_cv is between 0 and 1, the validation accuracy according to the corresponding test size.
    """

    def __init__(self, data_x, data_y, limit_search=False, opt_cv=3, eval_metric="f1", min_gamma=0):
        self.data_x = data_x
        self.data_y = data_y
        self.limit_search = limit_search
        self.opt_cv = opt_cv 
        self.eval_metric = eval_metric 
        self.min_gamma = min_gamma

        if self.min_gamma >= 5:
            raise ValueError('The min_gamma parameter must be less than 5!')

        if self.opt_cv is None:
            self.opt_cv = 5 # This is the default behavior of cross_validate, setting here to avoid NoneType errors

    def __call__(self, trial):

        params = {"objective": "binary:logistic", "eval_metric": self.eval_metric} 
    
        if self.opt_cv < 1:
            train_x, valid_x, train_y, valid_y = train_test_split(self.data_x, self.data_y, test_size=self.opt_cv, random_state=190977)#np.random.randint(1, 1e9))
            dtrain, dvalid = DMatrix(train_x, label=train_y), DMatrix(valid_x, label=valid_y)
            #print('Initializing XGBoost Pruner...')
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-" + self.eval_metric)

        if self.limit_search:
            params['n_estimators'] = trial.suggest_int('n_estimators', 100, 500)
            params['booster'] = trial.suggest_categorical('booster', ['gbtree', 'dart'])
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 2.0)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 2.0)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
            params['eta'] = trial.suggest_float('eta', 0.01, 0.3)
            params['gamma'] = trial.suggest_float('gamma', float(self.min_gamma), 5.0)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

            if params['booster'] == "dart":
                params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                params['rate_drop'] = trial.suggest_float('rate_drop', 0.1, 0.5)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 0.5)
                if self.opt_cv >= 1:
                    clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                        reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                        grow_policy=params['grow_policy'], sample_type=params['sample_type'], normalize_type=params['normalize_type'],
                        rate_drop=params['rate_drop'], skip_drop=params['skip_drop'], random_state=190977)#, tree_method='hist')
            
            elif params['booster'] == 'gbtree':
                params['subsample'] = trial.suggest_loguniform('subsample', 1e-6, 1.0)
                if self.opt_cv >= 1:
                    clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                        reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                        grow_policy=params['grow_policy'], subsample=params['subsample'], random_state=190977)#, tree_method='hist')

            if self.opt_cv < 1:
                bst = train(params, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
                preds = bst.predict(dvalid)
                pred_labels = np.rint(preds)
                accuracy = accuracy_score(valid_y, pred_labels)
            else:
                #FROM SKLEARN DOCUMENTATION: For int/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, Fold is used.
                cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv) 
                accuracy = np.mean(cv['test_score'])

            return accuracy

        params['booster'] = trial.suggest_categorical('booster', ['gbtree', 'dart'])
        params['n_estimators'] = trial.suggest_int('n_estimators', 100, 500)
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 2.0)
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 2.0)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        params['eta'] = trial.suggest_float('eta', 0.01, 0.3)
        params['gamma'] = trial.suggest_float('gamma', float(self.min_gamma), 5.0)
        params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
        params['max_delta_step'] = trial.suggest_int('max_delta_step', 1, 10)
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)

        if params['booster'] == "dart":
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 0.1, 0.5)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 0.5)
            if self.opt_cv >= 1:
                clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'], 
                    reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], 
                    gamma=params['gamma'], grow_policy=params['grow_policy'], min_child_weight=params['min_child_weight'], 
                    max_delta_step=params['max_delta_step'], subsample=params['subsample'], sample_type=params['sample_type'], 
                    normalize_type=params['normalize_type'], rate_drop=params['rate_drop'], skip_drop=params['skip_drop'], random_state=190977)#, tree_method='hist')
        elif params['booster'] == 'gbtree':
            if self.opt_cv >= 1:
                clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'],  reg_lambda=params['reg_lambda'], 
                    reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], grow_policy=params['grow_policy'], 
                    min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'], subsample=params['subsample'], random_state=190977)#, tree_method='hist')
            
        if self.opt_cv < 1:
            bst = train(params, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
            preds = bst.predict(dvalid)
            pred_labels = np.rint(preds)
            accuracy = accuracy_score(valid_y, pred_labels)
        else:
            cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
            accuracy = np.mean(cv['test_score'])
        
        return accuracy

class objective_nn(object):
    """
    Optimization objective function for the scikit-learn implementatin of the
    MLP classifier. The Optuna software for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902.

    The total number of hidden layers to test is limited to 10, with 10-100 possible 
    number of neurons in each.

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels. 
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. 

    Returns:
        The performance metric, determined using the cross-fold validation method.
    """

    def __init__(self, data_x, data_y, opt_cv):
        self.data_x = data_x
        self.data_y = data_y
        self.opt_cv = opt_cv

    def __call__(self, trial):

        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 0.3, step=1e-3)
        solver = trial.suggest_categorical("solver", ["sgd", "adam"]) #"lbfgs"
        activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        alpha = trial.suggest_float("alpha", 1e-7, 1, step=1e-3)
        n_layers = trial.suggest_int('hidden_layer_sizes', 1, 10)
        #batch_size = trial.suggest_int('batch_size', 1, 1000)

        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_{i}', 10, 100))

        try:
            clf = MLPClassifier(hidden_layer_sizes=tuple(layers),learning_rate_init=learning_rate_init, 
                solver=solver, activation=activation, alpha=alpha, batch_size='auto', max_iter=200, random_state=1909)
        except:
            print("Invalid hyperparameter combination, skipping trial")
            return 0.0

        cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
        final_score = np.mean(cv['test_score'])

        return final_score

class objective_rf(object):
    """
    Optimization objective function for the scikit-learn implementatin of the
    Random Forest classifier. The Optuna software for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels. 
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. 

    Returns:
        The performance metric, determined using the cross-fold validation method.
    """
    
    def __init__(self, data_x, data_y, opt_cv):

        self.data_x = data_x
        self.data_y = data_y
        self.opt_cv = opt_cv

    def __call__(self, trial):

        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 2, 25)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
        max_features = trial.suggest_int('max_features', 1, self.data_x.shape[1])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        try:
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                max_features=max_features, bootstrap=bootstrap, random_state=1909)
        except:
            print("Invalid hyperparameter combination, skipping trial")
            return 0.0

        cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
        final_score = np.mean(cv['test_score'])

        return final_score

class ObjectiveOneClassSVM(object):
    """
    Optimization objective function for the scikit-learn implementation of the
    One-Class SVM. The Optuna software for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the number of samples,
            and m is the number of features.
        data_y (ndarray, str): 1D array containing the corresponding labels.
        opt_cv (int): Cross-validations to perform when assessing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross-validation accuracy.

    Returns:
        The performance metric, determined using the cross-fold validation method.
    """

    def __init__(self, data_x, data_y, opt_cv):
        self.data_x = data_x
        self.data_y = data_y
        self.opt_cv = opt_cv

    def __call__(self, trial):
        """
        Define the optimization objective function for the One-Class SVM.

        Args:
            trial (optuna.trial.Trial): A Trial object containing the current state of the optimization.

        Returns:
            float: The performance metric, determined using cross-validation.
        """

        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
        degree = trial.suggest_int('degree', 0, 10)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        coef0 = trial.suggest_float('coef0', 0.0, 1.0)
        tol = trial.suggest_float('tol', 1e-6, 1e-2, log=True)
        nu = trial.suggest_float('nu', 0.1, 1.0)
        shrinking = trial.suggest_categorical('shrinking', [True, False])
        cache_size = trial.suggest_float('cache_size', 100, 1000)

        try:
            clf = OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol,
                nu=nu, shrinking=shrinking, cache_size=cache_size)
        except:
            print("Invalid hyperparameter combination, skipping trial")
            return 0.0

        cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
        final_score = np.mean(cv['test_score'])

        return final_score


def hyper_opt(data_x=None, data_y=None, clf='rf', n_iter=25, opt_cv=None, balance=True, limit_search=True, min_gamma=0, return_study=True): 
    """
    Optimizes hyperparameters using a k-fold cross validation splitting strategy.
    If save_study=True, the Optuna study object will be the third output. This
    object can be used for various analysis, including optimization visualizations.
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
        data_x (ndarray): 2D array of size (n x m), where n is the number of samples, and m the number of features. 
        data_y (ndarray, str): 1D array containing the corresponing labels. 
        clf (str): The machine learning classifier to optimize. Can either be 'rf' for Random Forest, 'nn' for Neural Network, 
            or 'xgb' for eXtreme Gradient Boosting.
        n_iter (int, optional): The maximum number of iterations to perform during the hyperparameter search. Defaults to 25.
        opt_cv (int): Cross-validations to perform when assesing the performance at each hyperparameter optimization trial. 
            For example, if cv=3, then each optimization trial will be assessed according to the 3-fold cross validation accuracy. 
            If clf='xgb' and this value is set between 0 and 1, this sets the size of the validation data, which will be chosen 
            randomly each trial. This is used to enable an early stopping callback which is not possible with the cross-validation method. 
            Defaults to 10. 
        balance (bool, optional): If True, a weights array will be calculated and used when fitting the classifier. This can improve 
            classification when classes are imbalanced. This is only applied if the classification is a binary task. Defaults to True. 
        limit_search (bool): If True the optimization search spaces will be limited, for quicker computation. Defaults to True.
        min_gamma (float): Controls the optimization of the gamma Tree Booster hyperparameter, applicable if clf='xgb'. The gamma parameter is the 
            lowest loss reduction needed on a tree leaf node in order to partition again. The algorithm's level of conservatism 
            increases with gamma, therefore it acts as a regularizer. By default, during the optimization routine will consider a miminum value 
            of 0 when tuning the gamma parameter, unless this min_gamma input is set. This parameter determines the lowest gamma value the 
            optimizer should consider. Must be less than 5. Defaults to 0. Consider increasing this to ~1 if the optimized models
            are overfitting.
        return_study (bool, optional): If True the Optuna study object will be returned. This can be used to review the method attributes, 
            such as optimization plots. Defaults to True.
        
    Returns:
        The first output is the classifier with the optimal hyperparameters.
        Second output is a dictionary containing the optimal hyperparameters.
        If save_study=True, the Optuna study object will be the third output.
    """

    if clf == 'rf':
        model_0 = RandomForestClassifier(random_state=1909)
    elif clf == 'nn':
        model_0 = MLPClassifier(random_state=1909)
    elif clf == 'xgb':
        model_0 = XGBClassifier(random_state=1909)
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
        raise ValueError('clf argument must either be "rf", "xgb", or "nn".')

    if n_iter == 0:
        if clf == 'rf' or clf == 'xgb' or clf == 'nn':
            print('No optimization trials configured (n_iter=0), returning base {} model...'.format(clf))
            return model_0 
        else:
            raise ValueError('No optimization trials configured, set n_iter > 0!')

    if clf == 'rf' or clf == 'xgb' or clf == 'nn':
        cv = cross_validate(model_0, data_x, data_y, cv=opt_cv)
        initial_score = np.mean(cv['test_score'])

    sampler = optuna.samplers.TPESampler(seed=1909)
    study = optuna.create_study(direction='maximize', sampler=sampler)#, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
    print('Starting hyperparameter optimization, this will take a while...')

    #If binary classification task, can deal with imbalance classes with weights hyperparameter
    if len(np.unique(data_y)) == 2:
        if clf == 'rf' or clf == 'xgb' or clf == 'nn':
            counter = Counter(data_y)
            if counter[np.unique(data_y)[0]] != counter[np.unique(data_y)[1]]:
                if balance:
                    print('Unbalanced dataset detected, will train classifier with weights! To disable, set balance=False')
                    if clf == 'xgb':
                        total_negative = len(np.where(data_y == counter.most_common(1)[0][0])[0])
                        total_positive = len(data_y) - total_negative
                        sample_weight = total_negative / total_positive
                    elif clf == 'rf':
                        sample_weight = 'balanced'
                    elif clf == 'nn':
                        print('WARNING: MLPClassifier() does not support sample weights.')
                else:
                    sample_weight = None
            else:
                sample_weight = None
        else:
            print('Unbalanced dataset detected but the selected clf does not support weights.')
    else:
        sample_weight = None

    if clf == 'rf':
        try:
            objective = objective_rf(data_x, data_y, opt_cv=opt_cv)
            study.optimize(objective, n_trials=n_iter, show_progress_bar=True)#, gc_after_trial=True)
            params = study.best_trial.params
            model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], 
                max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], 
                min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], 
                bootstrap=params['bootstrap'], class_weight=sample_weight, random_state=1909)
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
            gs = BayesSearchCV(n_iter=n_iter, estimator=model_0, search_spaces=params, 
                optimizer_kwargs={'base_estimator': 'RF'}, cv=opt_cv)
            gs.fit(data_x, data_y)
            best_est, best_score = gs.best_estimator_, np.round(gs.best_score_, 4)
            print('Highest mean accuracy: {}'.format(best_score))
            return gs.best_estimator_, gs.best_params_

    elif clf == 'nn':
        try:
            objective = objective_nn(data_x, data_y, opt_cv=opt_cv)
            study.optimize(objective, n_trials=n_iter, show_progress_bar=True)#, gc_after_trial=True)
            params = study.best_trial.params
            layers = [param for param in params if 'n_units_' in param]
            layers = tuple(params[layer] for layer in layers)
            model = MLPClassifier(hidden_layer_sizes=tuple(layers), learning_rate_init=params['learning_rate_init'], 
                activation=params['activation'], learning_rate=params['learning_rate'], alpha=params['alpha'], 
                batch_size='auto', solver=params['solver'], max_iter=200, random_state=1909)
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
            gs = BayesSearchCV(n_iter=n_iter, estimator=model_0, search_spaces=params, cv=opt_cv)
            gs.fit(data_x, data_y)
            best_est, best_score = gs.best_estimator_, np.round(gs.best_score_, 4)
            print('Highest mean accuracy: {}'.format(best_score))
            return gs.best_estimator_, gs.best_params_
          
    elif clf == 'xgb':
        objective = objective_xgb(data_x, data_y, limit_search=limit_search, opt_cv=opt_cv, min_gamma=min_gamma)
        if limit_search:
            print('NOTE: To expand hyperparameter search space, set limit_search=False, although this will increase the optimization time significantly.')
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True)#, gc_after_trial=True)
        params = study.best_trial.params
        if limit_search:
            if params['booster'] == 'dart':
                model = XGBClassifier(booster=params['booster'],  n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                    reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], sample_type=params['sample_type'], normalize_type=params['normalize_type'],
                    rate_drop=params['rate_drop'], skip_drop=params['skip_drop'], scale_pos_weight=sample_weight, random_state=1909)
            elif params['booster'] == 'gbtree':
                model = XGBClassifier(booster=params['booster'],  n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                    reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], subsample=params['subsample'], scale_pos_weight=sample_weight, random_state=1909)
        else:
            if params['booster'] == 'dart':
                model = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'], 
                    reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], sample_type=params['sample_type'], normalize_type=params['normalize_type'],rate_drop=params['rate_drop'], 
                    skip_drop=params['skip_drop'], min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'], subsample=params['subsample'],
                    scale_pos_weight=sample_weight, random_state=1909)
            elif params['booster'] == 'gbtree':
                model = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'], 
                    reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], subsample=params['subsample'], min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'],
                    scale_pos_weight=sample_weight, random_state=1909)

    final_score = study.best_value

    if clf == 'rf' or clf == 'xgb' or clf == 'nn':

        if initial_score > final_score:
            print('Hyperparameter optimization complete! Optimal performance of {} is LOWER than the base performance of {}, try increasing the value of n_iter and run again.'.format(np.round(final_score, 8), np.round(initial_score, 8)))
        else:
            print('Hyperparameter optimization complete! Optimal performance of {} is HIGHER than the base performance of {}.'.format(np.round(final_score, 8), np.round(initial_score, 8)))
        
        if return_study:
            return model, params, study
        return model, params

    else:

        print('Hyperparameter optimization complete! Optimal performance: {}'.format(np.round(final_score, 8)))
        
        if return_study:
            return params, study
        return params

def borutashap_opt(data_x, data_y, boruta_trials=50, model='rf', importance_type='gain'):
    """
    Applies a combination of the Boruta algorithm and
    Shapley values, a method developed by Eoghan Keany (2020).

    See: https://doi.org/10.5281/zenodo.4247618

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        boruta_trials (int): The number of trials to run. A larger number is
            better as the distribution will be more robust to random fluctuations. 
            Defaults to 50.
        model (str): The ensemble method to use when fitting and calculating
            the feature importance metric. Only two options are currently
            supported, 'rf' for Random Forest and 'xgb' for Extreme Gradient Boosting.
            Defaults to 'rf'.
        importance_type (str): The feature importance type to use, only applicable
            when using clf='xgb'. The options include “gain”, “weight”, “cover”,
            “total_gain” or “total_cover”. Defaults to 'gain'.

    Returns:
        First output is a 1D array containing the indices of the selected features. 
        These indices can then be used to select the columns in the data_x array.
        Second output is the feature selection object, which contains feature selection
        history information and visualization options.
    """
    
    if boruta_trials == 0: #This is the flag that the ensemble_model.Classifier class uses to disable feature selection
        return np.arange(data_x.shape[1]), None

    if boruta_trials < 20:
        print('WARNING: Results are unstable if boruta_trials is too low!')
    if np.any(np.isnan(data_x)):
        #print('NaN values detected, applying Strawman imputation...')
        data_x = Strawman_imputation(data_x)

    if model == 'rf':
        classifier = RandomForestClassifier(random_state=1909)
    elif model == 'xgb':
        classifier = XGBClassifier(random_state=1909)#tree_method='exact', max_depth=20, importance_type=importance_type)
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
        print('Boruta with Shapley values failed, switching to original Boruta...')
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

    classifier = RandomForestClassifier(random_state=1909)

    feat_selector = BorutaPy(classifier, n_estimators='auto', random_state=1909)
    print('Running feature selection...')
    feat_selector.fit(data_x, data_y)

    feats = np.array([str(feat) for feat in feat_selector.support_])
    index = np.where(feats == 'True')[0]

    print('Feature selection complete, {} selected out of {}!'.format(len(index),len(feat_selector.support_)))

    return index

def impute_missing_values(data, imputer=None, strategy='knn', k=3, constant_value=0, nan_threshold=0.5):
    """
    Impute missing values in the input data array using various imputation strategies.
    By default the imputer will be created and returned, unless
    the imputer argument is set, in which case only the transformed
    data is output. 

    The function first identifies the columns with mostly missing values based on the nan_threshold parameter. 
    It replaces the values in those columns with zeros before performing the imputation step using the selected 
    imputation strategy. This way, the ignored columns will have zeros and won't be taken into account during imputation. 
    The resulting imputed_data will have all columns preserved, with missing values filled according to the chosen imputation strategy.
    This is required because the imputation techniques employed remove columns that have too many nans!
    
    Note:
        As the KNN imputation method bundles neighbors according to their eucledian distance,
        it is sensitive to outliers. Furthermore, it can also yield weak predictions if the
        training features are heaviliy correlated. Tang & Ishwaran 2017 reported that if there is 
        low to medium correlation in the dataset, Random Forest imputation algorithms perform 
        better than KNN imputation

    Args:
        data (ndarray): Input data array with missing values.
        imputer (optional): A KNNImputer class instance, configured using sklearn.impute.KNNImputer.
            Defaults to None, in which case the transformation is created using
            the data itself. 
        strategy (str, optional): Imputation strategy to use. Defaults to 'knn'.
            - 'mean': Fill missing values with the mean of the non-missing values in the same column.
            - 'median': Fill missing values with the median of the non-missing values in the same column.
            - 'mode': Fill missing values with the mode (most frequent value) of the non-missing values in the same column.
            - 'constant': Fill missing values with a constant value provided by the user.
            - 'knn': Fill missing values using k-Nearest Neighbor imputation.
        k (int, optional): Number of nearest neighbors to consider for k-Nearest Neighbor imputation.
            Only applicable if the imputation strategy is set to 'knn'. Defaults to 3.
        constant_value (float or int, optional): Constant value to use for constant imputation.
            Only applicable if the imputation strategy is set to 'constant'. Defaults to 0.
        nan_threshold (float): Columns with nan values greater than this ratio will be zeroed out before the imputation.
            Defualts to 0.5.

    Returns:
        The first output is the data array with with the missing values filled in. 
        The second output is the KNN Imputer that should be used to transform
        new data, prior to predictions. 
    """

    if imputer is None:

        column_missing_ratios = np.mean(np.isnan(data), axis=0)
        columns_to_ignore = np.where(column_missing_ratios > nan_threshold)[0]
        if len(columns_to_ignore) > 0:
            print(f"WARNING: At least one data column has too many nan values according to the following threshold: {nan_threshold}. These columns have been zeroed out completely: {columns_to_ignore}")
            data[:,columns_to_ignore] = 0
        
        if strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
        elif strategy == 'constant':
            if constant_value is None:
                raise ValueError("The constant_value parameter must be provided if strategy='constant'.")
            imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=k)
        else:
            raise ValueError("Invalid imputation strategy. Please choose from 'mean', 'median', 'mode', 'constant', or 'knn'.")

        imputer.fit(data)
        imputed_data = imputer.transform(data)
        return imputed_data, imputer

    return imputer.transform(data) 

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
