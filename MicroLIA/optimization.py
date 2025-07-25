#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  25 10:39:23 2023

@author: daniel
"""
#import os, sys
#os.environ['PYTHONHASHSEED'], os.environ["TF_DETERMINISTIC_OPS"] = '0', '1'
import numpy as np
#import random as python_random
#np.random.seed(1909), python_random.seed(1909)
#import joblib   
from pandas import DataFrame
from collections import Counter 
#import sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from xgboost import XGBClassifier
from MicroLIA import feature_selection 

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)


class objective_xgb(object):
    """
    Optuna objective class for optimizing an XGBoost classifier using cross-validation.

    This class defines the optimization logic for tuning XGBoost hyperparameters using
    the Optuna framework. It supports limited or broad search spaces depending on
    the `limit_search` flag, and returns the cross-validated performance metric for 
    each trial.

    Parameters
    ----------
    data_x : ndarray
        Feature matrix of shape (n_samples, n_features).
    data_y : ndarray or array-like
        Corresponding class labels of shape (n_samples,).
    limit_search : bool, optional
        If True, restricts the hyperparameter search space to a narrower range.
        Defaults to False (broad search).
    opt_cv : int, optional
        Number of cross-validation folds. Must be >= 2. Default is 3.
    scoring_metric : str, optional
        Evaluation metric used during optimization. Options are:
        ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']. Default is 'f1'.
    SEED_NO : int, optional
        Random seed for reproducibility. Default is 1909.

    Returns
    -------
    float
        Cross-validated score (mean across folds) for the given trial configuration.
    """
    def __init__(self, data_x, data_y, limit_search=False, opt_cv=3, scoring_metric="f1", SEED_NO=1909):
        self.data_x = data_x
        self.data_y = data_y
        self.limit_search = limit_search
        self.opt_cv = opt_cv
        self.SEED_NO = SEED_NO

        if opt_cv < 2:
            raise ValueError("opt_cv must be >= 2 for StratifiedKFold.")

        # Determine number of classes
        self.n_classes = np.unique(data_y).size

        # Upgrade scorer if multiclass
        if self.n_classes > 2:
            if scoring_metric in ("f1", "precision", "recall"):
                self.scoring_metric = f"{scoring_metric}_macro"
            elif scoring_metric == "roc_auc":
                self.scoring_metric = "roc_auc_ovr"
            else:
                self.scoring_metric = scoring_metric
        else:
            self.scoring_metric = scoring_metric

    def __call__(self, trial):
        """
        Run a single optimization trial by training the XGBoost model on cross-validation folds
        and returning the mean performance metric.

        Parameters
        ----------
        trial : optuna.Trial
            A trial object provided by Optuna to suggest hyperparameters.

        Returns
        -------
        float
            Mean cross-validated score for the trial.
        """
        if self.limit_search:
            # The hyperparam search space
            n_estimators = trial.suggest_int('n_estimators', 100, 300)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            eta = trial.suggest_float('eta', 1e-3, 0.3, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 2.0, log=True)
            reg_alpha = trial.suggest_float('reg_alpha', 1e-3, 2.0, log=True)
            gamma = trial.suggest_float('gamma', 0.0, 10.0)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)

            clf = XGBClassifier(
                booster='gbtree',
                n_estimators=n_estimators,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                max_depth=max_depth,
                eta=eta,
                gamma=gamma,
                subsample=subsample,
                random_state=self.SEED_NO
            )
        else:
            n_estimators = trial.suggest_int('n_estimators', 50, 2000)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            eta = trial.suggest_float('eta', 1e-3, 0.3, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
            reg_alpha = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
            gamma = trial.suggest_float('gamma', 0.0, 10.0)
            min_child_weight = trial.suggest_float('min_child_weight', 1e-3, 50.0, log=True)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

            clf = XGBClassifier(
                booster='gbtree',
                n_estimators=n_estimators,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                max_depth=max_depth,
                eta=eta,
                gamma=gamma,
                min_child_weight=min_child_weight,
                subsample=subsample,
                random_state=self.SEED_NO
            )

        # Set objective based on class count
        if self.n_classes > 2:
            clf.set_params(objective='multi:softprob', num_class=self.n_classes)
        else:
            clf.set_params(objective='binary:logistic')

        cv_splitter = StratifiedKFold(n_splits=self.opt_cv, shuffle=True, random_state=self.SEED_NO)
        cross_val = cross_validate(clf, self.data_x, self.data_y, cv=cv_splitter, scoring=self.scoring_metric)
        trial_performance = np.mean(cross_val['test_score'])

        return trial_performance

class objective_nn(object):
    """
    Optuna objective class for optimizing an MLP classifier using cross-validation.

    This class defines the optimization logic for tuning XGBoost hyperparameters using
    the Optuna framework. It supports limited or broad search spaces depending on
    the `limit_search` flag, and returns the cross-validated performance metric for 
    each trial.

    Parameters
    ----------
    data_x : ndarray
        Feature matrix of shape (n_samples, n_features).
    data_y : ndarray or array-like
        Corresponding class labels of shape (n_samples,).
    limit_search : bool, optional
        If True, restricts the hyperparameter search space to a narrower range.
        Defaults to False (broad search).
    opt_cv : int, optional
        Number of cross-validation folds. Must be >= 2. Default is 3.
    scoring_metric : str, optional
        Evaluation metric used during optimization. Options are:
        ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']. Default is 'f1'.
    SEED_NO : int, optional
        Random seed for reproducibility. Default is 1909.

    Returns
    -------
    float
        Cross-validated score (mean across folds) for the given trial configuration.
    """
    def __init__(self, data_x, data_y, opt_cv, scoring_metric="f1", SEED_NO=1909):
        self.data_x = data_x
        self.data_y = data_y
        self.opt_cv = opt_cv
        self.SEED_NO = SEED_NO

        if opt_cv < 2:
            raise ValueError("opt_cv must be >= 2 for StratifiedKFold.")

        n_classes = np.unique(data_y).size
        if n_classes > 2:
            if scoring_metric in ("f1", "precision", "recall"):
                self.scoring_metric = f"{scoring_metric}_macro"
            elif scoring_metric == "roc_auc":
                self.scoring_metric = "roc_auc_ovr"
            else:
                self.scoring_metric = scoring_metric
        else:
            self.scoring_metric = scoring_metric

    def __call__(self, trial):
        """
        Run a single optimization trial by training the XGBoost model on cross-validation folds
        and returning the mean performance metric.

        Parameters
        ----------
        trial : optuna.Trial
            A trial object provided by Optuna to suggest hyperparameters.

        Returns
        -------
        float
            Mean cross-validated score for the trial.
        """
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 3e-1, log=True)
        solver = trial.suggest_categorical("solver", ["sgd", "adam"])
        activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        alpha = trial.suggest_float("alpha", 1e-7, 1e0, log=True)
        n_layers = trial.suggest_int('hidden_layer_sizes', 1, 10)

        layers = tuple(trial.suggest_int(f'n_units_{i}', 10, 200) for i in range(n_layers))

        clf = MLPClassifier(
            hidden_layer_sizes=layers,
            learning_rate_init=learning_rate_init,
            learning_rate=learning_rate,
            solver=solver,
            activation=activation,
            alpha=alpha,
            batch_size='auto',
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=self.SEED_NO
        )

        cv = StratifiedKFold(n_splits=self.opt_cv, shuffle=True, random_state=self.SEED_NO)
        cross_val = cross_validate(clf, self.data_x, self.data_y, cv=cv, scoring=self.scoring_metric)
        trial_performance = np.mean(cross_val['test_score'])

        return trial_performance

class objective_rf(object):
    """
    Optuna objective class for optimizing a RF classifier using cross-validation.

    This class defines the optimization logic for tuning XGBoost hyperparameters using
    the Optuna framework. It supports limited or broad search spaces depending on
    the `limit_search` flag, and returns the cross-validated performance metric for 
    each trial.

    Parameters
    ----------
    data_x : ndarray
        Feature matrix of shape (n_samples, n_features).
    data_y : ndarray or array-like
        Corresponding class labels of shape (n_samples,).
    limit_search : bool, optional
        If True, restricts the hyperparameter search space to a narrower range.
        Defaults to False (broad search).
    opt_cv : int, optional
        Number of cross-validation folds. Must be >= 2. Default is 3.
    scoring_metric : str, optional
        Evaluation metric used during optimization. Options are:
        ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']. Default is 'f1'.
    SEED_NO : int, optional
        Random seed for reproducibility. Default is 1909.

    Returns
    -------
    float
        Cross-validated score (mean across folds) for the given trial configuration.
    """
    def __init__(self, data_x, data_y, opt_cv, scoring_metric='f1', SEED_NO=1909):
        self.data_x = data_x
        self.data_y = data_y
        self.opt_cv = opt_cv
        self.SEED_NO = SEED_NO

        if opt_cv < 2:
            raise ValueError("opt_cv must be >= 2 for StratifiedKFold.")

        n_classes = np.unique(data_y).size
        if n_classes > 2:
            if scoring_metric in ("f1", "precision", "recall"):
                self.scoring_metric = f"{scoring_metric}_macro"
            elif scoring_metric == "roc_auc":
                self.scoring_metric = "roc_auc_ovr"
            else:
                self.scoring_metric = scoring_metric
        else:
            self.scoring_metric = scoring_metric

    def __call__(self, trial):
        """
        Run a single optimization trial by training the XGBoost model on cross-validation folds
        and returning the mean performance metric.

        Parameters
        ----------
        trial : optuna.Trial
            A trial object provided by Optuna to suggest hyperparameters.

        Returns
        -------
        float
            Mean cross-validated score for the trial.
        """
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        max_depth = trial.suggest_int('max_depth', 2, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 'auto'])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])

        max_samples = None
        if bootstrap:
            max_samples = trial.suggest_float('max_samples', 0.3, 1.0)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            max_samples=max_samples,
            class_weight=class_weight,
            random_state=self.SEED_NO,
        )

        cv = StratifiedKFold(n_splits=self.opt_cv, shuffle=True, random_state=self.SEED_NO)
        cross_val = cross_validate(clf, self.data_x, self.data_y, cv=cv, scoring=self.scoring_metric)
        trial_performance = np.mean(cross_val['test_score'])

        return trial_performance

def hyper_opt(
    data_x=None, 
    data_y=None, 
    clf='rf', 
    n_iter=25, 
    opt_cv=3, 
    scoring_metric='f1', 
    balance=True, 
    limit_search=True, 
    return_study=True, 
    SEED_NO=1909
    ): 
    """
    Optimizes hyperparameters using a k-fold cross-validation splitting strategy.

    This function constructs a classification engine based on the input classifier (`clf`)
    and tunes its hyperparameters using Optuna. If `return_study=True`, the Optuna Study
    object will be returned for further analysis or visualization.

    Parameters
    ----------
    data_x : ndarray, optional
        Feature matrix of shape (n_samples, n_features).
    data_y : ndarray or list of str, optional
        Corresponding class labels of shape (n_samples,).
    clf : str, optional
        Classifier to optimize. Options are:
        'rf' (Random Forest), 'nn' (Neural Network), 'xgb' (XGBoost).
        Default is 'rf'.
    n_iter : int, optional
        Maximum number of optimization iterations (trials). Default is 25.
    opt_cv : int, optional
        Number of cross-validation folds used per trial. Must be >= 2. Default is 3.
    scoring_metric : str, optional
        Evaluation metric used during optimization. Options are:
        ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']. Default is 'f1'.
    balance : bool, optional
        If True, class weights will be computed and applied to help address class imbalance.
        Only applies to binary classification. Default is True.
    limit_search : bool, optional
        If True, restricts the hyperparameter search space for quicker optimization.
        Default is True.
    return_study : bool, optional
        If True, also returns the Optuna Study object used during optimization.
        Default is True.
    SEED_NO : int, optional
        Random seed for reproducibility. Default is 1909.

    Returns
    -------
    model : BaseEstimator
        Trained classifier with optimal hyperparameters.
    params : dict
        Dictionary of the best hyperparameter combination found during optimization.
    study : optuna.study.Study, optional
        Only returned if `return_study=True`. The Optuna study object used for optimization.
    """

    if clf == 'rf':
        model_0 = RandomForestClassifier(random_state=SEED_NO)
    elif clf == 'nn':
        model_0 = MLPClassifier(random_state=SEED_NO)
    elif clf == 'xgb':
        model_0 = XGBClassifier(random_state=SEED_NO)
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
        print(f'No optimization trials configured (n_iter=0), returning base {clf} model...')
        return model_0 
    
    # Beginning optimization, but first define a baseline model (defaul hyperparams)
    n_classes = np.unique(data_y).size
    if n_classes > 2:
        scoring_map = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro", "roc_auc": "roc_auc_ovr"}
        scoring_metric = scoring_map.get(scoring_metric, scoring_metric)

    cv = StratifiedKFold(n_splits=opt_cv, shuffle=True, random_state=SEED_NO)
    cross_val = cross_validate(model_0, data_x, data_y, cv=cv, scoring=scoring_metric)
    initial_score = np.mean(cross_val['test_score'])

    sampler = optuna.samplers.TPESampler(seed=SEED_NO)
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
        objective = objective_rf(data_x, data_y, opt_cv=opt_cv, scoring_metric=scoring_metric, SEED_NO=SEED_NO)
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True)#, gc_after_trial=True)
        params = study.best_trial.params
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'], 
            criterion=params['criterion'], 
            max_depth=params['max_depth'], 
            min_samples_split=params['min_samples_split'], 
            min_samples_leaf=params['min_samples_leaf'], 
            max_features=params['max_features'], 
            bootstrap=params['bootstrap'], 
            class_weight=sample_weight, 
            random_state=SEED_NO
            )
        
    elif clf == 'nn':
        objective = objective_nn(
            data_x, 
            data_y, 
            opt_cv=opt_cv, 
            scoring_metric=scoring_metric, 
            SEED_NO=SEED_NO
            )

        study.optimize(objective, n_trials=n_iter, show_progress_bar=True)#, gc_after_trial=True)
        params = study.best_trial.params
        layers = [param for param in params if 'n_units_' in param]
        layers = tuple(params[layer] for layer in layers)

        model = MLPClassifier(
            hidden_layer_sizes=tuple(layers), 
            learning_rate_init=params['learning_rate_init'], 
            activation=params['activation'], 
            learning_rate=params['learning_rate'], 
            alpha=params['alpha'], 
            solver=params['solver'], 
            max_iter=2500, 
            random_state=SEED_NO
            )

          
    elif clf == 'xgb':
        objective = objective_xgb(
            data_x, 
            data_y, 
            limit_search=limit_search, 
            opt_cv=opt_cv, 
            scoring_metric=scoring_metric, 
            SEED_NO=SEED_NO
            )

        if limit_search:
            print('NOTE: To expand hyperparameter search space, set limit_search=False, although this will increase the optimization time significantly.')
        
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True)#, gc_after_trial=True)
        params = study.best_trial.params

        if limit_search:
            model = XGBClassifier(
                booster='gbtree',  
                n_estimators=params['n_estimators'], 
                reg_lambda=params['reg_lambda'], 
                reg_alpha=params['reg_alpha'], 
                max_depth=params['max_depth'], 
                eta=params['eta'], 
                gamma=params['gamma'], 
                subsample=params['subsample'], 
                scale_pos_weight=sample_weight, 
                random_state=SEED_NO
                )
        else:
            model = XGBClassifier(
                booster='gbtree', 
                n_estimators=params['n_estimators'], 
                colsample_bytree=params['colsample_bytree'], 
                reg_lambda=params['reg_lambda'], 
                reg_alpha=params['reg_alpha'], 
                max_depth=params['max_depth'], 
                eta=params['eta'], 
                gamma=params['gamma'], 
                subsample=params['subsample'], 
                min_child_weight=params['min_child_weight'], 
                scale_pos_weight=sample_weight, 
                random_state=SEED_NO
                )

    final_score = study.best_value

    if initial_score > final_score:
        print('Hyperparameter optimization complete! Optimal performance of {} is LOWER than the base performance of {}, try increasing the value of n_iter and run again.'.format(np.round(final_score, 8), np.round(initial_score, 8)))
    else:
        print('Hyperparameter optimization complete! Optimal performance of {} is HIGHER than the base performance of {}.'.format(np.round(final_score, 8), np.round(initial_score, 8)))
    
    if return_study:
        return model, params, study
    return model, params


def borutashap_opt(
    data_x, 
    data_y, 
    boruta_trials=50, 
    model='rf', 
    importance_type='gain', 
    SEED_NO=1909
    ):
    """
    Applies a combination of the Boruta algorithm and SHAP values for feature selection.

    This method was developed by Eoghan Keany (2020) and integrates model-based
    feature selection with Shapley values to yield a stable, interpretable set of features.

    See: https://doi.org/10.5281/zenodo.4247618

    Parameters
    ----------
    data_x : ndarray
        Feature matrix of shape (n_samples, n_features).
    data_y : ndarray or list of str
        Corresponding class labels of shape (n_samples,).
    boruta_trials : int, optional
        Number of trials to run. A higher value increases the robustness of feature selection.
        Defaults to 50.
    model : str, optional
        Model to use for computing feature importance. Options are:
        'rf' (Random Forest) or 'xgb' (XGBoost). Defaults to 'rf'.
    importance_type : str, optional
        XGBoost-specific feature importance metric. Options are:
        ['gain', 'weight', 'cover', 'total_gain', 'total_cover']. Default is 'gain'.
    SEED_NO : int, optional
        Random seed for reproducibility. Default is 1909.

    Returns
    -------
    selected_indices : ndarray
        1D array of indices corresponding to selected features.
    feat_selector : BorutaSHAP
        The feature selection object, containing selection history and plotting methods.
    """
    
    if boruta_trials == 0: #This is the flag that the ensemble_model.Classifier class uses to disable feature selection
        return np.arange(data_x.shape[1]), None

    if boruta_trials < 20:
        print('WARNING: Results are unstable if boruta_trials is too low!')
    if np.any(np.isnan(data_x)):
        #print('NaN values detected, applying Strawman imputation...')
        data_x = Strawman_imputation(data_x)

    if model == 'rf':
        classifier = RandomForestClassifier(random_state=SEED_NO)
    elif model == 'xgb':
        classifier = XGBClassifier(random_state=SEED_NO)#tree_method='exact', max_depth=20, importance_type=importance_type)
    else:
        raise ValueError('Model argument must either be "rf" or "xgb".')
    
    #BorutaShap program requires input to have the columns attribute
    #Converting to Pandas dataframe
    cols = [str(i) for i in np.arange(data_x.shape[1])]
    X = DataFrame(data_x, columns=cols)
    y = np.zeros(len(data_y))

    #Below is to convert categorical labels to numerical, as per BorutaShap requirements
    for i, label in enumerate(np.unique(data_y)):
        mask = np.where(data_y == label)[0]
        y[mask] = i

    feat_selector = feature_selection.BorutaSHAP(model=classifier, importance_measure='shap', classification=True)
    print('Running feature selection...')
    feat_selector.fit(X=X, y=y, n_trials=boruta_trials, verbose=False, random_state=SEED_NO)

    index = np.array([int(feat) for feat in feat_selector.accepted])
    index.sort()
    print('Feature selection complete, {} selected out of {}!'.format(len(index), data_x.shape[1]))

    return index, feat_selector

def standardize_data(
    data_x, 
    method='min-max', 
    return_scaler=True
    ):
    """
    Normalizes the data using the specified method.

    Tree-based ensembles do not require standardized inputs, but methods such as 
    neural networks or PCA (which are sensitive to feature ranges) benefit from standardization.

    Parameters
    ----------
    data_x : ndarray
        Training data feature matrix of shape (n_samples, n_features).
    method : str, optional
        Normalization method. Options are:
        'min-max' (default), 'robust', or 'standard'.
    return_scaler : bool, optional
        If True, returns both the normalized data and the fitted scaler.
        If False, returns only the normalized data. Default is True.

    Returns
    -------
    norm_data_x : ndarray
        Normalized feature matrix.
    scaler : MinMaxScaler or RobustScaler or StandardScaler, optional
        The fitted scaler object. Only returned if `return_scaler` is True.

    Raises
    ------
    ValueError
        If an unknown method is specified.
    """

    if method == 'min-max':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'standard':
        scaler = StandardScaler()

    scaler.fit(data_x)

    norm_data_x = scaler.transform(data_x)

    if return_scaler:
        return norm_data_x, scaler
    else:
        return norm_data_x

def impute_missing_values(data, imputer=None, strategy='knn', k=3, constant_value=0, nan_threshold=0.5):
    """
    Impute missing values in the input data array using various imputation strategies.

    This function identifies columns with a high fraction of NaNs (as defined by
    `nan_threshold`) and replaces them with zeros before applying imputation. This avoids
    issues where imputation algorithms would otherwise remove those columns.

    Notes
    -----
    - KNN imputation is sensitive to outliers and performs worse when features are highly correlated.
      Tang & Ishwaran (2017) report that in such cases, Random Forest-based methods may be superior.

    Parameters
    ----------
    data : ndarray
        Input data array with missing values. Shape (n_samples, n_features).
    imputer : SimpleImputer or KNNImputer, optional
        A pre-configured imputer object. If provided, only transformation is applied.
        If None, a new imputer is created and returned. Default is None.
    strategy : str, optional
        Strategy to use for imputation. Options are:
        'mean', 'median', 'mode', 'constant', or 'knn'. Default is 'knn'.
    k : int, optional
        Number of neighbors for k-Nearest Neighbor imputation. Only used if `strategy='knn'`.
        Default is 3.
    constant_value : float or int, optional
        Value to use if `strategy='constant'`. Default is 0.
    nan_threshold : float, optional
        Columns with NaN ratios above this threshold will be filled with zeros before imputation.
        Default is 0.9.

    Returns
    -------
    imputed_data : ndarray
        Data with missing values filled in.
    imputer : SimpleImputer or KNNImputer
        The fitted imputer used for the transformation.
        Only returned if `imputer` was None at input.

    Raises
    ------
    ValueError
        If an invalid strategy is given or required parameters are missing.
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
    Perform Strawman imputation, a time-efficient algorithm in which missing data values
    are replaced with the median value of the entire non-NaN sample.

    If the data is one-hot encoded boolean (e.g., 0/1), the median will correspond to
    the most frequent value, which is sufficient for random forests that do not accept
    True/False input.

    This is the baseline imputation algorithm used in:
    Tang & Ishwaran (2017), https://arxiv.org/pdf/1701.05305.pdf

    Notes
    -----
    - This function assumes each row corresponds to a sample and missing values
      are encoded as either `np.nan` or `np.inf`.
    - For 1D arrays, the overall median of finite values is used.
    - For 2D arrays, the median is computed independently for each column.

    Parameters
    ----------
    data : ndarray or list
        Input array of shape (n,) or (n_samples, n_features) with missing values
        encoded as NaN or Inf.

    Returns
    -------
    imputed_data : ndarray
        The input data with missing values replaced using median-based imputation.
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
