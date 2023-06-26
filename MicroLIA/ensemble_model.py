# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import os
import copy 
import joblib 
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
from cycler import cycler
from warnings import warn
from pathlib import Path
from collections import Counter  

from sklearn import decomposition
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, auc, RocCurveDisplay
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from scikitplot.metrics import plot_roc
from sklearn.manifold import TSNE

from optuna.importance import get_param_importances, FanovaImportanceEvaluator
from MicroLIA.optimization import hyper_opt, borutashap_opt, impute_missing_values
from MicroLIA import extract_features

class Classifier:
    """
    Creates a machine learning classifier object. The built-in methods can be used to optimize the engine and output visualizations.

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        clf (str): The machine learning classifier to optimize. Can either be
            'rf' for Random Forest, 'nn' for Neural Network, or 'xgb' for Extreme Gradient Boosting. 
            Defaults to 'rf'.
        optimize (bool): If True the Boruta algorithm will be run to identify the features
            that contain useful information (if boruta_trials > 0), after which the optimal engine 
            hyperparameters will be determined using Bayesian optimization (if n_iter > 0).
        opt_cv (int): Cross-validations to perform when assesing the performance during the
            hyperparameter optimization. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. Defaults to 10.
            NOTE: The higher this number, the longer the optimization will take.
        limit_search (bool): If False, the search space for the parameters will be expanded,
            as there are some hyperparameters that can range from 0 to inf. Defaults to True to
            limit the search and speed up the optimization routine.
        impute (bool): If True data imputation will be performed to replace NaN values. Defaults to False.
            If set to True, the imputer attribute will be saved for future transformations. 
        imp_method (str, optional): Imputation strategy to use if impute is set to True.
            Defaults to 'knn'. The imputation methods supported include:
            ('knn'): Fill missing values using k-Nearest Neighbor imputation.
            ('mean'): Fill missing values with the mean of the non-missing values in the same column.
            ('median'): Fill missing values with the median of the non-missing values in the same column.
            ('mode'): Fill missing values with the mode (most frequent value) of the non-missing values in the same column.
            ('constant'): Fill missing values with a constant value provided by the user.
        n_iter (int): The maximum number of iterations to perform during the hyperparameter search. 
            Defaults to 25. Can be set to 0 to avoid this optimization routine.
        boruta_trials (int): The number of trials to run when running Boruta for feature selection. 
            Can be set to 0 for no feature selection. Defaults to 50.
        boruta_model (str): The ensemble algorithm to use when calculating the feature importance metrics
            for the features, which is utilized by the Boruta algorithm to construct the distributions. 
            Can either be 'rf' or 'xgb'. In practice setting this to 'xgb' will result in a more agressive 
            feature selection. Defaults to 'rf'.
        balance (bool, optional): If True, a weights array will be calculated and used when fitting the classifier. 
            This can improve classification when classes are imbalanced and is ignored otherwise. This is only applied if 
            the classification is a binary task. Defaults to True.        
        csv_file (DataFrame, optional): The csv file output after generating the training set. This can be
            input in lieu of the data_x and data_y arguments. Note that the csv_file must have a "label" column,
            and is intended to be used after executing the MicroLIA.training_set routine.
    """

    def __init__(self, data_x=None, data_y=None, clf='rf', optimize=False, opt_cv=10, 
        limit_search=True, impute=False, imp_method='knn', n_iter=25, 
        boruta_trials=50, boruta_model='rf', balance=True, csv_file=None):

        self.data_x = data_x
        self.data_y = data_y
        self.clf = clf
        self.optimize = optimize 
        self.opt_cv = opt_cv 
        self.limit_search = limit_search
        self.impute = impute
        self.imp_method = imp_method
        self.n_iter = n_iter
        self.boruta_trials = boruta_trials
        self.boruta_model = boruta_model 
        self.balance = balance 
        self.csv_file = csv_file

        self.model = None
        self.imputer = None
        self.feats_to_use = None

        self.feature_history = None  
        self.optimization_results = None 
        self.best_params = None 

        if self.csv_file is not None:
            self.data_x = np.array(csv_file[csv_file.columns[:-1]])
            self.data_y = csv_file.label
            print('Successfully loaded the data_x and data_y arrays from the input csv_file!')
        else:
            if self.data_x is None or self.data_y is None:
                print('NOTE: data_x and data_y parameters are required to output visualizations.')
        
        if self.data_y is not None:
            self.data_y_ = copy.deepcopy(self.data_y) #For plotting purposes, save the original label array as it will be overwritten with the numerical labels when plotting
            if self.clf == 'xgb':
                if all(isinstance(val, (int, str)) for val in self.data_y):
                    print('XGBoost classifier requires numerical class labels! Converting class labels as follows:')
                    print('________________________________')
                    y = np.zeros(len(self.data_y))
                    for i in range(len(np.unique(self.data_y))):
                        print(str(np.unique(self.data_y)[i]).ljust(10)+'  ------------->     '+str(i))
                        index = np.where(self.data_y == np.unique(self.data_y)[i])[0]
                        y[index] = i
                    self.data_y = y 
                    print('________________________________')
        else:
            self.data_y_ = None 

    def create(self, overwrite_training=False):
        """
        Creates the machine learning engine, current options are either a
        Random Forest, XGBoost, or a Neural Network classifier. 

        overwrite_training (bool): Whether to replace the original input data_x with the pre-processed
            data_x. Defaults to False. 
        
        Returns:
            Trained and optimized classifier.
        """

        if self.optimize is False:
            if len(np.unique(self.data_y)) == 2:
                counter = Counter(self.data_y)
                if counter[np.unique(self.data_y)[0]] != counter[np.unique(self.data_y)[1]]:
                    if self.balance: #If balance is True but optimize is False
                        print('Unbalanced dataset detected, to apply weights set optimize=True.')

        if self.clf == 'rf':
            model = RandomForestClassifier(random_state=1909)
        elif self.clf == 'nn':
            model = MLPClassifier(random_state=1909)
        elif self.clf == 'xgb':
            model = XGBClassifier(random_state=1909)
            if all(isinstance(val, (int, str)) for val in self.data_y):
                print('XGBoost classifier requires numerical class labels! Converting class labels as follows:')
                print('________________________________')
                y = np.zeros(len(self.data_y))
                for i in range(len(np.unique(self.data_y))):
                    print(str(np.unique(self.data_y)[i]).ljust(10)+'  ------------->     '+str(i))
                    index = np.where(self.data_y == np.unique(self.data_y)[i])[0]
                    y[index] = i
                self.data_y = y 
                print('________________________________')
        elif self.clf == 'ocsvm':
            if self.data_y is not None:
                if len(np.unique(self.data_y)) != 1:
                    raise ValueError('The clf parameter has been set to "ocsvm" but OneClassSVM requires that only the positive class be input!')
            model = OneClassSVM()
        else:
            raise ValueError('clf argument must either be "rf", "nn", "ocsvm", or "xgb".')
        
        self.data_x[np.isinf(self.data_x)] = np.nan

        if self.impute is False and self.optimize is False:
            data = copy.deepcopy(self.data_x)
            data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
            if np.any(np.isfinite(self.data_x)==False):
                raise ValueError('data_x array contains nan values but impute is set to False! Set impute=True and run again.')
            print("Returning base {} model...".format(self.clf))
            model.fit(data, self.data_y)
            self.model = model
            self.data_x = data if overwrite_training else self.data_x

            return

        if self.impute:
            data, self.imputer = impute_missing_values(self.data_x, strategy=self.imp_method)
            data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
            if self.optimize is False:
                print("Returning base {} model...".format(self.clf))
                model.fit(data, self.data_y)
                self.model = model 
                self.data_x = data if overwrite_training else self.data_x

                return
        else:
            data = copy.deepcopy(self.data_x)
            data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7

        if self.feats_to_use is None:
            self.feats_to_use, self.feature_history = borutashap_opt(data, self.data_y, boruta_trials=self.boruta_trials, model=self.boruta_model)
            if len(self.feats_to_use) == 0:
                print('No features selected, increase the number of n_trials when running MicroLIA.optimization.borutashap_opt(). Using all features...')
                self.feats_to_use = np.arange(data.shape[1])
        else:
            print('The feats_to_use attribute already exists, skipping feature selection...')

        #Re-construct the imputer with the selected features as new predictions will only compute these metrics, so need to fit again!
        data_x, self.imputer = impute_missing_values(self.data_x[:,self.feats_to_use], strategy=self.imp_method) if self.impute else self.data_x[:,self.feats_to_use]

        if self.n_iter > 0:
            self.model, self.best_params, self.optimization_results = hyper_opt(data_x, self.data_y, clf=self.clf, n_iter=self.n_iter, balance=self.balance, 
                return_study=True, limit_search=self.limit_search, opt_cv=self.opt_cv)
        else:
            print("Fitting and returning final model...")
            self.model = hyper_opt(data_x, self.data_y, clf=self.clf, n_iter=self.n_iter, balance=self.balance, return_study=True, limit_search=self.limit_search, opt_cv=self.opt_cv)

        self.model.fit(data_x, self.data_y)
        self.data_x = data_x if overwrite_training else self.data_x

        return
        
    def save(self, dirname=None, path=None, overwrite=False):
        """
        Saves the trained classifier in a new directory named 'MicroLIA_ensemble_model', 
        as well as the imputer and the features to use attributes, if not None.
        
        Args:
            dirname (str): The name of the directory where the model folder will be saved.
                This directory will be created, and therefore if it already exists
                in the system an error will appear.
            path (str): Absolute path where the data folder will be saved
                Defaults to None, in which case the directory is saved to the
                local home directory.
            overwrite (bool, optional): If True the 'MicroLIA_ensemble_model' folder this
                function creates in the specified path will be deleted if it exists
                and created anew to avoid duplicate files. 
        """

        if self.model is None and self.imputer is None and self.feats_to_use is None:
            raise ValueError('The models have not been created! Run the create() method first.')

        path = str(Path.home()) if path is None else path 
        path = path + '/' if path[-1] != '/' else path 
        
        if dirname is not None:
            dirname = dirname + '/' if dirname[-1] != '/' else dirname
            path = path + dirname
            try:
                os.makedirs(path)
            except FileExistsError:
                raise ValueError('The dirname folder already exists!')

        try:
            os.mkdir(path + 'MicroLIA_ensemble_model')
        except FileExistsError:
            if overwrite:
                try:
                    os.rmdir(path+'MicroLIA_ensemble_model')
                except OSError:
                    for file in os.listdir(path+'MicroLIA_ensemble_model'):
                        os.remove(path+'MicroLIA_ensemble_model/'+file)
                    os.rmdir(path+'MicroLIA_ensemble_model')
                os.mkdir(path+'MicroLIA_ensemble_model')
            else:
                raise ValueError('Tried to create "MicroLIA_ensemble_model" directory in specified path but folder already exists! If you wish to overwrite set overwrite=True.')
        
        path += 'MicroLIA_ensemble_model/'
        if self.model is not None:
            joblib.dump(self.model, path+'Model')
        if self.imputer is not None:
            joblib.dump(self.imputer, path+'Imputer')
        if self.feats_to_use is not None:
            joblib.dump(self.feats_to_use, path+'Feats_Index')
        if self.optimization_results is not None:
            joblib.dump(self.optimization_results, path+'HyperOpt_Results')
        if self.best_params is not None:
            joblib.dump(self.best_params, path+'Best_Params')
        if self.feature_history is not None:
            joblib.dump(self.feature_history, path+'FeatureOpt_Results')

        print('Files saved in: {}'.format(path))

        self.path = path

        return 

    def load(self, path=None):
        """ 
        Loads the model, imputer, and feats to use, if created and saved.
        This function will look for a folder named 'MicroLIA_models' in the
        local home directory, unless a path argument is set. 

        Args:
            path (str): Path where the directory 'MicroLIA_models' is saved. 
                Defaults to None, in which case the folder is assumed to be in the 
                local home directory.
        """

        path = str(Path.home()) if path is None else path 
        path = path+'/' if path[-1] != '/' else path 
        path += 'MicroLIA_ensemble_model/'

        try:
            self.model = joblib.load(path+'Model')
            model = 'model'
        except FileNotFoundError:
            model = ''
            pass

        try:
            self.imputer = joblib.load(path+'Imputer')
            imputer = ', imputer'
        except FileNotFoundError:
            imputer = ''
            pass 

        try:
            self.feats_to_use = joblib.load(path+'Feats_Index')
            feats_to_use = ', feats_to_use'
        except FileNotFoundError:
            feats_to_use = ''
            pass

        try:
            self.best_params = joblib.load(path+'Best_Params')
            best_params = ', best_params'
        except FileNotFoundError:
            best_params = ''
            pass

        try:
            self.feature_history = joblib.load(path+'FeatureOpt_Results')
            feature_opt_results = ', feature_selection_results'
        except FileNotFoundError:
            feature_opt_results = ''
            pass

        try:
            self.optimization_results = joblib.load(path+'HyperOpt_Results')
            optimization_results = ', optimization_results'
        except FileNotFoundError:
            optimization_results = '' 
            pass

        print('Successfully loaded the following class attributes: {}{}{}{}{}{}'.format(model, imputer, feats_to_use, best_params, feature_opt_results, optimization_results))
        
        self.path = path

        return

    def predict(self, time, mag, magerr, convert=True, apply_weights=True, zp=24):
        """
        Predics the class label of new, unseen data.

        Args:
            time (ndarray): Array of observation timestamps.
            mag (ndarray): Array of observed magnitudes.
            magerr (ndarray): Array of corresponding magnitude errors.
            convert (bool): If False the features are computed with the input magnitudes.
                Defaults to True to convert and compute in flux. 
            apply_weights (bool): Whether to apply the photometric errors when calculating the features. 
                Defaults to True. Note that this assumes that the erros are Gaussian and uncorrelated. 
            zp (float): Zeropoint of the instrument, used to convert from magnitude
                to flux. Defaults to 24.

        Returns:
            Array containing the classes and the corresponding probability predictions.
        """

        if len(mag) < 30:
            warn('The number of data points is low -- results may be unstable!')

        #classes = ['CONSTANT', 'CV', 'LPV', 'ML', 'VARIABLE']
        classes = self.model.classes_
        stat_array=[]
        
        if self.imputer is None and self.feats_to_use is None:
            stat_array.append(extract_features.extract_all(time, mag, magerr, convert=convert, apply_weights=apply_weights, zp=zp))
            pred = self.model.predict_proba(stat_array)
            return np.c_[classes, pred[0]]
        
        stat_array.append(extract_features.extract_all(time, mag, magerr, convert=convert, apply_weights=apply_weights, zp=zp, feats_to_use=self.feats_to_use))        
        stat_array = self.imputer.transform(stat_array) if self.imputer is not None else stat_array

        pred = self.model.predict_proba(stat_array)

        return np.c_[classes, pred[0]]

    def plot_tsne(self, data_y=None, special_class=None, norm=True, pca=False, 
        legend_loc='upper center', title='Feature Parameter Space', savefig=False):
        """
        Plots a t-SNE projection using the sklearn.manifold.TSNE() method.

        Note:
            To highlight individual samples, use the data_y optional input
            and set that sample's data_y value to a unique name, and set that 
            same label in the special_class variable so that it can be highlighted 
            clearly in the plot.

        Args:
            data_y (ndarray, optional): A custom labels array, that coincides with
                the labels in model.data_y. Defaults to None, in which case the
                model.data_y labels are used.
            special_class (optional): The class label that you wish to highlight,
                setting this optional parameter will increase the size and alpha parameter
                for these points in the plot.
            norm (bool): If True the data will be min-max normalized. Defaults
                to True.
            pca (bool): If True the data will be fit to a Principal Component
                Analysis and all of the corresponding principal components will 
                be used to generate the t-SNE plot. Defaults to False.
            legend_loc (str): Location of legend, using matplotlib style.
            title (str): Title of the figure.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False. 

        Returns:
            AxesImage. 
        """

        if self.feats_to_use is not None:
            data = self.data_x[self.feats_to_use].reshape(1,-1) if len(self.data_x.shape) == 1 else self.data_x[:,self.feats_to_use] 
        else:
            data = copy.deepcopy(self.data_x)

        if np.any(np.isnan(data)):
            data = impute_missing_values(data, self.imputer) if self.imputer is not None else impute_missing_values(data, strategy=self.imp_method)[0]
            
        data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
        
        method = 'barnes_hut' if len(data) > 5e3 else 'exact' #bh Scales with O(N), exact scales with O(N^2)
        
        if norm:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)

        if pca:
            pca_transformation = decomposition.PCA(n_components=data.shape[1], whiten=True, svd_solver='auto')
            pca_transformation.fit(data) 
            data = pca_transformation.transform(data)

        feats = TSNE(n_components=2, method=method, learning_rate=1000, perplexity=35, init='random').fit_transform(data)
        x, y = feats[:,0], feats[:,1]
     
        markers = ['o', 's', '+', 'v', '.', 'x', 'h', 'p', '<', '>', '*']
        #color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c']
        color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#e41a1c', '#377eb8'] #Update the last two!

        if data_y is None:
            if self.data_y_ is None:
                if self.csv_file is None:
                    if data_y is None:
                        data_y = self.data_y
                        feats = np.unique(self.data_y)
                    else:
                        if isinstance(data_y, np.ndarray) is False: 
                            if type(data_y) == list:
                                data_y = np.array(data_y)
                            else:
                                raise ValueError('data_y argument must either be a list or an array!')
                        feats = np.unique(data_y)
                else:
                    if len(self.csv_file) == len(self.data_y):
                        data_y = np.array(self.csv_file.label)
                        feats = np.unique(data_y)
                    else:
                        data_y = self.data_y
                        feats = np.unique(self.data_y)
            else:
                if len(self.data_y_) == len(self.data_y):
                    data_y = self.data_y_ 
                    feats = np.unique(self.data_y_)
                else:
                    data_y = self.data_y
                    feats = np.unique(self.data_y)
        else:
            if isinstance(data_y, list):
                data_y = np.array(data_y)
            feats = np.unique(data_y) 

        for count, feat in enumerate(feats):
            if count+1 > len(markers):
                count = -1
            mask = np.where(data_y == feat)[0]
            if feat == special_class:
                pass
            else:
                plt.scatter(x[mask], y[mask], marker=markers[count], c=color[count], label=str(feat), alpha=0.44)

        if special_class is not None:
            mask = np.where(data_y == special_class)[0]
            if len(mask) == 0:
                raise ValueError('The data_y array does not contain the value input in the special_class parameter.')
            plt.scatter(x[mask], y[mask], marker='*', c='red', label=special_class, s=200, alpha=1.0)
        
        plt.legend(loc=legend_loc, ncol=len(np.unique(data_y)), frameon=False, handlelength=2)
        plt.title(title); plt.ylabel('t-SNE Dimension 1'); plt.xlabel('t-SNE Dimension 2')
        plt.xticks(); plt.yticks()

        if savefig:
            _set_style_()
            plt.savefig('tSNE_Projection.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        return

    def plot_conf_matrix(self, data_y=None, norm=False, pca=False, k_fold=10, normalize=True, 
        title='Confusion Matrix', savefig=False):
        """
        Returns a confusion matrix with k-fold validation.

        Args:
            data_y (ndarray, str, optional): 1D array containing the corresponing labels.
                Only use if using XGB algorithm as this method converts labels to numerical,
                in which case it may be desired to input the original label array using
                this parameter. Defaults to None, which uses the data_y attribute.
            norm (bool): If True the data will be min-max normalized. Defaults
                to False. NOTE: Set this to True if pca=True.
            pca (bool): If True the data will be fit to a Principal Component
                Analysis and all of the corresponding principal components will 
                be used to evaluate the classifier and construct the matrix. 
                Defaults to False.
            k_fold (int, optional): The number of cross-validations to perform.
                The output confusion matrix will display the mean accuracy across
                all k_fold iterations. Defaults to 10.
            normalize (bool, optional): If False the confusion matrix will display the
                total number of objects in the sample. Defaults to True, in which case
                the values are normalized between 0 and 1.
            title (str): Title of the figure.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False. 

        Returns:
            AxesImage.
        """

        if self.data_x is None or self.data_y is None:
            raise ValueError('The data_x and data_y have not been input!')

        if self.model is None:
            raise ValueError('No model has been created! Run .create() first.')

        if data_y is not None:
            classes = [str(label) for label in np.unique(data_y)]
        else:
            if self.data_y_ is None:
                if self.csv_file is None:
                    classes = [str(label) for label in np.unique(self.data_y)]
                else:
                    classes = [str(label) for label in np.unique(np.array(self.csv_file.label))]
            else:
                classes = [str(label) for label in np.unique(self.data_y_)]

        if self.feats_to_use is not None:
            if len(self.data_x.shape) == 1:
                data = self.data_x[self.feats_to_use].reshape(1,-1)
            else:
                data = self.data_x[:,self.feats_to_use]
        else:
            data = copy.deepcopy(self.data_x)

        if np.any(np.isnan(data)):
            data = impute_missing_values(data, self.imputer) if self.imputer is not None else impute_missing_values(data, strategy=self.imp_method)[0]
          
        data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7

        if norm:
            scaler = MinMaxScaler()
            scaler.fit_transform(data)

        if pca:
            pca_transformation = decomposition.PCA(n_components=data.shape[1], whiten=True, svd_solver='auto')
            pca_transformation.fit(data) 
            pca_data = pca_transformation.transform(data)
            data = np.asarray(pca_data).astype('float64')

        predicted_target, actual_target = evaluate_model(self.model, data, self.data_y, normalize=normalize, k_fold=k_fold)
        generate_matrix(predicted_target, actual_target, normalize=normalize, classes=classes, title=title, savefig=savefig)

    def plot_roc_curve(self, k_fold=10, pca=False, title="Receiver Operating Characteristic Curve", 
        savefig=False):
        """
        Plots ROC curve with k-fold cross-validation, as such the 
        standard deviation variations are also plotted.
        
        Args:
            k_fold (int, optional): The number of cross-validations to perform.
                The output confusion matrix will display the mean accuracy across
                all k_fold iterations. Defaults to 10.
            pca (bool): If True the data will be fit to a Principal Component
                Analysis and all of the corresponding principal components will 
                be used to evaluate the classifier and construct the matrix. 
                Defaults to False.
            title (str, optional): The title of the output plot.
            savefig (bool): If True the figure will not disply but will be saved instead. Defaults to False. 
            
        Returns:
            AxesImage
        """

        if self.model is None:
            raise ValueError('No model has been created! Run model.create() first.')

        if self.feats_to_use is not None:
            data = self.data_x[self.feats_to_use].reshape(1,-1) if len(self.data_x.shape) == 1 else self.data_x[:,self.feats_to_use]
        else:
            data = copy.deepcopy(self.data_x)

        if np.any(np.isnan(data)):
            data = impute_missing_values(data, self.imputer) if self.imputer is not None else impute_missing_values(data, strategy=self.imp_method)[0]
          
        data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7

        if pca:
            pca_transformation = decomposition.PCA(n_components=data.shape[1], whiten=True, svd_solver='auto')
            pca_transformation.fit(data) 
            pca_data = pca_transformation.transform(data)
            data = np.asarray(pca_data).astype('float64')
        
        model0 = copy.deepcopy(self.model)

        if len(np.unique(self.data_y)) != 2:
            test_size = 1. / k_fold
            X_train, X_test, y_train, y_test = train_test_split(data, self.data_y, test_size=test_size, random_state=0)
            model0.fit(X_train, y_train)
            y_probas = model0.predict_proba(X_test)
            plot_roc(y_test, y_probas, text_fontsize='large', title='ROC Curve', cmap='nipy_spectral', plot_macro=False, plot_micro=False)
            
            if savefig:
                _set_style_()
                plt.savefig('Ensemble_ROC_Curve.png', bbox_inches='tight', dpi=300)
                plt.clf(); plt.style.use('default')
            else:
                plt.show()

            return 
            
        cv = StratifiedKFold(n_splits=k_fold)
        
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()

        for i, (data_x, test) in enumerate(cv.split(data, self.data_y)):
            model0.fit(data[data_x], self.data_y[data_x])
            viz = RocCurveDisplay.from_estimator(model0, data[test], self.data_y[test], alpha=0, lw=1, ax=ax, name="ROC fold {}".format(i+1))
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr); aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc, std_auc = auc(mean_fpr, mean_tpr), np.std(aucs)
        lns1, = ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean (AUC = %0.2f)" % (mean_auc), lw=2, alpha=0.8) #label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),

        std_tpr = np.std(tprs, axis=0)
        tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)
        lns_sigma = ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1$\sigma$")

        ax.set(xlim=[0, 1.0], ylim=[0.0, 1.0], title="Receiver Operating Characteristic Curve")
        lns2, = ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random (AUC=0.5)", alpha=0.8)

        ax.legend([lns2, (lns1, lns_sigma)], ['Random (AUC = 0.5)', r"Mean (AUC = %0.2f)" % (mean_auc)], loc='lower center', ncol=2, frameon=False, handlelength=2)
        plt.title(label=title); plt.ylabel('True Positive Rate'); plt.xlabel('False Positive Rate')
        ax.set_facecolor("white")

        if savefig:
            _set_style_()
            plt.savefig('Ensemble_ROC_Curve.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        return

    def plot_hyper_opt(self, baseline=None, xlim=None, ylim=None, xlog=True, ylog=False, savefig=False):
        """
        Plots the hyperparameter optimization history.

        Note:
            The Optuna API has its own plot function: plot_optimization_history(self.optimization_results)
    
        Args:
            baseline (float): Baseline accuracy achieved when using only
                the default engine hyperparameters. If input a vertical
                line will be plot to indicate this baseline accuracy.
                Defaults to None.
            xlim (tuple): Limits for the x-axis, e.g. xlim = (0, 1000)
            ylim (tuple): Limits for the y-axis. e.g. ylim = (0.9, 0.94)
            xlog (bool): If True the x-axis will be log-scaled. Defaults to True.
            ylog (bool): If True the y-axis will be log-scaled. Defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False. 

        Returns:
            AxesImage
        """

        trials = self.optimization_results.get_trials()
        trial_values, best_value = [], []
        for trial in range(len(trials)):
            value = trials[trial].values[0]
            trial_values.append(value)
            if trial == 0:
                best_value.append(value)
            else:
                if any(y > value for y in best_value): #If there are any numbers in best values that are higher than current one
                    best_value.append(np.array(best_value)[trial-1])
                else:
                    best_value.append(value)

        best_value, trial_values = np.array(best_value), np.array(trial_values)
        best_value[1] = trial_values[1] #Make the first trial the best model, since technically it is.
        for i in range(2, len(trial_values)):
            if trial_values[i] < best_value[1]:
                best_value[i] = best_value[1]
            else:
                break

        if baseline is not None:
            plt.axhline(y=baseline, color='k', linestyle='--', label='Baseline Model')
            ncol=3
        else:
            ncol=2

        plt.plot(range(len(trials)), best_value, color='r', alpha=0.83, linestyle='-', label='Optimized Model')
        plt.scatter(range(len(trials)), trial_values, c='b', marker='+', s=35, alpha=0.45, label='Trial')
        plt.xlabel('Trial #', alpha=1, color='k')

        if self.opt_cv > 0:
            plt.ylabel(str(self.opt_cv)+'-Fold CV Accuracy', alpha=1, color='k')
        else:
            plt.ylabel('Accuracy', alpha=1, color='k')
        
        if self.clf == 'xgb':
            plt.title('XGBoost Hyperparameter Optimization')
        elif self.clf == 'rf':
            plt.title('RF Hyperparameter Optimization')
        elif self.clf == 'ocsvm':
            plt.title('OneClass SVM Hyperparameter Optimization')
        elif self.clf == 'nn':
            plt.title('Neural Network Hyperparameter Optimization')

        plt.legend(loc='upper center', ncol=ncol, frameon=False)
        plt.rcParams['axes.facecolor']='white'
        plt.grid(False)

        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim((1, len(trials)))
        if ylim is not None:
            plt.ylim(ylim)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        
        if savefig:
            _set_style_()
            plt.savefig('Ensemble_Hyperparameter_Optimization.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        return

    def plot_feature_opt(self, feat_names=None, top='all', include_other=True, include_shadow=True, 
        include_rejected=False, flip_axes=True, save_data=False, savefig=False):
        """
        Returns whisker plot displaying the z-score distribution of each feature
        across all trials.
    
        Note:
            The following can be used to output the plot from the original BorutaShap API.

            model.feature_history.plot(which_features='accepted', X_size=14)

            Can designate to display either 'all', 'accepted', or 'tentative'

        Args: 
            feat_names (ndarry, optional): A list or array containing the names
                of the features in the data_x matrix, in order. Defaults to None,
                in which case the respective indices will appear instead. Can be set to
                'default' which will the features in MicroLIA.features module, so only to be used
                if all the features were extracted using MicroLIA.extract_features.
            top (float, optional): Designates how many features to plot. If set to 3, it 
                will plot the top 3 performing features. Can be 'all' in which casee all features
                that were accepted are plotted. Defaults to 'all'.
            include_other (bool): Whether to include the features that are not in the top designation,
                if True these features will be averaged out and displayed. Defaults to True.
            include_shadow (bool): Whether to include the mean shadow feature that was used as a 
                baseline for 'random' behavior. Defaults to True.
            include_rejected (bool): Whether to include the rejected features, if False these features 
                will not be shown. If set to True or 'all', all the rejected features will show. If set to
                a number, only the designated top rejected features will show. Defaults to False.
            flip_axes (bool): Whether transpose the figure. Defaults to True.
            save_data (bool): Whether to save the feature importances as a csv file, defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False. 

        Returns:
            AxesImage
        """

        if feat_names is None and self.csv_file is not None:
            feat_names = self.csv_file.columns[:-1]

        if feat_names == 'default':
            feat_names = ['Anderson-Darling', 'FluxPctRatioMid20', 'FluxPctRatioMid35', 'FluxPctRatioMid50', 'FluxPctRatioMid65', 'FluxPctRatioMid80', 
                'Median-Based Skew', 'Linear Trend', 'Max Slope', 'Pair Slope Trend', 'Percent Amp.', 'Percent DiffFluxPct', 'Above 1', 'Above 3', 
                'Above 5', 'Abs. Energy', 'Abs. Sum Changes', 'Amplitude', 'Autocorrelation', 'Below 1', 'Below 3', 'Below 5', 'Benford Correlation', 
                'C3 Non-Linearity', 'Dup. Val. Check', 'Max Val. Dup. Check', 'Min. Val. Dup. Check', 'Max. Last Loc. Check', 'Min. Last Loc. Check', 
                'Complexity', 'Consec. Cluster Count', 'Num. Points Above', 'Num. Points Below', 'Cumulative Sum', 'First Loc. Max', 'First Loc. Min', 
                'Half Mag. Amp. Ratio', 'Mass Quant. Index', 'Integration', 'Kurtosis', 'Large Std. Dev.', 'Longest Strike Above', 'Longest Strike Below', 
                'Mean Magnitude', 'Mean Abs. Change', 'Mean Change', 'Mean of Abs. Maxima', 'Mean Second Deriv.', 'Median Abs. Dev.', 'Median Buffer Range', 
                'Median Distance', 'Num. CWT Peaks', 'Num. Crossings', 'Num. Peaks', 'Peaks Detection', 'Permutation Entropy', 'Quantile', 'Recurring Pts. Ratio', 
                'Root Mean Squared', 'Sample Entropy', 'Shannon Entropy', 'Shapiro-Wilk', 'Skewness', 'Std. Dev. over Mean', 'Stetson J', 'Stetson K', 'Stetson L', 
                'Sum of Vals.', 'Symmetry Looking', 'Time Reversal Asym.', 'Variance', 'Var. Exceeds Std. Dev.', 'Variation Coeff.', 'vonNeumannRatio', 
                'Deriv-Anderson-Darling', 'Deriv-FluxPctRatioMid20', 'Deriv-FluxPctRatioMid35', 'Deriv-FluxPctRatioMid50', 'Deriv-FluxPctRatioMid65', 'Deriv-FluxPctRatioMid80', 
                'Deriv-Median-Based Skew', 'Deriv-Linear Trend', 'Deriv-Max Slope', 'Deriv-Pair Slope Trend', 'Deriv-Percent Amp.', 'Deriv-Percent DiffFluxPct', 'Deriv-Above 1', 'Deriv-Above 3', 
                'Deriv-Above 5', 'Deriv-Abs. Energy', 'Deriv-Abs. Sum Changes', 'Deriv-Amplitude', 'Deriv-Autocorrelation', 'Deriv-Below 1', 'Deriv-Below 3', 'Deriv-Below 5', 'Deriv-Benford Correlation', 
                'Deriv-C3 Non-Linearity', 'Deriv-Dup. Val. Check', 'Deriv-Max Val. Dup. Check', 'Deriv-Min. Val. Dup. Check', 'Deriv-Max. Last Loc. Check', 'Deriv-Min. Last Loc. Check', 
                'Deriv-Complexity', 'Deriv-Consec. Cluster Count', 'Deriv-Num. Points Above', 'Deriv-Num. Points Below', 'Deriv-Cumulative Sum', 'Deriv-First Loc. Max', 'Deriv-First Loc. Min', 
                'Deriv-Half Mag. Amp. Ratio', 'Deriv-Mass Quant. Index', 'Deriv-Integration', 'Deriv-Kurtosis', 'Deriv-Large Std. Dev.', 'Deriv-Longest Strike Above', 'Deriv-Longest Strike Below', 
                'Deriv-Mean Magnitude', 'Deriv-Mean Abs. Change', 'Deriv-Mean Change', 'Deriv-Mean of Abs. Maxima', 'Deriv-Mean Second Deriv.', 'Deriv-Median Abs. Dev.', 'Deriv-Median Buffer Range', 
                'Deriv-Median Distance', 'Deriv-Num. CWT Peaks', 'Deriv-Num. Crossings', 'Deriv-Num. Peaks', 'Deriv-Peaks Detection', 'Deriv-Permutation Entropy', 'Deriv-Quantile', 'Deriv-Recurring Pts. Ratio', 
                'Deriv-Root Mean Squared', 'Deriv-Sample Entropy', 'Deriv-Shannon Entropy', 'Deriv-Shapiro-Wilk', 'Deriv-Skewness', 'Deriv-Std. Dev. over Mean', 'Deriv-Stetson J', 'Deriv-Stetson K', 'Deriv-Stetson L', 
                'Deriv-Sum of Vals.', 'Deriv-Symmetry Looking', 'Deriv-Time Reversal Asym.', 'Deriv-Variance', 'Deriv-Var. Exceeds Std. Dev.', 'Deriv-Variation Coeff.', 'Deriv-vonNeumannRatio']

        fname = str(Path.home()) + '/__borutaimportances__' #Temporary file

        try:
            self.feature_history.results_to_csv(filename=fname)
        except AttributeError:
            raise ValueError('No optimization history found for feature selection, run .create() with optimize=True!')

        csv_data = pd.read_csv(fname+'.csv')
        if save_data is False:
            os.remove(fname+'.csv')

        accepted_indices = np.where(csv_data.Decision == 'Accepted')[0]
        if top == 'all':
            top = len(accepted_indices)
        else:
            if top > len(accepted_indices):
                top = len(accepted_indices)
                print('The top parameter exceeds the number of accepted variables, setting to the maximum value of {}'.format(str(top)))

        x, y, y_err = [], [], []

        for i in accepted_indices[:top]:
            x.append(int(csv_data.iloc[i].Features))
            y.append(float(csv_data.iloc[i]['Average Feature Importance']))
            y_err.append(float(csv_data.iloc[i]['Standard Deviation Importance']))

        include_other = False if len(accepted_indices) == top else include_other

        if include_other:
            mean, std = [], []
            for i in accepted_indices[top:]:
                mean.append(float(csv_data.iloc[i]['Average Feature Importance']))
                std.append(float(csv_data.iloc[i]['Standard Deviation Importance']))
            x.append(0), y.append(np.mean(mean)), y_err.append(np.mean(std))

        if include_shadow:
            ix = np.where(csv_data.Features == 'Mean_Shadow')[0]
            y.append(float(csv_data.iloc[ix]['Average Feature Importance']))
            y_err.append(float(csv_data.iloc[ix]['Standard Deviation Importance']))
            x.append(0) #Just a placeholder

        if feat_names is not None:  
            feat_names = np.array(feat_names) if isinstance(feat_names, np.ndarray) is False else feat_names
            if include_shadow is False:
                x_names = feat_names[x] if include_other is False else np.r_[feat_names[x[:-1]], ['Other Accepted']] #By default x is the index of the feature
            else:
                x_names = np.r_[feat_names[x[:-1]], ['Mean Shadow']] if include_other is False else np.r_[feat_names[x[:-2]], ['Other Accepted'], ['Mean Shadow']]
        else:
            if include_other is False:
                x_names = csv_data.iloc[x].Features if include_shadow is False else np.r_[csv_data.iloc[x[:-1]].Features, ['Mean Shadow']]
            else:
                x_names = np.r_[csv_data.iloc[x[:-1]].Features, ['Other Accepted']] if include_shadow is False else np.r_[csv_data.iloc[x[:-2]].Features, ['Other Accepted'], ['Mean Shadow']]
        

        if include_rejected is not False:
            x = []
            rejected_indices = np.where(csv_data.Decision == 'Rejected')[0]
            if include_rejected == 'all' or include_rejected == True:
                for i in rejected_indices:
                    x.append(int(csv_data.iloc[i].Features))
                    y.append(float(csv_data.iloc[i]['Average Feature Importance']))
                    y_err.append(float(csv_data.iloc[i]['Standard Deviation Importance']))
            
            else:
                if include_rejected > len(rejected_indices):
                    include_rejected = len(rejected_indices)
                    print('The include_rejected parameter exceeds the number of rejected features, setting to the maximum value of {}'.format(str(include_rejected)))

                for i in rejected_indices[:include_rejected]:
                    x.append(int(csv_data.iloc[i].Features))
                    y.append(float(csv_data.iloc[i]['Average Feature Importance']))
                    y_err.append(float(csv_data.iloc[i]['Standard Deviation Importance']))
            
            rejected_names = csv_data.iloc[x].Features if feat_names is None else feat_names[x]
            x_names = np.r_[x_names, rejected_names] if feat_names is None else np.r_[x_names, rejected_names]

        
        y, y_err = np.array(y), np.array(y_err)

        fig, ax = plt.subplots()
        if flip_axes:
            lns, = ax.plot(y, np.arange(len(x_names)), 'k*--', lw=0.77)
            lns_sigma = ax.fill_betweenx(np.arange(len(x_names)), y-y_err, y+y_err, color="grey", alpha=0.2)
            ax.set_xlabel('Z Score', alpha=1, color='k'); ax.set_yticks(np.arange(len(x_names)), x_names)#, rotation=90)
            
            for t in ax.get_yticklabels():
                txt = t.get_text()
                if 'Mean Shadow' in txt:
                    t.set_color('red')
                    if include_rejected is False:
                        idx = 1
                    elif include_rejected == 'all' or include_rejected == True:
                        idx = 1 + len(rejected_indices)
                    else:
                        idx = 1 + len(rejected_indices[:include_rejected])
                    ax.plot(y[-idx], np.arange(len(x_names))[-idx], marker='*', color='red')

            ax.set_ylim((np.arange(len(x_names))[0]-0.5, np.arange(len(x_names))[-1]+0.5))
            ax.set_xlim((np.min(y)-1, np.max(y)+1))
            ax.invert_yaxis(); ax.invert_xaxis()
        else:
            lns, = ax.plot(np.arange(len(x_names)), y, 'k*--', lw=0.77)#, label='XGBoost', lw=0.77)
            lns_sigma = ax.fill_between(np.arange(len(x_names)), y-y_err, y+y_err, color="grey", alpha=0.2)
            ax.set_ylabel('Z Score', alpha=1, color='k'); ax.set_xticks(np.arange(len(x_names)), x_names, rotation=90)
            for t in ax.get_xticklabels():
                txt = t.get_text()
                if 'Mean Shadow' in txt:
                    t.set_color('red')
                    if include_rejected is False:
                        idx = 1
                    elif include_rejected == 'all' or include_rejected == True:
                        idx = 1 + len(rejected_indices)
                    else:
                        idx = 1 + len(rejected_indices[:include_rejected])
                    ax.plot(np.arange(len(x_names))[-idx], y[-idx], marker='*', color='red')

            ax.set_xlim((np.arange(len(x_names))[0]-0.5, np.arange(len(x_names))[-1]+0.5))
            ax.set_ylim((np.min(y)-1, np.max(y)+1))

        ax.legend([(lns, lns_sigma)], [r'$\pm$ 1$\sigma$'], loc='upper right', ncol=1, frameon=False, handlelength=2)
        ax.set_title('Feature Importance')

        if savefig:
            _set_style_()
            plt.savefig('Feature_Importance.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        return

    def plot_hyper_param_importance(self, plot_time=True, savefig=False):
        """
        Plots the hyperparameter optimization history.
    
        Note:
            The Optuna API provides its own plotting function: plot_param_importances(self.optimization_results)

        Args:
            plot_tile (bool): If True, the importance on the duration will also be included. Defaults to True.
            savefig (bool): If True the figure will not disply but will be saved instead. Defaults to False. 

        Returns:
            AxesImage
        """

        try:
            if isinstance(self.path, str):
                try:
                    hyper_importances = joblib.load(self.path+'Hyperparameter_Importance')
                except FileNotFoundError:
                    raise ValueError('Could not find the importance file in the '+self.path+' folder')

                try:
                    duration_importances = joblib.load(self.path+'Duration_Importance')
                except FileNotFoundError:
                    raise ValueError('Could not find the importance file in the '+self.path+' folder')
            else:
                raise ValueError('Call the save_hyper_importance() attribute first.')
        except:
            raise ValueError('Call the save_hyper_importance() attribute first.')

        params, importance, duration_importance = [], [], []
        for key in hyper_importances:       
            params.append(key)

        for name in params:
            importance.append(hyper_importances[name])
            duration_importance.append(duration_importances[name])

        xtick_labels = format_labels(params)

        fig, ax = plt.subplots()
        ax.barh(xtick_labels, importance, label='Importance for Classification', color=mcolors.TABLEAU_COLORS["tab:blue"], alpha=0.87)
        if plot_time:
            ax.barh(xtick_labels, duration_importance, label='Impact on Engine Speed', color=mcolors.TABLEAU_COLORS["tab:orange"], alpha=0.7, hatch='/')

        ax.set_ylabel("Hyperparameter"); ax.set_xlabel("Importance Evaluation")
        ax.legend(ncol=2, frameon=False, handlelength=2, bbox_to_anchor=(0.5, 1.1), loc='upper center')
        ax.set_xscale('log'); plt.xlim((0, 1.))
        plt.gca().invert_yaxis()

        if savefig:
            _set_style_()
            if plot_time:
                plt.savefig('Ensemble_Hyperparameter_Importance.png', bbox_inches='tight', dpi=300)
            else:
                plt.savefig('Ensemble_Hyperparameter_Duration_Importance.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        return

    def save_hyper_importance(self):
        """
        Calculates and saves binary files containing
        dictionaries with importance information, one
        for the importance and one for the duration importance

        Note:
            This procedure is time-consuming but must be run once before
            plotting the importances. This function will save
            two files in the model folder for future use. 

        Returns:
            Saves two binary files, importance and duration importance.
        """
        
        if self.n_iter >= 500:
            print('Calculating and saving importances, this could take up to an hour...')

        try:
            path = self.path if isinstance(self.path, str) else str(Path.home())
        except:
            path = str(Path.home())

        hyper_importance = get_param_importances(self.optimization_results)
        joblib.dump(hyper_importance, path+'Hyperparameter_Importance')

        importance = FanovaImportanceEvaluator()
        duration_importance = importance.evaluate(self.optimization_results, target=lambda t: t.duration.total_seconds())
        joblib.dump(duration_importance, path+'Duration_Importance')
        
        print(f"Files saved in: {path}")

        self.path = path

        return  

#Helper functions below to generate confusion matrix
def format_labels(labels: list) -> list:
    """
    Takes a list of labels and returns the list with all words capitalized and underscores removed.
    Also replaces 'eta' with 'Learning Rate' and 'n_estimators' with 'Number of Trees'.
    
    Args:
        labels (list): A list of strings.
    
    Returns:
        Reformatted list, of same lenght.
    """

    new_labels = []
    for label in labels:
        label = label.replace("_", " ")
        if label == "eta":
            new_labels.append("Learning Rate"); continue
        if label == "n estimators":
            new_labels.append("Num of Trees"); continue
        if label == "colsample bytree":
            new_labels.append("ColSample ByTree"); continue
        new_labels.append(label.title())

    return new_labels

def evaluate_model(classifier, data_x, data_y, normalize=True, k_fold=10):
    """
    Cross-checks model accuracy and outputs both the predicted
    and the true class labels. 

    Args:
        classifier: The machine learning classifier to optimize.
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m is the number of features.
        data_y (ndarray, str): 1D array containing the corresponding labels.
        normalize (bool, optional): If False, the confusion matrix will display the
            total number of objects in the sample. Defaults to True, in which case
            the values are normalized between 0 and 1. 
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.

    Returns:
        The first output is the 1D array of the true class labels.
        The second output is the 1D array of the predicted class labels.
    """

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    predicted_targets = []
    actual_targets = []

    for train_index, test_index in kf.split(data_x):
        classifier.fit(data_x[train_index], data_y[train_index])
        predicted_targets.extend(classifier.predict(data_x[test_index]))
        actual_targets.extend(data_y[test_index])

    predicted_targets = np.array(predicted_targets)
    actual_targets = np.array(actual_targets)

    return predicted_targets, actual_targets


def generate_matrix(predicted_labels_list, actual_targets, classes, normalize=True, 
    title='Confusion Matrix', savefig=False):
    """
    Generates the confusion matrix using the output from the evaluate_model() function.

    Args:
        predicted_labels_list: 1D array containing the predicted class labels.
        actual_targets: 1D array containing the actual class labels.
        classes (list): A list containing the label of the two training bags. This
            will be used to set the axis. Ex) classes = ['ML', 'OTHER']
        normalize (bool, optional): If True the matrix accuracy will be normalized
            and displayed as a percentage accuracy. Defaults to True.
        title (str, optional): The title of the output plot. 
        savefig (bool): If True the figure will not disply but will be saved instead. Defaults to False. 

    Returns:
        AxesImage.
    """

    conf_matrix = confusion_matrix(actual_targets, predicted_labels_list)
    np.set_printoptions(precision=2)

    plt.figure()
    if normalize:
        generate_plot(conf_matrix, classes=classes, normalize=normalize, title=title)
    else:
        generate_plot(conf_matrix, classes=classes, normalize=normalize, title=title)
    
    if savefig:
        _set_style_()
        plt.savefig('Ensemble_Confusion_Matrix.png', bbox_inches='tight', dpi=300)
        plt.clf(); plt.style.use('default')
    else:
        plt.show()
    
def generate_plot(conf_matrix, classes, normalize=False, title='Confusion Matrix'):
    """
    Generates the confusion matrix figure object, but does not plot.
    
    Args:
        conf_matrix: The confusion matrix generated using the generate_matrix() function.
        classes (list): A list containing the label of the two training bags. This
            will be used to set the axis. Defaults to a list containing 'ML' & 'OTHER'. 
        normalize (bool, optional): If True the matrix accuracy will be normalized
            and displayed as a percentage accuracy. Defaults to True.
        title (str, optional): The title of the output plot. 

    Returns:
        AxesImage object. 
    """

    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title); plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, alpha=1, color='k'); plt.yticks(tick_marks, classes, alpha=1, color='k', rotation=90)

    fmt = '.4f' if normalize is True else 'd'
    thresh = conf_matrix.max() / 2.

    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label', alpha=1, color='k'); plt.xlabel('Predicted label',alpha=1, color='k')
    plt.grid(False); plt.tight_layout()

    return conf_matrix

def min_max_norm(data_x):
    """
    Normalizes the data to be between 0 and 1. NaN values are ignored.
    The transformation matrix will be returned as it will be needed
    to consitently normalize new data.
    
    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the number of samples, and m the number of features.

    Returns:
        Normalized data array.
    """

    Ny, Nx = data_x.shape
    new_array = np.zeros((Ny, Nx))
    
    for i in range(Nx):
        new_array[:,i] = (data_x[:,i] - np.min(data_x[:,i])) / (np.max(data_x[:,i]) - np.min(data_x[:,i]))

    return new_array
    
def _set_style_():
    """
    Function to configure the matplotlib.pyplot style. This function is called before any images are saved,
    after which the style is reset to the default.
    """

    plt.rcParams["xtick.color"] = "323034"
    plt.rcParams["ytick.color"] = "323034"
    plt.rcParams["text.color"] = "323034"
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["patch.facecolor"] = "#bc80bd"  # Replace with a valid color code
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["patch.linewidth"] = 0.8
    plt.rcParams["scatter.edgecolors"] = "black"
    plt.rcParams["grid.color"] = "#b1afb5"  # Replace with a valid color code
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.title_fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.prop_cycle"] = (cycler('color', ['#bc80bd', '#fb8072', '#b3de69', '#fdb462', '#fccde5', '#8dd3c7', '#ffed6f', '#bebada', '#80b1d3', '#ccebc5', '#d9d9d9']))  # Replace with valid color codes
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.8
    plt.rcParams["legend.fontsize"] = 13
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.borderpad"] = 0.2
    plt.rcParams["legend.columnspacing"] = 1.5
    plt.rcParams["legend.labelspacing"] = 0.4
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.labelsize"] = 17
    plt.rcParams["axes.titlelocation"] = "center"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    plt.rcParams["axes.labelpad"] = 3
    plt.rcParams["axes.formatter.limits"] = (-4, 4)
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["figure.titlesize"] = 18
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300

    return