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
from warnings import warn
from pathlib import Path
from collections import Counter  

from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, auc, RocCurveDisplay
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.manifold import TSNE

from optuna.importance import get_param_importances, FanovaImportanceEvaluator
from MicroLIA.optimization import hyper_opt, borutashap_opt, KNN_imputation
from MicroLIA import extract_features
from xgboost import XGBClassifier
import scikitplot as skplt

class Classifier:
    """
    Creates a machine learning classifier object. The built-in methods can be used to optimize the engine and output visualizations.

    Note:
        test_size is an optional parameter to speed up the XGB optimization training.
        If input a random validation data will be generated according to this size,
        which will replace the cross-validation method used by default during the
        optimization procedure. Need more testing to make this more robust, recommended
        option is test_size = None. The opt_cv parameter should be used instead to set
        the number of folds to use when assessing optimization trial performance. 

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        clf (str): The machine learning classifier to optimize. Can either be
            'rf' for Random Forest, 'nn' for Neural Network, or 'xgb' for Extreme Gradient Boosting. 
            Defaults to 'rf'.
        optimize (bool): If True the Boruta algorithm will be run to identify the features
            that contain useful information, after which the optimal Random Forest hyperparameters
            will be calculated using Bayesian optimization. 
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. Defaults to 10.
            NOTE: The higher this number, the longer the optimization will take.
        test_size (float, optional): The size of the validation data, will be chosen
            randomly each trial. Must be between 0 and 1. Defaults to None, in which
            case the opt_cv parameter will be used instead.
        limit_search (bool): If True, the search space for the parameters will be expanded,
            as there are some hyperparameters that can range from 0 to inf. Defaults to False.
        impute (bool): If False no data imputation will be performed. Defaults to True,
            which will result in two outputs, the classifier and the imputer to save
            for future transformations. 
        imp_method (str): The imputation techinque to apply, can either be 'KNN' for k-nearest
            neighbors imputation, or...?. Defaults to 'KNN'.
        n_iter (int): The maximum number of iterations to perform during 
            the hyperparameter search. Defaults to 25. 
        boruta_trials (int): The number of trials to run when running Boruta for
            feature selection. Set to 0 for no feature selection. Defaults to 50.
        boruta_model (str): The ensemble to use when calculating the feature importance
            to be utilized by the Boruta algorithm. Can either be 'rf' or 'xgb'. Note
            that this does not have to be the same as the machine learning classifier, clf.
        balance (bool, optional): If True, a weights array will be calculated and used
            when fitting the classifier. This can improve classification when classes
            are imbalanced. This is only applied if the classification is a binary task. 
            Defaults to True.        
        csv_file (DataFrame, optional): The csv file output after generating the training set. This can be
            input in lieu of the data_x and data_y arguments.
        
    Attributes:
        model (object): The machine learning model that is created
        imputer (object): The imputer created during model creation
        feats_to_use (ndarray): Array of indices containing the metrics
            that contribute to classification accuracy.
        feature_history (object): The Boruta feature selection history result.
        optimization_results (object): The Optuna hyperparameter optimization history result.
        best_params (dict): The best parameters output from the optimization routine.
    """

    def __init__(self, data_x=None, data_y=None, clf='rf', optimize=False, opt_cv=10, 
        test_size=None, limit_search=True, impute=True, imp_method='KNN', n_iter=25, 
        boruta_trials=50, boruta_model='rf', balance=True, csv_file=None):

        self.data_x = data_x
        self.data_y = data_y
        self.clf = clf
        self.optimize = optimize 
        self.opt_cv = opt_cv 
        self.test_size = test_size
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
            self.data_y_ = copy.deepcopy(self.data_y) #For plotting purposes, save the original label array as it will be overwrrite with the numerical labels
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

    def create(self):
        """
        Creates the machine learning engine, current options are either a
        Random Forest, XGBoost, or a Neural Network classifier. 
        
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

        else:
            raise ValueError('clf argument must either be "rf", "nn", or "xgb".')
        
        if self.impute is False and self.optimize is False:
            if np.any(np.isfinite(self.data_x)==False):
                raise ValueError('data_x array contains nan values but impute is set to False! Set impute=True and run again.')
            print("Returning base {} model...".format(self.clf))
            model.fit(self.data_x, self.data_y)
            self.model = model
            return

        if self.impute:
            if self.imp_method == 'KNN':
                data, self.imputer = KNN_imputation(data=self.data_x, imputer=None)
            else:
                raise ValueError('Invalid imputation method, currently only k-NN is supported.')
            
            if self.optimize is False:
                data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
                print("Returning base {} model...".format(self.clf))
                model.fit(data, self.data_y)
                self.model = model 
                return
                
        else:
            data = self.data_x[:]
            data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7

        if self.feats_to_use is None:
            self.feats_to_use, self.feature_history = borutashap_opt(data, self.data_y, boruta_trials=self.boruta_trials, model=self.boruta_model)
            if len(self.feats_to_use) == 0:
                print('No features selected, increase the number of n_trials when running MicroLIA.optimization.borutashap_opt(). Using all features...')
                self.feats_to_use = np.arange(data.shape[1])
        else:
            print('feats_to_use attribute already exists, skipping feature selection...')

        #Re-construct the imputer with the selected features as
        #new predictions will only compute these metrics, need to fit again!
        if self.imp_method == 'KNN':
            data_x, self.imputer = KNN_imputation(data=self.data_x[:,self.feats_to_use], imputer=None)
        else: 
            data_x = self.data_x[:,self.feats_to_use]

        self.model, self.best_params, self.optimization_results = hyper_opt(data_x, self.data_y, clf=self.clf, n_iter=self.n_iter, 
            balance=self.balance, return_study=True, limit_search=self.limit_search, opt_cv=self.opt_cv, test_size=self.test_size)
        print("Fitting and returning final model...")
        self.model.fit(data_x, self.data_y)
        
    def save(self, dirname=None, path=None, overwrite=False):
        """
        Saves the trained classifier in a new directory named 'MicroLIA_models', 
        as well as the imputer and the features to use attributes, if not None.
        
        Args:
            dirname (str): The name of the directory where the model folder will be saved.
                This directory will be created, and therefore if it already exists
                in the system an error will appear.
            path (str): Absolute path where the data folder will be saved
                Defaults to None, in which case the directory is saved to the
                local home directory.
            overwrite (bool, optional): If True the 'MicroLIA_models' folder this
                function creates in the specified path will be deleted if it exists
                and created anew to avoid duplicate files. 
        """
        if self.model is None and self.imputer is None and self.feats_to_use is None:
            raise ValueError('The models have not been created! Run the create() method first.')

        if path is None:
            path = str(Path.home())
        if path[-1] != '/':
            path+='/'

        if dirname is not None:
            if dirname[-1] != '/':
                dirname+='/'
            path = path+dirname
            try:
                os.makedirs(path)
            except FileExistsError:
                raise ValueError('The dirname folder already exists!')

        try:
            os.mkdir(path+'MicroLIA_ensemble_model')
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

        if path is None:
            path = str(Path.home())
        if path[-1] != '/':
            path+='/'

        path += 'MicroLIA_ensemble_model/'

        try:
            self.model = joblib.load(path+'Model')
            model = 'model'
        except FileNotFoundError:
            model = ''
            pass

        try:
            self.imputer = joblib.load(path+'Imputer')
            imputer = 'imputer'
        except FileNotFoundError:
            imputer = ''
            pass 

        try:
            self.feats_to_use = joblib.load(path+'Feats_Index')
            feats_to_use = 'feats_to_use'
        except FileNotFoundError:
            feats_to_use = ''
            pass

        try:
            self.best_params = joblib.load(path+'Best_Params')
            best_params = 'best_params'
        except FileNotFoundError:
            best_params = ''
            pass

        try:
            self.feature_history = joblib.load(path+'FeatureOpt_Results')
            feature_opt_results = 'feature_selection_results'
        except FileNotFoundError:
            feature_opt_results = ''
            pass

        try:
            self.optimization_results = joblib.load(path+'HyperOpt_Results')
            optimization_results = 'optimization_results'
        except FileNotFoundError:
            optimization_results = '' 
            pass

        print('Successfully loaded the following class attributes: {}, {}, {}, {}, {}, {}'.format(model, imputer, feats_to_use, best_params, feature_opt_results, optimization_results))
        
        self.path = path

        return

    def predict(self, time, mag, magerr, convert=True, zp=24):
        """
        Predics the class label of new, unseen data.

        Args:
            time (ndarray): Array of observation timestamps.
            mag (ndarray): Array of observed magnitudes.
            magerr (ndarray): Array of corresponding magnitude errors.
            model (object): The machine learning model to use for predictions.
            convert (bool, optional): If False the features are computed with the input magnitudes.
                Defaults to True to convert and compute in flux. 
            zp (float): Zeropoint of the instrument, used to convert from magnitude
                to flux. Defaults to 24.

        Returns:
            Array containing the classes and the corresponding probability predictions.
        """

        if len(mag) < 30:
            warn('The number of data points is low -- results may be unstable')

        #classes = ['CONSTANT', 'CV', 'LPV', 'ML', 'VARIABLE']
        classes = self.model.classes_
        stat_array=[]
        
        if self.imputer is None and self.feats_to_use is None:
            stat_array.append(extract_features.extract_all(time, mag, magerr, convert=convert, zp=zp))
            pred = self.model.predict_proba(stat_array)
            return np.c_[classes, pred[0]]
        
        stat_array.append(extract_features.extract_all(time, mag, magerr, convert=convert, zp=zp, feats_to_use=self.feats_to_use))
        
        if self.imputer is not None:
            stat_array = self.imputer.transform(stat_array)
        pred = self.model.predict_proba(stat_array)

        return np.c_[classes, pred[0]]

    def plot_tsne(self, data_y=None, special_class=None, norm=True, pca=False, 
        legend_loc='upper center', title='Feature Parameter Space', savefig=False):
        """
        Plots a t-SNE projection using the sklearn.manifold.TSNE() method.

        Note:
            To highlight individual samples, use the data_y optional input
            and set that sample's data_y value to a unique name, and set that 
            same label in the special_class variable so that it can be visualized 
            clearly.

        Args:
            data_y (ndarray, optional): If using XGBoost then the
            special_class (optional): The class label that you wish to highlight,
                setting this optional parameter will 
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
            if len(self.data_x.shape) == 1:
                data = self.data_x[self.feats_to_use].reshape(1,-1)
            else:
                data = self.data_x[:,self.feats_to_use]
        else:
            data = self.data_x[:]

        if np.any(np.isnan(data)):
            #print('Automatically imputing NaN values with KNN imputation...')
            if self.imputer is not None and self.imp_method == 'KNN':
                data = KNN_imputation(data=data, imputer=self.imputer)
            else:
                data = KNN_imputation(data=data)[0]

        data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
        
        if len(data) > 5e3:
            method = 'barnes_hut' #Scales with O(N)
        else:
            method = 'exact' #Scales with O(N^2)

        if norm:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)

        if pca:
            pca_transformation = decomposition.PCA(n_components=data.shape[1], whiten=True, svd_solver='auto')
            pca_transformation.fit(data) 
            data = pca_transformation.transform(data)

        feats = TSNE(n_components=2, method=method, learning_rate=1000, 
            perplexity=35, init='random').fit_transform(data)
        x, y = feats[:,0], feats[:,1]
     
        markers = ['o', 's', '+', 'v', '.', 'x', 'h', 'p', '<', '>', '*']
        #color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c']
        color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

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
                    data_y = np.array(self.csv_file.label)
                    feats = np.unique(data_y)
            else:
                data_y = self.data_y_ 
                feats = np.unique(self.data_y_)
        else:
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
        
        plt.legend(loc=legend_loc, ncol=len(np.unique(data_y)), frameon=False, handlelength=2)#prop={'size': 14}
        plt.title(title)#, size=18)
        plt.xticks()#fontsize=14)
        plt.yticks()#fontsize=14)
        plt.ylabel('t-SNE Dimension 1')
        plt.xlabel('t-SNE Dimension 2')

        if savefig:
            plt.savefig('tSNE_Projection.png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()

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
            raise ValueError('Input data_x and data_y!')
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
            data = self.data_x[:]

        if np.any(np.isnan(data)):
            #print('Automatically imputing NaN values with KNN imputation...')
            if self.imputer is not None and self.imp_method == 'KNN':
                data = KNN_imputation(data=data, imputer=self.imputer)
            else:
                data = KNN_imputation(data=data)[0]

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

    def plot_hyper_opt(self, baseline=None, xlim=None, ylim=None, xlog=True, ylog=False, 
        savefig=False):
        """
        Plots the hyperparameter optimization history.
    
        Args:
            baseline (float): Baseline accuracy achieved when using only
                the default engine hyperparameters. If input a vertical
                line will be plot to indicate this baseline accuracy.
                Defaults to None.
            xlim: Limits for the x-axis. Ex) xlim = (0, 1000)
            ylim: Limits for the y-axis. Ex) ylim = (0.9, 0.94)
            xlog (boolean): If True the x-axis will be log-scaled.
                Defaults to True.
            ylog (boolean): If True the y-axis will be log-scaled.
                Defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False. 

        Returns:
            AxesImage
        """

        #fig = plot_optimization_history(self.optimization_results)
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

        plt.plot(range(len(trials)), best_value, color='r', alpha=0.83, linestyle='-', label='Best Model')
        plt.scatter(range(len(trials)), trial_values, c='b', marker='+', s=35, alpha=0.45, label='Trial')
        plt.xlabel('Trial #', alpha=1, color='k')
        plt.ylabel('Accuracy', alpha=1, color='k')
       # if self.clf == 'xgb':
        plt.title('Hyperparameter Optimization')#, size=18) Make this a f" string option!!
        #plt.xticks(fontsize=14)#, color='k')
        #plt.yticks(fontsize=14)#, color='k')
        #plt.grid(True, color='k', alpha=0.35, linewidth=1.5, linestyle='--')
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
        #plt.tight_layout()
        #plt.legend(prop={'size': 12}, loc='upper left')
        plt.legend(loc='upper center', ncol=ncol, frameon=False)#, handlelength=4)#prop={'size': 14}
        plt.rcParams['axes.facecolor']='white'
        
        if savefig:
            plt.savefig('Ensemble_Hyperparameter_Optimization.png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()

    def plot_feature_opt(self, feat_names=None, top=3, include_other=True, include_shadow=True, 
        include_rejected=False, flip_axes=True, save_data=False, savefig=False):
        """
        Returns whisker plot displaying the z-score distribution of each feature
        across all trials.
    
        Note:
            The following can be used to output the plot from the original API.

            model.feature_history.plot(which_features='accepted', X_size=14)

            Can designate to display either 'all', 'accepted', or 'tentative'


        Args: 
            feat_names (ndarry, optional): A list or array containing the names
                of the features in the data_x matrix, in order. Defaults to None,
                in which case the respective indices will appear instead.
            top (float, optional): Designates how many features to plot. Defaults to 3, which 
                will plot the top 3 performing features.
            include_other (bool): Whether to include the features that are not in the top designation,
                if True these features will be averaged out and displayed. Defaults to True.
            include_shadow (bool): Whether to include the max shadow feature that was used as a 
                baseline for 'random' behavior. Defaults to True.
            include_rejected (bool): Whether to include the rejected features, if True these features 
                will be averaged out and displayed.
            flip_axes (bool): Whether transpose the figure. Defaults to True.
            save_data (bool): Whether to save the feature importances as a csv file, defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False. 

        Returns:
            AxesImage
        """

        fname = str(Path.home())+'/__borutaimportances__' #Temporary file
        try:
            self.feature_history.results_to_csv(filename=fname)
        except AttributeError:
            raise ValueError('No optimization history found for feature selection, run .create() with optimize=True!')

        csv_data = pd.read_csv(fname+'.csv')
        #os.remove(fname+'.csv')
        accepted_indices = np.where(csv_data.Decision == 'Accepted')[0]
        if top > len(accepted_indices):
            top = len(accepted_indices)
            print('The top parameter exceeds the number of accepted variables, setting to the maximum value of {}'.format(str(top)))

        x, y, y_err = [], [], []

        for i in accepted_indices[:top]:
            if feat_names is None:
                x.append(int(i))
            else:
                x.append(int(csv_data.iloc[i].Features))
            y.append(float(csv_data.iloc[i]['Average Feature Importance']))
            y_err.append(float(csv_data.iloc[i]['Standard Deviation Importance']))
            
        if len(accepted_indices) == top:
            include_other = False

        if include_other:
            mean = []
            std = []
            for j in accepted_indices[top:]:
                mean.append(float(csv_data.iloc[j]['Average Feature Importance']))
                std.append(float(csv_data.iloc[j]['Standard Deviation Importance']))
            x.append(0), y.append(np.mean(mean)), y_err.append(np.mean(std))

        if include_shadow:
            ix = np.where(csv_data.Features == 'Max_Shadow')[0]
            y.append(float(csv_data.iloc[ix]['Average Feature Importance']))
            y_err.append(float(csv_data.iloc[ix]['Standard Deviation Importance']))
            x.append(int(ix))

        if feat_names is not None:  
            if isinstance(feat_names, np.ndarray) is False: 
                feat_names = np.array(feat_names)
            if include_shadow is False:
                if include_other is False:
                    x_names = feat_names[x] #By default x is the index of the feature
                else:
                    x_names = np.r_[feat_names[x[:-1]], ['Other Accepted']]
            else:
                if include_other is False:
                    x_names = np.r_[feat_names[x[:-1]], ['Max Shadow']]
                else:
                    x_names = np.r_[feat_names[x[:-2]], ['Other Accepted'], ['Max Shadow']]
        else:
            if self.csv_file is None:
                if include_other is False:
                    if include_shadow is False:
                        x_names = csv_data.iloc[x].Features
                    else:
                        x_names = np.r_[csv_data.iloc[x[:-1]], ['Max Shadow']]
                else:
                    if include_shadow is False:
                        x_names = np.r_[csv_data.iloc[x[:-1]].Features, ['Max Shadow']]
                    else:
                        x_names = np.r_[csv_data.iloc[x[:-2]].Features, ['Other Accepted'], ['Max Shadow']]
            else:
                if include_other is False:
                    if include_shadow is False:
                        x_names = self.self.csv_file.columns[x[:-1]]
                    else:
                        x_names = np.r_[self.csv_file.columns[x[:-1]], ['Max Shadow']]
                else:
                    if include_shadow is False:
                        x_names = np.r_[self.csv_file.columns[x[:-1]], ['Max Shadow']]
                    else:
                        x_names = np.r_[self.csv_file.columns[x[:-2]], ['Other Accepted'], ['Max Shadow']]


        if include_rejected:
            x = []
            rejected_indices = np.where(csv_data.Decision == 'Rejected')[0]
            for i in rejected_indices:
                if feat_names is None:
                    x.append(int(i))
                else:
                    x.append(int(csv_data.iloc[i].Features))
                y.append(float(csv_data.iloc[i]['Average Feature Importance']))
                y_err.append(float(csv_data.iloc[i]['Standard Deviation Importance']))
            if feat_names is not None:
                x_names = np.r_[x_names, feat_names[x]]
            else:
                x_names = np.r_[x_names, csv_data.iloc[x].Features]
        
        y, y_err = np.array(y), np.array(y_err)
        fig, ax = plt.subplots()
        if flip_axes:
            lns, = ax.plot(y, np.arange(len(x_names)), 'k*--', lw=0.77)
            lns_sigma = ax.fill_betweenx(np.arange(len(x_names)), y-y_err, y+y_err, color="grey", alpha=0.2)
            ax.set_xlabel('Z Score', alpha=1, color='k')
            ax.set_yticks(np.arange(len(x_names)), x_names)#, rotation=90)
            for t in ax.get_yticklabels():
                txt = t.get_text()
                if 'Max Shadow' in txt:
                    t.set_color('red')
                    ax.plot(y[-1], np.arange(len(x_names))[-1], marker='*', color='red')
            ax.set_ylim((np.arange(len(x_names))[0]-0.5, np.arange(len(x_names))[-1]+0.5))
            ax.set_xlim((np.min(y)-1, np.max(y)+1))
            ax.invert_yaxis(), ax.invert_xaxis()
        else:
            lns, = ax.plot(np.arange(len(x_names)), y, 'k*--', lw=0.77)#, label='XGBoost', lw=0.77)
            lns_sigma = ax.fill_between(np.arange(len(x_names)), y-y_err, y+y_err, color="grey", alpha=0.2)
            ax.set_ylabel('Z Score', alpha=1, color='k')
            ax.set_xticks(np.arange(len(x_names)), x_names, rotation=90)
            for t in ax.get_xticklabels():
                txt = t.get_text()
                if 'Max Shadow' in txt:
                    t.set_color('red')
                    ax.plot(np.arange(len(x_names))[-1], y[-1], marker='*', color='red')
            ax.set_xlim((np.arange(len(x_names))[0]-0.5, np.arange(len(x_names))[-1]+0.5))
            ax.set_ylim((np.min(y)-1, np.max(y)+1))

        ax.set_title('Feature Importance')#, size=18)
        ax.legend([(lns, lns_sigma)], [r'$\pm$ 1$\sigma$'], loc='upper right', ncol=1, frameon=False, handlelength=2)

        if savefig:
            plt.savefig('Feature_Importance.png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()

    def plot_hyper_param_importance(self, plot_time=True, savefig=False):
        """
        Plots the hyperparameter optimization history.
    
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
        #fig.subplots_adjust(top=0.8)
        ax.barh(xtick_labels, importance, label='Importance for Classification', color=mcolors.TABLEAU_COLORS["tab:blue"], alpha=0.87)
        if plot_time:
            ax.barh(xtick_labels, duration_importance, label='Impact on Engine Speed', color=mcolors.TABLEAU_COLORS["tab:orange"], alpha=0.7, hatch='/')

        ax.set_ylabel("Hyperparameter")
        ax.set_xlabel("Importance Evaluation")
        ax.legend(ncol=2, frameon=False, handlelength=2, bbox_to_anchor=(0.5, 1.1), loc='upper center')
        ax.set_xscale('log')
        plt.gca().invert_yaxis()
        plt.xlim((0, 1.))#np.max(importance+duration_importance)))#np.max(importance+duration_importance)))
        #fig = plot_param_importances(self.optimization_results)
        #fig = plot_param_importances(self.optimization_results, target=lambda t: t.duration.total_seconds(), target_name="duration")
        #plt.tight_layout()
        if savefig:
            if plot_time:
                plt.savefig('Ensemble_Hyperparameter_Importance.png', bbox_inches='tight', dpi=300)
            else:
                plt.savefig('Ensemble_Hyperparameter_Duration_Importance.png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()

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
        
        print('Calculating and saving importances, this could take up to an hour...')

        try:
            if isinstance(self.path, str):
                path = self.path  
            else:
                path = str(Path.home())
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
            new_labels.append("Learning Rate")
            continue
        if label == "n estimators":
            new_labels.append("Num of Trees")
            continue
        if label == "colsample bytree":
            new_labels.append("ColSample ByTree")
            continue
        new_labels.append(label.title())

    return new_labels

def evaluate_model(classifier, data_x, data_y, normalize=True, k_fold=10):
    """
    Cross-checks model accuracy and outputs both the predicted
    and the true class labels. 

    Args:
        classifier: The machine learning classifier to optimize.
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        normalize (bool, optional): If False the confusion matrix will display the
            total number of objects in the sample. Defaults to True, in which case
            the values are normalized between 0 and 1. 
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.

    Returns:
        The first output is the 1D array of the true class labels.
        The second output is the 1D array of the predicted class labels.
    """

    k_fold = KFold(k_fold, shuffle=True)#, random_state=1)
    #k_fold = StratifiedKFold(k_fold, shuffle=False)#, random_state=8)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train_ix, test_ix in k_fold.split(data_x, data_y):
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]
        # Fit the classifier
        classifier.fit(train_x, train_y)
        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(test_x)
        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)

    return predicted_targets, actual_targets

def generate_matrix(predicted_labels_list, actual_targets, classes, normalize=True, 
    title='Confusion Matrix', savefig=False):
    """
    Generates the confusion matrix using the output from the evaluate_model() function.

    Args:
        predicted_labels_list: 1D array containing the predicted class labels.
        actual_targets: 1D array containing the actual class labels.
        classes (list): A list containing the label of the two training bags. This
            will be used to set the axis. Ex) classes = ['DIFFUSE', 'OTHER']
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
    elif normalize == False:
        generate_plot(conf_matrix, classes=classes, normalize=normalize, title=title)
    
    if savefig:
        plt.savefig('Ensemble_Confusion_Matrix.png', bbox_inches='tight', dpi=300)
        plt.clf()
    else:
        plt.show()
    
def generate_plot(conf_matrix, classes, normalize=False, title='Confusion Matrix'):
    """
    Generates the confusion matrix figure object, but does not plot.
    
    Args:
        conf_matrix: The confusion matrix generated using the generate_matrix() function.
        classes (list): A list containing the label of the two training bags. This
            will be used to set the axis. Defaults to a list containing 'DIFFUSE' & 'OTHER'. 
        normalize (bool, optional): If True the matrix accuracy will be normalized
            and displayed as a percentage accuracy. Defaults to True.
        title (str, optional): The title of the output plot. 

    Returns:
        AxesImage object. 
    """

    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)#, fontsize=20)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, alpha=1, color='k')#rotation=45, fontsize=14,
    plt.yticks(tick_marks, classes, alpha=1, color='k', rotation=90)#fontsize=14,
    #plt.xticks(tick_marks, ['DIFFUSE','OTHER'], rotation=45, fontsize=14)
    #plt.yticks(tick_marks, ['DIFFUSE','OTHER'], fontsize=14)

    fmt = '.4f' if normalize is True else 'd'
    thresh = conf_matrix.max() / 2.

    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")#fontsize=14,

    plt.grid(False)
    plt.ylabel('True label', alpha=1, color='k')#fontsize=18
    plt.xlabel('Predicted label',alpha=1, color='k')#fontsize=18
    plt.tight_layout()

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
        print((np.max(data_x[:,i]) - np.min(data_x[:,i])))
        new_array[:,i] = (data_x[:,i] - np.min(data_x[:,i])) / (np.max(data_x[:,i]) - np.min(data_x[:,i]))

    return new_array
    



