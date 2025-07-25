# -*- coding: utf-8 -*-
"""
    Created on Thu Jul 24 21:25:44 2025
    
    @author: danielgodinez
"""
import shap
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import binomtest, ks_2samp
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
import warnings; warnings.filterwarnings("ignore")


class BorutaSHAP:
    """
    Feature selection wrapper combining Boruta and SHAP-based importance metrics.

    The `BorutaSHAP` class extends the Boruta feature selection methodology using model-specific 
    importance metrics (SHAP, Gini, or permutation-based), making it compatible with both classification 
    and regression tasks. It introduces a shadow feature mechanism to iteratively compare feature 
    importances against randomized features, statistically testing for their relevance. This class 
    supports flexible importance measures, sample-based selection using Isolation Forest, and automatic 
    integration with scikit-learn-style pipelines via `fit`, `transform`, and `set_params`.

    Parameters
    ----------
    model : object, optional
        A scikit-learn-compatible model with `fit` and `predict` methods. If not provided, a default 
        RandomForestClassifier or RandomForestRegressor is used depending on the task type.

    importance_measure : str, default='Shap'
        Metric used to compute feature importances. One of ['Shap', 'gini', 'perm'].

    classification : bool, default=True
        Whether the task is classification (`True`) or regression (`False`).

    percentile : int, default=100
        Percentile of shadow feature importances to use as a threshold for significance testing.
        Lower values make the algorithm more lenient.

    pvalue : float, default=0.05
        Significance level for hypothesis testing. Lower values result in stricter feature rejection,
        potentially increasing runtime.

    Attributes
    ----------
    accepted : list
        Final list of features accepted as important.

    rejected : list
        Final list of features rejected as unimportant.

    tentative : list
        List of features whose importance is undetermined.

    history_x : pd.DataFrame
        DataFrame recording feature importances over each iteration.

    history_shadow : np.ndarray
        Historical shadow feature importances.

    hits : np.ndarray
        Array tracking the number of times each feature beats the shadow threshold.

    all_columns : np.ndarray
        List of original column names in `X`.

    X_boruta : pd.DataFrame
        DataFrame containing both original and shadow features.

    Methods
    -------
    fit(X, y, ...)
        Runs the BorutaSHAP feature selection algorithm on the provided dataset.

    results_to_csv(filename='feature_importance')
        Saves the importance scores and feature decisions to CSV.

    create_mapping_of_features_to_attribute(maps)
        Creates a dictionary mapping each feature to a visual label (e.g., color or decision).

    Notes
    -----
    SHAP-based explanations are intended for tree-based models, such as XGBoost, LightGBM, or CatBoost.

    Permutation importance is computed using scikit-learn’s `permutation_importance` function.

    Gini importance requires the model to expose a `feature_importances_` attribute, such as in RandomForest models.

    Missing values are supported only for models that can handle them internally. Otherwise, a ValueError will be raised.

    When sample-based SHAP explanations are enabled, the method uses an Isolation Forest to select a representative subset 
    of the data that preserves the original anomaly score distribution.

    References
    ----------
    - Kursa, M. B., & Rudnicki, W. R. (2010). Feature selection with the Boruta package.
    - Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions (SHAP).
    - https://github.com/Ekeany/Boruta-Shap
    """
    def __init__(self, model=None, importance_measure='Shap', classification=True, percentile=100, pvalue=0.05):
        """
        Initialize the BorutaSHAP feature selection object.

        Parameters
        ----------
        model : object, optional
            A scikit-learn-compatible model instance that implements `fit` and `predict`. 
            If not provided, a default RandomForestClassifier (for classification) or 
            RandomForestRegressor (for regression) is used.

        importance_measure : str, default='Shap'
            Method used to calculate feature importance.
            Valid options are:
            'Shap'  — SHAP values using shap.TreeExplainer
            'gini'  — Gini importance via the model's feature_importances_ attribute
            'perm'  — Permutation importance using sklearn's permutation_importance

        classification : bool, default=True
            Whether the task is a classification problem. If False, regression is assumed.

        percentile : int, default=100
            Percentile (0 to 100) used to define the shadow feature threshold. 
            Lower values make the algorithm more lenient by reducing the cutoff.

        pvalue : float, default=0.05
            Significance level for the statistical tests used to accept or reject features. 
            Smaller values increase strictness and may increase runtime.
        """
        self.importance_measure = importance_measure
        self.percentile = percentile
        self.pvalue = pvalue
        self.classification = classification
        self.model = model
        self.check_model()

    @classmethod
    def _get_param_names(cls):
        """
        Retrieve the parameter names from the class constructor (__init__).

        This utility method introspects the constructor signature to extract all explicitly
        defined parameter names, excluding 'self' and variable keyword arguments (**kwargs).
        It ensures compatibility with scikit-learn-style estimators by enforcing that no 
        variable positional arguments (*args) are used.

        Returns
        -------
        list of str
            A sorted list of parameter names defined in the constructor.

        Raises
        ------
        RuntimeError
            If the class defines variable positional arguments (*args), which violates
            scikit-learn's estimator API convention.
        """
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get the parameters of this estimator.

        This method returns a dictionary of all parameters in the estimator. If `deep=True`,
        it will recursively retrieve parameters of sub-estimators (i.e., model attributes 
        that implement `get_params` themselves), using a double-underscore naming convention.

        Parameters
        ----------
        deep : bool, default=True
            If True, include parameters from nested objects (such as the wrapped model).
            If False, only return parameters directly set on this estimator.

        Returns
        -------
        params : dict
            Dictionary mapping parameter names to their current values.
            Nested parameters are flattened using the format: 'component__param'.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        This method updates the estimator’s parameters using the provided dictionary. 
        It supports both top-level parameters and nested parameters using the 
        scikit-learn convention of double underscores (e.g., 'model__n_estimators') 
        for sub-estimators.

        Parameters
        ----------
        **params : dict
            Dictionary of parameter names mapped to their new values. Nested parameters 
            can be updated using double-underscore notation.

        Returns
        -------
        self : object
            The updated estimator instance.

        Raises
        ------
        ValueError
            If a parameter name is invalid or does not match any parameter in the estimator.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return 

    def check_model(self):
        """
        Validate and initialize the model used for feature importance evaluation.

        If no model was provided at initialization, this method assigns a default 
        RandomForestClassifier (for classification) or RandomForestRegressor (for regression). 
        It also checks that the model has the required methods and attributes based 
        on the selected importance measure.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If the provided model does not implement both `fit` and `predict`, or if the
            `gini` importance measure is selected but the model lacks the `feature_importances_` attribute.
        """
        check_fit = hasattr(self.model, 'fit')
        check_predict_proba = hasattr(self.model, 'predict')

        try:
            check_feature_importance = hasattr(self.model, 'feature_importances_')

        except:
            check_feature_importance = True

        if self.model is None:

            if self.classification:
                self.model = RandomForestClassifier()
            else:
                self.model = RandomForestRegressor()

        elif check_fit is False and check_predict_proba is False:
            raise AttributeError('Model must contain both the fit() and predict() methods')

        elif check_feature_importance is False and self.importance_measure == 'gini':
            raise AttributeError('Model must contain the feature_importances_ method to use Gini try Shap instead')

        else:
            pass

    def check_X(self):
        """
        Verify that the input feature data `X` is a pandas DataFrame.

        This method ensures that the feature matrix provided to the BorutaSHAP 
        instance is of the correct type before proceeding with feature selection.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If `X` is not a pandas DataFrame.
        """
        if isinstance(self.X, pd.DataFrame) is False:
            raise AttributeError('X must be a pandas Dataframe')

        else:
            pass

    def missing_values_y(self):
        """
        Check for missing values in the target variable `y`.

        Supports pandas Series, DataFrame, or NumPy array inputs. Returns True if 
        any missing values are found.

        Returns
        -------
        bool
            True if `y` contains missing values, False otherwise.

        Raises
        ------
        AttributeError
            If `y` is not a pandas Series, DataFrame, or NumPy array.
        """
        if isinstance(self.y, pd.Series) or isinstance(self.y, pd.DataFrame):
            return self.y.isnull().any().any()

        elif isinstance(self.y, np.ndarray):
            return np.isnan(self.y).any()

        else:
            raise AttributeError('Y must be a pandas Dataframe, Series, or a numpy array')

    def check_missing_values(self):
        """
        Check for missing values in the feature matrix `X` and target variable `y`.

        This method verifies that no missing values are present in the input data. 
        If missing values are found, a warning is issued for models that support them 
        (e.g., XGBoost, LightGBM, CatBoost). Otherwise, a ValueError is raised.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If missing values are detected and the model does not support them.

        Notes
        -----
        Models known to support missing values include: XGBoost, CatBoost, and LightGBM.
        """
        X_missing = self.X.isnull().any().any()
        Y_missing = self.missing_values_y()

        models_to_check = ('xgb', 'catboost', 'lgbm', 'lightgbm')

        model_name = str(type(self.model)).lower()
        if X_missing or Y_missing:

            if any([x in model_name for x in models_to_check]):
                print('Warning there are missing values in your data !')

            else:
                raise ValueError('There are missing values in your Data')

        else:
            pass

    def Check_if_chose_train_or_test_and_train_model(self):
        """
        Split the data and train the model based on the `train_or_test` strategy.

        If `train_or_test='test'`, the method splits the Boruta-augmented dataset into training 
        and testing sets (70/30 split) using the specified `random_state` and optional stratification.
        The model is trained on the training portion.

        If `train_or_test='train'`, the model is trained on the full dataset without splitting.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If stratification is requested for a regression task, or if `train_or_test` is not 
            one of the accepted values ("train" or "test").

        Notes
        -----
        For a detailed discussion on training vs. testing data when computing feature importance, 
        see: https://slds-lmu.github.io/iml_methods_limitations/pfi-data.html
        """
        if self.stratify is not None and not self.classification:
            raise ValueError('Cannot take a strtified sample from continuous variable please bucket the variable and try again !')

        if self.train_or_test.lower() == 'test':
            # keeping the same naming convention as to not add complexit later on
            self.X_boruta_train, self.X_boruta_test, self.y_train, self.y_test, self.w_train, self.w_test = train_test_split(self.X_boruta,
                                                                                                                                self.y,
                                                                                                                                self.sample_weight,
                                                                                                                                test_size=0.3,
                                                                                                                                random_state=self.random_state,
                                                                                                                                stratify=self.stratify)
            self.Train_model(self.X_boruta_train, self.y_train, sample_weight = self.w_train)

        elif self.train_or_test.lower() == 'train':
            # model will be trained and evaluated on the same data
            self.Train_model(self.X_boruta, self.y, sample_weight = self.sample_weight)

        else:
            raise ValueError('The train_or_test parameter can only be "train" or "test"')

    def Train_model(self, X, y, sample_weight = None):
        """
        Fit the model to the provided data.

        This method trains the model using the given features `X` and targets `y`. 
        It handles special cases for certain models like CatBoost, which require 
        categorical feature specifications. It also gracefully handles models that do 
        not accept the `verbose` parameter.

        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame containing the feature matrix.

        y : pandas.Series or numpy.ndarray
            Array or Series containing the target variable.

        sample_weight : pandas.Series or numpy.ndarray, optional
            Sample weights to apply during model training.

        Returns
        -------
        None
        """
        if 'catboost' in str(type(self.model)).lower():
            self.model.fit(X, y, sample_weight = sample_weight, cat_features = self.X_categorical,  verbose=False)

        else:

            try:
                self.model.fit(X, y, sample_weight = sample_weight, verbose=False)

            except:
                self.model.fit(X, y, sample_weight = sample_weight)

    def fit(self, X, y, sample_weight = None, n_trials = 20, random_state=0, sample=False,
            train_or_test = 'test', normalize=True, verbose=True, stratify=None):
        """
        Run the BorutaSHAP feature selection process.

        This is the core method that performs iterative feature selection by comparing 
        real features against shadow (randomized) features using the chosen importance 
        measure (SHAP, Gini, or permutation). Features are repeatedly tested against the 
        maximum shadow importance, and classified as accepted, rejected, or tentative.

        The algorithm proceeds as follows:
        1. Extend the dataset by adding shuffled copies of original features (shadow features).
        2. Train the model and compute feature importances.
        3. Identify features that outperform the maximum shadow importance threshold.
        4. Track hit counts and statistically test features against the null hypothesis.
        5. Accept, reject, or defer decision on each feature.
        6. Repeat until a decision is made for all features or the trial limit is reached.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        y : pandas.Series or numpy.ndarray
            Target variable.

        sample_weight : pandas.Series or numpy.ndarray, optional
            Observation-level weights used during model training.

        n_trials : int, default=20
            Maximum number of iterations to run the feature selection process.

        random_state : int, default=0
            Random seed for reproducibility.

        sample : bool, default=False
            If True, a representative sample of the data will be selected using 
            Isolation Forest for SHAP value estimation.

        train_or_test : {'train', 'test'}, default='test'
            Specifies whether feature importances should be computed on training data 
            or a held-out test split.

        normalize : bool, default=True
            Whether to normalize feature importances using z-score transformation.

        verbose : bool, default=True
            If True, prints the final list of accepted, rejected, and tentative features.

        stratify : array-like, optional
            Class labels used for stratified splitting during train-test division.

        Returns
        -------
        self : object
            Returns the fitted BorutaSHAP instance.

        Notes
        -----
        For a detailed discussion on the implications of computing feature importances 
        on training vs. test data, see:
        https://compstat-lmu.github.io/iml_methods_limitations/pfi-data.html
        """
        if sample_weight is None:
            sample_weight = np.ones(len(X))

        np.random.seed(random_state)

        self.starting_X = X.copy()
        self.X = X.copy()
        self.y = y.copy()
        self.sample_weight = sample_weight.copy()
        self.n_trials = n_trials
        self.random_state = random_state
        self.ncols = self.X.shape[1]
        self.all_columns = self.X.columns.to_numpy()
        self.rejected_columns = []
        self.accepted_columns = []

        self.check_X()
        self.check_missing_values()
        self.sample = sample
        self.train_or_test = train_or_test
        self.stratify = stratify

        self.features_to_remove = []
        self.hits  = np.zeros(self.ncols)
        self.order = self.create_mapping_between_cols_and_indices()
        self.create_importance_history()

        if self.sample: self.preds = self.isolation_forest(self.X, self.sample_weight)

        for trial in tqdm(range(self.n_trials)):

            self.remove_features_if_rejected()
            self.columns = self.X.columns.to_numpy()
            self.create_shadow_features()

            # early stopping
            if self.X.shape[1] == 0:
                break

            else:

                self.Check_if_chose_train_or_test_and_train_model()

                self.X_feature_import, self.Shadow_feature_import = self.feature_importance(normalize=normalize)
                self.update_importance_history()
                hits = self.calculate_hits()
                self.hits += hits
                self.history_hits = np.vstack((self.history_hits, self.hits))
                self.test_features(iteration=trial+1)

        self.store_feature_importance()
        self.calculate_rejected_accepted_tentative(verbose=verbose)
        
    def calculate_rejected_accepted_tentative(self, verbose):
        """
        Finalize feature decisions: accepted, rejected, or tentative.

        This method processes the accumulated hit statistics across all trials to determine 
        which features are:
        - Accepted: consistently more important than shadow features.
        - Rejected: consistently less important than shadow features.
        - Tentative: not confidently accepted or rejected.

        Parameters
        ----------
        verbose : bool
            If True, prints the number and names of accepted, rejected, and tentative features.

        Returns
        -------
        None
        """
        self.rejected  = list(set(self.flatten_list(self.rejected_columns))-set(self.flatten_list(self.accepted_columns)))
        self.accepted  = list(set(self.flatten_list(self.accepted_columns)))
        self.tentative = list(set(self.all_columns) - set(self.rejected + self.accepted))

        if verbose:
            print(str(len(self.accepted))  + ' attributes confirmed important: ' + str(self.accepted))
            print(str(len(self.rejected))  + ' attributes confirmed unimportant: ' + str(self.rejected))
            print(str(len(self.tentative)) + ' tentative attributes remains: ' + str(self.tentative))

    def create_importance_history(self):
        """
        Initialize arrays to store historical feature importance scores.

        This method sets up internal storage for tracking the shadow importances, 
        original feature importances, and cumulative hit counts across all iterations.

        Returns
        -------
        None
        """
        self.history_shadow = np.zeros(self.ncols)
        self.history_x = np.zeros(self.ncols)
        self.history_hits = np.zeros(self.ncols)

    def update_importance_history(self):
        """
        Update historical records of feature importances for the current iteration.

        This method appends the current shadow and actual feature importances to their 
        respective history arrays, ensuring they remain aligned with the original 
        column order using a mapping index.

        Returns
        -------
        None
        """
        padded_history_shadow  = np.full((self.ncols), np.nan)
        padded_history_x = np.full((self.ncols), np.nan)

        for (index, col) in enumerate(self.columns):
            map_index = self.order[col]
            padded_history_shadow[map_index] = self.Shadow_feature_import[index]
            padded_history_x[map_index] = self.X_feature_import[index]

        self.history_shadow = np.vstack((self.history_shadow, padded_history_shadow))
        self.history_x = np.vstack((self.history_x, padded_history_x))

    def store_feature_importance(self):
        """
        Finalize and store historical feature importance statistics.

        This method reshapes the accumulated feature importance history into a pandas DataFrame 
        and appends summary statistics for the shadow features, including maximum, minimum, 
        mean, and median importance values.

        Returns
        -------
        None
        """
        self.history_x = pd.DataFrame(data=self.history_x, columns=self.all_columns)

        self.history_x['Max_Shadow']    =  [max(i) for i in self.history_shadow]
        self.history_x['Min_Shadow']    =  [min(i) for i in self.history_shadow]
        self.history_x['Mean_Shadow']   =  [np.nanmean(i) for i in self.history_shadow]
        self.history_x['Median_Shadow'] =  [np.nanmedian(i) for i in self.history_shadow]

    def results_to_csv(self, filename='feature_importance'):
        """
        Save feature importance summary statistics and decisions to a CSV file.

        This method compiles the average and standard deviation of each feature's 
        importance across all iterations, appends its final classification 
        (Accepted, Rejected, Tentative, or Shadow), and exports the result to disk.

        Parameters
        ----------
        filename : str, default='feature_importance'
            The base name for the output CSV file. The file will be saved as 
            '<filename>.csv' in the current working directory.

        Returns
        -------
        None
        """
        features = pd.DataFrame(data={'Features':self.history_x.iloc[1:].columns.values,
        'Average Feature Importance':self.history_x.iloc[1:].mean(axis=0).values,
        'Standard Deviation Importance':self.history_x.iloc[1:].std(axis=0).values})

        decision_mapper = self.create_mapping_of_features_to_attribute(maps=['Tentative','Rejected','Accepted', 'Shadow'])
        features['Decision'] = features['Features'].map(decision_mapper)
        features = features.sort_values(by='Average Feature Importance',ascending=False)

        features.to_csv(filename + '.csv', index=False)

    def remove_features_if_rejected(self):
        """
        Remove rejected features from the dataset.

        This method drops features from `self.X` that have been marked for removal 
        based on the outcome of statistical tests in the current iteration.

        Returns
        -------
        None
        """
        if len(self.features_to_remove) != 0:
            for feature in self.features_to_remove:
                try:
                    self.X.drop(feature, axis = 1, inplace=True)
                except:
                    pass

        else:
            pass

    @staticmethod
    def flatten_list(array):
        """
        Flatten a list of lists into a single list.

        Parameters
        ----------
        array : list of lists
            A nested list to be flattened.

        Returns
        -------
        list
            A single flattened list containing all elements from the sublists.
        """
        return [item for sublist in array for item in sublist]

    def create_mapping_between_cols_and_indices(self):
        """
        Create a mapping from feature names to their column indices.

        This mapping preserves the original order of columns in `self.X` and is used 
        to align importance values across iterations.

        Returns
        -------
        dict
            Dictionary mapping column names to their corresponding integer indices.
        """
        return dict(zip(self.X.columns.to_list(), np.arange(self.X.shape[1])))

    def calculate_hits(self):
        """
        Compute hit counts for each feature based on shadow feature comparison.

        A feature is assigned a "hit" if its importance exceeds the specified percentile 
        threshold of the shadow feature importances. Hits are padded and aligned to the 
        full column index order.

        Returns
        -------
        numpy.ndarray
            Array of length `ncols` containing the updated hit counts for each feature.
        """
        shadow_threshold = np.percentile(self.Shadow_feature_import, self.percentile)

        padded_hits = np.zeros(self.ncols)
        hits = self.X_feature_import > shadow_threshold

        for (index, col) in enumerate(self.columns):
            map_index = self.order[col]
            padded_hits[map_index] += hits[index]

        return padded_hits

    def create_shadow_features(self):
        """
        Generate shadow features by shuffling each original feature column.

        Shadow features are created by independently permuting each column of the input data `X`. 
        These are used as a baseline for comparing feature importances. The resulting shadow 
        features are renamed with a 'shadow_' prefix and concatenated with the original data 
        to form the extended dataset used during model training.

        This method also identifies categorical columns for models (e.g., CatBoost) that 
        require explicit specification of categorical features.

        Returns
        -------
        None
        """
        self.X_shadow = self.X.apply(np.random.permutation)
        
        if isinstance(self.X_shadow, pd.DataFrame):
            # append
            obj_col = self.X_shadow.select_dtypes("object").columns.tolist()
            if obj_col ==[] :
                 pass
            else :
                 self.X_shadow[obj_col] =self.X_shadow[obj_col].astype("category")

        self.X_shadow.columns = ['shadow_' + feature for feature in self.X.columns]
        self.X_boruta = pd.concat([self.X, self.X_shadow], axis = 1)

        col_types = self.X_boruta.dtypes
        self.X_categorical = list(col_types[(col_types=='category' ) | (col_types=='object')].index)

    @staticmethod
    def calculate_Zscore(array):
        """
        Compute the z-score normalization of a numeric array.

        Each element is standardized by subtracting the mean and dividing by the standard deviation.

        Parameters
        ----------
        array : array-like
            Input array of numeric values.

        Returns
        -------
        list of float
            Z-score normalized values of the input array.
        """
        mean_value = np.mean(array)
        std_value  = np.std(array)

        return [(element-mean_value)/std_value for element in array]

    def feature_importance(self, normalize):
        """
        Compute feature importance scores for both original and shadow features.

        This method calculates importance values based on the specified `importance_measure`:
        - 'shap': Uses SHAP values computed via `shap.TreeExplainer`
        - 'perm': Uses permutation importance via `sklearn.inspection.permutation_importance`
        - 'gini': Uses built-in `feature_importances_` from the model (e.g., RandomForest)

        Importance values can optionally be normalized using z-score transformation.

        Parameters
        ----------
        normalize : bool
            If True, importance scores are z-score normalized.

        Returns
        -------
        X_feature_import : array-like
            Importance scores for the original features.

        Shadow_feature_import : array-like
            Importance scores for the shadow (randomized) features.

        Raises
        ------
        ValueError
            If `importance_measure` is not one of {'shap', 'perm', 'gini'}.
        """
        if self.importance_measure == 'shap':

            self.explain()
            vals = self.shap_values

            if normalize:
                vals = self.calculate_Zscore(vals)

            X_feature_import = vals[:len(self.X.columns)]
            Shadow_feature_import = vals[len(self.X_shadow.columns):]
            
        elif self.importance_measure == 'perm':
            
            # set default scoring as f1, can be changed to an argument for customizability
            perm_importances_ =  permutation_importance(self.model, self.X, self.y, scoring='f1')
            perm_importances_ = perm_importance.importances_mean

            if normalize:
                perm_importances_ = self.calculate_Zscore(perm_importances_)

            X_feature_import = perm_importances_[:len(self.X.columns)]
            Shadow_feature_import = perm_importances_[len(self.X.columns):]

        elif self.importance_measure == 'gini':

                feature_importances_ =  np.abs(self.model.feature_importances_)

                if normalize:
                    feature_importances_ = self.calculate_Zscore(feature_importances_)

                X_feature_import = feature_importances_[:len(self.X.columns)]
                Shadow_feature_import = feature_importances_[len(self.X.columns):]

        else:

            raise ValueError('No Importance_measure was specified select one of (shap, perm, gini)')

        return X_feature_import, Shadow_feature_import

    @staticmethod
    def isolation_forest(X, sample_weight):
        """
        Fit an Isolation Forest to the dataset and compute anomaly scores.

        This method trains an Isolation Forest on the input feature matrix `X` and 
        returns anomaly scores for each sample. Higher scores indicate more typical 
        (less anomalous) samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Input feature matrix.

        sample_weight : array-like
            Sample weights to apply during model fitting.

        Returns
        -------
        numpy.ndarray
            Anomaly scores for each sample, as returned by `IsolationForest.score_samples`.
        """
        clf = IsolationForest().fit(X, sample_weight = sample_weight)
        preds = clf.score_samples(X)

        return preds

    @staticmethod
    def get_5_percent(num):
        """
        Compute 5 percent of a given number.

        Parameters
        ----------
        num : int or float
            Input number.

        Returns
        -------
        int
            Value corresponding to 5% of the input, rounded to the nearest integer.
        """
        return round(5  / 100 * num)

    def get_5_percent_splits(self, length):
        """
        Generate index positions at 5% intervals of a given length.

        This method returns an array of indices that split a dataset into 
        successive 5% chunks, based on the total number of samples.

        Parameters
        ----------
        length : int
            Total number of samples in the dataset.

        Returns
        -------
        numpy.ndarray
            Array of index positions at 5% intervals.
        """
        five_percent = self.get_5_percent(length)

        return np.arange(five_percent,length,five_percent)

    def find_sample(self):
        """
        Select a representative sample of the dataset using KS-test on anomaly scores.

        This method iteratively draws random samples of increasing size and compares the 
        distribution of their anomaly scores (from Isolation Forest) to the original 
        distribution using the Kolmogorov-Smirnov (KS) test. It starts at 5% of the dataset 
        and increases in 5% increments until a sample is found with a KS p-value > 0.95, 
        indicating statistical similarity.

        Returns
        -------
        pandas.DataFrame
            A representative sample of `self.X_boruta` with similar anomaly score distribution 
            to the full dataset.
        """
        loop = True
        iteration = 0
        size = self.get_5_percent_splits(self.X.shape[0])
        element = 1
        while loop:

            sample_indices = np.random.choice(np.arange(self.preds.size),  size=size[element], replace=False)
            sample = np.take(self.preds, sample_indices)
            if ks_2samp(self.preds, sample).pvalue > 0.95:
                break
            
            iteration+=1

            if iteration == 20:
                element  += 1
                iteration = 0

        return self.X_boruta.iloc[sample_indices]

    def explain(self):
        """
        Compute SHAP values for the model using TreeExplainer.

        This method uses the SHAP package to calculate feature importances based on 
        Shapley values. It selects `TreeExplainer` with path-dependent perturbations 
        for efficiency on tree-based models. If `self.sample` is True, a representative 
        subset of the data (selected via `find_sample`) is used; otherwise, the full 
        `self.X_boruta` dataset is used.

        For classification tasks, SHAP values across classes are aggregated to compute
        a single importance value per feature. For regression, absolute SHAP values are 
        averaged directly.

        Returns
        -------
        None
            The computed SHAP values are stored internally in `self.shap_values`.

        Raises
        ------
        ValueError
            If the model is not compatible with SHAP's TreeExplainer (though in practice,
            the method currently assumes tree-based models only).
        """
        explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent", approximate=True)

        if self.sample:

            if self.classification:
                # for some reason shap returns values wraped in a list of length 1

                self.shap_values = np.array(explainer.shap_values(self.find_sample()))
                if isinstance(self.shap_values, list):

                    class_inds = range(len(self.shap_values))
                    shap_imp = np.zeros(self.shap_values[0].shape[1])
                    for i, ind in enumerate(class_inds):
                        shap_imp += np.abs(self.shap_values[ind]).mean(0)
                    self.shap_values /= len(self.shap_values)

                elif len(self.shap_values.shape) == 3:
                    self.shap_values = np.abs(self.shap_values).sum(axis=0)
                    self.shap_values = self.shap_values.mean(1)

                else:
                    self.shap_values = np.abs(self.shap_values).mean(0)

            else:
                self.shap_values = explainer.shap_values(self.find_sample())
                self.shap_values = np.abs(self.shap_values).mean(0)

        else:

            if self.classification:
                # for some reason shap returns values wraped in a list of length 1
                self.shap_values = np.array(explainer.shap_values(self.X_boruta))
                if isinstance(self.shap_values, list):

                    class_inds = range(len(self.shap_values))
                    shap_imp = np.zeros(self.shap_values[0].shape[1])
                    for i, ind in enumerate(class_inds):
                        shap_imp += np.abs(self.shap_values[ind]).mean(0)
                    self.shap_values /= len(self.shap_values)

                elif len(self.shap_values.shape) == 3:
                    self.shap_values = np.abs(self.shap_values).sum(axis=0)
                    self.shap_values = self.shap_values.mean(1)

                else:
                    self.shap_values = np.abs(self.shap_values).mean(0)

            else:
                self.shap_values = explainer.shap_values(self.X_boruta)
                self.shap_values = np.abs(self.shap_values).mean(0)

    @staticmethod
    def binomial_H0_test(array, n, p, alternative):
        """
        Perform a binomial test for each element in an array.

        This method tests the null hypothesis that the probability of success is `p` 
        in a Bernoulli trial, using a binomial test. Each element in the input array 
        is treated as the number of observed successes out of `n` trials.

        Parameters
        ----------
        array : array-like
            Array of observed success counts (can be float; will be rounded).

        n : int
            Number of trials per test.

        p : float
            Null hypothesis probability of success.

        alternative : {'two-sided', 'greater', 'less'}
            Defines the alternative hypothesis.

        Returns
        -------
        list of float
            List of p-values from the binomial tests for each element in the input array.
        """
        return [binomtest(int(round(x)), n=n, p=p, alternative=alternative).pvalue for x in array]

    @staticmethod
    def find_index_of_true_in_array(array):
        """
        Return the indices of elements that are True in a boolean array.

        Parameters
        ----------
        array : array-like of bool
            Boolean array indicating which elements to select.

        Returns
        -------
        list of int
            Indices where the array has True values.
        """
        length = len(array)

        return list(filter(lambda x: array[x], range(length)))

    @staticmethod
    def bonferoni_corrections(pvals, alpha=0.05, n_tests=None):
        """
        Perform statistical tests to accept or reject features based on hit counts.

        This method compares the number of times each feature outperformed the shadow 
        features ("hits") to the expected distribution under the null hypothesis (p = 0.5), 
        using a binomial test. It applies Bonferroni correction to control for multiple 
        comparisons and classifies features as accepted, rejected, or tentative.

        Parameters
        ----------
        iteration : int
            Current iteration number, used as the number of trials in the binomial test.

        Returns
        -------
        None
            Updates internal attributes:
            - `features_to_remove`: list of features to drop in the next iteration.
            - `accepted_columns`: list of newly accepted features.
            - `rejected_columns`: list of newly rejected features.
        """
        pvals = np.array(pvals)

        if n_tests is None:
            n_tests = len(pvals)
        else:
            pass

        alphacBon = alpha / float(n_tests)
        reject = pvals <= alphacBon
        pvals_corrected = pvals * float(n_tests)

        return reject, pvals_corrected

    def test_features(self, iteration):
        """
        Perform statistical tests to accept or reject features based on accumulated hit counts.

        For each feature, this method performs two binomial hypothesis tests:
        - A right-tailed test to check if the feature is significantly better than random (acceptance).
        - A left-tailed test to check if the feature is significantly worse than random (rejection).

        The tests use the number of times each feature outperformed the shadow feature (stored in `self.hits`)
        over `iteration` trials. Bonferroni correction is applied to control for multiple comparisons.

        Parameters
        ----------
        iteration : int
            The current iteration count, used as the number of trials in the binomial test.

        Returns
        -------
        None
            Updates the following internal attributes:
            - `self.accepted_columns`: list of accepted feature names for this iteration.
            - `self.rejected_columns`: list of rejected feature names for this iteration.
            - `self.features_to_remove`: list of features to remove from `self.X` in the next iteration.
        """
        acceptance_p_values = self.binomial_H0_test(self.hits, n=iteration, p=0.5, alternative='greater')

        regect_p_values = self.binomial_H0_test(self.hits, n=iteration, p=0.5, alternative='less')

        # [1] as function returns a tuple
        modified_acceptance_p_values = self.bonferoni_corrections(acceptance_p_values, alpha=0.05, n_tests=len(self.columns))[1]

        modified_regect_p_values = self.bonferoni_corrections(regect_p_values, alpha=0.05, n_tests=len(self.columns))[1]

        # Take the inverse as we want true to keep featrues
        rejected_columns = np.array(modified_regect_p_values) < self.pvalue
        accepted_columns = np.array(modified_acceptance_p_values) < self.pvalue

        rejected_indices = self.find_index_of_true_in_array(rejected_columns)
        accepted_indices = self.find_index_of_true_in_array(accepted_columns)

        rejected_features = self.all_columns[rejected_indices]
        accepted_features = self.all_columns[accepted_indices]

        self.features_to_remove = rejected_features

        self.rejected_columns.append(rejected_features)
        self.accepted_columns.append(accepted_features)

    @staticmethod
    def create_list(array, color):
        """
        Create a list of repeated color labels for visualization or mapping.

        Parameters
        ----------
        array : array-like
            List of elements (used only to determine length).

        color : str
            The color label or string to repeat.

        Returns
        -------
        list of str
            A list of the same `color` repeated to match the length of `array`.
        """
        colors = [color for x in range(len(array))]
        return colors

    def create_mapping_of_features_to_attribute(self, maps = []):
        """
        Create a dictionary mapping features to attribute labels (e.g., for color or status tagging).

        This method maps each feature—tentative, rejected, accepted, and shadow summary features—
        to a corresponding label or value provided in the `maps` list. It is typically used for
        visualization or export purposes.

        Parameters
        ----------
        maps : list of str
            A list of four strings corresponding to labels for:
            [0] Tentative features  
            [1] Rejected features  
            [2] Accepted features  
            [3] Shadow features (e.g., Max_Shadow, Min_Shadow, etc.)

        Returns
        -------
        dict
            Dictionary mapping each feature name to its corresponding label.
        """
        rejected = list(self.rejected)
        tentative = list(self.tentative)
        accepted = list(self.accepted)
        shadow = ['Max_Shadow','Median_Shadow','Min_Shadow','Mean_Shadow']

        tentative_map = self.create_list(tentative, maps[0])
        rejected_map  = self.create_list(rejected, maps[1])
        accepted_map  = self.create_list(accepted, maps[2])
        shadow_map = self.create_list(shadow, maps[3])

        values = tentative_map + rejected_map + accepted_map + shadow_map
        keys = tentative + rejected + accepted + shadow

        return self.to_dictionary(keys, values)

    @staticmethod
    def to_dictionary(list_one, list_two):
        """
        Create a dictionary by zipping two lists together.

        Parameters
        ----------
        list_one : list
            List of keys.

        list_two : list
            List of values.

        Returns
        -------
        dict
            Dictionary mapping each element in `list_one` to the corresponding element in `list_two`.
        """
        return dict(zip(list_one, list_two))
