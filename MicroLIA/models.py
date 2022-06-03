# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
from warnings import warn

from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, auc, RocCurveDisplay
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.manifold import TSNE

from MicroLIA.optimization import hyper_opt, boruta_opt, KNN_imputation, MissForest_imputation
from MicroLIA import extract_features

def create(data_x, data_y, model='rf', optimize=True, impute=True, imp_method='KNN', n_iter=25,
    save_model=False, path=None):
    """Creates the machine learning engine, current options are either a
    Random Forest or a Neural Network classifier. 
    
    Example:
        If the impute and optimize arguments are set to False, only the Random Forest
        classifier will be returned.

        >>> classifier = create(data_x, data_y, impute=False, optimize=False)

        If impute=True, the imputer will also be returned which is necessary to 
        properly transform unseen, new data.

        >>> classifier, imputer = create(data_x, data_y, impute=True, optimize=False)

        Lastly, if optimize=True, the index of the useful features is also returned.

        >>> classifier, index = create(data_x, data_y, impute=False, optimize=True)

        By default, impute and optimize are set to True, therefore three items are returned.

        >>> classfier, imputer, index = create(data_x, data_y)

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        model (str): Model to use for classification, can either be 'rf' for Random Forest 
            or 'nn' for Neural Network.
        optimize (bool): If True the Boruta algorithm will be run to identify the features
            that contain useful information, after which the optimal Random Forest hyperparameters
            will be calculated using Bayesian optimization. 
        impute (bool): If False no data imputation will be performed. Defaults to True,
            which will result in two outputs, the classifier and the imputer to save
            for future transformations. 
        imp_method (str): The imputation techinque to apply, can either be 'KNN' for k-nearest
            neighbors imputation, or 'MissForest' for the MissForest machine learning imputation
            algorithm. Defaults to 'MissForest'.
        n_iter (int, optional): The maximum number of iterations to perform during 
            the hyperparameter search. Defaults to 25. 
        save_model (bool, optional): If True the machine learning model will be saved to the
            designated path. If path is not set the file will be saved to the local home dir.
            Defaults to False.
        path (str, optional): Absolute path of where to save the model. This path must include
            the desired filename. Defaults to None, in which case the file is saved locally 
            to the home directory.
    
    Note:
        If save_model=True, the path argument must be the absolute path, including the filename,
        of where to save the model. If path=None the file will be saved to the local home directory.

        This file can be loaded in the future and used to make predictions
        
        >>> import joblib
        >>> 
        >>> model = joblib.load(path)
        >>> model.predict(time, mag, magerr, model=model)

    Returns:
        Random Forest classifier model created with scikit-learn. If optimize=True, this
        model will already include the optimal hyperparameters. 
    """

    if impute is False and optimize is False:
        if model == 'rf':
            model = RandomForestClassifier()
        elif model == 'nn':
            model = MLPClassifier()

        model.fit(data_x, data_y)
        Print("Returning base model...")
        return model 

    if impute:
        if imp_method == 'KNN':
            data, imputer = KNN_imputation(data=data_x, imputer=None)
        elif imp_method == 'MissForest':
            warn('MissForest imputation refits every time, do not use to transform new, unseen data; use KNN instead.')
            data, imputer = MissForest_imputation(data=data_x, imputer=None)
        else:
            raise ValueError('Invalid imputation method, currently only k-NN and MissForest algorithms are supported.')
        
        if optimize:
            data_x = data
        else:
            if model == 'rf':
                model = RandomForestClassifier()
            elif model == 'nn':
                model = MLPClassifier()
            model.fit(data, data_y)
            return model, imputer 

    features_index = boruta_opt(data_x, data_y)
    model, best_params = hyper_opt(data_x[:,features_index], data_y, clf=model, n_iter=n_iter)
    print('Hyperparameter optimization complete! Optimal parameters:{}'.format(best_params))
    model.fit(data_x[:,features_index], data_y)

    if save_model:
        if path is None:
            print("No path specified, saving model to local home directory.")
            path = '~/MicroLIA_Model'
        joblib.dump(model, path)

    return model, imputer, features_index

def predict(time, mag, magerr, model, imputer=None, feats_to_use=None):
    """
    Predics the class label of new, unseen data

    Args:
        time (ndarray): Array of observation timestamps.
        mag (ndarray): Array of observed magnitudes.
        magerr (ndarray): Array of corresponding magnitude errors.
        model: The machine learning model to use for predictions.
        imputer: The imputer to use for imputation transformations.
            Defaults to None, in which case no imputation is performed.
        feats_to_use (ndarray): Array containing indices of features
            to use. This will be used to index the columns in the data array.
            Defaults to None, in which case all columns in the data array are used.

    Returns:
        Array containing the classes and the corresponding probability predictions.
    """

    if len(mag) < 30:
        warn('The number of data points is low -- results may be unstable')

    classes = ['CONSTANT', 'CV', 'LPV', 'ML', 'VARIABLE']

    stat_array=[]
    stat_array.append(extract_features.extract_all(time, mag, magerr, convert=True))
    
    if imputer is None and feats_to_use is None:
        pred = model.predict_proba(stat_array)
        return np.c_[classes, pred[0]]

    if imputer is not None:
        stat_array = imputer.transform(stat_array)

    if feats_to_use is not None:
        pred = model.predict_proba(stat_array[:,feats_to_use])
        return np.c_[classes, pred[0]]

    pred = model.predict_proba(stat_array)
    return np.c_[classes, pred[0]]

def plot_tsne(data_x, data_y, norm=False, pca=False, title='Segmentation Parameter Space'):
    """
    Plots a t-SNE projection using the sklearn.manifold.TSNE() method
    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        norm (bool): If True the data will be min-max normalized. Defaults
            to False.
        pca (bool): If True the data will be fit to a Principal Component
            Analysis and all of the corresponding principal components will 
            be used to generate the t-SNE plot. Defaults to False.
        title (str): Title 
    Returns:
        AxesImage. 
    """
    if len(data_x) > 1e3:
        method = 'barnes_hut' #Scales with O(N)
    else:
        method = 'exact' #Scales with O(N^2)

    if norm:
        scaler = MinMaxScaler()
        data_x = scaler.fit_transform(data_x)

    if np.any(np.isnan(data_x)):
        print('Automatically imputing NaN values with the MissForest algorithm.')
        data_x = MissForest_imputation(data=data_x)

    if pca:
        pca_transformation = decomposition.PCA(n_components=data_x.shape[1], whiten=True, svd_solver='auto')
        pca_transformation.fit(data_x) 
        data_x = pca_transformation.transform(data_x)
    
    feats = TSNE(n_components=2, method=method, learning_rate=1000, 
        perplexity=35, init='random').fit_transform(data_x)
    x, y = feats[:,0], feats[:,1]
 
    markers = ['o', '+', '*', 's', 'v', '.', 'x', 'h', 'p', '<', '>']
    feats = np.unique(data_y)

    for count, feat in enumerate(feats):
        if count+1 > len(markers):
            count = -1
        mask = np.where(data_y == feat)[0]
        plt.scatter(x[mask], y[mask], marker=markers[count], label=str(feat), alpha=0.7)

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title(title)
    plt.show()

def plot_conf_matrix(classifier, data_x, data_y, norm=False, pca=False, k_fold=10, normalize=True, classes=["DIFFUSE","OTHER"], title='Confusion matrix'):
    """
    Returns a confusion matrix with k-fold validation.

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        norm (bool): If True the data will be min-max normalized. Defaults
            to False.
        pca (bool): If True the data will be fit to a Principal Component
            Analysis and all of the corresponding principal components will 
            be used to evaluate the classifier and construct the matrix. 
            Defaults to False.
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.
        classes (list): A list containing the label of the two training bags. This
            will be used to set the axis. Defaults to a list containing 'DIFFUSE' & 'OTHER'. 
        title (str, optional): The title of the output plot. 

    Returns:
        AxesImage.
    """

    if len(classes) != 2:
        raise ValueError('Only two training classes are currently supported.')

    if norm:
        data_x = min_max_norm(data_x)

    if pca:
        pca_transformation = decomposition.PCA(n_components=len(data_x[0]), whiten=True, svd_solver='auto')
        pca_transformation.fit(data_x) 
        pca_data = pca_transformation.transform(data_x)
        data_x = np.asarray(pca_data).astype('float64')
    
    predicted_target, actual_target = evaluate_model(classifier, data_x, data_y, normalize=normalize, k_fold=k_fold)
    generate_matrix(predicted_target, actual_target, normalize=normalize, classes=classes, title=title)

def plot_roc_curve(classifier, data_x, data_y, k_fold=10, title="Receiver Operating Characteristic Curve"):
    """
    Plots ROC curve with k-fold cross-validation, as such the 
    standard deviation variations are plotted.
    
    Example:
        To assess the performance of a random forest classifier (created
        using the scikit-learn implementation) we can run the following:
        
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> classifier = RandomForestClassifier()
        >>> plot_roc_curve(classifier, data_x, data_y)
    
    Args:
        classifier: The machine learning classifier to optimize.
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.
        title (str, optional): The title of the output plot. 
    
    Returns:
        AxesImage
    """
    
    cv = StratifiedKFold(n_splits=k_fold)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    train = data_x
    fig, ax = plt.subplots()
    for i, (data_x, test) in enumerate(cv.split(train, data_y)):
        classifier.fit(train[data_x], data_y[data_x])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            train[test],
            data_y[test],
            name="ROC fold {}".format(i+1),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[0, 1.0],
        ylim=[0.0, 1.0],
        title="Receiver Operating Characteristic Curve",
    )
    ax.legend(loc="lower right")
    plt.ylabel('True Positive Rate', size=14)
    plt.xlabel('False Positive Rate', size=14)
    plt.title(label=title,fontsize=18)
    plt.show()

def evaluate_model(classifier, data_x, data_y, normalize=True, k_fold=10):
    """
    Cross-checks model accuracy and outputs both the predicted
    and the true class labels. 

    Args:
        classifier: The machine learning classifier to optimize.
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.

    Returns:
        The first output is the 1D array of the true class labels.
        The second output is the 1D array of the predicted class labels.
    """

    k_fold = KFold(k_fold, shuffle=True, random_state=1)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train_ix, test_ix in k_fold.split(data_x):
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]
        # Fit the classifier
        classifier.fit(train_x, train_y)
        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(test_x)
        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)

    return predicted_targets, actual_targets

def generate_matrix(predicted_labels_list, actual_targets, normalize=True, classes=["DIFFUSE","OTHER"], title='Confusion matrix'):
    """
    Generates the confusion matrix using the output from the evaluate_model() function.

    Args:
        predicted_labels_list: 1D array containing the predicted class labels.
        actual_targets: 1D array containing the actual class labels.
        normalize (bool, optional): If True the matrix accuracy will be normalized
            and displayed as a percentage accuracy. Defaults to True.
        classes (list): A list containing the label of the two training bags. This
            will be used to set the axis. Defaults to a list containing 'DIFFUSE' & 'OTHER'. 
        title (str, optional): The title of the output plot. 

    Returns:
        AxesImage.
    """

    conf_matrix = confusion_matrix(actual_targets, predicted_labels_list)
    np.set_printoptions(precision=2)

    plt.figure()
    if normalize == True:
        generate_plot(conf_matrix, classes=classes, normalize=normalize, title='Confusion matrix, without normalization')
    elif normalize == False:
        generate_plot(conf_matrix, classes=classes, normalize=normalize, title='Normalized Confusion Matrix')
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
    plt.title(title, fontsize=20)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize is True else 'd'
    thresh = conf_matrix.max() / 2.

    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=16)
    plt.xlabel('Predicted label',fontsize=16)

    return conf_matrix

def min_max_norm(data_x):
    """
    Normalizes the data to be between 0 and 1. NaN values are ignored.
    The transformation matrix will be returned as it will be needed
    to consitently normalize new data.
    
    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.

    Returns:
        Normalized data array.
    """

    Ny, Nx = data_x.shape
    new_array = np.zeros((Ny, Nx))
    
    for i in range(Nx):
        print((np.max(data_x[:,i]) - np.min(data_x[:,i])))
        new_array[:,i] = (data_x[:,i] - np.min(data_x[:,i])) / (np.max(data_x[:,i]) - np.min(data_x[:,i]))

    return new_array
