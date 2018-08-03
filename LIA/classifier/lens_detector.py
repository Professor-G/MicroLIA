# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from warnings import warn

import sys
sys.path.append('../features')
from extract_features import extract_all

def create_models(all_feats, pca_feats):
    """Creates the RF model and PCA tranformation used for classificaition.
    
    Parameters
    ----------
    all_feats : name of txt_file
        Text file containing all features and class label.
    pca_stats : name of txt_file
        Text file containing PCA features and class label.  

    Returns
    -------
    rf_model : fn
        Trained random forest ensemble. 
    pca_model : fn
        PCA transformation.
    """
    coeffs = np.loadtxt(all_feats,usecols=np.arange(1,48))
    pca = decomposition.PCA(n_components=int(np.ceil(min(coeffs.shape)/2.))+2, whiten=True, svd_solver='auto')
    pca.fit(coeffs)
    #feat_strengths = pca.explained_variance_ratio_
    training_set = np.loadtxt(pca_feats, dtype = str)
    rf=RandomForestClassifier(n_estimators=1500, max_depth = 16, max_features=5, min_samples_split=16)
    rf.fit(training_set[:,np.arange(1,27)].astype(float),training_set[:,0])

    return rf, pca

def predict_class(mag, magerr, rf_model, pca_model):
    """This function uses machine learning to classify any given lightcurve as either
        a Cataclysmic Variable (CV), a Variable source, Microlensing, or a constant star 
        displaying no variability.
        
    Parameters
    ----------
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    convert : boolean, optional 
        If False the features are computed with the inpute magnitudes,
        defaults to True to convert and compute in flux. 
  
    Returns
    -------
    prediction : str
        Predicted class.
    ml_pred : float
        Probability source is microlensing
    cons_pred : float
        Probability source is constant
    cv_pred : float
        Probability source is CV
    var_pred : float
        Probability source is variable
    """
    if len(mag) < 10:
        warn('The number of data points low -- results may be unstable')

    array=[]
    stat_array = array.append(extract_all(mag, magerr, convert=False))
    array=np.array(array)
    stat_array = pca_model.transform(array)
    
    prediction =rf_model.predict(stat_array)
    cons_pred = rf_model.predict_proba(stat_array)[:,0] #CONSTANT
    cv_pred = rf_model.predict_proba(stat_array)[:,1] #CV
    ml_pred = rf_model.predict_proba(stat_array)[:,2] #ML
    var_pred = rf_model.predict_proba(stat_array)[:,3] #VARIABLE
    
    return prediction, ml_pred, cons_pred, cv_pred, var_pred
"""
def predict_and_fit(mjd, mag, magerr):

    
    prediction, p1,p2,p3,p4 = predict_class(mag,magerr)
    
    if prediction == 'ML':
        print 'Event detected with', np.float(p1)*100,'% confidence. Now fitting with pyLIMA...'
        #creates temporary file to avoid error when inputting data to pylima#
        with open('temporary.txt', 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            mjd = [np.float(x) for x in mjd]
            mag = [np.float(x) for x in mag]
            magerr = [np.float(x) for x in magerr]
            writer.writerows(zip(mjd, mag, magerr))
        
        data = np.loadtxt('temporary.txt')
        your_event = event.Event()
        your_event.name = 'Detection'
        
        telescope_1 = telescopes.Telescope(name='Detection', camera_filter='i', light_curve_magnitude=data)
        your_event.telescopes.append(telescope_1)
        model_1 = microlmodels.create_model('PSPL', your_event)
        your_event.fit(model_1,'LM')
        errors = your_event.fits[0].fit_covariance.diagonal()**0.5
        to_err, uo_err, tE_err = [errors[i] for i in range(3)]
        
        if to_err*tE_err*uo_err == 0.0:
            your_event.fit(model_1,'DE', DE_population_size=1000)
            #errors = your_event.fits[0].fit_covariance.diagonal()**0.5
            params = your_event.fits[0].fit_results
            to, uo, tE, Chi2 = [params[i] for i in [1,2,3,5]]
            reduced_chi = 10.0
        
        else:
            params = your_event.fits[0].fit_results
            to, uo, tE, Chi2 = [params[i] for i in [1,2,3,5]]
            reduced_chi = Chi2/(len(mjd)-4.0)
        
        os.remove('temporary.txt')
        if tE >= 1 and uo != 0 and uo < 2.0 and reduced_chi <= 10.0:# and len(np.argwhere(((to - np.abs(tE)) < mjd) & ((to + np.abs(tE)) > mjd))) >= 2:
            print 'Microlensing candidate detected with PSPL parameters { uo:', uo,'|tE:', tE,'|to:', to,'|RedChi:',reduced_chi,'}'
        else:
            prediction = 'OTHER'
            print 'False alert -- fitted PSPL parameters not within range. |RedChi:', reduced_chi

    return prediction, np.float(probability_prediction)#, np.float(cv_pred), np.float(lyr_pred), np.float(other_pred), np.float(cons_pred)

"""
