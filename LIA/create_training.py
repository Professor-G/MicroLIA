# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""
import numpy as np
import random
from astropy.io import fits
from sklearn import decomposition
import os

import sys
sys.path.append('/lib')
import simulate
from noise_models import add_noise, add_gaussian_noise, create_noise
from quality_check import test_microlensing, test_cv
from extract_features import extract_all
    
def create_training(timestamps, min_base=14, max_base=21, fn=None, q=1000):
    """Creates a training dataset using adaptive cadence.
    Simulates each class q times, adding errors from
    a noise model either defined using the create_noise
    function, or Gaussian by default

    Parameters
    __________
    timestamps : array of arrays
		Times at which to simulate the different lightcurves.
		Must be an array containing all possible timestamps combinations.
    min_base : float, optional
		Minimum baseline magnitude for simulating lightcurves.
		Defaults to 14. 
    max_base : float, optional 
		Maximum baseline magnitude for simulating lightcurves.
		Defaults to 21.
    fn : function, optional 
		Noise model, can be created using the create_noise function.
		If None it defaults to adding Gaussian noise. 
    q : int, optional
		The amount of lightcurve (per class) to simulate.
        Defaults to 1000. 

    Outputs
    _______
    dataset : FITS
		All simulated lightcurves in a FITS file, sorted by class and ID
    all_features : txt file
		A txt file containing all the features plus class label.
    pca_stats : txt file
        A txt file containing all PCA features plus class label. 
    """
    if q < 12:
        raise ValueError("q must be at least 12 for principal components to be calculated.")

    times_list=[]
    mag_list=[]
    magerr_list=[]
    id_list = []
    source_class_list=[]
    stats_list = []

    print "Now simulating variables..."
    for i in range(1,q+1):
        time = random.choice(timestamps)
        baseline = np.random.uniform(min_base,max_base)
        mag, amplitude, period = simulate.variable(time,baseline)
           
        if fn is not None:
		mag, magerr = add_noise(mag, fn)
        if fn is None:
           mag, magerr = add_gaussian_noise(mag)

        source_class = ['VARIABLE']*len(time)
        source_class_list.append(source_class)

        id_num = [i]*len(time)
        id_list.append(id_num)

        times_list.append(time)
        mag_list.append(mag)
        magerr_list.append(magerr)
        
        stats = extract_all(mag,magerr)
        stats = [i for i in stats]
        stats = ['VARIABLE'] + stats
        stats_list.append(stats)
        
    print "Variables successfully simulated"
    print "Now simulating constants..."
    for i in range(1,q+1):
        time = random.choice(timestamps)
        baseline = np.random.uniform(min_base, max_base)
        mag = simulate.constant(time, baseline)
        
        if fn is not None:
		mag, magerr = add_noise(mag, fn)
        if fn is None:
           mag, magerr = add_gaussian_noise(mag)
           
        source_class = ['CONSTANT']*len(time)
        source_class_list.append(source_class)

        id_num = [2*q+i]*len(time)
        id_list.append(id_num)

        times_list.append(time)
        mag_list.append(mag)
        magerr_list.append(magerr)
        
        stats = extract_all(mag,magerr)
        stats = [i for i in stats]
        stats = ['CONSTANT'] + stats
        stats_list.append(stats)
        
    print "Constants successfully simulated"
    print "Now simulating CV..."
    for i in range(1,q+1):
        for k in range(10000):
            time = random.choice(timestamps)
            baseline = np.random.uniform(min_base, max_base)
            mag, burst_start_times, burst_end_times, end_rise_times, end_high_times= simulate.cv(time, baseline)
            
            quality = test_cv(time, burst_start_times, burst_end_times, end_rise_times, end_high_times)
            if quality is True:
                try:
                    if fn is not None:
                        mag, magerr=add_noise(mag,fn)
                    if fn is None:
                        mag, magerr=add_gaussian_noise(mag)
                except ValueError:
                    continue
                
                source_class = ['CV']*len(time)
                source_class_list.append(source_class)
                id_num = [3*q+i]*len(time)
                id_list.append(id_num)
            
                times_list.append(time)
                mag_list.append(mag)
                magerr_list.append(magerr)
                
                stats = extract_all(mag,magerr)
                stats = [i for i in stats]
                stats = ['CV'] + stats
                stats_list.append(stats)
                break
            if k == 9999:
                raise RuntimeError('Unable to simulate proper CV in 10k tries with current cadence -- inspect cadence and try again.')
    
    print "CVs successfully simulated"
    print "Now simulating microlensing..."
    for i in range(1,q+1):
        for k in range(10000):
            time = random.choice(timestamps)
            baseline = np.random.uniform(min_base, max_base)
            mag, baseline, u_0, t_0, t_e, blend_ratio = simulate.microlensing(time, baseline)
            try:
                if fn is not None:
                    mag, magerr=add_noise(mag,fn)
                if fn is None:
                    mag, magerr=add_gaussian_noise(mag)
            except ValueError:
                continue
                
            quality = test_microlensing(time, mag, magerr, baseline, u_0, t_0, t_e, blend_ratio)
            if quality is True:
                
                source_class = ['ML']*len(time)
                source_class_list.append(source_class)
                id_num = [4*q+i]*len(time)
                id_list.append(id_num)
            
                times_list.append(time)
                mag_list.append(mag)
                magerr_list.append(magerr)
                
                stats = extract_all(mag,magerr)
                stats = [i for i in stats]
                stats = ['ML'] + stats
                stats_list.append(stats)
                break
            if k == 9999:
                raise RuntimeError('Unable to simulate proper ML in 10k tries with current cadence -- inspect cadence and/or noise model and try again.')
                    
    print "Microlensing events successfully simulated"
    print "Writing files..."
    col0 = fits.Column(name='Class', format='20A', array=np.hstack(source_class_list))
    col1 = fits.Column(name='ID', format='E', array=np.hstack(id_list))
    col2 = fits.Column(name='time', format='E', array=np.hstack(times_list))
    col3 = fits.Column(name='mag', format='E', array=np.hstack(mag_list))
    col4 = fits.Column(name='magerr', format='E', array=np.hstack(magerr_list))
    cols = fits.ColDefs([col0, col1, col2, col3, col4])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto('lightcurves.fits')
    
    "Saving features..."
    output_file = open('feats.txt','w')
    print(output_file)
    for line in stats_list:
        print >>output_file, line
    output_file.close()
    
    with open(r'feats.txt', 'r') as infile, open(r'all_features.txt', 'w') as outfile:
         
         data = infile.read()
         data = data.replace("'", "")
         data = data.replace(",", "")
         data = data.replace("[", "")
         data = data.replace("]", "")
         outfile.write(data)
         
    os.remove('feats.txt')
    "Computing principal components..."
    coeffs = np.loadtxt('all_features.txt',usecols=np.arange(1,48))
    pca = decomposition.PCA(n_components=int(np.ceil(min(coeffs.shape)/2.))+2, whiten=True, svd_solver='auto')
    pca.fit(coeffs)
    #feat_strengths = pca.explained_variance_ratio_
    X_pca = pca.transform(coeffs) 
    
    f = open('pca_features.txt', 'w')
    classes = ["VARIABLE"]*q+["CONSTANT"]*q+["ML"]*q+["CV"]*q
    np.savetxt(f,np.column_stack(
        ((classes), (np.array(X_pca[:,0])), (np.array(X_pca[:,1])), (np.array(X_pca[:,2])),
         (np.array(X_pca[:,3])), (np.array(X_pca[:,4])), (np.array(X_pca[:,5])), (np.array(X_pca[:,6])), 
         (np.array(X_pca[:,7])),(np.array(X_pca[:,8])), (np.array(X_pca[:,9])), (np.array(X_pca[:,10])),
         (np.array(X_pca[:,11])),(np.array(X_pca[:,12])),(np.array(X_pca[:,13])),(np.array(X_pca[:,14])),
         (np.array(X_pca[:,15])),(np.array(X_pca[:,16])),(np.array(X_pca[:,17])),(np.array(X_pca[:,18])),
         (np.array(X_pca[:,19])),(np.array(X_pca[:,20])),(np.array(X_pca[:,21])),(np.array(X_pca[:,22])),
         (np.array(X_pca[:,23])),(np.array(X_pca[:,24])), (np.array(X_pca[:,25])))), fmt='%5s')
         
    # For unknown reasons np.savetxt does not always print the final lines, this iteration 
    # is to circumnavigate this bug -- embarrasing, I know.
    for i in range(100):
        try:
            np.loadtxt('pca_features.txt',dtype=str)
            break
        except ValueError:
            np.savetxt(f,np.column_stack(
            ((classes), (np.array(X_pca[:,0])), (np.array(X_pca[:,1])), (np.array(X_pca[:,2])),
             (np.array(X_pca[:,3])), (np.array(X_pca[:,4])), (np.array(X_pca[:,5])), (np.array(X_pca[:,6])), 
             (np.array(X_pca[:,7])),(np.array(X_pca[:,8])), (np.array(X_pca[:,9])), (np.array(X_pca[:,10])),
             (np.array(X_pca[:,11])),(np.array(X_pca[:,12])),(np.array(X_pca[:,13])),(np.array(X_pca[:,14])),
             (np.array(X_pca[:,15])),(np.array(X_pca[:,16])),(np.array(X_pca[:,17])),(np.array(X_pca[:,18])),
             (np.array(X_pca[:,19])),(np.array(X_pca[:,20])),(np.array(X_pca[:,21])),(np.array(X_pca[:,22])),
             (np.array(X_pca[:,23])),(np.array(X_pca[:,24])), (np.array(X_pca[:,25])))), fmt='%5s')
    print "Complete!"