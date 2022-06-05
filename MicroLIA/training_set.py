# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""
import os
import random
from pathlib import Path
import pkg_resources
from warnings import warn
from pathlib import Path
from inspect import getmembers, isfunction

import numpy as np
from astropy.io import fits
from progress import bar
from astropy.io.votable import parse_single_table
from sklearn import decomposition

from MicroLIA import simulate
from MicroLIA import noise_models
from MicroLIA import quality_check
from MicroLIA import extract_features
from MicroLIA import features
    
def create(timestamps, min_mag=14, max_mag=21, noise=None, zp=24, exptime=60, 
    n_class=500, ml_n1=7, cv_n1=7, cv_n2=1, t0_dist=None, u0_dist=None, tE_dist=None, 
    filename='', test=False, save_file=True):
    """
    Creates a training dataset using adaptive cadence.
    Simulates each class n_class times, adding errors from
    a noise model either defined using the create_noise
    function, or Gaussian by default

    Parameters
    __________
    timestamps : list of arrays
        Times at which to simulate the different lightcurves.
        Must be an array containing all possible timestamps combinations.
    min_mag : float, optional
        Minimum baseline magnitude for simulating lightcurves.
        Defaults to 14. 
    max_mag : float, optional 
        Maximum baseline magnitude for simulating lightcurves.
        Defaults to 21.
    noise : function, optional 
        Noise model, can be created using the create_noise function.
        If None it defaults to adding Gaussian noise. 
    n_class : int, optional
        The amount of lightcurve (per class) to simulate.
        Defaults to 500. 
    ml_n1 : int, optional
        The mininum number of measurements that should be within the 
        microlensing signal when simulating the lightcurves. 
    cv_n1 : int, optional
        The mininum number of measurements that should be within 
        at least one CV outburst when simulating the lightcurves.
    cv_n2 : int, optional
        The mininum number of measurements that should be within the 
        rise or drop of at least one CV outburst when simulating the lightcurves.
    t0_dist: array, optional
        An array containing the minumum and maximum t0 value to be 
        considered during the microlensing simulations. The indivial
        t0 per simulation will be selected from a uniform distribution
        between these two values.
    u0_dist: array, optional
        An array containing the minumum and maximum u0 value to be 
        considered during the microlensing simulations. The indivial
        u0 per simulation will be selected from a uniform distribution
        between these two values.
    te_dist: array, optional
        An array containing the minumum and maximum tE value to be 
        considered during the microlensing simulations. The indivial
        tE per simulation will be selected from a uniform distribution
        between these two values.
    test: bool, optional
        If False there will be no classification reports after training.
        Defaults to True.
    save_file: bool
        If True the lightcurve.fits and all_features.txt files will be
        saved to the home directory. Defaults to True.

    Outputs
    _______
    dataset : FITS
        All simulated lightcurves in a FITS file, sorted by class and ID
    all_features : txt file
        A txt file containing all the features plus class label and ID.
    """

    if len(getmembers(features, isfunction))*2 > n_class*5:
        raise ValueError("Parameter n_class must be at least "+str(int(1+len(getmembers(features, isfunction))*2//5))+" for principal components to be computed.")
    
    while True:
        try:
            len(timestamps[0])
            break
        except TypeError:
            raise ValueError("Incorrect format -- append the timestamps to a list and try again.")

    times_list=[]
    mag_list=[]
    magerr_list=[]
    id_list = []
    source_class_list=[]
    stats_list = []

    progess_bar = bar.FillingSquaresBar('Simulating variables...', max=n_class)
    for k in range(1,n_class+1):
        time = random.choice(timestamps)
        baseline = np.random.uniform(min_mag,max_mag)
        mag, amplitude, period = simulate.variable(time,baseline)
           
        if noise is not None:
            mag, magerr = noise_models.add_noise(mag, noise, zp=zp, exptime=exptime)
        if noise is None:
           mag, magerr = noise_models.add_gaussian_noise(mag, zp=zp, exptime=exptime)
           
        source_class = ['VARIABLE']*len(time)
        source_class_list.append(source_class)

        id_num = [k]*len(time)
        id_list.append(id_num)

        times_list.append(time)
        mag_list.append(mag)
        magerr_list.append(magerr)
        
        stats = extract_features.extract_all(time, mag, magerr, convert=True, zp=zp)
        stats = [i for i in stats]
        stats = ['VARIABLE'] + [k] + stats
        stats_list.append(stats)
        progess_bar.next()
    progess_bar.finish()

    progess_bar = bar.FillingSquaresBar('Simulating constants...', max=n_class)
    for k in range(1,n_class+1):
        time = random.choice(timestamps)
        baseline = np.random.uniform(min_mag,max_mag)
        mag = simulate.constant(time, baseline)
        
        if noise is not None:
            mag, magerr = noise_models.add_noise(mag, noise, zp=zp, exptime=exptime)
        if noise is None:
           mag, magerr = noise_models.add_gaussian_noise(mag, zp=zp, exptime=exptime)
           
        source_class = ['CONSTANT']*len(time)
        source_class_list.append(source_class)

        id_num = [1*n_class+k]*len(time)
        id_list.append(id_num)

        times_list.append(time)
        mag_list.append(mag)
        magerr_list.append(magerr)
        
        stats = extract_features.extract_all(time, mag, magerr, convert=True, zp=zp)
        stats = [i for i in stats]
        stats = ['CONSTANT'] + [1*n_class+k] + stats
        stats_list.append(stats) 
        progess_bar.next()  
    progess_bar.finish()

    progess_bar = bar.FillingSquaresBar('Simulating CV...', max=n_class)   
    for k in range(1,n_class+1):
        for j in range(100):
            if j > 20:
                warn('Taking longer than usual to simulate CV... this happens if the timestamps are too sparse \
                as it takes longer to simulate lightcurves that pass the quality check. The process will break after \
                one hundred attempts, if this happens you can try setting the outburst parameter cv_n1 to a value between 2 and 6.')
            time = random.choice(timestamps)
            baseline = np.random.uniform(min_mag,max_mag)
            mag, burst_start_times, burst_end_times, end_rise_times, end_high_times = simulate.cv(time, baseline)
            
            quality = quality_check.test_cv(time, burst_start_times, burst_end_times, end_rise_times, end_high_times, n1=cv_n1, n2=cv_n2)
            if quality is True:
                try:
                    if noise is not None:
                        mag, magerr = noise_models.add_noise(mag, noise, zp=zp, exptime=exptime)
                    if noise is None:
                        mag, magerr = noise_models.add_gaussian_noise(mag, zp=zp, exptime=exptime)
                except ValueError:
                    continue
                
                source_class = ['CV']*len(time)
                source_class_list.append(source_class)
                id_num = [2*n_class+k]*len(time)
                id_list.append(id_num)
            
                times_list.append(time)
                mag_list.append(mag)
                magerr_list.append(magerr)
                
                stats = extract_features.extract_all(time, mag, magerr, convert=True, zp=zp)
                stats = [i for i in stats]
                stats = ['CV'] + [2*n_class+k] + stats
                stats_list.append(stats)
                progess_bar.next()
                break

            if j == 99:
                raise RuntimeError('Unable to simulate proper CV in 100 tries with current cadence -- inspect cadence and try again.')
    progess_bar.finish()

    progess_bar = bar.FillingSquaresBar('Simulating microlensing...', max=n_class)  
    for k in range(1,n_class+1):
        for j in range(100):
            if j > 20:
                warn('Taking longer than usual to simulate ML... this happens if the timestamps are too sparse \
                as it takes longer to simulate lightcurves that pass the quality check. The process will break after \
                one hundred attempts, if this happens you can try setting the event parameter ml_n1 to a value between 2 and 6.')
            time = random.choice(timestamps)
            baseline = np.random.uniform(min_mag,max_mag)
            mag, baseline, u_0, t_0, t_e, blend_ratio = simulate.microlensing(time, baseline, t0_dist, u0_dist, tE_dist)

            try:
                if noise is not None:
                    mag, magerr = noise_models.add_noise(mag, noise, zp=zp, exptime=exptime)
                if noise is None:             
                    mag, magerr= noise_models.add_gaussian_noise(mag, zp=zp, exptime=exptime)
            except ValueError:
                continue
                
            quality = quality_check.test_microlensing(time, mag, magerr, baseline, u_0, t_0, t_e, blend_ratio, n=ml_n1)
            if quality is True:          
                source_class = ['ML']*len(time)
                source_class_list.append(source_class)
                id_num = [3*n_class+k]*len(time)
                id_list.append(id_num)
            
                times_list.append(time)
                mag_list.append(mag)
                magerr_list.append(magerr)
               
                stats = extract_features.extract_all(time, mag, magerr, convert=True, zp=zp)
                stats = [i for i in stats]
                stats = ['ML'] + [3*n_class+k] + stats
                stats_list.append(stats)
                progess_bar.next()
                break

            if j == 99:
                raise RuntimeError('Unable to simulate proper ML in 100 tries with current cadence -- inspect cadence and/or noise model and try again.')
    progess_bar.finish()
    
    progess_bar = bar.FillingSquaresBar('Simulating LPV...', max=n_class)  
    resource_package = __name__
    resource_path = '/'.join(('data', 'Miras_vo.xml'))
    template = pkg_resources.resource_filename(resource_package, resource_path)
    mira_table = parse_single_table(template)
    primary_period = mira_table.array['col4'].data
    amplitude_pp = mira_table.array['col5'].data
    secondary_period = mira_table.array['col6'].data
    amplitude_sp = mira_table.array['col7'].data
    tertiary_period = mira_table.array['col8'].data
    amplitude_tp = mira_table.array['col9'].data

    for k in range(1,n_class+1):
        time = random.choice(timestamps)
        baseline = np.random.uniform(min_mag,max_mag)
        mag = simulate.simulate_mira_lightcurve(time, baseline, primary_period, amplitude_pp, secondary_period, amplitude_sp, tertiary_period, amplitude_tp)
    
        try:
            if noise is not None:
                mag, magerr = noise_models.add_noise(mag, noise, zp=zp, exptime=exptime)
            if noise is None:             
                mag, magerr = noise_models.add_gaussian_noise(mag, zp=zp, exptime=exptime)
        except ValueError:
            continue
                
        source_class = ['LPV']*len(time)
        source_class_list.append(source_class)

        id_num = [4*n_class+k]*len(time)
        id_list.append(id_num)

        times_list.append(time)
        mag_list.append(mag)
        magerr_list.append(magerr)
        
        stats = extract_features.extract_all(time, mag, magerr, convert=True, zp=zp)
        stats = [i for i in stats]
        stats = ['LPV'] + [4*n_class+k] + stats
        stats_list.append(stats)
        progess_bar.next()
    progess_bar.finish()

    if save_file:
        print('Writing files to home directory...')
        path = str(Path.home())+'/'

        col0 = fits.Column(name='Class', format='20A', array=np.hstack(source_class_list))
        col1 = fits.Column(name='ID', format='E', array=np.hstack(id_list))
        col2 = fits.Column(name='time', format='D', array=np.hstack(times_list))
        col3 = fits.Column(name='mag', format='E', array=np.hstack(mag_list))
        col4 = fits.Column(name='magerr', format='E', array=np.hstack(magerr_list))
        cols = fits.ColDefs([col0, col1, col2, col3, col4])
        hdu = fits.BinTableHDU.from_columns(cols)

        fname = Path('lightcurves'+filename+'.fits')
        if fname.exists(): #To avoid error if file already exists
            fname.unlink()
        hdu.writeto(path+str(fname),overwrite=True)

        np.savetxt(path+'feats.txt',np.array(stats_list).astype(str),fmt='%s')
        with open(path+'feats.txt', 'r') as infile, open(path+'all_features'+filename+'.txt', 'w') as outfile:    
             data = infile.read()
             data = data.replace("'", "")
             data = data.replace(",", "")
             data = data.replace("[", "")
             data = data.replace("]", "")
             outfile.write(data)
        os.remove(path+'feats.txt')
    print("Simulation complete!")

    """
    if test == True:
        print("")
        print("------------------------------")
        print("Creating testing data set...")
        print("------------------------------")
        quality_check.create_test(timestamps, min_mag, max_mag, noise, zp, exptime, n_class, ml_n1, cv_n1, cv_n2, t0_dist, u0_dist, tE_dist, 
            'all_features'+filename+'.txt', 'pca_features'+filename+'.txt')
    """
