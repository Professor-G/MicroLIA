# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 14:30:12 2017
    
    @author: danielgodinez
"""
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Union, List, Tuple
from inspect import getmembers, isfunction

from MicroLIA import features

def extract_all(
    time: ArrayLike,
    mag: ArrayLike,
    magerr: ArrayLike,
    apply_weights: bool = True,
    feats_to_use: Optional[List[int]] = None,
    convert: bool = True,
    zp: float = 24,
    return_names: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    This function will compute the statistics used to train the RF.
    Amplitude dependent features are computed first, after which the
    mag/flux is normalized by the maximum value to compute the remanining
    features. By default a conversion from mag to flux is performed. If input
    is in flux or you wish to work in mag, set convert to False.  
    
    Parameters:
    ----------
    time : array
        Time of observations
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    apply_weights : bool, optional
        Whether to apply weights based on the magnitude errors. Defaults to True.
    feats_to_use : array
        Array containing indices of features to use. This will be used to index the columns 
        in the data array. Defaults to None, in which case all columns in the data array are used.
    convert : boolean, optional 
        If False the features are computed with the input magnitudes,
        defaults to True to convert and compute in flux. 
    zp : float
        Zeropoint of the instrument, only used if convert=True. Defaults to 24.
    return_names : bool
        If True the first output will be the stats array, and the second
        will be the list of corresponding feature names. Defaults to False,
        in which case only the stats array is returned.

    Returns:
    -------
    array
        All features to use for classification.
    """

    if isinstance(time, np.ndarray) is False:
        if type(time) == list:
            time = np.array(time)
        else:
            raise ValueError('The time argument must be a list or array.')

    if isinstance(mag, np.ndarray) is False:
        if type(mag) == list:
            mag = np.array(mag)
        else:
            raise ValueError('The mag argument must be a list or array.')

    if isinstance(magerr, np.ndarray) is False:
        if type(magerr) == list:
            magerr = np.array(magerr)
        else:
            raise ValueError('The magerr argument must be a list or array.')

    #Remove the nan and inf values, if present in the lightcurve
    mask = np.where(np.isfinite(time) & np.isfinite(mag) & np.isfinite(magerr))[0]
    time, mag, magerr = time[mask], mag[mask], magerr[mask]

    #Ensure lightcurve is sorted by timestamps
    if len(time) > 1 and not np.all(np.diff(time) > 0):
        print("WARNING: time array is not sorted! Sorting automatically...")
        sort_idx = np.argsort(time)
        time, mag, magerr = time[sort_idx], mag[sort_idx], magerr[sort_idx]

    if convert is True:
        flux = 10**(-(mag - zp) / 2.5)
        flux_err = (magerr * flux) / (2.5) * np.log(10)
    elif convert is False:
        flux, flux_err = mag, magerr

    # Normalize by max flux
    norm_flux = flux / np.max(flux)
    norm_fluxerr = flux_err * (norm_flux / flux)
    
    # Retrive all the statistical metrics from the features module
    all_features_functions = getmembers(features, isfunction)

    stats, feature_names = [], []

    #Normal space
    counter = 0
    for func in all_features_functions:
        if feats_to_use is not None:
            if counter not in feats_to_use:
                counter += 1; continue

        if func[0] == 'amplitude' or func[0] == 'median_buffer_range':
            try:
                feature = func[1](time, flux, flux_err, apply_weights=apply_weights)  #amplitude dependent features use non-normalized flux
            except:# (ZeroDivisionError, ValueError, IndexError):
                feature = np.nan
        else:
            try:
                feature = func[1](time, norm_flux, norm_fluxerr, apply_weights=apply_weights)
            except:# (ZeroDivisionError, ValueError, IndexError):
                feature = np.nan

        feature_names.append(func[0]); stats.append(feature)
        counter += 1

    # Derivative space
    dx, dy = np.gradient(time), np.gradient(flux)

    flux_deriv = dy / dx
    flux_deriv_err = np.sqrt(
        (np.gradient(flux_deriv, time) / flux_deriv) ** 2 * flux_err**2
    )
    mask_1 = np.where(
        np.isfinite(time) & np.isfinite(flux_deriv) & np.isfinite(flux_deriv_err)
    )[0]

    norm_flux_deriv = flux_deriv[mask_1] / np.max(flux_deriv[mask_1])
    norm_flux_deriv_err = np.sqrt(
        (np.gradient(norm_flux_deriv, time[mask_1]) / norm_flux_deriv) ** 2
        * norm_fluxerr[mask_1] ** 2
    )
    mask_2 = np.where(
        np.isfinite(time[mask_1])
        & np.isfinite(norm_flux_deriv)
        & np.isfinite(norm_flux_deriv_err)
    )[0]

    for func in all_features_functions:
        if feats_to_use is not None:
            if counter not in feats_to_use:
                counter += 1; continue

        if func[0] == 'amplitude' or func[0] == 'median_buffer_range':
            try:
                feature = func[1](
                    time[mask_1],
                    flux_deriv[mask_1],
                    flux_deriv_err[mask_1],
                    apply_weights=apply_weights,
                )  # amplitude dependent features use non-normalized flux
            except:  # (ZeroDivisionError, ValueError, IndexError):
                feature = np.nan
        else:
            try:
                feature = func[1](
                    time[mask_1][mask_2],
                    norm_flux_deriv[mask_2],
                    norm_flux_deriv_err[mask_2],
                    apply_weights=apply_weights,
                )
            except:  # (ZeroDivisionError, ValueError, IndexError):
                feature = np.nan

        feature_names.append(func[0]+'_deriv'); stats.append(feature)
        counter += 1

    stats = np.array(stats)

    # Ensure non-finite values are set to NaN
    stats[np.isfinite(stats) == False] = np.nan
    
    # Float limits (models break otherwise)
    stats[stats > 1e10], stats[(stats<1e-10) & (stats>0)], stats[stats < -1e10] = 1e10, 1e-10, -1e10

    if return_names is False:
        return stats
    else:
        return stats, feature_names
