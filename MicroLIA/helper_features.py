#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 10:05:51 2023

@author: daniel
"""
import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Sequence


def _safe_weights(magerr: ArrayLike) -> np.ndarray:
    """
    Inverse-variance weights that are finite and positive.

    Parameters
    ----------
    magerr : array-like
        Photometric errors.

    Returns
    -------
    w : ndarray
        Array of weights (1 / magerr²), with invalid entries set to 0.
    """

    w = np.zeros_like(magerr, dtype=float)
    good = np.isfinite(magerr) & (magerr > 0)
    w[good] = 1.0 / magerr[good]**2

    return w

def _weighted_percentiles(x: ArrayLike, w: ArrayLike, q: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
    """
    Weighted *value* percentiles.

    Parameters
    ----------
    x : 1-D array (already sorted ascending)
    w : matching 1-D weights (non-negative, same order as x)
    q : scalar or sequence in [0,1]   (e.g. 0.95 or [0.05, 0.4, 0.6, 0.95])

    Returns
    -------
    percentiles : ndarray
    """
    x   = np.asarray(x, float)
    w   = np.asarray(w, float)
    q   = np.atleast_1d(q)

    if x.size == 0 or w.sum() == 0:
        return np.full_like(q, np.nan, dtype=float)

    cdf = np.cumsum(w) / w.sum()
    return np.interp(q, cdf, x)

def _frac_sigma(mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, sign: int = 1) -> float:
    """
    Fraction of points more than 1σ above (sign=+1) or below (sign=−1) the median.

    Parameters
    ----------
    mag : array-like
        Magnitude or flux values.
    magerr : array-like
        Associated errors.
    sign : int, default=1
        Use +1 for upper tail or −1 for lower tail.
    apply_weights : bool, default=True
        Whether to use weighted average.

    Returns
    -------
    frac : float
        Fraction of outliers.
    """

    median = np.median(mag)
    sigma = np.std(mag, ddof=0)

    if sigma == 0:
        return 0.0

    sel = sign * (mag - median) > sigma   # bool mask

    if not apply_weights:
        return sel.mean()

    w = _safe_weights(magerr)

    return np.sum(w[sel]) / w.sum() if w.sum() > 0 else sel.mean()

def _weighted_percentile(data: ArrayLike, weights: ArrayLike, percentile: float) -> float:
    """
    Compute the weighted percentile of a 1D array.

    Parameters
    ----------
    data : array-like
        Input data values.
    weights : array-like
        Non-negative weights associated with each data point. Must be the same length as `data`.
    percentile : float
        Desired percentile in the range [0, 100].

    Returns
    -------
    value : float
        The weighted percentile value of the input data.
    """

    data = np.asarray(data)
    weights = np.asarray(weights)

    sorter = np.argsort(data)
    data = data[sorter]
    weights = weights[sorter]

    cumsum = np.cumsum(weights)
    cutoff = percentile / 100.0 * cumsum[-1]

    return data[np.searchsorted(cumsum, cutoff)]

def _first_sig_digit(arr: ArrayLike) -> np.ndarray:
    """
    First significant digit (1–9) of absolute values.

    Parameters
    ----------
    arr : array-like

    Returns
    -------
    digits : int array
        Vectorized output, same shape as input.
    """

    arr = np.abs(arr).astype(float)
    out = np.zeros_like(arr, dtype=int)

    good = np.isfinite(arr) & (arr > 0)
    if not good.any():
        return out

    exps = np.floor(np.log10(arr[good])).astype(int) # integer orders of magnitude
    base = np.power(10.0, exps) # **10.0** not 10 !
    out[good] = np.floor(arr[good] / base).astype(int)

    return out

def _dup_with_tol(values: ArrayLike, errs: ArrayLike, tol_factor: float = 2.0) -> int:
    """
    Detect duplicate values within a tolerance defined by errors.

    Parameters
    ----------
    values : array-like
        Magnitude or flux values.
    errs : array-like
        Associated errors.
    tol_factor : float
        Scaling factor for the tolerance.

    Returns
    -------
    int
        1 if duplicates are found within tolerance, else 0.
    """

    n = values.size

    for i in range(n - 1):
        d = np.abs(values[i+1:] - values[i])
        tol = tol_factor * np.sqrt(errs[i]**2 + errs[i+1:]**2)
        if np.any(d <= tol):
            return 1

    return 0

def _weighted_median(x: ArrayLike, w: ArrayLike) -> float:
    """
    Weighted median of data `x` with weights `w`.

    Parameters
    ----------
    x : array-like
        Data values.
    w : array-like
        Associated weights.

    Returns
    -------
    median : float
        Weighted median.
    """

    idx = np.argsort(x)

    return _weighted_percentiles(x[idx], w[idx], 0.5)[0]

def _flux_percentile_ratio(mag: ArrayLike, magerr: ArrayLike, p_lo: float, p_hi: float, apply_weights: bool = True) -> float:
    """
    (p_hi − p_lo) / (95th − 5th) percentile flux ratio, with optional weighting.

    Parameters
    ----------
    mag : array-like
        Magnitude or flux values.
    magerr : array-like
        Associated errors.
    p_lo, p_hi : float
        Percentile bounds, e.g., 0.4 and 0.6.
    apply_weights : bool
        Whether to apply inverse-variance weights.

    Returns
    -------
    ratio : float
        Flux percentile ratio.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        p5, plo, phi, p95 = np.percentile(mag, [5, p_lo*100, p_hi*100, 95])
    else:
        idx = np.argsort(mag)
        w = _safe_weights(magerr)[idx]
        x = mag[idx]
        p5, plo, phi, p95 = _weighted_percentiles(x, w, [0.05, p_lo, p_hi, 0.95])

    num = phi - plo
    den = p95 - p5

    return num / den if den != 0 else np.nan

def _longest_true_run(mask: np.ndarray) -> int:
    """
    Length of the longest consecutive True subsequence.

    Parameters
    ----------
    mask : 1D bool array

    Returns
    -------
    max_run : int
        Length of the longest run of True values.
    """

    if not mask.any():
        return 0

    # run-length encoding via np.diff
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    lengths = idx[1::2] - idx[::2]
    
    return lengths.max()

def _delta(mag: ArrayLike, magerr: ArrayLike) -> np.ndarray:
    """
    Stetson normalized residuals (single-band).

    Parameters
    ----------
    mag : array-like
        Magnitudes.
    magerr : array-like
        Magnitude errors.

    Returns
    -------
    delta : ndarray
        Normalized residuals.
    """

    n = mag.size

    return np.sqrt(n / (n - 1.0)) * (mag - np.median(mag)) / magerr

