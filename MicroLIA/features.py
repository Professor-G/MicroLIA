# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 14:30:12 2017
    
    @author: danielgodinez
"""
import numpy as np
from scipy.special import erf         
from scipy.stats import invgauss    
from scipy import signal as ssignal
import scipy.stats as sstats
from numpy.typing import ArrayLike

from MicroLIA import helper_features

def abs_energy(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Compute the absolute energy of a light curve.

    This statistic measures the total signal energy, defined as the sum of squared amplitudes.
    It is useful as a general indicator of overall variability, but does not consider time structure.

    Parameters
    ----------
    time : array-like
        Time values corresponding to each measurement. Not used in this function, but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with `mag`.
    apply_weights : bool, optional (default=True)
        Whether to apply inverse-variance weighting using `magerr`.

    Returns
    -------
    float
        Absolute energy of the light curve.

    Notes
    -----
    - If `apply_weights=True`, the energy is computed as a weighted sum using weights ∝ 1 / magerr².
    - Input can be magnitudes or fluxes, but **normalized flux** (e.g., min-max scaling) is recommended to ensure consistent interpretation across objects.
    - This feature is insensitive to time sampling and **does not require even sampling**.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if not apply_weights:
        return np.dot(mag, mag)

    w = helper_features._safe_weights(magerr)

    return np.sum(w * mag**2)

def abs_sum_changes(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Compute the absolute sum of changes in a light curve.

    This metric quantifies the total variation by summing the absolute differences 
    between consecutive measurements. It serves as a simple indicator of 
    short-timescale variability.

    Parameters
    ----------
    time : array-like
        Time values corresponding to each measurement. Not used in this function, but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with `mag`.
    apply_weights : bool, optional (default=True)
        If True, differences are weighted by the inverse uncertainty in each pair of measurements (1 / sqrt(σ_i² + σ_{i+1}²)).

    Returns
    -------
    float
        Sum of absolute changes, optionally weighted by uncertainty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Useful for identifying short-term variability or stochastic behavior.
    - Not robust to outliers unless pre-processed or normalized.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size < 2:
        return 0.0

    dmag = np.abs(np.diff(mag))

    if not apply_weights:
        return np.sum(dmag)

    derr = np.sqrt(magerr[:-1]**2 + magerr[1:]**2)
    valid = derr > 0

    return np.sum(dmag[valid] / derr[valid])

def above1(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of points more than 1σ above the median.

    This statistic measures the proportion of light curve points that lie 
    more than one standard deviation above the median value. It can be used 
    as a simple indicator of asymmetric variability or outburst-like behavior.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted fraction using inverse-variance weighting.

    Returns
    -------
    float
        Fraction of data points that are >1σ above the median.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Best used with **normalized flux**, but compatible with magnitudes or raw flux values.
    - Sensitive to outliers and skewed distributions; robust normalization is recommended if noise is high.
    """

    return helper_features._frac_sigma(np.asarray(mag, float), np.asarray(magerr, float), apply_weights=apply_weights, sign=+1)

def above3(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of points more than 3σ above the median.

    This statistic quantifies the fraction of light curve values that exceed the
    median by more than three times the local uncertainty (3σ). It highlights strong
    positive deviations that may correspond to flares, outbursts, or anomalies.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted fraction using inverse-variance weighting.

    Returns
    -------
    float
        Fraction of data points that are >3σ above the median.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Best used with **normalized flux**, but compatible with magnitudes or raw flux values.
    - Sensitive to outliers and skewed distributions; robust normalization is recommended if noise is high.
    """

    return helper_features._frac_sigma(np.asarray(mag, float), np.asarray(magerr, float), apply_weights=apply_weights, sign=+3)

def above5(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of points more than 5σ above the median.

    This statistic calculates the proportion of light curve values that are more than
    five standard deviations above the median. It is useful for identifying extreme 
    positive outliers such as strong flares, transients, or artifacts.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted fraction using inverse-variance weighting.

    Returns
    -------
    float
        Fraction of data points that are >5σ above the median.

    Notes
    -----
    - Suitable for light curves with **uneven sampling**, since time information is not used.
    - Best used with **normalized flux**, but compatible with magnitudes or raw flux values.
    - Sensitive to outliers and skewed distributions; robust normalization is recommended if noise is high.
    """

    return helper_features._frac_sigma(np.asarray(mag, float), np.asarray(magerr, float), apply_weights=apply_weights, sign=+5)

def amplitude(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, pct_clip: float = 1.0) -> float:
    """
    Estimate the amplitude of a light curve using clipped percentiles.

    This statistic computes the difference between the upper and lower percentile 
    bounds of the light curve, which serves as a robust proxy for amplitude. 
    The percentile clipping (e.g., 1st–99th) reduces the influence of outliers.

    Parameters
    ----------
    time : array-like
        Time values corresponding to each measurement. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, applies inverse-variance weighting when computing percentiles.
    pct_clip : float, optional (default=1.0)
        Lower and upper percentiles used to define amplitude. For example, `pct_clip=1.0` computes the 1st–99th percentile range. Must be between 0 and 50.

    Returns
    -------
    float
        Estimated amplitude (difference between high and low clipped percentiles).

    Notes
    -----
    - Robust to noise and outliers due to percentile-based clipping.
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Best used with **normalized flux**, but compatible with magnitudes or raw flux values.
    - If `magerr` is poorly estimated or zero everywhere, unweighted percentiles are used.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        lo = np.percentile(mag, pct_clip)
        hi = np.percentile(mag, 100.0 - pct_clip)
        return hi - lo

    # weighted version
    w = helper_features._safe_weights(magerr)
    if w.sum() == 0:
        lo = np.percentile(mag, pct_clip)
        hi = np.percentile(mag, 100.0 - pct_clip)
    else:
        lo = helper_features._weighted_percentile(mag, w, pct_clip)
        hi = helper_features._weighted_percentile(mag, w, 100.0 - pct_clip)

    return hi - lo

def AndersonDarling(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Compute the Anderson–Darling statistic as a measure of normality.

    This function evaluates how well the distribution of light curve values fits a 
    Gaussian profile, using the Anderson–Darling test. A logistic transformation 
    is applied to the statistic to bound the output between 0 and 1, with higher 
    values indicating stronger deviation from normality.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, the Anderson–Darling test is applied to the standardized residuals using inverse-variance weights.

    Returns
    -------
    float
        A logistic-transformed Anderson–Darling score between 0 and 1, where higher values
        indicate greater deviation from normality.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Best applied to **normalized flux** to reduce scaling sensitivity.
    - The output is **nonlinearly scaled** using a logistic transformation of the A² statistic, bounding it between 0 and 1.
    - When `apply_weights=True`, the test uses weighted estimates for the mean and variance.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        a2 = sstats.anderson(mag, dist='norm').statistic
    else:
        w = helper_features._safe_weights(magerr)
        mu = np.sum(w * mag) / w.sum() if w.sum() else np.mean(mag)
        var = np.sum(w * (mag - mu)**2) / w.sum() if w.sum() else np.var(mag, ddof=0)
        z = np.sort((mag - mu) / np.sqrt(var))
        n = z.size
        eps = np.finfo(float).eps # smallest representable positive number to avoid nans below
        cdf = np.clip(sstats.norm.cdf(z), eps, 1 - eps)
        i = np.arange(1, n + 1)
        a2 = -n - np.sum((2 * i - 1) * (np.log(cdf) + np.log1p(-cdf[::-1]))) / n

    return 1.0 / (1.0 + np.exp(-10.0 * (a2 - 0.3)))

def auto_corr(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Compute the lag-1 autocorrelation of a light curve.

    This metric measures the correlation between adjacent points in the light curve,
    i.e., how similar each value is to its immediate neighbor. A high value indicates
    smooth, slowly varying behavior; a low or negative value suggests more random
    or rapidly changing signals.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, applies inverse-variance weighting when computing the autocorrelation.

    Returns
    -------
    float
        Lag-1 autocorrelation coefficient. Returns NaN if the input is too short
        or the denominator is zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Best used with **normalized flux**, but compatible with magnitudes or raw flux values.
    - If `apply_weights=True`, weighted means and covariances are used.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    n = mag.size
    if n < 2:
        return np.nan

    if not apply_weights:
        return np.corrcoef(mag[:-1], mag[1:])[0, 1]

    w = helper_features._safe_weights(magerr)
    wp = np.sqrt(w[:-1] * w[1:])
    if wp.sum() == 0:
        return np.corrcoef(mag[:-1], mag[1:])[0, 1]

    mu = np.sum(w * mag) / w.sum()
    num = np.sum(wp * (mag[:-1] - mu) * (mag[1:] - mu))
    den = np.sum(w * (mag - mu) ** 2)

    return num / den if den > 0 else np.nan

def below1(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of points more than 1σ below the median.

    This statistic measures the proportion of light curve values that lie more than 
    one standard deviation below the median. It serves as an indicator of 
    dimming events or asymmetric variability skewed toward fainter fluxes.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted fraction using inverse-variance weighting.

    Returns
    -------
    float
        Fraction of data points that are >1σ fainter than the median.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    return helper_features._frac_sigma(np.asarray(mag, float), np.asarray(magerr, float), apply_weights=apply_weights, sign=-1)

def below3(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of points more than 3σ below the median.

    This statistic measures the proportion of light curve values that are more than
    three standard deviations fainter than the median. It is designed to capture
    deep dimming events, eclipses, or significant negative outliers.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted fraction using inverse-variance weighting.

    Returns
    -------
    float
        Fraction of data points that are >3σ fainter than the median.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    return helper_features._frac_sigma(np.asarray(mag, float), np.asarray(magerr, float), apply_weights=apply_weights, sign=-3)

def below5(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of points more than 5σ below the median.

    This statistic measures the proportion of light curve values that are more than
    five standard deviations fainter than the median. It is particularly useful for
    detecting rare or extreme dimming events such as deep eclipses or dropouts.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted fraction using inverse-variance weighting.

    Returns
    -------
    float
        Fraction of data points that are >5σ fainter than the median.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    return helper_features._frac_sigma(np.asarray(mag, float), np.asarray(magerr, float), apply_weights=apply_weights, sign=-5)

def benford_correlation(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Correlation with Benford's Law based on first significant digit frequencies.

    This statistic compares the distribution of first significant digits in the 
    light curve values to the expected distribution from Benford's Law. A high 
    correlation indicates that the data follow Benford-like behavior, which has 
    been proposed as a signature of natural, noise-like variability.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes weighted digit frequencies using inverse-variance weights.

    Returns
    -------
    float
        Pearson correlation coefficient between the observed digit distribution and 
        the theoretical Benford distribution. Returns NaN if input is empty or 
        contains no valid digits.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Best used with **raw or log-transformed fluxes**; normalized flux may suppress digit diversity.
    - First significant digits are extracted by computing: floor(|x| / 10^floor(log10(|x|))) for each nonzero, finite value x in `mag`. For example: 345.2 --> 3, 0.012 --> 1
    - Benford's Law expects first digits to follow a logarithmic distribution:  `P(d) = log10(1 + 1/d)`, for `d ∈ {1,...,9}`.
    - May serve as a statistical regularity check or anomaly detector.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    digits = helper_features._first_sig_digit(mag)
    mask = digits > 0
    digits = digits[mask]
    if digits.size == 0:
        return np.nan

    benford = np.log10(1 + 1.0 / np.arange(1, 10))

    if not apply_weights:
        data_freq = np.array([(digits == d).mean() for d in range(1, 10)])
    else:
        w = helper_features._safe_weights(magerr)[mask]
        if w.sum() == 0:
            data_freq = np.array([(digits == d).mean() for d in range(1, 10)])
        else:
            data_freq = np.array([w[digits == d].sum() for d in range(1, 10)])
            data_freq = data_freq / data_freq.sum()

    return np.corrcoef(benford, data_freq)[0, 1]

def c3(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, lag: int = 1) -> float:
    """
    Third-order autocorrelation statistic (C₃) of a light curve.

    This feature measures the average product of triplets of light curve values 
    separated by a fixed lag. It captures higher-order temporal structure, such as 
    phase correlations and coherent trends beyond simple pairwise autocorrelation.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, each triplet is weighted by the inverse variance of the combined uncertainty.
    lag : int, optional (default=1)
        Time step (in array index units) between elements in the triplet. The function evaluates the product of values at positions (i, i+lag, i+2*lag).

    Returns
    -------
    float
        Mean third-order product of lagged triplets, optionally weighted. Returns NaN
        if the array is too short for the specified lag.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Can detect asymmetries and nonlinear temporal correlations not captured by standard autocorrelation.
    - Requires at least `2 * lag + 1` points to be valid.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    n = mag.size
    if n < 2 * lag + 1:
        return np.nan

    x0 = mag[:-2*lag]
    x1 = mag[lag:-lag]
    x2 = mag[2*lag:]
    triple = x0 * x1 * x2

    if not apply_weights:
        return triple.mean()

    w = 1.0 / (magerr[:-2*lag]**2 + magerr[lag:-lag]**2 + magerr[2*lag:]**2)

    if w.sum() == 0:
        return triple.mean()

    return np.sum(w * triple) / w.sum()

def check_for_duplicate(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Check for duplicate or nearly duplicate light curve values.

    This function detects whether the input light curve contains repeated values. 
    If `apply_weights=True`, values are considered duplicates if they are 
    indistinguishable within photometric uncertainty using a tolerance-based comparison.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses a tolerance-based comparison that accounts for measurement errors. If False, checks for exact duplicates.

    Returns
    -------
    int
        1 if any duplicates are detected, 0 otherwise.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Useful for detecting flat or non-variable signals. 
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    return float(helper_features._dup_with_tol(mag, magerr)) if apply_weights else float(mag.size != np.unique(mag).size)

def check_for_max_duplicate(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Check for duplicate maximum values in a light curve.

    This function determines whether the **maximum value** in the light curve appears 
    more than once. If `apply_weights=True`, values are considered equal if they are 
    within 3σ of each other, accounting for photometric uncertainty.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, duplicates of the maximum value are detected within a tolerance of 3 times the corresponding measurement error. If False, an exact match is required.

    Returns
    -------
    int
        1 if the maximum value (within tolerance) appears more than once, 0 otherwise.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Useful for detecting plateau-like maxima or saturation effects in light curves.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    max_val = np.max(mag)
    idx = np.where(np.isclose(mag, max_val, atol=(3 * magerr if apply_weights else 0), rtol=0))[0]

    return float(idx.size > 1)

def check_for_min_duplicate(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Check for duplicate minimum values in a light curve.

    This function determines whether the **minimum value** in the light curve appears 
    more than once. If `apply_weights=True`, values are considered equal if they are 
    within 3σ of each other, accounting for photometric uncertainty.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, duplicates of the minimum value are detected within a tolerance of 3 times the corresponding measurement error. If False, an exact match is required.

    Returns
    -------
    int
        1 if the minimum value (within tolerance) appears more than once, 0 otherwise.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Useful for identifying repeated dimming events, eclipses, or floor effects in light curves.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    min_val = np.min(mag)
    idx = np.where(np.isclose(mag, min_val, atol=(3 * magerr if apply_weights else 0), rtol=0))[0]

    return float(idx.size > 1)

def check_max_last_loc(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Relative position of the last occurrence of the maximum value in a light curve.

    This function checks where (in normalized index units) the last occurrence of the 
    maximum value appears in the light curve. If `apply_weights=True`, matches to the 
    maximum value are allowed within a 3σ photometric uncertainty tolerance.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, the maximum match is evaluated using a tolerance of 3 times the measurement error. If False, an exact match is used.

    Returns
    -------
    float
        A value between 0 and 1 indicating how close to the end of the time series the 
        last occurrence of the maximum value occurs. Returns NaN if the input is empty 
        or no match is found.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Useful for detecting whether a light curve ends with a peak (e.g., rising events, flares).
    - A value close to 1 means the max value appears near the **start** of the time series; close to 0 means it appears near the **end**.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    max_val = np.max(mag)
    tol = 3 * magerr if apply_weights else 0
    idx = np.where(np.isclose(mag, max_val, atol=tol, rtol=0))[0]

    return 1.0 - idx[-1] / mag.size if idx.size else np.nan

def check_min_last_loc(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Relative position of the last occurrence of the minimum value in a light curve.

    This function determines where (in normalized index units) the last occurrence of 
    the minimum value appears in the light curve. If `apply_weights=True`, values 
    within 3σ of the minimum are treated as equivalent matches.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, the minimum match is evaluated using a tolerance of 3 times the 
        measurement error. If False, an exact match is used.

    Returns
    -------
    float
        A value between 0 and 1 indicating how close to the end of the time series the 
        last occurrence of the minimum value appears. Returns NaN if the input is empty 
        or no match is found.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Useful for detecting events where dimming or troughs occur at the end of the observation window.
    - A value near 0 implies the minimum occurred near the **end**; near 1 implies it occurred near the **start** of the light curve.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    min_val = np.min(mag)
    tol = 3 * magerr if apply_weights else 0
    idx = np.where(np.isclose(mag, min_val, atol=tol, rtol=0))[0]

    return 1.0 - idx[-1] / mag.size if idx.size else np.nan

def complexity(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Estimate the local complexity of a light curve using root-mean-square of differences.

    This feature quantifies short-timescale variability by computing the RMS of 
    first-order differences in the light curve. It reflects the degree of irregularity in the signal 
    and is sensitive to noise, flickering, and fast variability.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, weights the squared differences using the inverse variance from `magerr`.

    Returns
    -------
    float
        Root-mean-square of first-order differences, optionally weighted. Returns NaN
        if the input length is less than 2 or weights are zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Sensitive to both real high-frequency variability and measurement noise.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size < 2:
        return np.nan

    diff = np.diff(mag)

    if not apply_weights:
        return np.sqrt(np.mean(diff**2))

    w = helper_features._safe_weights(magerr[:-1])

    if w.sum() == 0:
        return np.sqrt(np.mean(diff**2))

    return np.sqrt(np.sum(w * diff**2) / w.sum())

def con_above1(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of clusters of ≥3 consecutive significantly bright points.

    This metric identifies and counts "clusters" of three or more consecutive light-curve 
    points that are ≥1σ brighter than the baseline magnitude. The result is normalized 
    by the total number of light-curve points.

    Parameters
    ----------
    time : array-like
        Time stamps of the light curve. Not used directly in the calculation.
    mag : array-like
        Light curve values. Can fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, applies a per-point threshold of 1 × `magerr` relative to the median. If False, uses a global 1σ threshold based on the standard deviation of `mag`.

    Returns
    -------
    float
        Fraction of light curve points that are part of clusters of ≥3 consecutive 
        ≥1σ bright excursions. Defined as `N_clusters / N_points`.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Used to detect **bursting** or **flare-like** variability patterns.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size < 3:
        return 0.0

    baseline = np.median(mag)

    if apply_weights:
        dev_mask = mag >= (baseline + 1.0 * magerr)
    else:
        sigma = np.std(mag, ddof=1)
        dev_mask = mag >= (baseline + 1.0 * sigma)

    clusters = 0
    run_len  = 0
    for bright in dev_mask:
        if bright:
            run_len += 1
            if run_len == 3:
                clusters += 1
        else:
            run_len = 0

    return clusters / mag.size

def con_above3(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of clusters of ≥3 consecutive significantly bright points.

    This metric identifies and counts "clusters" of three or more consecutive light-curve 
    points that are ≥3σ brighter than the baseline magnitude. The result is normalized 
    by the total number of light-curve points.

    Parameters
    ----------
    time : array-like
        Time stamps of the light curve. Not used directly in the calculation.
    mag : array-like
        Light curve values. Can be fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, applies a per-point threshold of 3 × `magerr` relative to the median. If False, uses a global 3σ threshold based on the standard deviation of `mag`.

    Returns
    -------
    float
        Fraction of light curve points that are part of clusters of ≥3 consecutive 
        ≥3σ bright excursions. Defined as `N_clusters / N_points`.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Used to detect **bursting** or **flare-like** variability patterns.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size < 3:
        return 0.0

    baseline = np.median(mag)

    if apply_weights:
        dev_mask = mag >= (baseline + 3.0 * magerr)
    else:
        sigma = np.std(mag, ddof=1)
        dev_mask = mag >= (baseline + 3.0 * sigma)

    clusters = 0
    run_len  = 0
    for bright in dev_mask:
        if bright:
            run_len += 1
            if run_len == 3:
                clusters += 1
        else:
            run_len = 0

    return clusters / mag.size

def con_above5(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of clusters of ≥3 consecutive significantly bright points.

    This metric identifies and counts "clusters" of three or more consecutive light-curve 
    points that are ≥5σ brighter than the baseline magnitude. The result is normalized 
    by the total number of light-curve points.

    Parameters
    ----------
    time : array-like
        Time stamps of the light curve. Not used directly in the calculation.
    mag : array-like
        Light curve values. Can fluxes, or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, applies a per-point threshold of 5 × `magerr` relative to the median. If False, uses a global 5σ threshold based on the standard deviation of `mag`.

    Returns
    -------
    float
        Fraction of light curve points that are part of clusters of ≥3 consecutive 
        ≥5σ bright excursions. Defined as `N_clusters / N_points`.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Used to detect **bursting** or **flare-like** variability patterns.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size < 3:
        return 0.0

    baseline = np.median(mag)

    if apply_weights:
        dev_mask = mag >= (baseline + 5.0 * magerr)
    else:
        sigma = np.std(mag, ddof=1)
        dev_mask = mag >= (baseline + 5.0 * sigma)

    clusters = 0
    run_len  = 0
    for bright in dev_mask:
        if bright:
            run_len += 1
            if run_len == 3:
                clusters += 1
        else:
            run_len = 0

    return clusters / mag.size

def con_below1(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of clusters of ≥3 consecutive significantly dim points.

    This metric identifies and counts "clusters" of three or more consecutive light-curve 
    points that are ≥1σ dimmer than the baseline magnitude. The result is normalized 
    by the total number of light-curve points.

    Parameters
    ----------
    time : array-like
        Time stamps of the light curve. Not used directly in the calculation.
    mag : array-like
        Light curve values. Can be fluxes or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, applies a per-point threshold of 1 × `magerr` relative to the median. If False, uses a global 1σ threshold based on the standard deviation of `mag`.

    Returns
    -------
    float
        Fraction of light curve points that are part of clusters of ≥3 consecutive 
        ≥1σ dim excursions. Defined as `N_clusters / N_points`.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Useful for detecting **dips**, **eclipses**, or **transits**.
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size < 3:
        return 0.0

    baseline = np.median(mag)

    if apply_weights:
        dev_mask = mag <= (baseline - 1.0 * magerr)
    else:
        sigma = np.std(mag, ddof=1)
        dev_mask = mag <= (baseline - 1.0 * sigma)

    clusters = 0
    run_len  = 0
    for dim in dev_mask:
        if dim:
            run_len += 1
            if run_len == 3:
                clusters += 1
        else:
            run_len = 0

    return clusters / mag.size

def con_below3(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of clusters of ≥3 consecutive significantly dim points.

    This metric identifies and counts "clusters" of three or more consecutive light-curve 
    points that are ≥3σ dimmer than the baseline magnitude. The result is normalized 
    by the total number of light-curve points.

    Parameters
    ----------
    time : array-like
        Time stamps of the light curve. Not used directly in the calculation.
    mag : array-like
        Light curve values. Can be fluxes or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, applies a per-point threshold of 3 × `magerr` relative to the median. If False, uses a global 3σ threshold based on the standard deviation of `mag`.

    Returns
    -------
    float
        Fraction of light curve points that are part of clusters of ≥3 consecutive 
        ≥3σ dim excursions. Defined as `N_clusters / N_points`.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Useful for detecting **dips**, **eclipses**, or **transits**.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size < 3:
        return 0.0

    baseline = np.median(mag)

    if apply_weights:
        dev_mask = mag <= (baseline - 3.0 * magerr)
    else:
        sigma = np.std(mag, ddof=1)
        dev_mask = mag <= (baseline - 3.0 * sigma)

    clusters = 0
    run_len  = 0
    for dim in dev_mask:
        if dim:
            run_len += 1
            if run_len == 3:
                clusters += 1
        else:
            run_len = 0

    return clusters / mag.size

def con_below5(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of clusters of ≥3 consecutive significantly dim points.

    This metric identifies and counts "clusters" of three or more consecutive light-curve 
    points that are ≥5σ dimmer than the baseline magnitude. The result is normalized 
    by the total number of light-curve points.

    Parameters
    ----------
    time : array-like
        Time stamps of the light curve. Not used directly in the calculation.
    mag : array-like
        Light curve values. Can be fluxes or normalized fluxes.
    magerr : array-like
        Measurement uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, applies a per-point threshold of 5 × `magerr` relative to the median. If False, uses a global 5σ threshold based on the standard deviation of `mag`.

    Returns
    -------
    float
        Fraction of light curve points that are part of clusters of ≥3 consecutive 
        ≥5σ dim excursions. Defined as `N_clusters / N_points`.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Useful for detecting **dips**, **eclipses**, or **transits**.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size < 3:
        return 0.0

    baseline = np.median(mag)

    if apply_weights:
        dev_mask = mag <= (baseline - 5.0 * magerr)
    else:
        sigma = np.std(mag, ddof=1)
        dev_mask = mag <= (baseline - 5.0 * sigma)

    clusters = 0
    run_len  = 0
    for dim in dev_mask:
        if dim:
            run_len += 1
            if run_len == 3:
                clusters += 1
        else:
            run_len = 0

    return clusters / mag.size

def count_above(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of light curve points above the median.

    This metric calculates the proportion of light curve values that are greater than
    the median. It is a simple measure of asymmetry or skewness in the distribution 
    of values. When `apply_weights=True`, the comparison is made to the weighted 
    median, and the result is a weighted fraction.

    Parameters
    ----------
    time : array-like
        Time stamps of the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weights to compute the median and resulting fraction. If False, performs an unweighted comparison to the simple median.

    Returns
    -------
    float
        Fraction of values greater than the (weighted) median. Returns a weighted 
        sum of points above the median if `apply_weights=True`.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if not apply_weights:
        return (mag > np.median(mag)).sum() / mag.size

    w = helper_features._safe_weights(magerr)
    if w.sum() == 0:
        return (mag > np.median(mag)).sum() / mag.size

    med = helper_features._weighted_median(mag, w)

    return w[mag > med].sum() / w.sum()

def count_below(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of light curve points below the median.

    This metric computes the proportion of light curve values that are less than
    the median. It is a simple measure of asymmetry or skewness in the distribution 
    of values. When `apply_weights=True`, the comparison is made to the weighted 
    median, and the result is a weighted fraction.

    Parameters
    ----------
    time : array-like
        Time stamps of the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weights to compute the median and resulting fraction. If False, performs an unweighted comparison to the simple median.

    Returns
    -------
    float
        Fraction of values less than the (weighted) median. Returns a weighted 
        sum of points below the median if `apply_weights=True`.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if not apply_weights:
        return (mag < np.median(mag)).sum() / mag.size

    w = helper_features._safe_weights(magerr)
    if w.sum() == 0:
        return (mag < np.median(mag)).sum() / mag.size

    med = helper_features._weighted_median(mag, w)

    return w[mag < med].sum() / w.sum()

def cusum(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Cumulative sum (CUSUM) variability index.

    This statistic quantifies the overall deviation from stationarity by computing 
    the normalized cumulative sum of residuals from the median. The final value is 
    the range between the maximum and minimum of the CUSUM series. It captures 
    gradual trends, systematic drifts, or long-term changes in the light curve.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting to compute a weighted standard deviation. If False, uses the unweighted standard deviation.

    Returns
    -------
    float
        The maximum cumulative deviation from the median, normalized by light curve 
        length and scatter. Returns NaN if the input is empty and 0.0 if the scatter is zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Sensitive to slow drifts or systematic trends in the light curve.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    med = np.median(mag)

    if not apply_weights:
        sig = np.std(mag, ddof=0)
    else:
        w = helper_features._safe_weights(magerr)
        sig = np.sqrt(np.sum(w * (mag - med)**2) / w.sum()) if w.sum() else np.std(mag, ddof=0)

    if sig == 0:
        return 0.0

    c = np.cumsum(mag - med) / (mag.size * sig)

    return np.max(c) - np.min(c)

def first_loc_max(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Normalized location of the first occurrence of the maximum value.

    This feature identifies the first (or weighted) index at which the light curve 
    reaches its maximum value and returns its position normalized by the total 
    number of points. It provides a simple temporal indicator of where brightening 
    events occur in the light curve.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, the product of `mag * weight` is used to determine the maximum value, where weight is the inverse-variance from `magerr`. If False, the raw maximum is used.

    Returns
    -------
    float
        Index of the first occurrence of the (weighted) maximum, normalized by array size.
        Returns NaN if the input is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        return np.argmax(mag) / mag.size

    w = helper_features._safe_weights(magerr)
    idx = np.argmax(mag * w)

    return idx / mag.size

def first_loc_min(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Normalized location of the first occurrence of the minimum value.

    This feature identifies the first (or weighted) index at which the light curve 
    reaches its minimum value and returns its position normalized by the total 
    number of points. It provides a simple temporal indicator of where dimming 
    events occur in the light curve.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, the product of `mag * weight` is used to determine the minimum value,
        where weight is the inverse-variance from `magerr`. If False, the raw minimum
        is used.

    Returns
    -------
    float
        Index of the first occurrence of the (weighted) minimum, normalized by array size.
        Returns NaN if the input is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        return np.argmin(mag) / mag.size

    w = helper_features._safe_weights(magerr)
    idx = np.argmin(mag * w)

    return idx / mag.size

def FluxPercentileRatioMid20(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Flux percentile ratio for the middle 20% of the light curve.

    This metric computes the ratio of the flux range between the 40th and 60th percentiles 
    to the full flux range (2nd to 98th percentiles). It captures the concentration of 
    values near the median and helps distinguish compact versus broad distributions.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting when computing percentiles.

    Returns
    -------
    float
        Ratio of the 40–60th percentile range to the 2–98th percentile range.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Higher values indicate that more of the light curve is concentrated near the median.
    """

    return helper_features._flux_percentile_ratio(mag, magerr, 0.40, 0.60, apply_weights)

def FluxPercentileRatioMid35(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Flux percentile ratio for the middle 35% of the light curve.

    Computes the ratio of the 32.5–67.5th percentile flux range to the total flux range 
    (2nd to 98th percentiles). This measures how tightly flux values cluster around 
    the center of the distribution.

    Parameters
    ----------
    time : array-like
        Time values. Not used in the calculation.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Associated 1σ uncertainties.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting for percentile calculations.

    Returns
    -------
    float
        Ratio of the 32.5–67.5 percentile range to the 2–98 percentile range.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Useful for identifying symmetric, compact light curve profiles.
    """

    return helper_features._flux_percentile_ratio(mag, magerr, 0.325, 0.675, apply_weights)

def FluxPercentileRatioMid50(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Flux percentile ratio for the middle 50% of the light curve.

    Computes the ratio of the interquartile range (25th to 75th percentile) to the 
    total flux range (2nd to 98th percentiles), measuring the spread of the 
    central portion of the distribution.

    Parameters
    ----------
    time : array-like
        Time values. Not used in the calculation.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Associated 1σ uncertainties.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting for percentile calculations.

    Returns
    -------
    float
        Ratio of the 25–75 percentile range to the 2–98 percentile range.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Equivalent to the **interquartile flux ratio** normalized to the full flux spread.
    """

    return helper_features._flux_percentile_ratio(mag, magerr, 0.25, 0.75, apply_weights)

def FluxPercentileRatioMid65(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Flux percentile ratio for the middle 65% of the light curve.

    Calculates the ratio between the 17.5–82.5th percentile flux range and the 
    total flux range (2nd to 98th percentiles). Captures more of the light curve's 
    distribution than narrower windows.

    Parameters
    ----------
    time : array-like
        Time values. Not used in the calculation.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Associated 1σ uncertainties.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting for percentile calculations.

    Returns
    -------
    float
        Ratio of the 17.5–82.5 percentile range to the 2–98 percentile range.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Useful for characterizing moderately broad light curve distributions.
    """

    return helper_features._flux_percentile_ratio(mag, magerr, 0.175, 0.825, apply_weights)

def FluxPercentileRatioMid80(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Flux percentile ratio for the middle 80% of the light curve.

    Calculates the ratio between the 10–90th percentile flux range and the 
    total flux range (2nd to 98th percentiles). Captures the majority of 
    the distribution while being robust to outliers.

    Parameters
    ----------
    time : array-like
        Time values. Not used in the calculation.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Associated 1σ uncertainties.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting for percentile calculations.

    Returns
    -------
    float
        Ratio of the 10–90 percentile range to the 2–98 percentile range.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Captures broad variability without being fully dominated by outliers.
    """

    return helper_features._flux_percentile_ratio(mag, magerr, 0.10, 0.90, apply_weights)

def Gskew(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Robust skewness estimator using extreme quantiles (G-skew).

    This statistic measures the asymmetry of the light curve distribution by comparing 
    the medians of the lower and upper 3% tails to the global median. It is more robust 
    to outliers and non-Gaussian noise than classical skewness metrics.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Associated 1σ uncertainties for each value in `mag`.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting when computing medians and quantiles. If False, computes the unweighted G-skew.

    Returns
    -------
    float
        G-skew value, defined as: `(median of bottom 3%) + (median of top 3%) - 2 × global median`.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - More robust than classical skewness for non-symmetric or noisy light curves.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        med = np.median(mag)
        p3, p97 = np.percentile(mag, [3, 97])
        return np.median(mag[mag <= p3]) + np.median(mag[mag >= p97]) - 2 * med

    idx = np.argsort(mag)
    w = helper_features._safe_weights(magerr)[idx]
    x = mag[idx]

    p3, med, p97 = helper_features._weighted_percentiles(x, w, [0.03, 0.5, 0.97])

    mq3 = helper_features._weighted_percentiles(x[x <= p3], w[x <= p3], 0.5)[0] if np.any(x <= p3) else np.nan
    mq97 = helper_features._weighted_percentiles(x[x >= p97], w[x >= p97], 0.5)[0] if np.any(x >= p97) else np.nan

    return mq3 + mq97 - 2 * med

def half_mag_amplitude_ratio(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Ratio of variability amplitude above vs. below the median.

    This statistic compares the root-mean-square (RMS) scatter of light curve points 
    that are greater than the median to those that are lower, by computing:

        sqrt( Σ[(Δm)²]_greater / Σ[(Δm)²]_lower )

    where Δm is the deviation from the median. This provides a compact measure 
    of flux asymmetry about the median. 

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting for the RMS computations.

    Returns
    -------
    float
        Ratio of RMS scatter in the fainter half to that in the brighter half, 
        relative to the median. Returns NaN if either side has zero variance 
        or no data.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    med = np.median(mag)
    mask_low = mag > med
    mask_high = ~mask_low

    if not apply_weights:
        var_low = np.mean((mag[mask_low]  - med)**2) if mask_low.any() else np.nan
        var_high = np.mean((mag[mask_high] - med)**2) if mask_high.any() else np.nan
    else:
        w_low = helper_features._safe_weights(magerr[mask_low])
        w_high = helper_features._safe_weights(magerr[mask_high])
        var_low = np.sum(w_low * (mag[mask_low]  - med)**2) / w_low.sum() if w_low.sum() else np.nan
        var_high = np.sum(w_high * (mag[mask_high] - med)**2) / w_high.sum() if w_high.sum() else np.nan

    return np.sqrt(var_low / var_high) if (var_low > 0 and var_high > 0) else np.nan

def index_mass_quantile(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, r: float = 0.5) -> float:
    """
    Index at which a given fraction of the absolute cumulative flux is reached.

    This metric computes the index location (normalized to [0,1]) at which a 
    specified fraction `r` of the total **absolute flux** (or amplitude) is accumulated 
    in the sorted light curve. It reflects how quickly the "mass" of the light curve 
    builds up and characterizes burstiness or concentration.

    Parameters
    ----------
    time : array-like
        Time stamps of the light curve. Not used directly in the calculation.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting (1 / magerr²) when computing cumulative mass.
    r : float, optional (default=0.5)
        The cumulative mass quantile to evaluate (must be in (0, 1)).

    Returns
    -------
    float
        The normalized index location at which the cumulative absolute "mass" 
        exceeds the specified fraction `r`. Returns NaN if input is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Values closer to 0 imply early accumulation of flux; closer to 1 indicates late accumulation.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    abs_mag = np.abs(mag)

    if not apply_weights:
        cdf = np.cumsum(abs_mag) / abs_mag.sum()
    else:
        w = 1.0 / magerr**2
        cdf = np.cumsum(abs_mag * w) / (abs_mag * w).sum()

    idx = np.searchsorted(cdf, r, side="left")

    return (idx + 1) / mag.size

def integrate(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Numerical integration of the light curve using the trapezoidal rule.

    This function computes the integral of the magnitude time series using 
    the trapezoidal rule. The integration is performed over the `time` array 
    with respect to the `mag` values. Magnitude errors (`magerr`) are ignored, 
    as they do not affect the integration of the signal itself.

    Parameters
    ----------
    time : array-like
        Time values corresponding to each magnitude measurement.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with `mag`. Ignored in this function.
    apply_weights : bool, optional (default=True)
        Currently unused. Included for API compatibility only.

    Returns
    -------
    float
        The integrated magnitude over time, computed via the trapezoidal rule.

    Notes
    -----
    - This is a purely geometric integration; `magerr` is not used.
    - Works for **unevenly sampled** data.
    """

    integrated_mag = np.trapezoid(mag, time)
    
    return integrated_mag

def kurtosis(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, fisher: bool = True) -> float:
    """
    Weighted or unweighted kurtosis of the light curve.

    This statistic measures the "tailedness" of the light curve distribution. By default, 
    the **Fisher definition** is used, where a normal distribution has kurtosis = 0. 
    The function supports both weighted (inverse-variance) and unweighted computation.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties for each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes kurtosis using inverse-variance weights (1 / magerr²). If False, computes standard unweighted kurtosis via `scipy.stats.kurtosis()`.
    fisher : bool, optional (default=True)
        If True, returns **excess kurtosis** (i.e., subtracts 3 so that Gaussian = 0). If False, returns raw kurtosis (Gaussian = 3).

    Returns
    -------
    float
        Kurtosis of the light curve distribution. Returns NaN if the input length 
        is < 4 or if the weighted variance is zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - High positive kurtosis indicates heavy tails (outliers); negative kurtosis indicates a flat-topped distribution.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size < 4:
        return np.nan

    if not apply_weights:
        return sstats.kurtosis(mag, fisher=fisher)

    w = helper_features._safe_weights(magerr)
    if w.sum() == 0:
        return np.nan

    mean = np.sum(w * mag) / w.sum()
    var = np.sum(w * (mag - mean) ** 2) / w.sum()

    if var == 0:
        return 0.0

    fourth = np.sum(w * (mag - mean) ** 4) / w.sum()
    g2 = fourth / var**2

    return g2 - 3.0 if fisher else g2

def large_standard_deviation(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, r: float = 0.3) -> float:
    """
    Binary indicator for large standard deviation relative to dynamic range.

    This metric returns 1 if the standard deviation of the light curve exceeds 
    the input fraction `r` of its total dynamic range (max - min), and 0 otherwise.
    It is intended as a coarse flag for strong variability.

    Parameters
    ----------
    time : array-like
        Time values for the light curve. Not used in the calculation but retained for API consistency.
    mag : array-like
        Light curve values. Can be magnitudes, fluxes, or normalized fluxes.
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted standard deviation using inverse-variance weights.
    r : float, optional (default=0.3)
        Threshold ratio. Returns 1 if std > `r × (max - min)`; otherwise 0.

    Returns
    -------
    int
        1 if the standard deviation exceeds `r × (max - min)`, otherwise 0.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    rng = mag.max() - mag.min()

    if rng == 0:
        return 0.0

    if not apply_weights:
        return float(np.std(mag, ddof=0) > r * rng)

    w = helper_features._safe_weights(magerr)
    mu = (w * mag).sum() / w.sum() if w.sum() else mag.mean()
    sigma = np.sqrt((w * (mag - mu) ** 2).sum() / w.sum()) if w.sum() else np.std(mag)
    
    return float(sigma > r * rng)

def LinearTrend(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Slope of the best-fit linear trend in the light curve.

    This feature quantifies the overall linear trend in the light curve by fitting a line 
    to `mag` as a function of `time`. The result is the **slope** of the fit, which indicates 
    whether the light curve is systematically brightening or dimming over time.

    Parameters
    ----------
    time : array-like
        Time values of the light curve.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties for each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, fits the line using weighted least squares (weights = 1 / magerr²). If False, uses an unweighted ordinary least squares fit.

    Returns
    -------
    float
        Slope of the best-fit line. A negative value indicates a brightening trend 
        (in magnitudes), while a positive value indicates fading. Returns NaN if input 
        is too short or the time variance is zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    time = np.asarray(time, float)
    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size < 2:
        return np.nan

    if not apply_weights:
        return sstats.linregress(time, mag).slope

    w = helper_features._safe_weights(magerr)
    if w.sum() == 0:
        return sstats.linregress(time, mag).slope

    xm = np.sum(w * time) / w.sum()
    ym = np.sum(w * mag) / w.sum()
    cov = np.sum(w * (time - xm) * (mag - ym))
    var = np.sum(w * (time - xm)**2)

    return cov / var if var > 0 else np.nan

def longest_strike_above(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Longest consecutive run of points significantly above the median.

    This metric computes the longest sequence of consecutive light curve points 
    that are above the (weighted or unweighted) median. If `apply_weights=True`, 
    a point is considered "above" only if it exceeds the median by more than its 
    associated 1σ uncertainty.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties for each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses `mag > (median + magerr)` to determine significance. If False, uses a simple comparison to the median.

    Returns
    -------
    float
        The longest sequence of consecutive "above-median" points, 
        normalized by the total number of points. Returns NaN if input is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    med = np.median(mag) if not apply_weights else helper_features._weighted_median(mag, helper_features._safe_weights(magerr))
    mask = mag > (med + magerr) if apply_weights else mag > med

    return helper_features._longest_true_run(mask) / mag.size

def longest_strike_below(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Longest consecutive run of points significantly below the median.

    This metric computes the longest sequence of consecutive light curve points 
    that are below the (weighted or unweighted) median. If `apply_weights=True`, 
    a point is considered "below" only if it is less than the median by more than 
    its associated 1σ uncertainty.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties for each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses `mag < (median - magerr)` to determine significance. If False, uses a simple comparison to the median.

    Returns
    -------
    float
        The longest sequence of consecutive "below-median" points, 
        normalized by the total number of points. Returns NaN if input is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    med = np.median(mag) if not apply_weights else helper_features._weighted_median(mag, helper_features._safe_weights(magerr))
    mask = mag < (med - magerr) if apply_weights else mag < med

    return helper_features._longest_true_run(mask) / mag.size

def MaxSlope(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Maximum or weighted average absolute slope between consecutive light curve points.

    This statistic computes the maximum (or weighted average) of the absolute slope 
    between consecutive time-adjacent measurements. It is designed to detect rapid 
    changes in the light curve and is sensitive to flares, eclipses, or steep rises/falls.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Must be in a consistent time unit (e.g., days).
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, returns the weighted average of absolute slopes using inverse-variance weights. If False, returns the maximum absolute slope.

    Returns
    -------
    float
        The maximum (or weighted average) absolute slope in units of `mag / time`.
        Returns NaN if the light curve has fewer than two points or invalid time steps.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - If any `dt` is zero or non-finite, it is automatically excluded from the calculation. Therefore time must be strictly increasing and free of duplicate values to avoid division by zero.
    """

    time = np.asarray(time, float)
    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size < 2:
        return np.nan

    dt = time[1:] - time[:-1]
    slope = np.abs((mag[1:] - mag[:-1]) / dt)
    valid = np.isfinite(slope) & (dt != 0)

    if not valid.any():
        return np.nan

    if not apply_weights:
        return np.max(slope[valid])

    w = 1.0 / (magerr[:-1]**2 + magerr[1:]**2)
    w = w[valid]
    s = slope[valid]

    return np.sum(w * s) / w.sum()

def mean_abs_change(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Mean or weighted mean of absolute changes between consecutive light curve points.

    This statistic computes the mean of the absolute value of successive differences 
    in the light curve values. It quantifies the typical fluctuation magnitude, 
    regardless of direction, and is useful for measuring overall variability.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted mean using inverse-sum error weights. If False, computes an unweighted mean of absolute changes.

    Returns
    -------
    float
        The mean (or weighted mean) of absolute changes in the light curve.
        Returns NaN if fewer than two data points are present.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Sensitive to noise; a noisy light curve may show large average absolute changes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size < 2:
        return np.nan

    diffs = np.abs(np.diff(mag))

    if not apply_weights:
        return diffs.mean()

    w = 1.0 / (magerr[:-1] + magerr[1:])

    return np.sum(w * diffs) / w.sum() if w.sum() else diffs.mean()

def mean_change(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Mean or weighted mean of signed changes between consecutive light curve points.

    This statistic computes the average of the signed differences between 
    adjacent light curve measurements. It reflects any long-term slope 
    or drift in brightness and can help detect slow trends.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted mean using inverse-variance weights from adjacent errors. If False, computes an unweighted mean.

    Returns
    -------
    float
        The mean (or weighted mean) of signed changes in the light curve.
        Returns NaN if fewer than two data points are present.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Ideal for trend detection in smoothed or denoised light curves.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size < 2:
        return np.nan

    diffs = np.diff(mag)

    if not apply_weights:
        return diffs.mean()

    w = 1.0 / (magerr[:-1]**2 + magerr[1:]**2)

    return np.sum(w * diffs) / w.sum() if w.sum() else diffs.mean()

def meanMag(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Mean or weighted mean of the light curve values.

    This statistic computes the average brightness of the light curve, either as a 
    simple arithmetic mean or as an inverse-variance weighted mean, depending on 
    whether `apply_weights` is enabled. It provides a measure of the central tendency 
    of the light curve, useful for normalization or baseline estimation.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted mean using inverse-variance weights. If False, computes a simple unweighted mean.

    Returns
    -------
    float
        The mean (or weighted mean) of the light curve values.
        Returns NaN if the array is empty.

    Notes
    -----
    - MAY NOT BE APPROPRIATE FOR MACHINE LEARNING CLASSIFICATION
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        return np.mean(mag)

    w = helper_features._safe_weights(magerr)

    return np.sum(w * mag) / w.sum() if w.sum() > 0 else np.mean(mag)

def mean_n_abs_max(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, number_of_maxima: int = 10) -> float:
    """
    Mean or weighted mean of the top-N largest absolute light curve values.

    This statistic computes the mean of the `N` largest absolute values in the light 
    curve, optionally applying inverse-variance weighting. It is useful for capturing 
    the contribution of extreme values (e.g., flares, outliers, strong variability) 
    in magnitude or flux measurements.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weighting. Otherwise, uses unweighted mean.
    number_of_maxima : int, optional (default=10)
        Number of largest absolute values to include in the average.

    Returns
    -------
    float
        Mean (or weighted mean) of the top-N absolute values.
        Returns NaN if `number_of_maxima` is not in [1, len(mag)] or if `mag` is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - A small `N` emphasizes only the most extreme events, while a large `N` includes broader variability.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    n = int(number_of_maxima)

    if n <= 0 or n > mag.size:
        return np.nan

    idx = np.argpartition(np.abs(mag), -n)[-n:]

    if not apply_weights:
        return np.abs(mag[idx]).mean()

    w = helper_features._safe_weights(magerr[idx])

    return np.sum(w * np.abs(mag[idx])) / w.sum() if w.sum() else np.abs(mag[idx]).mean()

def mean_second_derivative(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Mean or weighted mean of the discrete second derivative in a light curve.

    This statistic estimates the average curvature of the light curve by computing 
    the second derivative at each internal point using finite differences. It is 
    sensitive to acceleration in brightness changes and can help identify sudden 
    shifts in slope, such as flares or eclipses.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Must be in a consistent time unit (e.g., days).
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes a weighted average using inverse-variance weights from the three points contributing to each second-derivative estimate.

    Returns
    -------
    float
        The (weighted) mean of the discrete second derivative estimates. 
        Returns NaN if fewer than three valid points exist or all weights are zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; each estimate uses three points: (i-1, i, i+1), assuming irregular sampling.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    time = np.asarray(time, float)
    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    n = mag.size
    if n < 3:
        return np.nan

    dt_fwd = time[2:] - time[1:-1]
    dt_bck = time[1:-1] - time[:-2]
    dt_tot = time[2:]  - time[:-2]

    valid = (dt_fwd > 0) & (dt_bck > 0) & (dt_tot > 0)
    if not valid.any():
        return np.nan

    # centre indices for valid triplets
    i = np.where(valid)[0] + 1

    # second-derivative estimates
    d2 = 2.0 / dt_tot[valid] * (
            (mag[i+1] - mag[i]) / dt_fwd[valid] -
            (mag[i] - mag[i-1]) / dt_bck[valid]
         )

    if not apply_weights:
        return d2.mean()

    # simple inverse-variance weights from the three contributing errors
    w = 1.0 / (magerr[i-1]**2 + magerr[i]**2 + magerr[i+1]**2)
    good_w = w > 0
    if not good_w.any():
        return d2.mean()

    return np.sum(w[good_w] * d2[good_w]) / w[good_w].sum()

def medianAbsDev(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Median Absolute Deviation (MAD) of the light curve.

    This statistic measures the median of the absolute deviations from the median value
    of the light curve. It provides a robust estimate of variability that is less sensitive 
    to outliers than the standard deviation.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes the weighted median and weighted MAD using inverse-variance weights. If False, uses standard unweighted medians.

    Returns
    -------
    float
        Median absolute deviation of the light curve. Returns NaN if input is empty or invalid.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        med = np.median(mag)
        return np.median(np.abs(mag - med))

    w = helper_features._safe_weights(magerr)
    if w.sum() == 0:
        med = np.median(mag)
        return np.median(np.abs(mag - med))

    idx = np.argsort(mag)
    med = helper_features._weighted_percentiles(mag[idx], w[idx], 0.5)[0]
    mad = helper_features._weighted_percentiles(np.abs(mag[idx] - med), w[idx], 0.5)[0]

    return mad

def median_buffer_range(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of light curve points within ±10% of the semi-amplitude around the central value.

    This metric quantifies the concentration of measurements near the central brightness level
    by computing the fraction of points that lie within a narrow buffer region centered on 
    the mean or median magnitude. The buffer has a width of 20% of the full amplitude.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses the inverse-variance weighted mean as the center value.
        If False, uses the unweighted median.

    Returns
    -------
    float
        Fraction of points within ±10% of the semi-amplitude around the center.
        Returns NaN if the amplitude is undefined or the input is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size == 0:
        return np.nan

    amp = amplitude(time, mag, magerr, apply_weights)
    if not np.isfinite(amp):
        return np.nan

    if apply_weights:
        w = helper_features._safe_weights(magerr)
        mean = np.sum(w * mag) / w.sum() if w.sum() > 0 else np.mean(mag)
    else:
        mean = np.median(mag)

    a = mean - 0.1 * amp
    b = mean + 0.1 * amp

    return np.count_nonzero((mag > a) & (mag < b)) / float(mag.size)

def median_distance(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Median Euclidean distance between consecutive light curve points in time-magnitude space.

    This statistic measures the typical spacing between adjacent measurements in the 
    (time, magnitude) plane. It captures the smoothness and sampling density of the 
    light curve trajectory.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Must be in a consistent time unit (e.g., days).
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, normalizes the squared magnitude difference by the sum of the variances of adjacent points to account for measurement uncertainty.

    Returns
    -------
    float
        Median Euclidean distance between consecutive points. Units are in (mag² + time²)⁰·⁵ 
        or (normalized) units depending on the input scale.
        Returns NaN if fewer than two points or if all distances are invalid.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)
    time = np.asarray(time, float)

    if mag.size < 2:
        return np.nan

    dmag2 = (mag[1:] - mag[:-1]) ** 2
    dt2 = (time[1:] - time[:-1]) ** 2

    if apply_weights:
        var_sum = magerr[1:] ** 2 + magerr[:-1] ** 2
        good = var_sum > 0
        dist = np.sqrt((dmag2[good] + dt2[good]) / var_sum[good])
    else:
        dist = np.sqrt(dmag2 + dt2)

    return np.median(dist) if dist.size else np.nan

def number_cwt_like_peaks(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Normalized count of CWT-like peaks in a light curve using simple prominence and width criteria.

    This statistic approximates a continuous wavelet transform (CWT)-style peak count by using 
    `scipy.signal.find_peaks`. It identifies medium-scale features in the light curve such as flares or bumps.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, weights the magnitude values by the inverse of the error prior to peak detection.

    Returns
    -------
    float
        Number of detected peaks divided by the total number of points.
        Returns 0.0 if the input array is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    - Intended as a lightweight proxy for more sophisticated wavelet-based peak detection.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return 0.0

    if apply_weights and magerr is not None:
        weights = 1.0 / np.maximum(magerr, 1e-6)
        mag = mag * weights  # crude weighted signal copy

    # CWT-like peak detection: medium prominence and width
    peaks, _ = ssignal.find_peaks(mag, prominence=0.5, width=5)

    return len(peaks) / mag.size

def number_of_crossings(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Fraction of sign changes relative to the median of the light curve.

    This statistic estimates how frequently the light curve crosses its median value, 
    indicating the presence of variability. When `apply_weights` is True, only consider 
    crossings with significant changes relative to photometric uncertainty.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, only count crossings where the absolute difference exceeds the measurement error.

    Returns
    -------
    float
        Fraction of crossings relative to the total number of data points.
        Returns 0.0 if fewer than two measurements are present.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves but best if time is evenly spaced; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size < 2:
        return 0.0

    med = np.median(mag)
    sign = mag > med
    basic_cross = np.diff(sign).astype(np.bool_)

    if not apply_weights:
        return basic_cross.sum() / mag.size

    strong = np.abs(np.diff(mag)) > magerr[:-1]

    return (basic_cross & strong).sum() / mag.size

def PairSlopeTrend(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, n_last: int = 30) -> float:
    """
    Trend fraction based on slopes of recent consecutive magnitude pairs.

    This statistic computes the fraction of upward trends (positive slopes)
    among the last `n_last` data points. When `apply_weights` is True, it returns 
    the weighted fraction of positive slopes using inverse-variance weights.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, uses inverse-variance weights to compare rising vs. falling slopes.
    n_last : int, optional (default=30)
        Number of most recent points to consider. If fewer are available, all are used.

    Returns
    -------
    float
        Fraction of increasing slopes (weighted or unweighted) among recent pairs.
        Returns NaN if fewer than two valid points or no valid weights.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if len(mag) < 2:
        return np.nan

    data = np.asarray(mag[-n_last:], float)
    err = np.asarray(magerr[-n_last:], float)

    diff = data[1:] - data[:-1]
    pos = diff > 0
    neg = diff < 0

    if not apply_weights:
        return pos.sum() / diff.size

    w = 1.0 / (err[:-1]**2 + err[1:]**2)
    w_pos = w[pos].sum()
    w_neg = w[neg].sum()

    return w_pos / (w_pos + w_neg) if (w_pos + w_neg) > 0 else np.nan

def PercentAmplitude(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Maximum percent deviation from the (weighted) median magnitude.

    This statistic computes the maximum absolute deviation from the median,
    normalized by the median itself: max(|m_i − median|) / median

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (e.g., magnitudes or fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes the weighted median using inverse-variance weights.

    Returns
    -------
    float
        Maximum fractional deviation from the median magnitude.
        Returns NaN if the input is empty, or inf if the median is zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        med = np.median(mag)
    else:
        w = helper_features._safe_weights(magerr)
        idx = np.argsort(mag)
        med = helper_features._weighted_percentiles(mag[idx], w[idx], 0.5)[0]

    amp = np.max(np.abs(mag - med))

    return amp / med if med != 0 else np.inf

def PercentDifferenceFluxPercentile(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Inter-percentile range divided by the median: (F95 − F5) / median.

    This variability metric captures the amplitude of the central 90% of the
    light curve distribution relative to the median, providing a robust estimate
    of variability that is less sensitive to outliers.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, computes weighted percentiles using inverse-variance weights.

    Returns
    -------
    float
        (95th percentile − 5th percentile) / median.
        Returns NaN if input is empty, or inf if the median is zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        p5, p95 = np.percentile(mag, [5, 95])
        med     = np.median(mag)
    else:
        idx = np.argsort(mag)
        w   = helper_features._safe_weights(magerr)[idx]
        p5, p95, med = helper_features._weighted_percentiles(mag[idx], w, [0.05, 0.95, 0.50])

    return (p95 - p5) / med if med != 0 else np.inf

def permutation_entropy(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, tau: int = 1, dimension: int = 3) -> float:
    """
    Permutation entropy of the light curve.

    This non-parametric measure captures the complexity or randomness in the 
    ordering of values over time by evaluating the distribution of ordinal patterns 
    in embedded vectors. A high value indicates a more disordered or complex signal, 
    while a low value indicates more regular behavior. No option to account for errors included. 

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (flux or magnitude).
    magerr : array-like
        Photometric uncertainties. Not used in the calculation but included for API consistency.
    apply_weights : bool, optional (default=True)
        Not used in the calculation, but included for API consistency.
    tau : int, optional (default=1)
        Time delay between elements in each embedded vector.
    dimension : int, optional (default=3)
        Embedding dimension (length of ordinal patterns).

    Returns
    -------
    float
        Permutation entropy value (non-negative).
        Returns NaN if the light curve is too short to compute the statistic.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    x = np.asarray(mag, float)
    n = x.size

    if n < (dimension - 1) * tau + 1:
        return np.nan

    idx = np.arange(0, n - (dimension - 1) * tau)
    patterns = np.array([np.argsort(x[i:i + dimension * tau:tau]) for i in idx])
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / counts.sum()

    return -np.sum(probs * np.log(probs))

def prominence_peaks(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, thres: float = 0.3, min_dist: int = 25, thres_abs: bool = False) -> float:
    """
    Detects prominent peaks in a light curve using slope changes and thresholding.

    This function identifies local maxima by detecting sign changes in the first derivative 
    of the light curve. Optionally, it weights the signal by inverse variance and applies 
    a threshold on the (weighted) amplitude to reject low-significance peaks. When multiple 
    peaks are too close (within `min_dist`), only the most prominent one is retained.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not directly used in peak detection.
    mag : 1-D array-like
        Light curve values (e.g., normalized fluxes or fluxes).
    magerr : 1-D array-like
        Photometric uncertainties corresponding to each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, signal values are weighted by inverse variance when computing the amplitude used for thresholding.
    thres : float, optional (default=0.3)
        Threshold level for accepting peaks. If `thres_abs=False`, this is relative:
        a value between 0 and 1 indicating how far between the min and max amplitude 
        a peak must lie. If `thres_abs=True`, it is an absolute threshold in the same units as `mag`.
    min_dist : int, optional (default=25)
        Minimum separation (in samples) required between two retained peaks. If multiple 
        peaks fall within `min_dist`, only the most prominent (weighted) one is kept.
    thres_abs : bool, optional (default=False)
        If True, `thres` is interpreted as an absolute threshold. If False, `thres` is relative.

    Returns
    -------
    float
        Fraction of light curve points identified as significant peaks.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Code adapted from the peakutils Python package (MIT license).
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.ndim != 1:
        raise ValueError("`y` must be a 1-D array")

    # Amplitude array used for thresholding / ranking
    if apply_weights:
        w = helper_features._safe_weights(np.asarray(magerr, float))
        amp = mag * w # weighted amplitude
        if w.sum() == 0: # fall back if all weights zero
            amp = mag.copy()
    else:
        amp = mag.copy()

    # Threshold level
    if not thres_abs:
        level = thres * (amp.max() - amp.min()) + amp.min()
    else:
        level = thres

    min_dist = int(min_dist)

    # First-order difference & plateau handling
    dy = np.diff(mag)
    zeros = np.flatnonzero(dy == 0)

    # completely flat signal
    if zeros.size == dy.size:
        return 0

    if zeros.size:
        zdiff = np.diff(zeros)
        breaks = np.flatnonzero(zdiff != 1) + 1
        plates = np.split(zeros, breaks)

        # left edge
        if plates and plates[0][0] == 0:
            dy[plates[0]] = dy[plates[0][-1] + 1]
            plates.pop(0)
        # right edge
        if plates and plates[-1][-1] == dy.size - 1:
            dy[plates[-1]] = dy[plates[-1][0] - 1]
            plates.pop()

        for p in plates: # internal plateaus
            mid = int(np.median(p))
            dy[p[p < mid]]  = dy[p[0] - 1]
            dy[p[p >= mid]] = dy[p[-1] + 1]

    # Raw peaks: +slope --> −slope and amp > threshold
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0) &
        (np.hstack([0.0, dy]) > 0) &
        (amp > level)
    )[0]

    # Enforce minimum separation
    if peaks.size > 1 and min_dist > 1:
        order = peaks[np.argsort(amp[peaks])][::-1] # highest-(weighted) first
        mask = np.ones(mag.size, dtype=bool)
        mask[peaks] = False

        for p in order:
            if not mask[p]:
                lo = max(0, p - min_dist)
                hi = p + min_dist + 1
                mask[lo:hi] = True
                mask[p] = False

        peaks = np.nonzero(~mask)[0]

    return len(peaks.astype(int)) / len(mag)

def quantile_5(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    5th percentile value of the light curve. This is useful for describing the brightness distribution 
    without being sensitive to outliers.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties. Not used in the calculation but included for API consistency.
    apply_weights : bool, optional (default=True)
        Currently unused; included for API consistency.

    Returns
    -------
    float
        The 5th percentile value of the light curve.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    """

    return np.quantile(np.asarray(mag, float), 0.05)

def quantile_25(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    25th percentile value of the light curve. This is useful for describing the brightness distribution 
    without being sensitive to outliers.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties. Not used in the calculation but included for API consistency.
    apply_weights : bool, optional (default=True)
        Currently unused; included for API consistency.

    Returns
    -------
    float
        The 25th percentile value of the light curve.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    """

    return np.quantile(np.asarray(mag, float), 0.25)

def quantile_50(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Median (50th percentile) value of the light curve. This is useful for describing the brightness distribution 
    without being sensitive to outliers.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties. Not used in the calculation but included for API consistency.
    apply_weights : bool, optional (default=True)
        Currently unused; included for API consistency.

    Returns
    -------
    float
        The median value of the light curve.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    """

    return np.quantile(np.asarray(mag, float), 0.50)

def quantile_75(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    75th percentile value of the light curve. This is useful for describing the brightness distribution 
    without being sensitive to outliers.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties. Not used in the calculation but included for API consistency.
    apply_weights : bool, optional (default=True)
        Currently unused; included for API consistency.

    Returns
    -------
    float
        The 75th percentile value of the light curve.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    """

    return np.quantile(np.asarray(mag, float), 0.75)

def quantile_95(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    95th percentile value of the light curve. This is useful for describing the brightness distribution 
    without being sensitive to outliers.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties. Not used in the calculation but included for API consistency.
    apply_weights : bool, optional (default=True)
        Currently unused; included for API consistency.

    Returns
    -------
    float
        The 95th percentile value of the light curve.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    """

    return np.quantile(np.asarray(mag, float), 0.95)

def ratio_recurring_points(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Ratio of recurring values in the light curve.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, default True
        If True, recurrence is counted using per-point tolerance from `magerr`.

    Returns
    -------
    float
        Fraction of unique values that appear more than once.
        Uses exact matching unless `apply_weights=True`, in which case
        values are matched within ±1σ.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return 0.0

    uniq, idx, counts = np.unique(mag, return_inverse=True, return_counts=True)

    if not apply_weights:
        return np.sum(counts > 1) / uniq.size

    # Weighted version: use point-by-point uncertainty
    recurring = 0
    for u in uniq:
        mask = np.isclose(mag, u, atol=magerr)
        if np.sum(mask) > 1:
            recurring += 1

    return recurring / uniq.size if uniq.size > 0 else 0.0

def root_mean_squared(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Root-mean-square (RMS) deviation of the light curve magnitudes.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, default True
        If True, compute weighted RMS using inverse-variance weights.

    Returns
    -------
    float
        RMS deviation of the light curve values. Returns NaN if input is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        return np.sqrt(np.mean((mag - np.mean(mag)) ** 2))

    w = helper_features._safe_weights(magerr)
    if np.sum(w) == 0:
        return np.sqrt(np.mean((mag - np.mean(mag)) ** 2))

    mean = np.sum(w * mag) / np.sum(w)
    rms = np.sqrt(np.sum(w * (mag - mean) ** 2) / np.sum(w))

    return rms

def sample_entropy(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Sample entropy of the light curve.

    Measures the negative log-likelihood that sequences of `m` consecutive points
    that are similar (within `r`) remain similar when extended to `m+1` points.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties. Not used in the calculation but included for API consistency.
    apply_weights : bool, default True
        Currently unused; included for API consistency.

    Returns
    -------
    float
        Sample entropy value, or NaN if undefined.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    x = np.asarray(mag, float)
    n = x.size

    if n < 4:
        return np.nan

    m = 2
    r = 0.2 * np.std(x, ddof=0)

    def _phi(m):
        try:
            templates = np.lib.stride_tricks.sliding_window_view(x, m)
        except ValueError:
            return 0
        D = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=2)
        return np.sum(D <= r) - len(templates)  # remove self-matches

    B = _phi(m)
    A = _phi(m + 1)

    return -np.log(A / B) if A > 0 and B > 0 else np.nan

def shannon_entropy(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, eps: float = 1e-12) -> float:
    """
    Shannon–type entropy of a light curve.

    This metric estimates the total information content of a light curve by 
    computing entropy contributions based on the Gaussian and inverse-Gaussian 
    cumulative distribution functions (CDFs) around each point, integrated over 
    a symmetric interval defined by the photometric uncertainty.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Must be in a consistent time unit (e.g., days).
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties corresponding to `mag`.
    apply_weights : bool, default=True
        If True, entropy contributions are weighted by inverse variance.
    eps : float, default=1e-12
        Small constant added to probabilities inside log terms to ensure numerical stability.

    Returns
    -------
    float
        Total Shannon entropy of the light curve, combining Gaussian and 
        inverse-Gaussian contributions. Returns NaN if the input is too small 
        or unstable (e.g., zero mean).

    Notes
    -----
    - Suitable for **unevenly sampled** light curves.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if not (time.size == mag.size == magerr.size):
        raise ValueError("`time`, `mag`, and `magerr` must be the same length.")
    if mag.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays.")

    mu = np.median(mag) # robust mean estimate
    sigma = root_mean_squared(time, mag, magerr, apply_weights=apply_weights) if apply_weights else np.sqrt(np.mean((mag - mu) ** 2))  # RMS scatter

    # Gaussian CDFs for upper & lower error bounds 
    z_hi = (mag + magerr - mu) / (sigma * np.sqrt(2.0))
    z_lo = (mag - magerr - mu) / (sigma * np.sqrt(2.0))
    p_hi = 0.5 * (1.0 + erf(z_hi))
    p_lo = 0.5 * (1.0 + erf(z_lo))

    # Inverse-Gaussian CDFs
    # scipy’s parameterisation: X ~ InvGauss(μ, λ) --> shape = λ/μ
    lam = sigma**2
    shape_param = lam / mu
    p_inv_hi = invgauss.cdf((mag + magerr) / mu, shape_param)
    p_inv_lo = invgauss.cdf((mag - magerr) / mu, shape_param)

    # differential element Δm_i
    delta = 2.0 * magerr

    # weights
    w = np.ones_like(mag) if not apply_weights else 1.0 / (magerr**2 + eps)

    # entropy contributions
    ent_gauss = -np.sum(w * delta * (np.log2(p_hi + eps) + np.log2(p_lo + eps)))
    ent_inv = -np.sum(w * delta * (np.log2(p_inv_hi + eps) + np.log2(p_inv_lo + eps)))

    return ent_gauss + ent_inv

def shapiro_wilk(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Shapiro–Wilk normality test statistic for light-curve data.

    This function returns the Shapiro–Wilk statistic, which tests the null hypothesis
    that the input data are drawn from a normal distribution. Values close to 1 indicate
    consistency with normality. No error propagation is included, as the underlying 
    `scipy.stats.shapiro` implementation does not support weighting or uncertainties.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties. Not used in the calculation but included for API consistency.
    apply_weights : bool, default True
        Not used in the calculation but included for API consistency.

    Returns
    -------
    float
        The Shapiro–Wilk W statistic. Values closer to 1 suggest normality.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    return sstats.shapiro(np.asarray(mag, float))[0]

def skewness(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Weighted or unweighted skewness of the light-curve distribution.

    Computes the third standardized moment (skewness) of the light-curve values,
    which quantifies the asymmetry of the distribution. If `apply_weights=True`,
    inverse-variance weights (1/σ²) are used to compute a weighted, unbiased estimator.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, apply inverse-variance weighting. Otherwise, use unweighted skewness.

    Returns
    -------
    float
        The skewness of the light-curve distribution. A value of 0 indicates
        a symmetric distribution. Returns NaN for insufficient data (fewer than 3 points).

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size < 3:
        return np.nan

    if not apply_weights:
        return sstats.skew(mag, bias=False)

    w = helper_features._safe_weights(magerr)
    if w.sum() == 0:
        return np.nan

    mean = np.sum(w * mag) / w.sum()
    var = np.sum(w * (mag - mean) ** 2) / w.sum()

    if var == 0:
        return 0.0

    std  = np.sqrt(var)
    third = np.sum(w * (mag - mean) ** 3) / w.sum()

    return third / std**3

def std_over_mean(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Coefficient of variation (σ / μ) of the light-curve values.

    Computes the ratio of the standard deviation to the mean of the light-curve values,
    which quantifies relative variability. If `apply_weights=True`, inverse-variance weights (1/σ²)
    are used for a weighted estimator of the mean and variance.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, apply inverse-variance weighting. Otherwise, use unweighted statistics.

    Returns
    -------
    float
        The coefficient of variation (standard deviation divided by mean).
        Returns NaN if input is empty, and Inf if the mean is zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        mean = np.mean(mag)
        std = np.std(mag, ddof=0)
        return std / mean if mean != 0 else np.inf

    w = helper_features._safe_weights(magerr)
    if w.sum() == 0:
        mean = np.mean(mag)
        std = np.std(mag, ddof=0)
        return std / mean if mean != 0 else np.inf

    mean = np.sum(w * mag) / w.sum()
    var = np.sum(w * (mag - mean) ** 2) / w.sum()

    return np.sqrt(var) / mean if mean != 0 else np.inf

def stetsonJ(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Stetson's J variability index.

    Measures the degree of correlated variability in a time series. For a light curve,
    it is sensitive to the persistence of bright or faint measurements over time.
    High values indicate consecutive measurements that deviate similarly from the mean.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        Ignored in this implementation, weights are always used.

    Returns
    -------
    float
        Stetson J index. Higher values indicate stronger correlated variability.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    if mag.size < 2:
        return np.nan

    d = helper_features._delta(np.asarray(mag, float), np.asarray(magerr, float))
    Pk = d[:-1] * d[1:]

    return np.sum(np.sign(Pk) * np.sqrt(np.abs(Pk))) / len(Pk) # mag.size

def stetsonK(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Stetson's K variability index.

    Measures the kurtosis-like behavior of the normalized residuals in a light curve.
    A Gaussian distribution yields K ≈ 0.798, while larger values indicate variability.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        Ignored in this implementation, weights are always used.

    Returns
    -------
    float
        Stetson K index. Higher values suggest stronger variability or outliers.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    d = helper_features._delta(np.asarray(mag, float), np.asarray(magerr, float))
    n = d.size

    if n == 0:
        return np.nan

    return (np.sum(np.abs(d)) / n) / np.sqrt(np.sum(d ** 2) / n)

def stetsonL(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Stetson's L variability index.

    Combines the Stetson J and K indices to give an overall measure of variability,
    normalized such that L ≈ 1 for Gaussian noise. Designed to detect both correlated
    deviations and excess kurtosis in photometric data.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        Ignored in this implementation, weights are always used.

    Returns
    -------
    float
        Stetson L index. Values significantly above 1 suggest variability.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    J = stetsonJ(time, mag, magerr, apply_weights)
    K = stetsonK(time, mag, magerr, apply_weights)

    return J * K / 0.798 if np.isfinite(J) and np.isfinite(K) else np.nan

def sum_values(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Mean value of the light curve (weighted or unweighted).

    Computes the average of the light-curve values, either as a simple arithmetic
    mean or as an inverse-variance weighted mean depending on `apply_weights`.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, apply inverse-variance weighting. Otherwise, use unweighted statistics.

    Returns
    -------
    float
        Weighted or unweighted mean of the input values.
        Returns NaN if input array is empty.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float) 
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        return mag.sum() / mag.size

    w = helper_features._safe_weights(magerr)

    return (w * mag).sum() / w.sum() if w.sum() else mag.mean()

def symmetry_looking(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, r: float = 0.5) -> float:
    """
    Symmetry indicator based on mean–median agreement.

    Returns 1 if the absolute difference between the (weighted) mean and 
    (weighted) median is less than `r` times the full range of the values; 
    otherwise returns 0. This is a simple heuristic for assessing the 
    symmetry of a light-curve distribution.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, apply inverse-variance weighting for mean and medium calculations. Otherwise, use unweighted statistics.
    r : float, optional (default=0.5)
        Tolerance factor for deciding whether the distribution is symmetric.
        The threshold is defined as `r × (max − min)`.

    Returns
    -------
    int
        1 if the distribution is considered symmetric under the specified criterion, 0 otherwise.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    rng = mag.max() - mag.min()
    if rng == 0:
        return 1.0

    if not apply_weights:
        return float(np.abs(mag.mean() - np.median(mag)) < r * rng)

    w = helper_features._safe_weights(magerr)
    mu = (w * mag).sum() / w.sum() if w.sum() else mag.mean()
    med = helper_features._weighted_median(mag, w) if w.sum() else np.median(mag)

    return float(np.abs(mu - med) < r * rng)

def time_reversal_asymmetry(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, lag: int = 1) -> float:
    """
    Time-reversal asymmetry statistic with τ-lag.

    Computes: ⟨(x_{t+2τ} − x_t) (x_{t+τ} − x_t)⟩

    which measures nonlinear time asymmetry in the light curve. A value of zero suggests time-reversibility.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, apply inverse-variance weighting. Otherwise, use unweighted statistics.
    lag : int, optional (default=1)
        The lag τ to use when evaluating the statistic.
    
    Returns
    -------
    float
        Time-reversal asymmetry statistic, or NaN if input too short.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    n = mag.size
    if n < 2 * lag + 1:
        return np.nan

    x0, x1, x2 = mag[:-2*lag], mag[lag:-lag], mag[2*lag:]
    stat = (x2 - x0) * (x1 - x0)

    if not apply_weights:
        return stat.mean()

    w = 1.0 / (magerr[:-2*lag]**2 + magerr[lag:-lag]**2 + magerr[2*lag:]**2)

    return (w * stat).sum() / w.sum() if w.sum() else stat.mean()

def time_reversal_asymmetry_normalized(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = False, tau: int = 3) -> float:
    """
    Normalized time-reversal asymmetry (TREV) statistic.

    Computes: trev(τ) = ⟨(x_{t+τ} − x_t)^3⟩ / ⟨(x_{t+τ} − x_t)^2⟩^{3/2}

    which is a skewness-like measure of time irreversibility.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=False)
        If True, applies inverse-variance weights derived from error propagation. Otherwise, use unweighted statistics.
    tau : int, optional (default=3)
        The lag τ to use when computing the difference terms.
    
    Returns
    -------
    float
        Normalized time-reversal asymmetry statistic. Returns NaN if too few data points.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)

    if mag.size < tau + 1:
        return np.nan

    diff = mag[tau:] - mag[:-tau]

    if not apply_weights:
        num = np.mean(diff**3)
        denom = np.mean(diff**2)**1.5
        return num / denom if denom > 0 else 0.0

    # Weighted version
    magerr = np.asarray(magerr, float)
    if magerr.size < tau + 1:
        return np.nan

    # Combine errors in quadrature for the difference
    err = np.sqrt(magerr[tau:]**2 + magerr[:-tau]**2)
    w = helper_features._safe_weights(err)

    num = np.sum(w * diff**3) / np.sum(w)
    denom = (np.sum(w * diff**2) / np.sum(w))**1.5

    return num / denom if denom > 0 else 0.0

def variance(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Weighted/unweighted variance of the light curve.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, default=True
        If True, apply inverse-variance weighting. Otherwise, use unweighted statistics.

    Returns
    -------
    float
        Estimated variance.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        return np.var(mag, ddof=0)

    w = helper_features._safe_weights(magerr)
    mu = (w * mag).sum() / w.sum() if w.sum() else mag.mean()

    return (w * (mag - mu) ** 2).sum() / w.sum() if w.sum() else np.var(mag, ddof=0)

def variance_larger_than_standard_deviation(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Binary test: is variance greater than standard deviation?

    Compares variance to its square root and returns 1 if true, 0 otherwise.
    
    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, apply inverse-variance weighting. Otherwise, use unweighted statistics.


    Returns
    -------
    int or float
        1 if var > sqrt(var), 0 if not, np.nan if invalid.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    var = variance(time, mag, magerr, apply_weights)

    return float(var > np.sqrt(var)) if np.isfinite(var) else np.nan

def variation_coefficient(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Coefficient of variation (σ / μ).

    Measures relative dispersion by dividing the standard deviation
    by the mean. Useful for comparing variability across magnitudes.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, apply inverse-variance weighting. Otherwise, use unweighted statistics.

    Returns
    -------
    float
        Variation coefficient. Returns NaN if mean is zero.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size == 0:
        return np.nan

    if not apply_weights:
        mu = mag.mean()
        return np.std(mag, ddof=0) / mu if mu else np.nan

    w = helper_features._safe_weights(magerr)
    mu = (w * mag).sum() / w.sum() if w.sum() else mag.mean()
    sigma = np.sqrt((w * (mag - mu) ** 2).sum() / w.sum()) if w.sum() else np.std(mag)

    return sigma / mu if mu else np.nan

def vonNeumannRatio(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True) -> float:
    """
    Von Neumann's η statistic: ratio of successive differences to variance.

    A low value suggests smoothness; high value suggests jumps or outliers.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, apply inverse-variance weighting. Otherwise, use unweighted statistics.

    Returns
    -------
    float
        Von Neumann ratio. Returns np.nan if lightcurve has fewer than 2 points.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, but compatible with magnitudes or fluxes.
    """

    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    n = mag.size
    if n < 2:
        return np.nan

    diff2 = np.diff(mag) ** 2

    if not apply_weights:
        return diff2.mean() / np.var(mag, ddof=0)

    w  = helper_features._safe_weights(magerr)
    wp = np.sqrt(w[1:] * w[:-1]) # weight for each difference
    if wp.sum() == 0:
        return diff2.mean() / np.var(mag, ddof=0)

    delta = np.sum(wp * diff2) / wp.sum()

    mean = np.sum(w * mag) / w.sum()
    var = np.sum(w * (mag - mean) ** 2) / w.sum()

    return delta / var if var > 0 else np.inf

def windowed_peak_fraction(time: ArrayLike, mag: ArrayLike, magerr: ArrayLike, apply_weights: bool = True, n: int = 7) -> float:
    """
    Normalized number of local peaks in the light curve.

    A point is considered a peak if it is greater than `n` neighbors on each side.
    When `apply_weights` is True, a point is a peak only if it exceeds its neighbors
    by more than the combined error budget.

    Parameters
    ----------
    time : array-like
        Time values of the light curve. Not used in the calculation but included for API consistency.
    mag : array-like
        Light curve values (magnitudes, fluxes, or normalized fluxes).
    magerr : array-like
        Photometric uncertainties associated with each `mag` value.
    apply_weights : bool, optional (default=True)
        If True, requires the peak to be significantly above neighbors based on the combined uncertainty.
    n : int, optional (default=7)
        Number of neighbors to compare on each side of the point. The light curve must have at least `2n+1` points.

    Returns
    -------
    float
        Fraction of peaks relative to the total number of data points.
        Returns 0.0 if there are fewer than `2n+1` data points.

    Notes
    -----
    - Suitable for **unevenly sampled** light curves; time is not used.
    - Most effective with **normalized flux**, and is designed for **flux space**.
    - Works best for high-cadence light curves with visible local maxima.
    """

    mag = np.asarray(mag, float)
    magerr = np.asarray(magerr, float)

    if mag.size < 2 * n + 1:
        return 0.0

    core = mag[n:-n]
    if not apply_weights:
        res = np.ones_like(core, dtype=bool)
        for i in range(1, n + 1):
            res &= core > np.roll(mag, i)[n:-n]
            res &= core > np.roll(mag, -i)[n:-n]

        return res.sum() / mag.size

    res = np.ones_like(core, dtype=bool)
    for i in range(1, n + 1):
        dl = core - np.roll(mag, i)[n:-n]
        dr = core - np.roll(mag, -i)[n:-n]
        el = np.sqrt(magerr[n:-n]**2 + np.roll(magerr, i)[n:-n]**2)
        er = np.sqrt(magerr[n:-n]**2 + np.roll(magerr, -i)[n:-n]**2)
        res &= (dl > el) & (dr > er)

    return res.sum() / mag.size

