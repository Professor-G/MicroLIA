# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""
from __future__ import division, print_function
from gatspy.datasets import fetch_rrlyrae_templates
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

import importlib.resources as resources
from pathlib import Path




import importlib.util
import numpy as np
import os
import sys

from numpy.typing import ArrayLike
from typing import Optional, Tuple

def microlensing(
    timestamps: ArrayLike,
    baseline: float,
    t0_dist: Tuple[float, float] | None = None,
    u0_dist: Tuple[float, float] | None = None,
    tE_dist: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Simulate a single-lens, point-source microlensing event with blending.

    If not defined, the microlensing parameter space is determined using data from an 
    analysis of the OGLE III microlensing survey from Y. Tsapras et al (2016).
    See: The OGLE-III planet detection efficiency from six years of microlensing observations (2003 to 2008).
    (https://arxiv.org/abs/1602.02519)

    Parameters
    ----------
    timestamps : array-like
        Observation epochs (days).
    baseline : float
        Baseline magnitude outside the event.
    t0_dist : (float, float), optional
        Uniform bounds for ``t0`` (time of peak).  Defaults to the middle
        98% of the supplied `timestamps` plus a cushion of 0.5 `t_E` on each
        side.
    u0_dist : (float, float), optional
        Uniform bounds for ``u0`` (impact parameter).  Default: (0, 1).
    tE_dist : (float, float), optional
        Mean and standard deviation for normally distributed ``t_E``
        (Einstein-radius crossing time).  Default: 30 plus/minus 10 days.

    Returns
    -------
    Tuple
        * `mag` – simulated magnitudes (np.ndarray)
        * `u0` – drawn impact parameter
        * `t0` – drawn peak time (days)
        * `tE` – drawn event timescale (days)
        * `blend_ratio` – flux blending ratio *f_b / f_s*
    """
    
    ts = np.asarray(timestamps, dtype=float)

    # Draw parameters
    u0_min, u0_max = u0_dist if u0_dist is not None else (0.0, 1.0)
    u0 = np.random.uniform(u0_min, u0_max)

    tE_mean, tE_std = tE_dist if tE_dist is not None else (30.0, 10.0)
    tE = np.random.normal(tE_mean, tE_std)

    if t0_dist is not None:
        t0_min, t0_max = t0_dist
    else:
        t0_min = np.percentile(ts, 1) - 0.5 * tE
        t0_max = np.percentile(ts, 99) + 0.5 * tE
    t0 = np.random.uniform(t0_min, t0_max)

    blend_ratio = np.random.uniform(0.0, 1.0)

    # Magnification curve
    u_t = np.sqrt(u0**2 + ((ts - t0) / tE) ** 2)
    A_t = (u_t**2 + 2.0) / (u_t * np.sqrt(u_t**2 + 4.0))

    # Convert to observed magnitudes with blending 
    mag_base = constant(ts, baseline)
    flux_base = 10.0 ** (-0.4 * mag_base)

    f_s = np.median(flux_base) / (1.0 + blend_ratio)  # source flux
    f_b = blend_ratio * f_s # blend flux
    flux_obs = f_s * A_t + f_b

    mag_obs = -2.5 * np.log10(flux_obs)

    return mag_obs, u0, t0, tE, blend_ratio
    
def cv(timestamps, baseline):
    """Simulates Cataclysmic Variable event.
    The outburst can be reasonably well represented as three linear phases: a steeply 
    positive gradient in the rising phase, a flat phase at maximum brightness followed by a declining 
    phase of somewhat shallower negative gradient. The period is selected from a normal distribution
    centered about 100 days with a standard deviation of 200 days. The outburtst amplitude ranges from
    0.5 to 5.0 mag, selected from a uniform random function. 

    Parameters:
    ----------
        timestamps : array
            Times at which to simulate the lightcurve.
        baseline : float
            Baseline magnitude at which to simulate the lightcurve.

    Returns:
    -------
        mag : array
            Simulated magnitudes given the timestamps and baseline. 
        outburst_start_times : array
            The start time of each outburst.
        outburst_end_times : array
            The end time of each outburst.
        end_rise_times : array
            The end time of each rise (start time of max amplitude).
        end_high_times : array
            The end time of each peak (end time of max amplitude).
    """

    period = abs(np.random.normal(100, 200))
    amplitude = np.random.uniform(0.5, 5.0)
    lc = np.zeros(len(timestamps))
    # First generate the times when outbursts start. Note that the
    # duration of outbursts can vary for a single object, so the t_end_outbursts will be added later.
    start_times = []

    min_start = min(timestamps)
    max_start = min((min(timestamps)+period),max(timestamps))

    first_outburst_time = np.random.uniform(min_start, max_start)

    start_times.append(first_outburst_time)
    t_start = first_outburst_time + period

    for t in np.arange(t_start,max(timestamps),period):
        start_times.append(t)

    outburst_end_times = []
    duration_times = []
    end_rise_times = []
    end_high_times = []    
    
    for t_start_outburst in start_times:
    # Since each outburst can be a different shape,
    # generate the lightcurve morphology parameters for each outburst:
        duration = np.random.uniform(3.0, (period/10.0))
        duration_times.append(duration)
        t_end_outburst = t_start_outburst + duration
        outburst_end_times.append(t_end_outburst)        
        rise_time = np.random.uniform(0.5,1.0)
        high_state_time = np.random.normal(0.4*duration, 0.2*duration)
        drop_time = duration - rise_time - high_state_time
        t_end_rise = t_start_outburst + rise_time
        t_end_high = t_start_outburst + rise_time + high_state_time
        end_rise_times.append(t_end_rise)
        end_high_times.append(t_end_high)  
        # Rise and drop is modeled as a straight lines with differing gradients
        rise_gradient = -1.0 * amplitude / rise_time

        drop_gradient = (amplitude / drop_time)

        for i in range(0,len(timestamps),1):
                if timestamps[i] >= t_start_outburst and timestamps[i] <= t_end_rise:
                        lc[i] = rise_gradient * (timestamps[i] -t_start_outburst)
                elif timestamps[i] >= t_end_rise and timestamps[i] <= t_end_high:
                        lc[i] = -1.0 * amplitude
                elif timestamps[i] > t_end_high and timestamps[i] <= t_end_outburst:
                        lc[i] = -amplitude + ( drop_gradient * (timestamps[i] - t_end_high))

    lc = lc + baseline 

    return np.array(lc), np.array(start_times), np.array(outburst_end_times), np.array(end_rise_times), np.array(end_high_times)



def constant(timestamps: ArrayLike, baseline: float) -> np.ndarray:
    """
    Generate a non-variable light curve.

    Parameters
    ----------
    timestamps : array-like
        Observation epochs (any shape) in days.
    baseline : float
        Constant magnitude to assign.

    Returns
    -------
    np.ndarray
        Array with the same shape as `timestamps`, filled with `baseline`.
    """
    return np.full_like(np.asarray(timestamps, dtype=float),
                        fill_value=baseline,
                        dtype=float)



def rrlyr_variable(timestamps, baseline, bailey=None):
    """Simulates a variable source. 
    This simulation is done using the gatspy module provided by AstroML. 
    This module implements a template-based model using RR Lyrae
    lightcurves from Sesar et al. 2010, which contains observations of
    RR Lyrae in Stripe 82 (from SDSS) measured approximately over a 
    decade. For the purpose of simulating variable stars we apply a 
    single band ('g') template model and modify only the period. We 
    currently only provide simulated RR Lyrae (periods < 1 day) or 
    Cepheid Variables which have an average period of 10 days.
    
    See:
    Template-based Period Fitting: https://www.astroml.org/gatspy/periodic/template.html
    Period distribution for RR Lyrae from Sesar et al. 2010 (https://arxiv.org/abs/0910.4611).
    Period distribution for Cepheids from Becker et al. 1977 (http://adsabs.harvard.edu/abs/1977ApJ...218..633B)

    Parameters:
    ----------
        timestamps : array
            Times at which to simulate the lightcurve.
        baseline : float
            Baseline at which to simulate the lightcurve.
        bailey : int, optional 
            The type of variable to simulate. A bailey
            value of 1 simulaes RR Lyrae type ab, a value
            of 2 simulates RR Lyrae type c, and a value of 
            3 simulates a Cepheid variable period, but note the amplitude
            is still derived from the RRLyrae template. If not provided
            it defaults to a random choice between 1 and 2. 

    Returns:
    -------
        mag : array
            Simulated magnitudes given the timestamps.
        amplitude : float
            Amplitude of the signal in mag. 
        period : float
            Period of the signal in days.
    """

    if bailey is None: 
        bailey = np.random.randint(1,3) #Removing Bailey=3 option
    if bailey < 0 or bailey > 3:
        raise RuntimeError("Bailey out of range, must be between 1 and 3.")

    if bailey == 1:
        period = np.random.normal(0.6, 0.15)
    elif bailey == 2:
        period = np.random.normal(0.33, 0.1)
    elif bailey == 3:
        period = np.random.lognormal(0., 0.2)
        period = 10**period
    
    period=np.abs(period) #on rare occassions period is negative which breaks the code

    #Fetch random RRLyrae template 
    rrlyrae = fetch_rrlyrae_templates(data_home=get_rrlyr_data_path())
    unique_id = np.unique([i[:-1] for i in rrlyrae.ids])
    inx = np.random.randint(len(unique_id)) #Random index to choose RR Lyrae lc

    filt = 'g' #All 23 RRLyr templates have g data! 
    lcid = unique_id[inx] + filt #rrlyrae.ids[inx]
    t, mag = rrlyrae.get_template(lcid)

    #Fit to our simulated cadence
    model = RRLyraeTemplateModeler(filt)
    model.fit(t, mag)
    mag_fit = model.predict(timestamps, period = period)

    #The following fit is only done to extract amplitude.
    mag_fit_amp = model.predict(np.arange(0, period, 0.01), period = period)
    amplitude = np.ptp(mag_fit_amp) / 2.0

    #Bring lc down to 0 and add input baseline
    mag_fit = mag_fit - np.mean(mag_fit_amp)
    mag_fit = mag_fit+baseline

    return np.array(mag_fit), amplitude, period


def get_rrlyr_data_path():
    """
    Retrieves the path to the RRLyrae template data directory within the MicroLIA package.

    Returns:
    --------
    str
        Path to the data directory.
    """

    #resource_package = __name__
    #resource_path = 'data'
    #data_path = pkg_resources.resource_filename(resource_package, resource_path)
    #return data_path
    return str(resources.files(__package__) / 'data')

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

BELOW IS CODE FROM THE GATSPY PROGRAM!!!

This was done because gatspy.period.RRLyraeTemplateModeler() does not allow users
to specify the path to the RRLyrae templates! Therefore, the below code manually sets the path to the program data directory.

This is based on the algorithm described in Sesar 2010.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


class PeriodicModeler(object):
    """Base class for periodic modeling"""
    def __init__(self, optimizer=None, fit_period=False,
                 optimizer_kwds=None, *args, **kwargs):
        if optimizer is None:
            kwds = optimizer_kwds or {}
            optimizer = LinearScanOptimizer(**kwds)
        elif optimizer_kwds:
            warnings.warn("Optimizer specified, so optimizer keywords ignored")

        if not hasattr(optimizer, 'best_period'):
            raise ValueError("optimizer must be a PeriodicOptimizer instance: "
                             "{0} has no best_period method".format(optimizer))
        self.optimizer = optimizer
        self.fit_period = fit_period
        self.args = args
        self.kwargs = kwargs
        self._best_period = None

    def fit(self, t, y, dy=None):
        """Fit the multiterm Periodogram model to the data.

        Parameters
        ----------
        t : array_like, one-dimensional
            sequence of observation times
        y : array_like, one-dimensional
            sequence of observed values
        dy : float or array_like (optional)
            errors on observed values
        """
        # For linear models, dy=1 is equivalent to no errors
        if dy is None:
            dy = 1

        self.t, self.y, self.dy = np.broadcast_arrays(t, y, dy)

        self._fit(self.t, self.y, self.dy)
        self._best_period = None  # reset best period in case of refitting

        if self.fit_period:
            self._best_period = self._calc_best_period()
            
        return self

    def predict(self, t, period=None):
        """Compute the best-fit model at ``t`` for a given frequency omega

        Parameters
        ----------
        t : float or array_like
            times at which to predict
        period : float (optional)
            The period at which to compute the model. If not specified, it
            will be computed via the optimizer provided at initialization.

        Returns
        -------
        y : np.ndarray
            predicted model values at times t
        """
        t = np.asarray(t)
        if period is None:
            period = self.best_period
        result = self._predict(t.ravel(), period=period)
        return result.reshape(t.shape)

    def score_frequency_grid(self, f0, df, N):
        """Compute the score on a frequency grid.

        Some models can compute results faster if the inputs are passed in this
        manner.

        Parameters
        ----------
        f0, df, N : (float, float, int)
            parameters describing the frequency grid freq = f0 + df * arange(N)
            Note that these are frequencies, not angular frequencies.

        Returns
        -------
        score : ndarray
            the length-N array giving the score at each frequency
        """
        return self._score_frequency_grid(f0, df, N)

    def periodogram_auto(self, oversampling=5, nyquist_factor=3,
                         return_periods=True):
        """Compute the periodogram on an automatically-determined grid

        This function uses heuristic arguments to choose a suitable frequency
        grid for the data. Note that depending on the data window function,
        the model may be sensitive to periodicity at higher frequencies than
        this function returns!

        The final number of frequencies will be
        Nf = oversampling * nyquist_factor * len(t) / 2

        Parameters
        ----------
        oversampling : float
            the number of samples per approximate peak width
        nyquist_factor : float
            the highest frequency, in units of the nyquist frequency for points
            spread uniformly through the data range.

        Returns
        -------
        period : ndarray
            the grid of periods
        power : ndarray
            the power at each frequency
        """
        N = len(self.t)
        T = np.max(self.t) - np.min(self.t)
        df = 1. / T / oversampling
        f0 = df
        Nf = int(0.5 * oversampling * nyquist_factor * N)
        freq = f0 + df * np.arange(Nf)
        return 1. / freq, self._score_frequency_grid(f0, df, Nf)

    def score(self, periods=None):
        """Compute the periodogram for the given period or periods

        Parameters
        ----------
        periods : float or array_like
            Array of angular frequencies at which to compute
            the periodogram.

        Returns
        -------
        scores : np.ndarray
            Array of normalized powers (between 0 and 1) for each frequency.
            Shape of scores matches the shape of the provided periods.
        """
        periods = np.asarray(periods)
        return self._score(periods.ravel()).reshape(periods.shape)

    periodogram = score

    @property
    def best_period(self):
        """Lazy evaluation of the best period given the model"""
        if self._best_period is None:
            self._best_period = self._calc_best_period()
        return self._best_period

    def find_best_periods(self, n_periods=5, return_scores=False):
        """Find the top several best periods for the model"""
        return self.optimizer.find_best_periods(self, n_periods,
                                                return_scores=return_scores)

    def _calc_best_period(self):
        """Compute the best period using the optimizer"""
        return self.optimizer.best_period(self)

    # The following methods should be overloaded by derived classes:

    def _score_frequency_grid(self, f0, df, N):
        freq = f0 + df * np.arange(N)
        return self._score(1. / freq)

    def _score(self, periods):
        """Compute the score of the model given the periods"""
        raise NotImplementedError()

    def _fit(self, t, y, dy):
        """Fit the model to the given data"""
        raise NotImplementedError()

    def _predict(self, t, period):
        """Predict the model values at the given times"""
        raise NotImplementedError()


class PeriodicModelerMultiband(PeriodicModeler):
    """Base class for periodic modeling on multiband data"""

    def fit(self, t, y, dy=None, filts=0):
        """Fit the multiterm Periodogram model to the data.

        Parameters
        ----------
        t : array_like, one-dimensional
            sequence of observation times
        y : array_like, one-dimensional
            sequence of observed values
        dy : float or array_like (optional)
            errors on observed values
        filts : array_like (optional)
            The array specifying the filter/bandpass for each observation.
        """
        self.unique_filts_ = np.unique(filts)

        # For linear models, dy=1 is equivalent to no errors
        if dy is None:
            dy = 1

        all_data = np.broadcast_arrays(t, y, dy, filts)
        self.t, self.y, self.dy, self.filts = map(np.ravel, all_data)

        self._fit(self.t, self.y, self.dy, self.filts)
        self._best_period = None  # reset best period in case of refitting

        if self.fit_period:
            self._best_period = self._calc_best_period()
        return self

    def predict(self, t, filts, period=None):
        """Compute the best-fit model at ``t`` for a given frequency omega

        Parameters
        ----------
        t : float or array_like
            times at which to predict
        filts : array_like (optional)
            the array specifying the filter/bandpass for each observation. This
            is used only in multiband periodograms.
        period : float (optional)
            The period at which to compute the model. If not specified, it
            will be computed via the optimizer provided at initialization.

        Returns
        -------
        y : np.ndarray
            predicted model values at times t
        """
        unique_filts = set(np.unique(filts))
        if not unique_filts.issubset(self.unique_filts_):
            raise ValueError("filts does not match training data: "
                             "input: {0} output: {1}"
                             "".format(set(self.unique_filts_),
                                       set(unique_filts)))
        t, filts = np.broadcast_arrays(t, filts)

        if period is None:
            period = self.best_period

        result = self._predict(t.ravel(), filts=filts.ravel(), period=period)
        return result.reshape(t.shape)

    # The following methods should be overloaded by derived classes:

    def _score(self, periods):
        """Compute the score of the model given the periods"""
        raise NotImplementedError()

    def _fit(self, t, y, dy, filts):
        """Fit the model to the given data"""
        raise NotImplementedError()

    def _predict(self, t, filts, period):
        """Predict the model values at the given times & filters"""
        raise NotImplementedError()

class PeriodicOptimizer(object):
    def find_best_periods(self, model, n_periods=5, return_scores=False):
        raise NotImplementedError()

    def best_period(self, model):
        periods = self.find_best_periods(model, n_periods=1,
                                         return_scores=False)
        return periods[0]


class LinearScanOptimizer(PeriodicOptimizer):
    """Optimizer based on a linear scan of candidate frequencies.

    Parameters / Attributes
    -----------------------
    period_range : tuple
        (min_period, max_period) for the linear scan
    quiet : bool (default = False)
        If true, then suppress printed output during optimization.
        By default, information is printed to stdout.
    first_pass_coverage : float (default = 5.0)
        estimated number of points across the width of a typical peak for the
        initial scan.
    final_pass_coverage : float (default = 500.0)
        estimated number of points across the width of a typical peak within
        the final scan.
    """
    def __init__(self, period_range=None, quiet=False,
                 first_pass_coverage=5, final_pass_coverage=500):
        self._period_range = period_range
        self.quiet = quiet
        self.first_pass_coverage = first_pass_coverage
        self.final_pass_coverage = final_pass_coverage

    def set(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def period_range(self):
        if self._period_range is None:
            raise ValueError("period_range must be set in optimizer in order "
                             "to find the best periods. For example:\n"
                             " >>> model = LombScargle(fit_period=True)\n"
                             " >>> model.optimizer.period_range = (0.2, 1.0)")
        return self._period_range

    @period_range.setter
    def period_range(self, value):
        value = tuple(value)
        assert len(value) == 2
        self._period_range = value

    def compute_grid_size(self, model):
        # compute the estimated peak width from the data range
        tmin, tmax = np.min(model.t), np.max(model.t)
        width = 2 * np.pi / (tmax - tmin)

        # our candidate steps in omega is controlled by period_range & coverage
        omega_step = width / self.first_pass_coverage
        omega_min = 2 * np.pi / np.max(self.period_range)
        omega_max = 2 * np.pi / np.min(self.period_range)
        N = (omega_max - omega_min) // omega_step

        return N

    def find_best_periods(self, model, n_periods=5, return_scores=False):
        """Find the `n_periods` best periods in the model"""

        # compute the estimated peak width from the data range
        tmin, tmax = np.min(model.t), np.max(model.t)
        width = 2 * np.pi / (tmax - tmin)

        # our candidate steps in omega is controlled by period_range & coverage
        omega_step = width / self.first_pass_coverage
        omega_min = 2 * np.pi / np.max(self.period_range)
        omega_max = 2 * np.pi / np.min(self.period_range)
        omegas = np.arange(omega_min, omega_max + omega_step, omega_step)
        periods = 2 * np.pi / omegas

        # print some updates if desired
        if not self.quiet:
            print("Finding optimal frequency:")
            print(" - Estimated peak width = {0:.3g}".format(width))
            print(" - Using {0} steps per peak; "
                  "omega_step = {1:.3g}".format(self.first_pass_coverage,
                                                omega_step))
            print(" - User-specified period range: "
                  " {0:.2g} to {1:.2g}".format(periods.min(), periods.max()))
            print(" - Computing periods at {0:.0f} steps".format(len(periods)))
            sys.stdout.flush()

        # Compute the score on the initial grid
        N = int(1 + width // omega_step)
        score = model.score_frequency_grid(omega_min / (2 * np.pi),
                                           omega_step / (2 * np.pi),
                                           len(omegas))

        # find initial candidates of unique peaks
        minscore = score.min()
        n_candidates = max(5, 2 * n_periods)
        candidate_freqs = np.zeros(n_candidates)
        candidate_scores = np.zeros(n_candidates)
        for i in range(n_candidates):
            j = np.argmax(score)
            candidate_freqs[i] = omegas[j]
            candidate_scores[i] = score[j]
            score[max(0, j - N):(j + N)] = minscore

        # If required, do a final pass on these unique at higher resolution
        if self.final_pass_coverage <= self.first_pass_coverage:
            best_periods = 2 * np.pi / candidate_freqs[:n_periods]
            best_scores = candidate_scores[:n_periods]
        else:
            f0 = -omega_step / (2 * np.pi)
            df = width / self.final_pass_coverage / (2 * np.pi)
            Nf = abs(2 * f0) // df
            steps = f0 + df * np.arange(Nf)
            candidate_freqs /= (2 * np.pi)

            freqs = steps + candidate_freqs[:, np.newaxis]
            periods = 1. / freqs

            if not self.quiet:
                print("Zooming-in on {0} candidate peaks:"
                      "".format(n_candidates))
                print(" - Computing periods at {0:.0f} "
                      "steps".format(periods.size))
                sys.stdout.flush()

            #scores = model.score(periods)
            scores = np.array([model.score_frequency_grid(c + f0, df, Nf)
                               for c in candidate_freqs])
            best_scores = scores.max(1)
            j = np.argmax(scores, 1)
            i = np.argsort(best_scores)[::-1]

            best_periods = periods[i, j[i]]
            best_scores = best_scores[i]

        if return_scores:
            return best_periods[:n_periods], best_scores[:n_periods]
        else:
            return best_periods[:n_periods]


__all__ = ['RRLyraeTemplateModeler', 'RRLyraeTemplateModelerMultiband']


class BaseTemplateModeler(PeriodicModeler):
    """
    Base class for single-band template models

    To extend this, overload the ``_template_ids`` and ``_get_template_by_id``
    methods.
    """

    def __init__(self, optimizer=None, fit_period=False, optimizer_kwds=None):
        self.templates = self._build_interpolated_templates()
        if len(self.templates) == 0:
            raise ValueError('No templates available!')
        PeriodicModeler.__init__(self, optimizer=optimizer,
                                 fit_period=fit_period,
                                 optimizer_kwds=optimizer_kwds)

    def _build_interpolated_templates(self):
        self.templates = [self._interpolated_template(tid)
                          for tid in self._template_ids()]
        return self.templates

    def _interpolated_template(self, templateid):
        """Return an interpolator for the given template"""
        phase, y = self._get_template_by_id(templateid)

        # double-check that phase ranges from 0 to 1
        assert phase.min() >= 0
        assert phase.max() <= 1

        # at the start and end points, we need to add ~5 points to make sure
        # the spline & derivatives wrap appropriately
        phase = np.concatenate([phase[-5:] - 1, phase, phase[:5] + 1])
        y = np.concatenate([y[-5:], y, y[:5]])

        # Univariate spline allows for derivatives; use this!
        return UnivariateSpline(phase, y, s=0, k=5)

    def _fit(self, t, y, dy):
        if dy.size == 1:
            ymean = np.mean(y)
        else:
            w = 1 / dy ** 2
            ymean = np.dot(y, w) / w.sum()
        self.chi2_0_ = np.sum((y - ymean) ** 2 / self.dy ** 2)

    def _score(self, periods):
        scores = np.zeros(periods.shape)

        for i, period in enumerate(periods.flat):
            theta_best, chi2 = self._eval_templates(period)
            scores.flat[i] = 1 - min(chi2) / self.chi2_0_

        return scores

    def _predict(self, t, period):
        theta_best, chi2 = self._eval_templates(period)
        i_best = np.argmin(chi2)
        return self._model(t, theta_best[i_best], period, i_best)

    def _eval_templates(self, period):
        """Evaluate the best template for the given period"""
        theta_best = [self._optimize(period, tmpid)
                      for tmpid, _ in enumerate(self.templates)]
        chi2 = [self._chi2(theta, period, tmpid)
                for tmpid, theta in enumerate(theta_best)]

        return theta_best, chi2

    def _model(self, t, theta, period, tmpid):
        """Compute model at t for the given parameters, period, & template"""
        template = self.templates[tmpid]
        phase = (t / period - theta[2]) % 1
        return theta[0] + theta[1] * template(phase)

    def _chi2(self, theta, period, tmpid, return_gradient=False):
        """
        Compute the chi2 for the given parameters, period, & template

        Optionally return the gradient for faster optimization
        """
        template = self.templates[tmpid]
        phase = (self.t / period - theta[2]) % 1
        model = theta[0] + theta[1] * template(phase)
        chi2 = (((model - self.y) / self.dy) ** 2).sum()

        if return_gradient:
            grad = 2 * (model - self.y) / self.dy ** 2
            gradient = np.array([np.sum(grad),
                                 np.sum(grad * template(phase)),
                                 -np.sum(grad * theta[1]
                                         * template.derivative(1)(phase))])
            return chi2, gradient
        else:
            return chi2

    def _optimize(self, period, tmpid, use_gradient=True):
        """Optimize the model for the given period & template"""
        theta_0 = [self.y.min(), self.y.max() - self.y.min(), 0]
        result = minimize(self._chi2, theta_0, jac=bool(use_gradient),
                          bounds=[(None, None), (0, None), (None, None)],
                          args=(period, tmpid, use_gradient))
        return result.x

    #------------------------------------------------------------
    # Overload the following two functions in base classes

    def _template_ids(self):
        """Return the list of template ids"""
        raise NotImplementedError()

    def _get_template_by_id(self, template_id):
        """Get a particular template

        Parameters
        ----------
        template_id : simple type
            Template ID used by base class to define templates
        
        Returns
        -------
        phase, y : ndarrays
            arrays containing the sorted phase and associated y-values.
        """
        raise NotImplementedError()
    

class RRLyraeTemplateModeler(BaseTemplateModeler):
    """Template-fitting periods for single-band RR Lyrae

    This class contains functionality to evaluate the fit of the Sesar 2010
    RR Lyrae templates to single-band data.

    Parameters
    ----------
    filts : list or iterable of characters (optional)
        The filters of the templates to be used. Items should be among 'ugriz'.
        Default is 'ugriz'; i.e. all available templates.
    optimizer : PeriodicOptimizer instance (optional)
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    fit_period : bool (optional)
        If True, then fit for the best period when fit() method is called.
    optimizer_kwds : dict (optional
        Dictionary of keyword arguments for constructing the optimizer

    See Also
    --------
    RRLyraeTemplateModelerMultiband : multiband version of template model
    """
    _raw_templates = fetch_rrlyrae_templates(data_home=get_rrlyr_data_path())

    def __init__(self, filts='ugriz', optimizer=None,
                 fit_period=False, optimizer_kwds=None):
        self.filts = list(filts)
        BaseTemplateModeler.__init__(self, optimizer=optimizer,
                                     fit_period=fit_period,
                                     optimizer_kwds=optimizer_kwds)

    def _template_ids(self):
        return (tid for tid in self._raw_templates.ids
                if tid[-1] in self.filts)

    def _get_template_by_id(self, tid):
        return self._raw_templates.get_template(tid)


class RRLyraeTemplateModelerMultiband(PeriodicModelerMultiband):
    """Multiband version of RR Lyrae template-fitting modeler

    This class contains functionality to evaluate the fit of the Sesar 2010
    RR Lyrae templates to multiband data.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance (optional)
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.

    See Also
    --------
    RRLyraeTemplateModeler : single band version of template model
    """

    def _fit(self, t, y, dy, filts):
        self.models_ = []
        for filt in self.unique_filts_:
            mask = (filts == filt)
            model = RRLyraeTemplateModeler(filts=filt)
            model.fit(t[mask], y[mask], dy[mask])
            self.models_.append(model)
        self.modeldict_ = dict(zip(self.unique_filts_, self.models_))

    def _score(self, periods):
        weights = [model.chi2_0_ for model in self.models_]
        scores = [model.score(periods) for model in self.models_]
        return np.dot(weights, scores) / np.sum(weights)

    def _predict(self, t, filts, period):
        result = np.zeros(t.shape)
        for filt in np.unique(filts):
            mask = (filts == filt)
            result[mask] = self.modeldict_[filt].predict(t[mask], period)
        return result
   
