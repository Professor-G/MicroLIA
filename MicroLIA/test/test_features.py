# -*- coding: utf-8 -*-
"""
    Created on Fri Jan 13 14:46:19 2017
    @author: danielgodinez
"""

import unittest
import numpy as np
import pkg_resources

import sys
sys.path.append('../../')
from MicroLIA.features import *
from MicroLIA.extract_features import extract_all

resource_package = __name__
file = pkg_resources.resource_filename(resource_package, 'test_ogle_lc.dat')

test_lc = np.loadtxt(file)
time, mag, magerr = test_lc[:,0], test_lc[:,1], test_lc[:,2]

#Remove the nan and inf values, if present in the lightcurve
mask = np.where(np.isfinite(time) & np.isfinite(mag) & np.isfinite(magerr))[0]
time, mag, magerr = time[mask], mag[mask], magerr[mask]

# Convert to flux
zp = 22
flux = 10**(-(mag-zp) / 2.5)
flux_err = (magerr * flux) / (2.5) * np.log(10)

# Normalize by max flux
norm_flux = flux / np.max(flux)
norm_fluxerr = flux_err * (norm_flux / flux)

class Test(unittest.TestCase):
    """Unittest to ensure all individual features work including the feature extraction function. 
    """

    def test_AndersonDarling(self):
        value = AndersonDarling(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.999999
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="AndersonDarling function with weights failed.")

        value = AndersonDarling(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.0
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="AndersonDarling function without weights failed.")

    def test_FluxPercentileRatioMid20(self):
        value = FluxPercentileRatioMid20(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.159674
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid20 function with weights failed.")

        value = FluxPercentileRatioMid20(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.1482182
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid20 function without weights failed.")

    def test_FluxPercentileRatioMid35(self):
        value = FluxPercentileRatioMid35(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.278154
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid35 function with weights failed.")

        value = FluxPercentileRatioMid35(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.250295
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid35 function without weights failed.")

    def test_FluxPercentileRatioMid50(self):
        value = FluxPercentileRatioMid50(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.42086381
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid50 function with weights failed.")

        value = FluxPercentileRatioMid50(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.37990401
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid50 function without weights failed.")

    def test_FluxPercentileRatioMid65(self):
        value = FluxPercentileRatioMid65(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.55882556
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid65 function with weights failed.")

        value = FluxPercentileRatioMid65(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.5510196
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid65 function without weights failed.")

    def test_FluxPercentileRatioMid80(self):
        value = FluxPercentileRatioMid80(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.8153054
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid80 function with weights failed.")

        value = FluxPercentileRatioMid80(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.78723416
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid80 function without weights failed.")

    def test_Gskew(self):
        value = Gskew(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = -0.0743417
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="Gskew function with weights failed.")

        value = Gskew(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0749738
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="Gskew function without weights failed.")

    def test_LinearTrend(self):
        value = LinearTrend(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 2.200513e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-5, msg="LinearTrend function with weights failed.")

        value = LinearTrend(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 2.061729e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-5, msg="LinearTrend function without weights failed.")

    def test_MaxSlope(self):
        value = MaxSlope(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0677382
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="MaxSlope function with weights failed.")

        value = MaxSlope(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.6859594
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="MaxSlope function without weights failed.")

    def test_PairSlopeTrend(self):
        value = PairSlopeTrend(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.44827586
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PairSlopeTrend function with weights failed.")

        value = PairSlopeTrend(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.43333333
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PairSlopeTrend function without weights failed.")

    def test_PercentAmplitude(self):
        value = PercentAmplitude(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.3733192
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PercentAmplitude function with weights failed.")

        value = PercentAmplitude(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.36772882
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PercentAmplitude function without weights failed.")

    def test_PercentDifferenceFluxPercentile(self):
        value = PercentDifferenceFluxPercentile(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.1492617
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PercentDifferenceFluxPercentile function with weights failed.")

        value = PercentDifferenceFluxPercentile(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.16703715
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PercentDifferenceFluxPercentile function without weights failed.")

    def test_above1(self):
        value = above1(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0003954281
        self.assertAlmostEqual(value, expected_value, delta=1e-5, msg="above1 function with weights failed.")

        value = above1(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.057377
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="above1 function without weights failed.")

    def test_above3(self):
        value = above3(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 4.5119902e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-6, msg="above3 function with weights failed.")

        value = above3(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0204918
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="above3 function without weights failed.")

    def test_above5(self):
        value = above5(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1.073958e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-6, msg="above5 function with weights failed.")

        value = above5(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0081967
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="above5 function without weights failed.")

    def test_abs_energy(self):
        value = abs_energy(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 411186.841739
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="abs_energy function with weights failed.")

        value = abs_energy(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 130.22714027
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="abs_energy function without weights failed.")

    def test_abs_sum_changes(self):
        value = abs_sum_changes(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 353.174336
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="abs_sum_changes function with weights failed.")

        value = abs_sum_changes(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 9.449282719
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="abs_sum_changes function without weights failed.")

    def test_amplitude(self):
        value = amplitude(time, flux, flux_err, apply_weights=True)
        expected_value = 3.3854549
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="amplitude function with weights failed.")

        value = amplitude(time, flux, flux_err, apply_weights=False)
        expected_value = 11.345267992
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="amplitude function without weights failed.")

    def test_auto_corr(self):
        value = auto_corr(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.001280894
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="auto_corr function with weights failed.")

        value = auto_corr(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.44277988
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="auto_corr function without weights failed.")

    def test_below1(self):
        value = below1(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.000749877
        self.assertAlmostEqual(value, expected_value, delta=1e-5, msg="below1 function with weights failed.")

        value = below1(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.12295081
        self.assertAlmostEqual(value, expected_value, delta=1e-5, msg="below1 function without weights failed.")

    def test_below3(self):
        value = below3(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0
        self.assertEqual(value, expected_value, msg="below3 function with weights failed.")

        value = below3(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0
        self.assertEqual(value, expected_value, msg="below3 function without weights failed.")

    def test_below5(self):
        value = below5(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0
        self.assertEqual(value, expected_value, msg="below5 function with weights failed.")

        value = below5(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0
        self.assertEqual(value, expected_value, msg="below5 function without weights failed.")

    def test_benford_correlation(self):
        value = benford_correlation(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = -0.2965806
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="benford_correlation function with weights failed.")

        value = benford_correlation(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = -0.3059369
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="benford_correlation function without weights failed.")

    def test_c3(self):
        value = c3(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.384324342
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="c3 function with weights failed.")

        value = c3(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.3896391
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="c3 function without weights failed.")

    def test_check_for_duplicate(self):
        value = check_for_duplicate(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1
        self.assertEqual(value, expected_value, msg="check_for_duplicate function with weights failed.")

        value = check_for_duplicate(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1
        self.assertEqual(value, expected_value, msg="check_for_duplicate function without weights failed.")

    def test_check_for_max_duplicate(self):
        value = check_for_max_duplicate(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1
        self.assertEqual(value, expected_value, msg="check_for_max_duplicate function with weights failed.")

        value = check_for_max_duplicate(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0
        self.assertEqual(value, expected_value, msg="check_for_max_duplicate function without weights failed.")

    def test_check_for_min_duplicate(self):
        value = check_for_min_duplicate(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1
        self.assertEqual(value, expected_value, msg="check_for_min_duplicate function with weights failed.")

        value = check_for_min_duplicate(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0
        self.assertEqual(value, expected_value, msg="check_for_min_duplicate function without weights failed.")

    def test_check_max_last_loc(self):
        value = check_max_last_loc(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.43442622
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="check_max_last_loc function with weights failed.")

        value = check_max_last_loc(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.5614754
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="check_max_last_loc function without weights failed.")

    def test_check_min_last_loc(self):
        value = check_min_last_loc(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.11475409
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="check_min_last_loc function with weights failed.")

        value = check_min_last_loc(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.59426229
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="check_min_last_loc function without weights failed.")

    def test_complexity(self):
        value = complexity(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0490217
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="complexity function with weights failed.")

        value = complexity(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.804158
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="complexity function without weights failed.")

    def test_con(self):
        value = con(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.19672131
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="con function with weights failed.")

        value = con(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.45901639
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="con function without weights failed.")

    def test_count_above(self):
        value = count_above(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.51072427
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="count_above function with weights failed.")

        value = count_above(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.49590163
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="count_above function without weights failed.")

    def test_count_below(self):
        value = count_below(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.444113
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="count_below function with weights failed.")

        value = count_below(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.4918032
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="count_below function without weights failed.")

    def test_cusum(self):
        value = cusum(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.135994224
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="cusum function with weights failed.")

        value = cusum(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.113665824
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="cusum function without weights failed.")

    def test_first_loc_max(self):
        value = first_loc_max(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.78688524
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="first_loc_max function with weights failed.")

        value = first_loc_max(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.55737704
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="first_loc_max function without weights failed.")

    def test_first_loc_min(self):
        value = first_loc_min(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.57377049
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="first_loc_min function with weights failed.")

        value = first_loc_min(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.59016393
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="first_loc_min function without weights failed.")

    def test_half_mag_amplitude_ratio(self):
        value = half_mag_amplitude_ratio(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.96093949
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="half_mag_amplitude_ratio function with weights failed.")

        value = half_mag_amplitude_ratio(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.18896064
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="half_mag_amplitude_ratio function without weights failed.")

    def test_index_mass_quantile(self):
        value = index_mass_quantile(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.54508196
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="index_mass_quantile function with weights failed.")

        value = index_mass_quantile(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.50819672
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="index_mass_quantile function without weights failed.")

    def test_integrate(self):
        value = integrate(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 647.3016737407153
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="integrate function with weights failed.")

        value = integrate(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 647.3016737407153
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="integrate function without weights failed.")

    def test_kurtosis(self):
        value = kurtosis(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 9.3216905
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="kurtosis function with weights failed.")

        value = kurtosis(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 8.8791982
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="kurtosis function without weights failed.")

    def test_large_standard_deviation(self):
        value = large_standard_deviation(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0
        self.assertEqual(value, expected_value, msg="large_standard_deviation function with weights failed.")

        value = large_standard_deviation(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0
        self.assertEqual(value, expected_value, msg="large_standard_deviation function without weights failed.")

    def test_longest_strike_above(self):
        value = longest_strike_above(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.02459016
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="longest_strike_above function with weights failed.")

        value = longest_strike_above(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.00409836
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="longest_strike_above function without weights failed.")

    def test_longest_strike_below(self):
        value = longest_strike_below(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.04918032
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="longest_strike_below function with weights failed.")

        value = longest_strike_below(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.00409836
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="longest_strike_below function without weights failed.")

    def test_meanMag(self):
        value = meanMag(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.7281628094835779
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="meanMag function with weights failed.")

        value = meanMag(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.7281628094835779
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="meanMag function without weights failed.")

    def test_mean_abs_change(self):
        value = mean_abs_change(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0401601158
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_abs_change function with weights failed.")

        value = mean_abs_change(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0388859371
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_abs_change function without weights failed.")

    def test_mean_change(self):
        value = mean_change(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0002078532
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_change function with weights failed.")

        value = mean_change(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0001021156
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_change function without weights failed.")

    def test_mean_n_abs_max(self):
        value = mean_n_abs_max(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.8470422594
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_n_abs_max function with weights failed.")

        value = mean_n_abs_max(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.8728018351
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_n_abs_max function without weights failed.")

    def test_mean_second_derivative(self):
        value = mean_second_derivative(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = -0.02545928
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_second_derivative function with weights failed.")

        value = mean_second_derivative(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = -1.44923422e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_second_derivative function without weights failed.")

    def test_medianAbsDev(self):
        value = medianAbsDev(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1.22668694
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="medianAbsDev function with weights failed.")

        value = medianAbsDev(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.02188771
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="medianAbsDev function without weights failed.")

    def test_median_buffer_range(self):
        value = median_buffer_range(time, flux, flux_err, apply_weights=True)
        expected_value = 0.0491803278
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="median_buffer_range function with weights failed.")

        value = median_buffer_range(time, flux, flux_err, apply_weights=False)
        expected_value = 0.2090163934
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="median_buffer_range function without weights failed.")

    def test_median_distance(self):
        value = median_distance(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 42.76261876
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="median_distance function with weights failed.")

        value = median_distance(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.027840578
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="median_distance function without weights failed.")

    def test_number_cwt_peaks(self):
        value = number_cwt_peaks(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.02459016393442623
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_cwt_peaks function with weights failed.")

        value = number_cwt_peaks(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.02459016393442623
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_cwt_peaks function without weights failed.")

    def test_number_of_crossings(self):
        value = number_of_crossings(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.36885245
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_of_crossings function with weights failed.")

        value = number_of_crossings(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.43032786
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_of_crossings function without weights failed.")

    def test_number_of_peaks(self):
        value = number_of_peaks(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.01639344
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_of_peaks function with weights failed.")

        value = number_of_peaks(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.06557377
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_of_peaks function without weights failed.")

    def test_peak_detection(self):
        value = peak_detection(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.004098360655737705
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="peak_detection function with weights failed.")

        value = peak_detection(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.004098360655737705
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="peak_detection function without weights failed.")

    def test_permutation_entropy(self):
        value = permutation_entropy(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 6.568915581
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="permutation_entropy function with weights failed.")

        value = permutation_entropy(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.784228458
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="permutation_entropy function without weights failed.")

    def test_quantile(self):
        value = quantile(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.7492040876180983
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="quantile function with weights failed.")

        value = quantile(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.7492040876180983
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="quantile function without weights failed.")

    def test_ratio_recurring_points(self):
        value = ratio_recurring_points(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.99305555
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="ratio_recurring_points function with weights failed.")

        value = ratio_recurring_points(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.38888888
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="ratio_recurring_points function without weights failed.")

    def test_root_mean_squared(self):
        value = root_mean_squared(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 2.09603896
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="root_mean_squared function with weights failed.")

        value = root_mean_squared(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.73113908
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="root_mean_squared function without weights failed.")

    def test_sample_entropy(self):
        value = sample_entropy(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1.7475882509149177
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="sample_entropy function with weights failed.")

        value = sample_entropy(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.7475882509149177
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="sample_entropy function without weights failed.")

    def test_shannon_entropy(self):
        value = shannon_entropy(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 29.10782792161686
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="shannon_entropy function with weights failed.")

        value = shannon_entropy(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 29.10782792161686
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="shannon_entropy function without weights failed.")

    def test_shapiro_wilk(self):
        value = shapiro_wilk(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.8506647348403931
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="shapiro_wilk function with weights failed.")

        value = shapiro_wilk(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.8506647348403931
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="shapiro_wilk function without weights failed.")

    def test_skewness(self):
        value = skewness(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1.162139172
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="skewness function with weights failed.")

        value = skewness(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.687363638
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="skewness function without weights failed.")

    def test_std_over_mean(self):
        value = std_over_mean(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.055833597
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="std_over_mean function with weights failed.")

        value = std_over_mean(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.066707594
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="std_over_mean function without weights failed.")

    def test_stetsonJ(self):
        value = stetsonJ(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 83.30990668739187
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonJ function with weights failed.")

        value = stetsonJ(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 83.30990668739187
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonJ function without weights failed.")

    def test_stetsonK(self):
        value = stetsonK(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.7246737139789522
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonK function with weights failed.")

        value = stetsonK(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.7246737139789522
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonK function without weights failed.")

    def test_stetsonL(self):
        value = stetsonL(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 75.65476126615566
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonL function with weights failed.")

        value = stetsonL(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 75.65476126615566
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonL function without weights failed.")

    def test_sum_values(self):
        value = sum_values(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.7281628094
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="sum_values function with weights failed.")

        value = sum_values(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.7289300572
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="sum_values function without weights failed.")

    def test_symmetry_looking(self):
        value = symmetry_looking(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1
        self.assertEqual(value, expected_value, msg="symmetry_looking function with weights failed.")

        value = symmetry_looking(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1
        self.assertEqual(value, expected_value, msg="symmetry_looking function without weights failed.")

    def test_time_reversal_asymmetry(self):
        value = time_reversal_asymmetry(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = -9.269384825779411e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-6, msg="time_reversal_asymmetry function with weights failed.")

        value = time_reversal_asymmetry(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = -9.269384825779422e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-6, msg="time_reversal_asymmetry function without weights failed.")

    def test_variance(self):
        value = variance(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0016534948
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="variance function with weights failed.")

        value = variance(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0023787596
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="variance function without weights failed.")

    def test_variance_larger_than_standard_deviation(self):
        value = variance_larger_than_standard_deviation(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0
        self.assertEqual(value, expected_value, msg="variance_larger_than_standard_deviation function with weights failed.")

        value = variance_larger_than_standard_deviation(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0
        self.assertEqual(value, expected_value, msg="variance_larger_than_standard_deviation function without weights failed.")

    def test_variation_coefficient(self):
        value = variation_coefficient(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0558335973
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="variation_coefficient function with weights failed.")

        value = variation_coefficient(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0669097526
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="variation_coefficient function without weights failed.")

    def test_vonNeumannRatio(self):
        value = vonNeumannRatio(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 24097.24856
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="vonNeumannRatio function with weights failed.")

        value = vonNeumannRatio(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.118731806
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="vonNeumannRatio function without weights failed.")

    def test_extract_all(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            
            value = extract_all(time, mag, magerr, apply_weights=True, convert=True, zp=zp)
            expected_value = [ 1.00000000e+00,  1.59674431e-01,  2.78154635e-01,  4.20863819e-01,
                5.58825565e-01,  8.15305445e-01, -7.43417014e-02,  2.20051317e-05,
                6.77382482e-02,  4.48275862e-01,  3.73319245e-01,  1.49261702e-01,
                3.95428167e-04,  4.51199030e-05,  1.07395800e-05,  4.11186842e+05,
                3.53174337e+02,  3.38545494e+00,  1.28089466e-03,  7.49877768e-04,
                0.00000000e+00,  0.00000000e+00, -2.96580618e-01,  3.84324342e-01,
                1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  4.34426230e-01,
                1.14754098e-01,  4.90217793e-02,  1.96721311e-01,  5.10724271e-01,
                4.44113089e-01,  1.35994224e-01,  7.86885246e-01,  5.73770492e-01,
                9.60939498e-01,  5.45081967e-01,  6.47301674e+02,  9.32169058e+00,
                0.00000000e+00,  2.45901639e-02,  4.91803279e-02,  7.28162809e-01,
                4.01601158e-02,  2.07853226e-04,  8.47042259e-01, -2.54592885e-02,
                1.22668694e+00,  4.91803279e-02,  4.27626188e+01,  2.45901639e-02,
                3.68852459e-01,  1.63934426e-02,  4.09836066e-03,  6.56891558e+00,
                7.49204088e-01,  9.93055556e-01,  2.09603897e+00,  1.74758825e+00,
                2.91078279e+01,  8.50664735e-01,  1.16213917e+00,  5.58335974e-02,
                8.33099067e+01,  7.24673714e-01,  7.56547613e+01,  7.28162809e-01,
                1.00000000e+00, -9.26938483e-05,  1.65349487e-03,  0.00000000e+00,
                5.58335974e-02,  2.40972486e+04,  1.00000000e+00,  1.59674431e-01,
                2.78154635e-01,  4.20863819e-01,  5.58825565e-01,  8.15305445e-01,
               -7.43417014e-02,  2.20051317e-05,  6.77382482e-02,  4.48275862e-01,
                3.73319245e-01,  1.49261702e-01,  3.95428167e-04,  4.51199030e-05,
                1.07395800e-05,  4.11186842e+05,  3.53174337e+02,  3.38545494e+00,
                1.28089466e-03,  7.49877768e-04,  0.00000000e+00,  0.00000000e+00,
               -2.96580618e-01,  3.84324342e-01,  1.00000000e+00,  1.00000000e+00,
                1.00000000e+00,  4.34426230e-01,  1.14754098e-01,  4.90217793e-02,
                1.96721311e-01,  5.10724271e-01,  4.44113089e-01,  1.35994224e-01,
                7.86885246e-01,  5.73770492e-01,  9.60939498e-01,  5.45081967e-01,
                6.47301674e+02,  9.32169058e+00,  0.00000000e+00,  2.45901639e-02,
                4.91803279e-02,  7.28162809e-01,  4.01601158e-02,  2.07853226e-04,
                8.47042259e-01, -2.54592885e-02,  1.22668694e+00,  4.91803279e-02,
                4.27626188e+01,  2.45901639e-02,  3.68852459e-01,  1.63934426e-02,
                4.09836066e-03,  6.56891558e+00,  7.49204088e-01,  9.93055556e-01,
                2.09603897e+00,  1.74758825e+00,  2.91078279e+01,  8.50664735e-01,
                1.16213917e+00,  5.58335974e-02,  8.33099067e+01,  7.24673714e-01,
                7.56547613e+01,  7.28162809e-01,  1.00000000e+00, -9.26938483e-05,
                1.65349487e-03,  0.00000000e+00,  5.58335974e-02,  2.40972486e+04]

            self.assertTrue(np.allclose(value, expected_value), "Extract all features function with weights failed.")

            value =  extract_all(time, mag, magerr, apply_weights=False, convert=True, zp=zp)
            expected_value = [ 1.00000000e+00,  1.48218206e-01,  2.50295260e-01,  3.79904016e-01,
                5.51019604e-01,  7.87234169e-01,  7.49738489e-02,  2.06172908e-05,
                6.85959450e-01,  4.33333333e-01,  3.67728826e-01,  1.67037156e-01,
                5.73770492e-02,  2.04918033e-02,  8.19672131e-03,  1.30227140e+02,
                9.44928272e+00,  1.13452680e+01,  4.42779887e-01,  1.22950820e-01,
                0.00000000e+00,  0.00000000e+00, -3.05936903e-01,  3.89639101e-01,
                1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  5.61475410e-01,
                5.94262295e-01,  8.04158048e-01,  4.59016393e-01,  4.95901639e-01,
                4.91803279e-01,  1.13665825e-01,  5.57377049e-01,  5.90163934e-01,
                1.18896064e+00,  5.08196721e-01,  6.47301674e+02,  8.87919820e+00,
                0.00000000e+00,  4.09836066e-03,  4.09836066e-03,  7.28162809e-01,
                3.88859371e-02,  1.02115612e-04,  8.72801835e-01, -1.44923422e-05,
                2.18877121e-02,  2.09016393e-01,  1.02784058e+00,  2.45901639e-02,
                4.30327869e-01,  6.55737705e-02,  4.09836066e-03,  1.78422846e+00,
                7.49204088e-01,  3.88888889e-01,  7.31139083e-01,  1.74758825e+00,
                2.91078279e+01,  8.50664735e-01,  1.68736364e+00,  6.67075949e-02,
                8.33099067e+01,  7.24673714e-01,  7.56547613e+01,  7.28930057e-01,
                1.00000000e+00, -9.26938483e-05,  2.37875967e-03,  0.00000000e+00,
                6.69097527e-02,  1.11873181e+00,  1.00000000e+00,  1.48218206e-01,
                2.50295260e-01,  3.79904016e-01,  5.51019604e-01,  7.87234169e-01,
                7.49738489e-02,  2.06172908e-05,  6.85959450e-01,  4.33333333e-01,
                3.67728826e-01,  1.67037156e-01,  5.73770492e-02,  2.04918033e-02,
                8.19672131e-03,  1.30227140e+02,  9.44928272e+00,  1.13452680e+01,
                4.42779887e-01,  1.22950820e-01,  0.00000000e+00,  0.00000000e+00,
               -3.05936903e-01,  3.89639101e-01,  1.00000000e+00,  0.00000000e+00,
                0.00000000e+00,  5.61475410e-01,  5.94262295e-01,  8.04158048e-01,
                4.59016393e-01,  4.95901639e-01,  4.91803279e-01,  1.13665825e-01,
                5.57377049e-01,  5.90163934e-01,  1.18896064e+00,  5.08196721e-01,
                6.47301674e+02,  8.87919820e+00,  0.00000000e+00,  4.09836066e-03,
                4.09836066e-03,  7.28162809e-01,  3.88859371e-02,  1.02115612e-04,
                8.72801835e-01, -1.44923422e-05,  2.18877121e-02,  2.09016393e-01,
                1.02784058e+00,  2.45901639e-02,  4.30327869e-01,  6.55737705e-02,
                4.09836066e-03,  1.78422846e+00,  7.49204088e-01,  3.88888889e-01,
                7.31139083e-01,  1.74758825e+00,  2.91078279e+01,  8.50664735e-01,
                1.68736364e+00,  6.67075949e-02,  8.33099067e+01,  7.24673714e-01,
                7.56547613e+01,  7.28930057e-01,  1.00000000e+00, -9.26938483e-05,
                2.37875967e-03,  0.00000000e+00,  6.69097527e-02,  1.11873181e+00]

            self.assertTrue(np.allclose(value, expected_value), "Extract all features function without weights failed.")

unittest.main()
    
