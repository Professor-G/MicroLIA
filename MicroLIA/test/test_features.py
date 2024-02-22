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
        expected_value = 1.0
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="AndersonDarling function with weights failed.")

        value = AndersonDarling(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.0
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="AndersonDarling function without weights failed.")

    def test_FluxPercentileRatioMid20(self):
        value = FluxPercentileRatioMid20(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0689803536382854
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid20 function with weights failed.")

        value = FluxPercentileRatioMid20(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.039877354974917505
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid20 function without weights failed.")

    def test_FluxPercentileRatioMid35(self):
        value = FluxPercentileRatioMid35(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.12590660235444978
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid35 function with weights failed.")

        value = FluxPercentileRatioMid35(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.07380961622247513
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid35 function without weights failed.")

    def test_FluxPercentileRatioMid50(self):
        value = FluxPercentileRatioMid50(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.19939074448918812
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid50 function with weights failed.")

        value = FluxPercentileRatioMid50(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.10982452469338858
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid50 function without weights failed.")

    def test_FluxPercentileRatioMid65(self):
        value = FluxPercentileRatioMid65(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.3024042113911416
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid65 function with weights failed.")

        value = FluxPercentileRatioMid65(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.16027966815051362
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid65 function without weights failed.")

    def test_FluxPercentileRatioMid80(self):
        value = FluxPercentileRatioMid80(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.4646719917349019
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid80 function with weights failed.")

        value = FluxPercentileRatioMid80(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.2815796582492684
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="FluxPercentileRatioMid80 function without weights failed.")

    def test_Gskew(self):
        value = Gskew(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.18492597448807718
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="Gskew function with weights failed.")

        value = Gskew(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.39793972732504634
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="Gskew function without weights failed.")

    def test_LinearTrend(self):
        value = LinearTrend(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1.6772905256302443e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-5, msg="LinearTrend function with weights failed.")

        value = LinearTrend(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 3.369656352424246e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-5, msg="LinearTrend function without weights failed.")

    def test_MaxSlope(self):
        value = MaxSlope(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.05098060418494285
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="MaxSlope function with weights failed.")

        value = MaxSlope(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 4.210248377290517
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="MaxSlope function without weights failed.")

    def test_PairSlopeTrend(self):
        value = PairSlopeTrend(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.6206896551724138
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PairSlopeTrend function with weights failed.")

        value = PairSlopeTrend(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.6
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PairSlopeTrend function without weights failed.")

    def test_PercentAmplitude(self):
        value = PercentAmplitude(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 5.263572387053168
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PercentAmplitude function with weights failed.")

        value = PercentAmplitude(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 5.65579596367229
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PercentAmplitude function without weights failed.")

    def test_PercentDifferenceFluxPercentile(self):
        value = PercentDifferenceFluxPercentile(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.2148371372468774
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PercentDifferenceFluxPercentile function with weights failed.")

        value = PercentDifferenceFluxPercentile(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.5044786027746446
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="PercentDifferenceFluxPercentile function without weights failed.")

    def test_above1(self):
        value = above1(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 3.9017319586328805e-06
        self.assertAlmostEqual(value, expected_value, delta=1e-5, msg="above1 function with weights failed.")

        value = above1(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.04194260485651214
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="above1 function without weights failed.")

    def test_above3(self):
        value = above3(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 6.834060977437952e-07
        self.assertAlmostEqual(value, expected_value, delta=1e-6, msg="above3 function with weights failed.")

        value = above3(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.020603384841795438
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="above3 function without weights failed.")

    def test_above5(self):
        value = above5(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 2.6732568592901866e-07
        self.assertAlmostEqual(value, expected_value, delta=1e-6, msg="above5 function with weights failed.")

        value = above5(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.013245033112582781
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="above5 function without weights failed.")

    def test_abs_energy(self):
        value = abs_energy(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 4500178.683024583
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="abs_energy function with weights failed.")

        value = abs_energy(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 47.90735517819772
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="abs_energy function without weights failed.")

    def test_abs_sum_changes(self):
        value = abs_sum_changes(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1884.1579764541495
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="abs_sum_changes function with weights failed.")

        value = abs_sum_changes(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 8.743579715855867
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="abs_sum_changes function without weights failed.")

    def test_amplitude(self):
        value = amplitude(time, flux, flux_err, apply_weights=True)
        expected_value = 4.123383618006657
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="amplitude function with weights failed.")

        value = amplitude(time, flux, flux_err, apply_weights=False)
        expected_value = 59.33707174544156
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="amplitude function without weights failed.")

    def test_auto_corr(self):
        value = auto_corr(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = -9.126623747543625e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="auto_corr function with weights failed.")

        value = auto_corr(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.9913296911919757
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="auto_corr function without weights failed.")

    def test_below1(self):
        value = below1(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0
        self.assertAlmostEqual(value, expected_value, delta=1e-5, msg="below1 function with weights failed.")

        value = below1(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0
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
        expected_value = 0.8726724176244784
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="benford_correlation function with weights failed.")

        value = benford_correlation(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.8748887126496152
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="benford_correlation function without weights failed.")

    def test_c3(self):
        value = c3(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.003699737078009774
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="c3 function with weights failed.")

        value = c3(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.012789480843882928
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
        expected_value = 0.07873436350257546
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="check_max_last_loc function with weights failed.")

        value = check_max_last_loc(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.9220014716703459
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="check_max_last_loc function without weights failed.")

    def test_check_min_last_loc(self):
        value = check_min_last_loc(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.010301692420897735
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="check_min_last_loc function with weights failed.")

        value = check_min_last_loc(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.32818248712288445
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="check_min_last_loc function without weights failed.")

    def test_complexity(self):
        value = complexity(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.009261672949689444
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="complexity function with weights failed.")

        value = complexity(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.4215801958181811
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="complexity function without weights failed.")

    def test_con(self):
        value = con(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.141280353200883
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="con function with weights failed.")

        value = con(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.41869021339220014
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="con function without weights failed.")

    def test_count_above(self):
        value = count_above(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.4580896499905867
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="count_above function with weights failed.")

        value = count_above(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.48712288447387786
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="count_above function without weights failed.")

    def test_count_below(self):
        value = count_below(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.8975326284780578
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="count_below function with weights failed.")

        value = count_below(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.4988962472406181
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="count_below function without weights failed.")

    def test_cusum(self):
        value = cusum(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.2618491120944599
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="cusum function with weights failed.")

        value = cusum(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.19662757396139072
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="cusum function without weights failed.")

    def test_first_loc_max(self):
        value = first_loc_max(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.9234731420161884
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="first_loc_max function with weights failed.")

        value = first_loc_max(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.9212656364974245
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="first_loc_max function without weights failed.")

    def test_first_loc_min(self):
        value = first_loc_min(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.12067696835908756
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="first_loc_min function with weights failed.")

        value = first_loc_min(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.3274466519499632
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="first_loc_min function without weights failed.")

    def test_half_mag_amplitude_ratio(self):
        value = half_mag_amplitude_ratio(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 19.2753592883238
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="half_mag_amplitude_ratio function with weights failed.")

        value = half_mag_amplitude_ratio(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 24.570053038121614
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="half_mag_amplitude_ratio function without weights failed.")

    def test_index_mass_quantile(self):
        value = index_mass_quantile(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.4775570272259014
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="index_mass_quantile function with weights failed.")

        value = index_mass_quantile(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.5599705665930832
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="index_mass_quantile function without weights failed.")

    def test_integrate(self):
        value = integrate(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 423.46510892202537
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="integrate function with weights failed.")

        value = integrate(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 423.46510892202537
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="integrate function without weights failed.")

    def test_kurtosis(self):
        value = kurtosis(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 95.78857757385587
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="kurtosis function with weights failed.")

        value = kurtosis(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 52.800944150884426
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
        expected_value = 0.09565857247976453
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="longest_strike_above function with weights failed.")

        value = longest_strike_above(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0007358351729212656
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="longest_strike_above function without weights failed.")

    def test_longest_strike_below(self):
        value = longest_strike_below(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0051508462104488595
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="longest_strike_below function with weights failed.")

        value = longest_strike_below(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0007358351729212656
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="longest_strike_below function without weights failed.")

    def test_meanMag(self):
        value = meanMag(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.15965329977937226
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="meanMag function with weights failed.")

        value = meanMag(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.15965329977937226
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="meanMag function without weights failed.")

    def test_mean_abs_change(self):
        value = mean_abs_change(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.006864001707190451
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_abs_change function with weights failed.")

        value = mean_abs_change(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.006438571219334217
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_abs_change function without weights failed.")

    def test_mean_change(self):
        value = mean_change(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.00011098405846421302
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_change function with weights failed.")

        value = mean_change(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 8.472624945244629e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_change function without weights failed.")

    def test_mean_n_abs_max(self):
        value = mean_n_abs_max(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.9262653150460143
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_n_abs_max function with weights failed.")

        value = mean_n_abs_max(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.9325255760714363
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_n_abs_max function without weights failed.")

    def test_mean_second_derivative(self):
        value = mean_second_derivative(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = -0.0015507818257872866
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_second_derivative function with weights failed.")

        value = mean_second_derivative(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 7.518984642801636e-06
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="mean_second_derivative function without weights failed.")

    def test_medianAbsDev(self):
        value = medianAbsDev(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1.2357186664592137
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="medianAbsDev function with weights failed.")

        value = medianAbsDev(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0037831245072696418
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="medianAbsDev function without weights failed.")

    def test_median_buffer_range(self):
        value = median_buffer_range(time, flux, flux_err, apply_weights=True)
        expected_value = 0.0235467255334805
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="median_buffer_range function with weights failed.")

        value = median_buffer_range(time, flux, flux_err, apply_weights=False)
        expected_value = 0.8977189109639441
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="median_buffer_range function without weights failed.")

    def test_median_distance(self):
        value = median_distance(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 187.2482670069835
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="median_distance function with weights failed.")

        value = median_distance(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.8356881379847432
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="median_distance function without weights failed.")

    def test_number_cwt_peaks(self):
        value = number_cwt_peaks(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.021339220014716703
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_cwt_peaks function with weights failed.")

        value = number_cwt_peaks(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.021339220014716703
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_cwt_peaks function without weights failed.")

    def test_number_of_crossings(self):
        value = number_of_crossings(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.3686534216335541
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_of_crossings function with weights failed.")

        value = number_of_crossings(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.4157468727005151
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_of_crossings function without weights failed.")

    def test_number_of_peaks(self):
        value = number_of_peaks(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.04856512141280353
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_of_peaks function with weights failed.")

        value = number_of_peaks(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.06181015452538632
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="number_of_peaks function without weights failed.")

    def test_peak_detection(self):
        value = peak_detection(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.0007358351729212656
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="peak_detection function with weights failed.")

        value = peak_detection(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.0007358351729212656
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="peak_detection function without weights failed.")

    def test_permutation_entropy(self):
        value = permutation_entropy(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 8.289452388743738
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="permutation_entropy function with weights failed.")

        value = permutation_entropy(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.7884623901136658
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="permutation_entropy function without weights failed.")

    def test_quantile(self):
        value = quantile(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.15431210656577926
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="quantile function with weights failed.")

        value = quantile(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.15431210656577926
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="quantile function without weights failed.")

    def test_ratio_recurring_points(self):
        value = ratio_recurring_points(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.8996655518394648
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="ratio_recurring_points function with weights failed.")

        value = ratio_recurring_points(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.5652173913043478
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="ratio_recurring_points function without weights failed.")

    def test_root_mean_squared(self):
        value = root_mean_squared(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 16.5824285280537
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="root_mean_squared function with weights failed.")

        value = root_mean_squared(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.15024499029989147
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="root_mean_squared function without weights failed.")

    def test_sample_entropy(self):
        value = sample_entropy(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.028073777758772216
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="sample_entropy function with weights failed.")

        value = sample_entropy(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.028073777758772216
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="sample_entropy function without weights failed.")

    def test_shannon_entropy(self):
        value = shannon_entropy(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 25.912950024330218
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="shannon_entropy function with weights failed.")

        value = shannon_entropy(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 25.912950024330218
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="shannon_entropy function without weights failed.")

    def test_shapiro_wilk(self):
        value = shapiro_wilk(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.23471534252166748
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="shapiro_wilk function with weights failed.")

        value = shapiro_wilk(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.23471534252166748
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="shapiro_wilk function without weights failed.")

    def test_skewness(self):
        value = skewness(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 9.203071908822924
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="skewness function with weights failed.")

        value = skewness(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 6.951458105661253
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="skewness function without weights failed.")

    def test_std_over_mean(self):
        value = std_over_mean(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.4042703086567313
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="std_over_mean function with weights failed.")

        value = std_over_mean(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.5781254633689681
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="std_over_mean function without weights failed.")

    def test_stetsonJ(self):
        value = stetsonJ(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 6006.476061062362
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonJ function with weights failed.")

        value = stetsonJ(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 6006.476061062362
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonJ function without weights failed.")

    def test_stetsonK(self):
        value = stetsonK(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.25181305525892006
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonK function with weights failed.")

        value = stetsonK(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.25181305525892006
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonK function without weights failed.")

    def test_stetsonL(self):
        value = stetsonL(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1895.374797337941
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonL function with weights failed.")

        value = stetsonL(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1895.374797337941
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="stetsonL function without weights failed.")

    def test_sum_values(self):
        value = sum_values(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.15965329977937262
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="sum_values function with weights failed.")

        value = sum_values(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.16645473379324044
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
        expected_value = 1.2178974822726409e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-6, msg="time_reversal_asymmetry function with weights failed.")

        value = time_reversal_asymmetry(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 1.2178974822726385e-05
        self.assertAlmostEqual(value, expected_value, delta=1e-6, msg="time_reversal_asymmetry function without weights failed.")

    def test_variance(self):
        value = variance(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 0.004212069813891517
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="variance function with weights failed.")

        value = variance(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.007544738579571028
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
        expected_value = 0.4042703086567313
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="variation_coefficient function with weights failed.")

        value = variation_coefficient(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.5218262806745012
        self.assertAlmostEqual(value, expected_value, delta=1e-3, msg="variation_coefficient function without weights failed.")

    def test_vonNeumannRatio(self):
        value = vonNeumannRatio(time, norm_flux, norm_fluxerr, apply_weights=True)
        expected_value = 1069921.7126451028
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="vonNeumannRatio function with weights failed.")

        value = vonNeumannRatio(time, norm_flux, norm_fluxerr, apply_weights=False)
        expected_value = 0.017346682763858742
        self.assertAlmostEqual(value, expected_value, delta=1e-1, msg="vonNeumannRatio function without weights failed.")

    def test_extract_all(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            
            value = extract_all(time, mag, magerr, apply_weights=True, convert=True, zp=zp)
            expected_value = [ 1.00000000e+00,  6.89803536e-02,  1.25906602e-01,  1.99390744e-01,
                3.02404211e-01,  4.64671992e-01,  1.84925974e-01,  1.67729053e-05,
                5.09806042e-02,  6.20689655e-01,  5.26357239e+00,  2.14837137e-01,
                3.90173196e-06,  6.83406098e-07,  2.67325686e-07,  4.50017868e+06,
                1.88415798e+03,  4.12338362e+00, -9.12662375e-05,  0.00000000e+00,
                0.00000000e+00,  0.00000000e+00,  8.72672418e-01,  3.69973708e-03,
                1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  7.87343635e-02,
                1.03016924e-02,  9.26167295e-03,  1.41280353e-01,  4.58089650e-01,
                8.97532628e-01,  2.61849112e-01,  9.23473142e-01,  1.20676968e-01,
                1.92753593e+01,  4.77557027e-01,  4.23465109e+02,  9.57885776e+01,
                0.00000000e+00,  9.56585725e-02,  5.15084621e-03,  1.59653300e-01,
                6.86400171e-03,  1.10984058e-04,  9.26265315e-01, -1.55078183e-03,
                1.23571867e+00,  2.35467255e-02,  1.87248267e+02,  2.13392200e-02,
                3.68653422e-01,  4.85651214e-02,  7.35835173e-04,  8.28945239e+00,
                1.54312107e-01,  8.99665552e-01,  1.65824285e+01,  2.80737778e-02,
                2.59129500e+01,  2.34715343e-01,  9.20307191e+00,  4.04270309e-01,
                6.00647606e+03,  2.51813055e-01,  1.89537480e+03,  1.59653300e-01,
                1.00000000e+00,  1.21789748e-05,  4.21206981e-03,  0.00000000e+00,
                4.04270309e-01,  1.06992171e+06,  1.00000000e+00,  3.42331787e-03,
                5.99080628e-03,  8.55829468e-03,  1.11257831e-02,  9.96347949e-01,
               -3.78242172e-01,  4.14932438e-06,  4.60042307e-03,  3.79310345e-01,
                1.73134297e+02,  3.38254210e+02,  1.00000000e-07,  1.00000000e-07,
                1.00000000e-07,  1.00000000e+07,  1.87570241e+02,  2.22154331e+00,
               -1.10496615e-03,  1.00000000e-07,  1.00000000e-07,  1.00000000e-07,
               -1.89062437e-01,  1.00000000e-07,  1.00000000e+00,  1.00000000e+00,
                1.00000000e+00,  7.40192450e-04,  7.40192450e-04,  9.72403863e-04,
                7.54996299e-02,  9.22402447e-01,  1.65638112e-01,  2.79908689e-01,
                8.94152480e-01,  8.49000740e-01,  2.69587089e+00,  8.94892672e-01,
                3.29124102e-02,  3.71371324e+00,  0.00000000e+00,  3.70096225e-03,
                8.14211695e-03,  5.74269412e-03,  1.50506683e-01, -2.06892955e-05,
                2.43483849e-01, -2.16377876e-05,  1.07375726e-01,  6.66173205e-03,
                2.04959893e+01,  1.62842339e-02,  7.10584752e-02,  0.00000000e+00,
                7.40192450e-04,  3.05940389e+00,  9.45536262e-04,  1.00000000e+00,
                1.65162030e+03,  1.49889437e-01,  5.25362145e+02,  2.13630557e-01,
               -2.01423257e+00,  5.14633269e-01,  5.91186447e+02,  3.58267204e-02,
                2.65416937e+01,  5.74269412e-03,  1.00000000e+00,  3.19501783e-04,
                2.84642815e-05,  0.00000000e+00,  5.14633269e-01,  1.00000000e+07]

            self.assertTrue(np.allclose(value, expected_value), "Extract all features function with weights failed.")

            value =  extract_all(time, mag, magerr, apply_weights=False, convert=True, zp=zp)

            expected_value = [ 1.00000000e+00,  3.98773550e-02,  7.38096162e-02,  1.09824525e-01,
                1.60279668e-01,  2.81579658e-01,  3.97939727e-01,  3.36965635e-05,
                4.21024838e+00,  6.00000000e-01,  5.65579596e+00,  5.04478603e-01,
                4.19426049e-02,  2.06033848e-02,  1.32450331e-02,  4.79073552e+01,
                8.74357972e+00,  5.93370717e+01,  9.91329691e-01,  0.00000000e+00,
                0.00000000e+00,  0.00000000e+00,  8.74888713e-01,  1.27894808e-02,
                1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.22001472e-01,
                3.28182487e-01,  4.21580196e-01,  4.18690213e-01,  4.87122884e-01,
                4.98896247e-01,  1.96627574e-01,  9.21265636e-01,  3.27446652e-01,
                2.45700530e+01,  5.59970567e-01,  4.23465109e+02,  5.28009442e+01,
                0.00000000e+00,  7.35835173e-04,  7.35835173e-04,  1.59653300e-01,
                6.43857122e-03,  8.47262495e-05,  9.32525576e-01,  7.51898464e-06,
                3.78312451e-03,  8.97718911e-01,  8.35688138e-01,  2.13392200e-02,
                4.15746873e-01,  6.18101545e-02,  7.35835173e-04,  1.78846239e+00,
                1.54312107e-01,  5.65217391e-01,  1.50244990e-01,  2.80737778e-02,
                2.59129500e+01,  2.34715343e-01,  6.95145811e+00,  5.78125463e-01,
                6.00647606e+03,  2.51813055e-01,  1.89537480e+03,  1.66454734e-01,
                1.00000000e+00,  1.21789748e-05,  7.54473858e-03,  0.00000000e+00,
                5.21826281e-01,  1.73466828e-02,  1.00000000e+00,  2.06895131e-02,
                4.39777779e-02,  8.16663566e-02,  1.48305490e-01,  3.98122507e-01,
                1.31742751e-02, -1.09493559e-07,  4.24524270e+02,  3.66666667e-01,
                3.44915659e+04,  7.96972664e+02,  2.07253886e-02,  9.62250185e-03,
                2.96076980e-03,  2.00321552e+00,  1.45874471e+01,  4.71996736e+01,
               -7.47045929e-02,  1.70244264e-02,  4.44115470e-03,  2.96076980e-03,
                9.97208088e-01, -3.71814238e-06,  0.00000000e+00,  0.00000000e+00,
                0.00000000e+00,  1.16950407e-01,  1.19911177e-01,  2.07384058e+00,
                4.53737972e-01,  4.99629904e-01,  4.99629904e-01,  4.67871895e-02,
                1.16210215e-01,  1.19170984e-01,  1.60700554e+00,  3.36787565e-01,
                3.29124102e-02,  3.52029159e+02,  0.00000000e+00,  7.40192450e-04,
                7.40192450e-04,  5.74269412e-03,  1.08055163e-02,  5.07961101e-07,
                3.29476157e-01, -2.21044235e-08,  9.23769784e-04,  7.88304959e-01,
                8.37570191e-01,  1.62842339e-02,  4.73723168e-01,  6.14359734e-02,
                7.40192450e-04,  1.76728574e+00,  9.45536262e-04,  0.00000000e+00,
                2.89917544e-05,  1.49889437e-01,  5.25362145e+02,  2.13630557e-01,
                1.19490858e+01,  1.32743635e+03,  5.91186447e+02,  3.58267204e-02,
                2.65416937e+01,  1.30084743e-03,  1.00000000e+00,  3.19501783e-04,
                1.48107280e-03,  0.00000000e+00,  2.95843370e+01,  2.15100075e+00]

            self.assertTrue(np.allclose(value, expected_value), "Extract all features function without weights failed.")

unittest.main()
    
