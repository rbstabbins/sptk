"""Unit tests for the Instrument and InstrumentBuilder classes of the
instrument.py module.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 28-09-2022
"""
import os
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sptk.instrument import InstrumentBuilder
from sptk.instrument import Instrument
from sptk.config import DATA_DIRECTORY, SAMPLE_RES, WVLS

def build_test_instrument(use_config_spectral_range: bool = False):
    instrument_name = 'test'
    instrument_type = 'filter-wheel'

    if not use_config_spectral_range:
        sampling = [450, 550, 650, 750, 850, 950, 1050]
        spectral_range = [400, 1100]
    else:
        spectral_range = [SAMPLE_RES['wvl_min'], SAMPLE_RES['wvl_max']]
        sampling = np.arange(
                    SAMPLE_RES['wvl_min']+50, SAMPLE_RES['wvl_max'], 100)
    resolution = 20
    test_instrument = InstrumentBuilder(
            instrument_name,
            instrument_type,
            sampling,
            resolution,
            spectral_range)
    test_df = test_instrument.generate_filter_band_table()
    test_instrument.export_instrument(test_df)
    return instrument_name

class TestInstrument(unittest.TestCase):
    """Class to test the Instrument class
    """
    def test_read_instrument_data(self):
        """Test the read_instrument_data function
        """
        test_name = build_test_instrument()
        # try to read the data from the test instrument
        result = Instrument.read_instrument_data(test_name)

        expected = pd.DataFrame(
            data={
            'filter_id': ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07'],
                'cwl': [450, 550, 650, 750, 850, 950, 1050],
                'fwhm': [22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5]
            },
        )
        expected = expected.set_index('filter_id')

        pd.testing.assert_frame_equal(result, expected)

        test_file = Path(DATA_DIRECTORY / 'instruments' /
                                        test_name).with_suffix('.csv')
        os.remove(test_file)

    def test_build_gauss_filter_single(self):
        """Testing the build_gauss_filter function for single profile.
        """
        test_cwl = np.mean(WVLS)
        test_fwhm = test_cwl / 50
        profile = Instrument.build_gauss_filter(test_cwl, test_fwhm)

        with self.subTest('profile shape'):
            expected = (1, len(WVLS))
            result = profile.shape
            self.assertEqual(result, expected)
        with self.subTest('centre-wavelength'):
            cwl = WVLS.dot((profile/profile.sum()).flatten())
            result = cwl
            expected = test_cwl
            self.assertAlmostEqual(result, expected, places=1)
        with self.subTest('fwhm'):
            mu2 = (WVLS.dot((profile/profile.sum()).flatten()))**2
            mom2 = np.power(WVLS, 2).dot((profile/profile.sum()).flatten())
            fwhm = np.sqrt(mom2 - mu2) * 2.355482004503
            result = fwhm
            expected = test_fwhm
            self.assertAlmostEqual(result, expected, places=1)

    def test_build_gauss_filter_batch(self):
        """Testing the build_gauss_filter function for batch of profiles
        """
        lower_cwl = np.mean([np.mean(WVLS),WVLS[0]])
        upper_cwl = np.mean([np.mean(WVLS),WVLS[1]])
        test_cwls = np.array([lower_cwl, upper_cwl])
        test_fwhms = test_cwls / 50
        profiles = Instrument.build_gauss_filter(test_cwls, test_fwhms)

        with self.subTest('shape'):
            result = profiles.shape
            expected = (2, len(WVLS))
            self.assertEqual(result, expected)

        profile_0_norm = (profiles[0]/profiles[0].sum()).flatten()
        profile_1_norm = (profiles[1]/profiles[1].sum()).flatten()

        with self.subTest('centre-wavelengths'):
            result = np.array(
                [WVLS.dot(profile_0_norm),
                 WVLS.dot(profile_1_norm)])
            expected = test_cwls
            np.testing.assert_array_almost_equal(result, expected, decimal=1)
        with self.subTest('fwhms'):
            mu2 = np.array(
                    [WVLS.dot(profile_0_norm),
                     WVLS.dot(profile_1_norm)])**2
            mom2 = np.array(
                    [np.power(WVLS, 2).dot(profile_0_norm),
                     np.power(WVLS, 2).dot(profile_1_norm)])
            fwhms = np.sqrt(mom2 - mu2) * 2.355482004503
            result = fwhms
            expected = test_fwhms
            np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_build_instrument_df(self):
        """Test the build_instrument_df function
        """
        test_name = build_test_instrument()
        test_data = Instrument.read_instrument_data(test_name)
        result = Instrument.build_instrument_df(test_data)

        with self.subTest('index'):
            pd.testing.assert_index_equal(result.index, test_data.index)
        with self.subTest('cwl'):
            pd.testing.assert_series_equal(result['cwl'], test_data['cwl'])
        with self.subTest('fwhm'):
            pd.testing.assert_series_equal(result['fwhm'], test_data['fwhm'])
        with self.subTest('wvls'):
            result_wvls = result.columns[2:].to_numpy()
            expected_wvls = WVLS
            np.testing.assert_array_equal(result_wvls, expected_wvls)
        with self.subTest('profile data'):
            expected_data = Instrument.build_gauss_filter(test_data['cwl'],
                                                            test_data['fwhm'])
            result_data = result[WVLS].to_numpy()
            np.testing.assert_array_equal(result_data, expected_data)

    def test_init_new(self):
        """Testing init function for a new instrument
        """
        test_name = build_test_instrument()
        test_data = Instrument.read_instrument_data(test_name)
        expected_main_df = Instrument.build_instrument_df(test_data)

        result = Instrument(test_name, project_name='test', export_df=False)

        pd.testing.assert_frame_equal(result.main_df, expected_main_df)
        result.__del__(rmdir=True)

    def test_init_existing(self):
        """Testing the init function for an existing instrument
        """
        test_name = build_test_instrument()
        initial_instrument = Instrument(
                                test_name,
                                project_name='test',
                                export_df=True)
        reload_instrument = Instrument(test_name, project_name='test')

        expected_main_df = initial_instrument.main_df
        result_main_df = reload_instrument.main_df

        pd.testing.assert_frame_equal(result_main_df, expected_main_df)

        initial_instrument.__del__(rmdir=True)
        reload_instrument.__del__(rmdir=True)


class TestInstrumentBuilder(unittest.TestCase):
    """Class to test the InstrumentBuilder class
    """

    def test_aotf_cwl_2_fwhm(self):
        """Testing the CWl to FWHM converter for AOTF type spectrometers
        """
        # expected results and test model extracted from Korablev et al 2017
        # (DOI:10.1089/ast.2016.1543)

        test_res = {'m': 0.0407, 'c': -6.246} # ISEM expected
        # test_res = {'m': 0.0539, 'c': -8.61} # LIS measured
        test_cwl_1 = 3300 # nm
        expected_fwhm_1 = 28 # nm
        result_fwhm_1 = InstrumentBuilder.aotf_cwl_2_fwhm(test_cwl_1, test_res)

        test_cwl_2 = 2500 # nm
        expected_fwhm_2 = 16 # nm
        result_fwhm_2 = InstrumentBuilder.aotf_cwl_2_fwhm(test_cwl_2, test_res)

        test_cwl_3 = 1150 # nm
        expected_fwhm_3 = 3.3 # nm
        result_fwhm_3 = InstrumentBuilder.aotf_cwl_2_fwhm(test_cwl_3, test_res)

        with self.subTest('3300 nm'):
            self.assertAlmostEqual(expected_fwhm_1, result_fwhm_1, places=0)
        with self.subTest('2500 nm'):
            self.assertAlmostEqual(expected_fwhm_2, result_fwhm_2, places=1)
        with self.subTest('1150 nm'):
            self.assertAlmostEqual(expected_fwhm_3, result_fwhm_3, places=1)

    def test_generate_filter_band_table(self):
        """Testing Filter-Wheel type spectrometer builder
        """
        instrument_name = 'test'
        instrument_type = 'filter-wheel'
        sampling = [450, 550, 650, 750, 850, 950, 1050]
        resolution = 20
        spectral_range = [0.4, 1.1]

        expected = pd.DataFrame(
            data={
                'filter_id': ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07'],
                'cwl': [450, 550, 650, 750, 850, 950, 1050],
                'fwhm': [22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5]
            }
        )

        test_instrument = InstrumentBuilder(
                instrument_name,
                instrument_type,
                sampling,
                resolution,
                spectral_range)
        result = test_instrument.generate_filter_band_table()

        pd.testing.assert_frame_equal(result, expected)

    def test_lvf_band_table(self):
        """Testing Linear Variable Filter type spectrometer builder
        """
        instrument_name = 'test'
        instrument_type = 'lvf'
        sampling = 'critical'
        resolution = 10
        spectral_range = [1000, 3000]

        exp_cwls = [1100.0, 1210.0, 1331.0, 1464.0, 1611.0,
                        1772.0, 1949.0, 2144.0, 2358.0, 2594.0]
        exp_fwhms = [110.0, 121.0, 133.1, 146.4, 161.1,
                        177.2, 194.9, 214.4, 235.8, 259.4]
        exp_filter_ids = ['S001', 'S002', 'S003', 'S004', 'S005',
                            'S006', 'S007', 'S008', 'S009', 'S010']

        expected = pd.DataFrame(data={
                        'filter_id':exp_filter_ids,
                        'cwl': exp_cwls,
                        'fwhm': exp_fwhms})

        test_instrument = InstrumentBuilder(
                instrument_name,
                instrument_type,
                sampling,
                resolution,
                spectral_range)
        result = test_instrument.generate_lvf_band_table()

        pd.testing.assert_frame_equal(result,expected, check_exact=False,atol=1)

    def test_aotf_band_table(self):
        """Testing Acousto-Tunable Filter type spectrometer builder
        """
        instrument_name = 'test'
        instrument_type = 'aotf'
        sampling = 'critical'
        resolution = {'m': 0.003, 'c': -5}
        spectral_range = [1000, 3000]

        exp_cwls = [1040.0, 1084.0, 1131.0, 1184.0, 1242.0, 1307.0, 1380.0,
                        1462.0, 1557.0, 1666.0, 1794.0, 1946.0, 2133.0, 2369.0]
        exp_fwhms = [43.6, 47.8, 52.6, 58.2, 64.9, 72.8, 82.4,
                        94.2, 109.0, 128.0, 153.0, 187.0, 235.0, 309.0]
        exp_filter_ids = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006',
                'S007', 'S008', 'S009', 'S010', 'S011', 'S012', 'S013', 'S014']

        expected = pd.DataFrame(data={
                        'filter_id':exp_filter_ids,
                        'cwl':exp_cwls,
                        'fwhm':exp_fwhms})

        test_instrument = InstrumentBuilder(
                instrument_name,
                instrument_type,
                sampling,
                resolution,
                spectral_range)
        result = test_instrument.generate_aotf_band_table()

        pd.testing.assert_frame_equal(result,expected, check_exact=False,atol=1)

    def test_export_instrument(self):
        """Testing export_instrument function
        """
        name = build_test_instrument()
        expected_file = Path(DATA_DIRECTORY / 'instruments' /
                                                    name).with_suffix('.csv')

        self.assertEqual(os.path.exists(expected_file), True)

        os.remove(expected_file)

if __name__ == '__main__':
    unittest.main(verbosity=2)