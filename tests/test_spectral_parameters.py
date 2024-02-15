"""Unit tests for the SpectralParameters Class of the
spectral_parameters.py module.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 28-09-2022
"""
import os
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sptk.material_collection import MaterialCollection
from sptk.instrument import Instrument
from sptk.observation import Observation
from sptk.spectral_parameters import SpectralParameters
from test_instrument import build_test_instrument, delete_test_instrument
from test_material_collection import generate_test_spectral_library, delete_test_spectral_library

test_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(test_dir)

def generate_test_objects(
        n_samples: int = 1,
        flat: bool = False,
        just_channels: bool=False):
    # build test data
    if flat:
        test_data = generate_test_spectral_library(
                                    flat_target=0.75,
                                    flat_background=0.25)
    else:
        test_data = generate_test_spectral_library(n_samples = n_samples)
    materials = {
            'target': [('test_target', '*')],
            'background': [('test_background', '*')]}
    library = 'test_library'
    test_matcol = MaterialCollection(
                materials,
                library,
                'test',
                load_existing=False,
                balance_classes=False,
                allow_out_of_bounds=False,
                plot_profiles=False,
                export_df=False)
    # build test instrument
    test_inst_name = build_test_instrument(use_config_spectral_range=True)
    test_inst = Instrument(
                    test_inst_name,
                    'test',
                    load_existing=False,
                    plot_profiles=False,
                    export_df=False)
    # make the test observation
    test_obs = Observation(
                test_matcol,
                test_inst,
                load_existing= False,
                plot_profiles=False,
                export_df=False)

    return test_data, test_matcol, test_inst, test_obs

class TestSpectralParameters(unittest.TestCase):
    """Class to test the SpectralParameters class
    """

    def test_init_new(self):
        """Testing the init function.
        """
        _, test_matcol, test_inst, test_obs = generate_test_objects()
        test_sp = SpectralParameters(test_obs, load_existing=False)

        with self.subTest('observation'):
            self.assertIs(test_sp.observation, test_obs)
        with self.subTest('material_collection'):
            self.assertIs(test_sp.material_collection, test_matcol)
        with self.subTest('instrument'):
            self.assertIs(test_sp.instrument, test_inst)
        with self.subTest('reflectance data'):
            result = test_sp.main_df[test_obs.chnl_lbls].to_numpy()
            expected = test_obs.main_df[test_obs.wvls].to_numpy()
            np.testing.assert_array_equal(result, expected)
        with self.subTest('sp_list'):
            result = test_sp.sp_list
            expected = test_obs.chnl_lbls
            self.assertListEqual(result, expected)
        with self.subTest('chnl_lbls'):
            result = test_sp.chnl_lbls
            expected = test_obs.chnl_lbls
            self.assertListEqual(result, expected)
        with self.subTest('sp_filters'):
            result = test_sp.sp_filters.to_list()
            expected = test_inst.filter_ids
            self.assertListEqual(result, expected)
        with self.subTest('ratio labels'):
            self.assertIsNone(test_sp.ratio_lbls)
        with self.subTest('slope labels'):
            self.assertIsNone(test_sp.slope_lbls)
        with self.subTest('band depth labels'):
            self.assertIsNone(test_sp.band_depth_lbls)
        with self.subTest('shoulder height labels'):
            self.assertIsNone(test_sp.shoulder_height_lbls)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_sp.__del__(rmproj=True)

    def test_init_existing(self):
        """Testing the init function.
        """
        _, test_matcol, test_inst, test_obs = generate_test_objects()
        test_sp_init = SpectralParameters(test_obs, load_existing=False)
        test_sp_init.export_main_df()
        test_sp = SpectralParameters(test_obs, load_existing=True)

        with self.subTest('observation'):
            self.assertIs(test_sp.observation, test_obs)
        with self.subTest('material_collection'):
            self.assertIs(test_sp.material_collection, test_matcol)
        with self.subTest('instrument'):
            self.assertIs(test_sp.instrument, test_inst)
        with self.subTest('reflectance data'):
            result = test_sp.main_df[test_obs.chnl_lbls].to_numpy()
            expected = test_obs.main_df[test_obs.wvls].to_numpy()
            np.testing.assert_array_equal(result, expected)
        with self.subTest('sp_list'):
            result = test_sp.sp_list
            expected = test_obs.chnl_lbls
            self.assertListEqual(result, expected)
        with self.subTest('chnl_lbls'):
            result = test_sp.chnl_lbls
            expected = test_obs.chnl_lbls
            self.assertListEqual(result, expected)
        with self.subTest('sp_filters'):
            result = test_sp.sp_filters.to_list()
            expected = test_inst.filter_ids
            self.assertListEqual(result, expected)
        with self.subTest('ratio labels'):
            self.assertIsNone(test_sp.ratio_lbls)
        with self.subTest('slope labels'):
            self.assertIsNone(test_sp.slope_lbls)
        with self.subTest('band depth labels'):
            self.assertIsNone(test_sp.band_depth_lbls)
        with self.subTest('shoulder height labels'):
            self.assertIsNone(test_sp.shoulder_height_lbls)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_sp.__del__(rmproj=True)

    def test_parse_sp_lbls(self):
        """Testing the parse_sp_lbls function.
        """
        _, _, test_inst, test_obs = generate_test_objects()
        test_sp = SpectralParameters(test_obs, load_existing=False)
        test_list = pd.Series(['R_400_500', 'BD_600_700',
                                        'S_500_600', 'SH_400_500_600'])
        with self.subTest('ratio'):
            result = test_sp.parse_sp_lbls(test_list, 'ratio')[0]
            expected = 'R_400_500'
            self.assertEqual(result, expected)
        with self.subTest('slope'):
            result = test_sp.parse_sp_lbls(test_list, 'slope')[0]
            expected = 'S_500_600'
            self.assertEqual(result, expected)
        with self.subTest('band depth'):
            result = test_sp.parse_sp_lbls(test_list, 'band_depth')[0]
            expected = 'BD_600_700'
            self.assertEqual(result, expected)
        with self.subTest('shoulder height'):
            result = test_sp.parse_sp_lbls(test_list, 'shoulder_height')[0]
            expected = 'SH_400_500_600'
            self.assertEqual(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_sp.__del__(rmproj=True)

    def test_get_channel_data(self):
        """Testing get_channel_data function.
        """
        _, _, test_inst, test_obs = generate_test_objects()
        test_sp = SpectralParameters(test_obs, load_existing=False)
        test_list = test_obs.chnl_lbls[0:2]
        result = test_sp.get_channel_data(test_list)
        print(result)

        # TODO write test

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_sp.__del__(rmproj=True)

    def test_compute_ratio_permutations(self):
        """Testing the compute_ratio_permutations function.
        """
        _, _, test_inst, test_obs = generate_test_objects(flat=True)
        test_sp = SpectralParameters(test_obs, load_existing=False)
        test_list = test_obs.chnl_lbls[0:3]
        test_cwls = test_obs.wvls[0:3]

        test_sp.compute_ratio_permutations(test_list)
        test_ratio_lbl = [
                f'R_{test_cwls[0]}_{test_cwls[1]}',
                f'R_{test_cwls[0]}_{test_cwls[2]}',
                f'R_{test_cwls[1]}_{test_cwls[0]}',
                f'R_{test_cwls[1]}_{test_cwls[2]}',
                f'R_{test_cwls[2]}_{test_cwls[0]}',
                f'R_{test_cwls[2]}_{test_cwls[1]}',]
        with self.subTest('ratio labels'):
            result = test_sp.ratio_lbls
            expected = test_ratio_lbl
            self.assertListEqual(result, expected)
        with self.subTest('dataframe'):
            result = test_sp.main_df[test_ratio_lbl]
            expected = pd.DataFrame(
                data = {test_ratio_lbl[0]: [1.0, 1.0],
                        test_ratio_lbl[1]: [1.0, 1.0],
                        test_ratio_lbl[2]: [1.0, 1.0],
                        test_ratio_lbl[3]: [1.0, 1.0],
                        test_ratio_lbl[4]: [1.0, 1.0],
                        test_ratio_lbl[5]: [1.0, 1.0]},
                index = test_obs.main_df.index)
            pd.testing.assert_frame_equal(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_sp.__del__(rmproj=True)

    def test_compute_slope_permutations(self):
        """Testing the compute_slope_permutations function.
        """
        _, _, test_inst, test_obs = generate_test_objects(flat=True)
        test_sp = SpectralParameters(test_obs, load_existing=False)
        test_list = test_obs.chnl_lbls[0:3]
        test_cwls = test_obs.wvls[0:3]

        test_sp.compute_slope_permutations(test_list)
        test_slope_lbl = [
                f'S_{test_cwls[0]}_{test_cwls[1]}',
                f'S_{test_cwls[0]}_{test_cwls[2]}',
                f'S_{test_cwls[1]}_{test_cwls[2]}']
        with self.subTest('slope labels'):
            result = test_sp.slope_lbls
            expected = test_slope_lbl
            self.assertListEqual(result, expected)
        with self.subTest('dataframe'):
            result = test_sp.main_df[test_slope_lbl]
            expected = pd.DataFrame(
                data = {test_slope_lbl[0]: [0.0, 0.0],
                        test_slope_lbl[1]: [0.0, 0.0],
                        test_slope_lbl[2]: [0.0, 0.0]},
                index = test_obs.main_df.index)
            pd.testing.assert_frame_equal(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_sp.__del__(rmproj=True)

    def test_compute_band_depth_permutations(self):
        """Testing the compute_band_depth_permutations function.
        """
        _, _, test_inst, test_obs = generate_test_objects(flat=True)
        test_sp = SpectralParameters(test_obs, load_existing=False)
        test_list = test_obs.chnl_lbls[0:4]
        test_cwls = test_obs.wvls[0:4]

        test_sp.compute_band_depth_permutations(test_list)
        test_band_depth_lbl = [
                f'BD_{test_cwls[0]}_{test_cwls[1]}_{test_cwls[2]}',
                f'BD_{test_cwls[0]}_{test_cwls[1]}_{test_cwls[3]}',
                f'BD_{test_cwls[0]}_{test_cwls[2]}_{test_cwls[3]}',
                f'BD_{test_cwls[1]}_{test_cwls[2]}_{test_cwls[3]}',
                ]
        with self.subTest('band_depth labels'):
            result = test_sp.band_depth_lbls
            expected = test_band_depth_lbl
            self.assertListEqual(result, expected)
        with self.subTest('dataframe'):
            result = test_sp.main_df[test_band_depth_lbl]
            expected = pd.DataFrame(
                data = {test_band_depth_lbl[0]: [0.0, 0.0],
                        test_band_depth_lbl[1]: [0.0, 0.0],
                        test_band_depth_lbl[2]: [0.0, 0.0],
                        test_band_depth_lbl[3]: [0.0, 0.0]},
                index = test_obs.main_df.index)
            pd.testing.assert_frame_equal(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_sp.__del__(rmproj=True)

    def test_compute_shoulder_height_permutations(self):
        """Testing the compute_shoulder_height_permutations function.
        """
        _, _, test_inst, test_obs = generate_test_objects(flat=True)
        test_sp = SpectralParameters(test_obs, load_existing=False)
        test_list = test_obs.chnl_lbls[0:4]
        test_cwls = test_obs.wvls[0:4]

        test_sp.compute_shoulder_height_permutations(test_list)
        test_shoulder_height_lbl = [
                f'SH_{test_cwls[0]}_{test_cwls[1]}_{test_cwls[2]}',
                f'SH_{test_cwls[0]}_{test_cwls[1]}_{test_cwls[3]}',
                f'SH_{test_cwls[0]}_{test_cwls[2]}_{test_cwls[3]}',
                f'SH_{test_cwls[1]}_{test_cwls[2]}_{test_cwls[3]}',
                ]
        with self.subTest('shoulder_height labels'):
            result = test_sp.shoulder_height_lbls
            expected = test_shoulder_height_lbl
            self.assertListEqual(result, expected)
        with self.subTest('dataframe'):
            result = test_sp.main_df[test_shoulder_height_lbl]
            expected = pd.DataFrame(
                data = {test_shoulder_height_lbl[0]: [0.0, 0.0],
                        test_shoulder_height_lbl[1]: [0.0, 0.0],
                        test_shoulder_height_lbl[2]: [0.0, 0.0],
                        test_shoulder_height_lbl[3]: [0.0, 0.0]},
                index = test_obs.main_df.index)
            pd.testing.assert_frame_equal(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_sp.__del__(rmproj=True)

    def test_compute_spectral_parameters(self):
        """Testing the compute_spectral_parameters function.
        """
        _, _, test_inst, test_obs = generate_test_objects(n_samples=10)
        test_sp = SpectralParameters(test_obs, load_existing=False)
        test_sp.compute_spectral_parameters()
        with self.subTest('ratios'):
            result = test_sp.ratio_lbls
            self.assertIsNotNone(result)
        with self.subTest('slope'):
            result = test_sp.slope_lbls
            self.assertIsNotNone(result)
        with self.subTest('band_depth'):
            result = test_sp.band_depth_lbls
            self.assertIsNotNone(result)
        with self.subTest('shoulder_height'):
            result = test_sp.shoulder_height_lbls
            self.assertIsNotNone(result)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_sp.__del__(rmproj=True)

    def test_test_train_random_split(self):
        """Testing the test_train_random_split function.
        """
        _, _, test_inst, test_obs = generate_test_objects(n_samples=10)
        root_sps = SpectralParameters(test_obs, load_existing=False)
        train_sps, test_sps = root_sps.train_test_random_split(0.2, 0)
        # check that the sizes are as expected, and check that the metadata
        # is the same
        with self.subTest('train size'):
            expected = int(0.8 * len(root_sps.main_df))
            result = len(train_sps.main_df)
            self.assertEqual(result, expected)
        with self.subTest('test size'):
            expected = int(0.2 * len(root_sps.main_df))
            result = len(test_sps.main_df)
            self.assertEqual(result, expected)
        with self.subTest('indices'):
            expected = root_sps.main_df.columns
            result = train_sps.main_df.columns
            pd.testing.assert_index_equal(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        root_sps.__del__(rmproj=True)

if __name__ == '__main__':
    unittest.main(verbosity=2)