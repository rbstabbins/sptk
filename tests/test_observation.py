"""Unit tests for the Observation2 Class of the
observation.py module.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 28-09-2022
"""
import os
import unittest
import numpy as np
import pandas as pd
from sptk.observation import Observation
from sptk.instrument import Instrument
from sptk.material_collection import MaterialCollection
from test_instrument import build_test_instrument, delete_test_instrument
from test_material_collection import generate_test_spectral_library, delete_test_spectral_library

test_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(test_dir)

def generate_test_objects(n_samples: int = 1):
    # build test data
    if n_samples > 1:
        test_data = generate_test_spectral_library(n_samples = n_samples)
    else:
        test_data = generate_test_spectral_library(
                                    flat_target=0.5, flat_background=0.5)
    materials = {
            'target': [('test_target', '*')],
            'background': [('test_background', '*')]}
    library = 'test_library'
    test_matcol = MaterialCollection(
                materials,
                library,
                'test',
                load_existing= False,
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
    return test_data, test_matcol, test_inst

class TestObservation(unittest.TestCase):
    """Class to test the Observation class
    """

    def test_init(self):
        """Testing the init function.
        """
        _, test_matcol, test_inst = generate_test_objects()
        test_obs = Observation(
                                test_matcol,
                                test_inst,
                                load_existing= False,
                                plot_profiles=False,
                                export_df=False)        
        
        # TODO tests for Observation
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_obs.__del__(rmproj=True)

    def test_build_new_observation(self):
        """Testing the build_new_observation function.
        """
        _, test_matcol, test_inst = generate_test_objects()
        test_wvls = test_inst.cwls().to_numpy()
        test_main_df = Observation.build_new_observation(test_matcol,
                                                    test_inst, test_wvls)
        # expected main df from this
        with self.subTest('wavelengths'):
            expected = list(test_wvls)
            result = test_main_df.columns[14:].to_list()
            self.assertListEqual(result, expected)
        with self.subTest('Data ID'):
            expected = test_matcol.main_df.index
            result = test_main_df.index
            pd.testing.assert_index_equal(result, expected)   

        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_matcol.__del__(rmproj=True)     

    def test_add_shot_noise(self):
        """Testing add_noise in shot mode.
        """
        test_data, test_matcol, test_inst = generate_test_objects()
        test_obs = Observation(
                            test_matcol,
                            test_inst,
                            load_existing= False,
                            plot_profiles=False,
                            export_df=False)
        noise = 0.1
        n_dups = 20
        noisy_df = test_obs.add_noise(
                                    noise,
                                    n_dups,
                                    noise_type='shot',
                                    apply=False)
        # check index label correct
        with self.subTest('index name'):
            expected = 'Data ID'
            result = noisy_df.index.name
            self.assertEqual(result, expected)
        # check for n duplicates
        with self.subTest('n_duplicates'):
            expected = len(test_data) * n_dups
            result = len(noisy_df)
            self.assertEqual(result, expected)
        # check for sigma of the data
        with self.subTest('sigma'):
            expected = noise
            test_cat = test_matcol.categories[0]
            cat_df = noisy_df[noisy_df['Category'] == test_cat]
            test_signal = test_obs.main_df[
                        test_obs.main_df['Category'] == test_cat][test_obs.wvls]
            result = (cat_df[test_obs.wvls].std() / test_signal).mean(axis=1)
            self.assertAlmostEqual(result.values[0], expected, delta=noise*0.1)
        # check for mean of the data
        with self.subTest('mean'):
            test_cat = test_matcol.categories[0]
            expected = test_obs.main_df[
                            test_obs.main_df['Category'] == test_cat][
                            test_obs.wvls].mean(axis=1)
            cat_df = noisy_df[noisy_df['Category'] == test_cat]
            result = cat_df[test_obs.wvls].mean().mean()
            self.assertAlmostEqual(result, expected.values[0], delta=noise*0.1)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_obs.__del__(rmproj=True)

    def test_get_refl_df(self):
        """Testing get_refl_df function for category and mineral selection.
        """
        # make test data set
        test_data, test_matcol, test_inst = generate_test_objects()
        test_obs = Observation(
                                test_matcol,
                                test_inst,
                                load_existing= False,
                                plot_profiles=False,
                                export_df=False)
        # test access of category
        with self.subTest('access category'):
            target_df = test_obs.get_refl_df(category = 'target')
            result = target_df.index.to_list()
            expected = [test_data[idx]['Data ID'] for idx in range(0,1)]
            self.assertListEqual(result, expected)
        # test access of mineral species
        with self.subTest('access mineral species'):
            target_df = test_obs.get_refl_df(mineral_name = 'test_target')
            result = target_df.index.to_list()
            expected = [test_data[idx]['Data ID'] for idx in range(0,1)]
            self.assertListEqual(result, expected)
        # test access category and mineral species
        with self.subTest('access category & mineral species'):
            target_df = test_obs.get_refl_df(category = 'target',
                            mineral_name = 'test_target')
            result = target_df.index.to_list()
            expected = [test_data[idx]['Data ID'] for idx in range(0,1)]
            self.assertListEqual(result, expected)
        # test access of all
        with self.subTest('access all'):
            all_df = test_obs.get_refl_df()
            result = all_df.index.to_list()
            expected = [entry['Data ID'] for entry in test_data]
            self.assertListEqual(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_obs.__del__(rmproj=True)

    """RMSE Tests
    """

    def test_resample_wavelengths(self):
        """Tesing the resample_wavelengths function.
        """
        # make test observation, with flat data
        _, test_matcol, test_inst = generate_test_objects()
        test_obs = Observation(
                                test_matcol,
                                test_inst,
                                load_existing= False,
                                plot_profiles=False,
                                export_df=False)
        result = test_obs.resample_wavelengths()
        # expect the data to match exactly with the material collection
        expected = test_matcol.get_refl_df().to_numpy()
        # - except for out of range values -what to do about these!
        # mask out the nan values
        result_ma = np.ma.masked_invalid(result)
        expected_ma = np.ma.array(expected, mask = result_ma.mask)

        np.testing.assert_allclose(result_ma, expected_ma)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_obs.__del__(rmproj=True)

    def test_compute_rmse(self):
        """Testing the compute_rmse function.
        """
        # make test observation with flat data
        _, test_matcol, test_inst = generate_test_objects()
        test_obs = Observation(
                                test_matcol,
                                test_inst,
                                load_existing= False,
                                plot_profiles=False,
                                export_df=False)
        # expect an rmse of 0 for all entries
        result = test_obs.compute_rmse().to_numpy()
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_obs.__del__(rmproj=True)

    """Statistics Tests
    """

    def test_channel_correlation(self):
        """Testing the channel_correlation function.
        """
        _, test_matcol, test_inst = generate_test_objects(n_samples=2)
        test_obs = Observation(
                                test_matcol,
                                test_inst,
                                load_existing= False,
                                plot_profiles=False,
                                export_df=False)
        corr_matrix = test_obs.channel_correlation()

        # perform weak tests only - trust the pandas correlation method works
        with self.subTest('matrix shape'):
            expected = (len(test_obs.wvls), len(test_obs.wvls))
            result = corr_matrix.shape
            self.assertTupleEqual(expected, result)
        with self.subTest('diagonal 1s'):
            expected = np.ones(len(test_obs.wvls))
            result = np.diagonal(corr_matrix.to_numpy())
            np.testing.assert_array_equal(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_obs.__del__(rmproj=True)

    def test_channel_pca(self):
        """Testing the channel_pca function.
        """
        test_data, test_matcol, test_inst = generate_test_objects(n_samples=35)
        test_obs = Observation(
                                test_matcol,
                                test_inst,
                                load_existing= False,
                                plot_profiles=False,
                                export_df=False)
        pca_matrix, _ = test_obs.channel_pca()

        # assume that the sklearn pca function performs correctly,
        # check the dimensions only
        with self.subTest('pca shape'):
            expected_ncols = min(len(test_data), len(test_obs.wvls))
            expected_nrows = len(test_data)
            expected = (expected_nrows, expected_ncols)
            result = pca_matrix.drop('Category',axis=1).shape
            self.assertTupleEqual(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_obs.__del__(rmproj=True)

    def test_channel_lda(self):
        """Testing the channel_lda function.
        """
        test_data, test_matcol, test_inst = generate_test_objects(n_samples=3)
        test_obs = Observation(
                                test_matcol,
                                test_inst,
                                load_existing= False,
                                plot_profiles=False,
                                export_df=False)
        lda_matrix = test_obs.channel_lda()

        # assume that the sklearn lda function performs correctly,
        # check the dimensions only
        with self.subTest('lda shape'):
            expected_ncols = len(test_matcol.categories) - 1
            expected_nrows = len(test_data)
            expected = (expected_nrows, expected_ncols)
            result = lda_matrix.drop('Category',axis=1).shape
            self.assertTupleEqual(result, expected)

        # cleanup
        delete_test_spectral_library()
        delete_test_instrument(test_inst)
        test_obs.__del__(rmproj=True)

if __name__ == '__main__':
    unittest.main(verbosity=2)