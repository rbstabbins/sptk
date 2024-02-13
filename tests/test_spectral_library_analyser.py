"""Unit tests for the SpectralLibraryAnalyser Class of the
spectral_library_analyser.py module.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 28-09-2022
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import os
from sptk.config import WVLS, PLT_FRMT
from sptk.spectral_library_analyser import SpectralLibraryAnalyser
from sptk.material_collection import MaterialCollection
from sptk.instrument import Instrument
from sptk.observation import Observation
from test_instrument import build_test_instrument
from test_material_collection import generate_test_spectral_library

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

class TestSpectralLibraryAnalyser(unittest.TestCase):
    """Class to test the SpectralLibraryAnalyser class
    """
    def test_init_material_collection(self):
        """Testing the init function for MaterialCollection.
        """
        _, test_matcol, _, _ = generate_test_objects()

        test_sla = SpectralLibraryAnalyser(test_matcol)

        with self.subTest('obj_type'):
            expected = 'material_collection'
            result = test_sla.obj_type
            self.assertEqual(result, expected)
        with self.subTest('wavelengths'):
            expected = WVLS
            result = test_sla.wvls
            np.testing.assert_array_equal(result, expected)
        with self.subTest('synthetic'):
            expected = False
            result = test_sla.synthetic
            self.assertEqual(result, expected)
        with self.subTest('spectra_obj'):
            expected = test_matcol
            result = test_sla.spectra_obj
            self.assertIs(result, expected)
        with self.subTest('band_info'):
            expected = pd.DataFrame()
            result = test_sla.band_info
            pd.testing.assert_frame_equal(result, expected)

    def test_plot_profiles_material_collection(self):
        """Testing the plot_profiles function for MaterialCollection.
        """
        _, test_matcol, _, _ = generate_test_objects(n_samples = 10)
        test_sla = SpectralLibraryAnalyser(test_matcol)

        test_sla.plot_profiles()
        # weakly test that a directory has been made to host the plots
        # if there is a problem, errors will be thrown in the test
        expected = Path(test_matcol.object_dir / 'plots')
        result = os.path.isdir(expected)
        self.assertTrue(result)

    def test_remove_continuum_material_collection(self):
        """Testing the remove_continuum function for MaterialCollection.
        """
        _, test_matcol, _, _ = generate_test_objects(n_samples = 1)
        test_sla = SpectralLibraryAnalyser(test_matcol)

        # performing cr on flat test profiles should produce and identical
        # dataframe to the original
        with self.subTest('cr obj'):
            expected = test_matcol.main_df
            expected[test_matcol.wvls] = 1.0
            result = test_sla.remove_continuum()
            pd.testing.assert_frame_equal(result, expected)
        # should also check that the original spectra_obj is unchanged
        with self.subTest('original obj'):
            expected = test_matcol.main_df
            test_sla.remove_continuum()
            result = test_sla.spectra_obj.main_df
            pd.testing.assert_frame_equal(result, expected)


    def test_visualise_spectrogram_material_collection(self):
        """Testing the visualise_spectrogram function for MaterialCollection.
        """
        _, test_matcol, _, _ = generate_test_objects(n_samples = 10)
        test_sla = SpectralLibraryAnalyser(test_matcol)

        test_sla.visualise_spectrogram()        
        out_file = Path(test_sla.project_dir, 'spectrogram').with_suffix(PLT_FRMT)
        self.assertTrue(os.path.isfile(out_file))

    def test_analyse_bands_material_collection(self):
        """Testing the analyse_bands function for MaterialCollection.
        TODO method is not complete.
        """
        _, test_matcol, _, _ = generate_test_objects(n_samples = 2)
        test_sla = SpectralLibraryAnalyser(test_matcol)

        result = test_sla.analyse_bands()
        print(result)
        pass

    def test_synthesize_spectra_from_band_info_material_collection(self):
        """Testing the synthesize_spectra_from_band_info function for
        MaterialCollection.
        """
        pass

    def test_build_gauss_feature_material_collection(self):
        """Testing the build_gauss_feature function for MaterialCollection.
        """
        pass

    """Observation tests
    """
    def test_init_observation(self):
        """Testing the init function for Observation.
        """
        pass

    def test_plot_profiles_observation(self):
        """Testing the plot_profiles function for Observation.
        """
        _, _, _, test_obs = generate_test_objects()

        with self.subTest('no noise'):
            test_sla = SpectralLibraryAnalyser(test_obs)
            test_sla.plot_profiles()
            # weakly test that a directory has been made to host the plots
            # if there is a problem, errors will be thrown in the test
            expected = Path(test_obs.object_dir / 'plots')
            result = os.path.isdir(expected)
            self.assertTrue(result)
        with self.subTest('with noise'):
            noise = 0.1
            n_dups = 10
            test_obs.add_noise(noise, n_dups)
            test_sla = SpectralLibraryAnalyser(test_obs)
            test_sla.plot_profiles(with_noise=True)
            # weakly test that a directory has been made to host the plots
            # if there is a problem, errors will be thrown in the test
            expected = Path(test_obs.object_dir / 'plots')
            result = os.path.isdir(expected)
            self.assertTrue(result)

    def test_remove_continuum_observation(self):
        """Testing the remove_continuum function for Observation.
        """
        _, _, _, test_obs = generate_test_objects(n_samples = 1)
        test_sla = SpectralLibraryAnalyser(test_obs)

        # performing cr on flat test profiles should produce and identical
        # dataframe to the original
        expected = test_obs.main_df
        expected[test_obs.wvls] = 1.0
        result = test_sla.remove_continuum()
        pd.testing.assert_frame_equal(result, expected)

    def test_visualise_spectrogram_observation(self):
        """Testing the visualise_spectrogram function for Observation.
        """
        _, _, _, test_obs = generate_test_objects(n_samples = 10)
        test_sla = SpectralLibraryAnalyser(test_obs)

        test_sla.visualise_spectrogram()
        out_file = Path(test_sla.project_dir, 'spectrogram').with_suffix(PLT_FRMT)
        self.assertTrue(os.path.isfile(out_file))

    def test_visualise_spectrogram_observation_with_noise(self):
        """Testing the visualise_spectrogram function for Observation with noise
        added.
        """
        _, _, _, test_obs = generate_test_objects(n_samples = 10)
        test_obs.add_noise(0.1, 10)
        test_sla = SpectralLibraryAnalyser(test_obs)

        test_sla.visualise_spectrogram()
        out_file = Path(test_sla.project_dir, 'spectrogram').with_suffix(PLT_FRMT)
        self.assertTrue(os.path.isfile(out_file))

    def test_analyse_bands_observation(self):
        """Testing the analyse_bands function for Observation.
        """
        pass

    def test_synthesize_spectra_from_band_info_observation(self):
        """Testing the synthesize_spectra_from_band_info function for
        Observation.
        """
        pass

    def test_build_gauss_feature_observation(self):
        """Testing the build_gauss_feature function for Observation.
        """
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)