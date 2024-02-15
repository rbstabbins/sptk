"""Unit tests for the MaterialCollection class of material_collection.py module.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 28-09-2022
"""
import os
import unittest
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
from sptk.config import OUTPUT_DIRECTORY, WVLS, SAMPLE_RES
from sptk.material_collection import HEADER_LIST, MaterialCollection
from sptk.instrument import Instrument
from test_instrument import build_test_instrument, delete_test_instrument
from test_data import generate_test_spectral_library, delete_test_spectral_library

test_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(test_dir)

class TestMaterialCollection(unittest.TestCase):
    """Class to test the material_collection module MaterialCollection class."""

    """Testing static methods"""

    def test_parse_materials(self):
        """Test the parse_materials function.
        """
        generate_test_spectral_library(n_samples = 2) # make test data set

        materials = {
                'target': [('test_target', '*')],
                'background': [('test_background','*')]
                }
        library = 'test_library'

        lib_dir = Path('spectral_library' , 'test_library')
        tg_dir = Path(lib_dir / 'test_target')
        bg_dir = Path(lib_dir / 'test_background')
        expected = {
            'target': [str(Path( tg_dir / 'test_target001.csv')),
                    str(Path( tg_dir / 'test_target002.csv'))],
            'background': [str(Path( bg_dir / 'test_background001.csv')),
                        str(Path( bg_dir / 'test_background002.csv'))]
        }

        result = MaterialCollection.parse_materials(materials, library)

        self.assertEqual(result, expected)

        # clean up test spectral library        
        delete_test_spectral_library()

    def test_init_frame(self):
        """Testing the init_frame function.
        """
        generate_test_spectral_library(n_samples = 2) # make test data set
        materials = {
                    'target': [('test_target', '*')],
                    'background': [('test_background','*')]
                    }
        library = 'test_library'
        categories = ['target', 'background']
        material_dict = MaterialCollection.parse_materials(materials, library)

        result = MaterialCollection.init_frame(categories, material_dict)

        with self.subTest():
            # check Filepath
            lib_dir = Path('spectral_library', 'test_library')
            tg_dir = Path(lib_dir / 'test_target')
            bg_dir = Path(lib_dir / 'test_background')
            expected = pd.Series([
                        str(Path( tg_dir / 'test_target001.csv')),
                        str(Path( tg_dir / 'test_target002.csv')),
                        str(Path( bg_dir / 'test_background001.csv')),
                        str(Path( bg_dir / 'test_background002.csv'))],
                        name='Filepath', dtype='str')
            pd.testing.assert_series_equal(result['Filepath'], expected)

        with self.subTest():
            # check Category
            expected = pd.Series(
                        ['target', 'target', 'background', 'background'],
                        name='Category', dtype='category')
            pd.testing.assert_series_equal(result['Category'], expected)

        for header in HEADER_LIST:
            with self.subTest():
                # check Headers are null
                expected = pd.Series(
                            ['', '', '', ''],
                            name=header, dtype='str')
                pd.testing.assert_series_equal(result[header], expected)

        with self.subTest():
            # check Wavelengths
            expected_columns = pd.Index(WVLS, dtype='object')
            wvl_start_idx_expected = 2 + len(HEADER_LIST) + 2
            result_columns = result.columns[wvl_start_idx_expected:]
            pd.testing.assert_index_equal(result_columns, expected_columns)
    
        # clean up test spectral library
        delete_test_spectral_library()

    def test_load_material(self):
        """Testing the load_material function
        """
        test_data = generate_test_spectral_library(n_samples = 2)
        materials = {
                    'target': [('test_target', '*')],
                    'background': [('test_background', '*')]}
        library = 'test_library'
        material_dict = MaterialCollection.parse_materials(materials, library)
        test_file = material_dict['target'][0]

        result = MaterialCollection.load_material(test_file)

        with self.subTest():
            today_str = date.today().strftime("%d/%m/%Y")
            expected_header = pd.DataFrame(data = {
                                    'Data ID': 'test_target001',
                                    'Sample ID': 'test_target001',
                                    'Mineral Name': 'test_target',
                                    'Sample Description': 'Synthetic Test',
                                    'Date Added': today_str,
                                    'Viewing Geometry': None,
                                    'Other Information': None,
                                    'Formula': None,
                                    'Composition': None,
                                    'Resolution': None,
                                    'Grain Size': None,
                                    'Locality': None,
                                    'Database of Origin': 'Test Library'
                                }, index = [0])
            result_header = result[['Data ID'] + HEADER_LIST]
            pd.testing.assert_frame_equal(result_header, expected_header)

        with self.subTest():
            expected_data = test_data[0][WVLS].to_frame().T
            result_data = result.loc[:,WVLS]
            pd.testing.assert_frame_equal(result_data, expected_data)
        
        # clean up test spectral library
        delete_test_spectral_library()

    def test_init_new_allow_out_of_bounds(self):
        """Testing the init function, for a new MaterialCollection
        """
        # build test spectral library
        test_data = generate_test_spectral_library(
                                            out_of_bounds=True, n_samples=1)

        materials = {
                    'target': [('test_target', '*')],
                    'background': [('test_background', '*')]}
        library = 'test_library'
        # make new MaterialCollection object
        result = MaterialCollection(
                    materials,
                    library,
                    'test',
                    load_existing= False,
                    balance_classes=False,
                    allow_out_of_bounds=True,
                    plot_profiles=False,
                    export_df=False)

        # use load_material to load in the material data in a way that can be
        # compared to the MaterialCollection dataframe
        with self.subTest(): # project_directory
            expected = OUTPUT_DIRECTORY / 'test'
            self.assertEqual(result.project_dir, expected)
        with self.subTest(): # project_name
            expected = 'test'
            self.assertEqual(result.project_name, expected)
        with self.subTest(): # material dictionary
            expected = MaterialCollection.parse_materials(materials, library)
            self.assertEqual(result.material_file_dict, expected)
        with self.subTest(): # categories
            expected = ['target', 'background']
            self.assertEqual(result.categories, expected)
        with self.subTest(): # wavelengths
            np.testing.assert_array_equal(result.wvls, WVLS)
        with self.subTest(): # allow_out_of_bounds
            self.assertEqual(result.allow_out_of_bounds, True)
        with self.subTest(): # header_list
            self.assertListEqual(result.header_list, HEADER_LIST)
        with self.subTest(): # data
            # get the data from the MaterialCollection, compare to test_data
            expected = test_data[0][7:].values
            valid_wvls = test_data[0].index[7:]
            result_data = result.main_df.iloc[0][15:]
            result_data_nonnan = result_data[valid_wvls].values
            np.testing.assert_array_equal(result_data_nonnan, expected)
            expected = np.array(list(set(WVLS) - set(valid_wvls.to_list())))
            result_nan_wvls = result_data[result_data.isnull()].index.to_numpy()
            np.testing.assert_array_equal(result_nan_wvls, expected)

        result.__del__(rmproj = True)
        
        # clean up test spectral library
        delete_test_spectral_library()

    def test_init_new_not_allow_out_of_bounds(self):
        """Testing the init function, for a new MaterialCollection
        """
        # build test spectral library
        generate_test_spectral_library(out_of_bounds=True, n_samples = 1)
        materials = {
                    'target': [('test_target', '*')],
                    'background': [('test_background', '*')]}
        library = 'test_library'
        # make new MaterialCollection object
        result = MaterialCollection(
                    materials,
                    library,
                    'test',
                    load_existing= False,
                    balance_classes=False,
                    allow_out_of_bounds=False,
                    plot_profiles=False,
                    export_df=False)

        # use load_material to load in the material data in a way that can
        # be compared to the MaterialCollection dataframe
        with self.subTest(): # project_directory
            expected = OUTPUT_DIRECTORY / 'test'
            self.assertEqual(result.project_dir, expected)
        with self.subTest(): # project_name
            expected = 'test'
            self.assertEqual(result.project_name, expected)
        with self.subTest(): # material dictionary
            expected = MaterialCollection.parse_materials(materials, library)
            self.assertEqual(result.material_file_dict, expected)
        with self.subTest(): # categories
            expected = ['target', 'background']
            self.assertEqual(result.categories, expected)
        with self.subTest(): # wavelengths
            np.testing.assert_array_equal(result.wvls, WVLS)
        with self.subTest(): # allow_out_of_bounds
            self.assertEqual(result.allow_out_of_bounds, False)
        with self.subTest(): # header_list
            self.assertListEqual(result.header_list, HEADER_LIST)
        with self.subTest(): # data
            # dataframe should be empty
            self.assertTrue(result.main_df.empty)

        result.__del__(rmproj = True)

        # clean up test spectral library
        delete_test_spectral_library()

    def test_init_new(self):
        """Testing the init function, for a new MaterialCollection
        """
        # build test spectral library
        test_data = generate_test_spectral_library(n_samples = 1)
        materials = {
                    'target': [('test_target', '*')],
                    'background': [('test_background', '*')]}
        library = 'test_library'
        # make new MaterialCollection object
        result = MaterialCollection(
                    materials,
                    library,
                    'test',
                    load_existing= False,
                    balance_classes=False,
                    allow_out_of_bounds=False,
                    plot_profiles=False,
                    export_df=False)

        # use load_material to load in the material data in a way that can be
        # compared to the MaterialCollection dataframe
        with self.subTest(): # project_directory
            expected = OUTPUT_DIRECTORY / 'test'
            self.assertEqual(result.project_dir, expected)
        with self.subTest(): # project_name
            expected = 'test'
            self.assertEqual(result.project_name, expected)
        with self.subTest(): # material dictionary
            expected = MaterialCollection.parse_materials(materials, library)
            self.assertEqual(result.material_file_dict, expected)
        with self.subTest(): # categories
            expected = ['target', 'background']
            self.assertEqual(result.categories, expected)
        with self.subTest(): # wavelengths
            np.testing.assert_array_equal(result.wvls, WVLS)
        with self.subTest(): # allow_out_of_bounds
            self.assertEqual(result.allow_out_of_bounds, False)
        with self.subTest(): # header_list
            self.assertListEqual(result.header_list, HEADER_LIST)
        with self.subTest(): # data
            # get the data from the MaterialCollection, compare to test_data
            expected_data = test_data[0][7:].values
            result_data = result.main_df.iloc[0][15:].values
            np.testing.assert_array_equal(result_data, expected_data)

        result.__del__(rmproj = True)

        # clean up test spectral library
        delete_test_spectral_library()

    def test_balance_class_sizes(self):
        """Testing the balance_class_sizes function.
        """
        # build an unbalanced test library
        generate_test_spectral_library(
            n_samples = {'test_target': 5, 'test_background': 10})
        materials = {
                    'target': [('test_target', '*')],
                    'background': [('test_background', '*')]}
        library = 'test_library'
        # make new MaterialCollection object
        result = MaterialCollection(
                    materials,
                    library,
                    'test',
                    load_existing= False,
                    balance_classes=False,
                    allow_out_of_bounds=True,
                    plot_profiles=False,
                    export_df=False)

        # check the classes are still imbalanced
        with self.subTest():
            tg_size = result.main_df.groupby('Category').size()['target']
            bg_size = result.main_df.groupby('Category').size()['background']
            self.assertEqual(tg_size, 5)
            self.assertEqual(bg_size, 10)

        result.balance_class_sizes()
        # check that the classes have been balanced in the final version
        with self.subTest():
            tg_size = result.main_df.groupby('Category').size()['target']
            bg_size = result.main_df.groupby('Category').size()['background']
            self.assertEqual(tg_size, bg_size)

        result.__del__(rmproj = True)

        # clean up test spectral library
        delete_test_spectral_library()

    def test_balance_class_sizes_repeatable(self):
        """Testing the balance_class_sizes random_state function.
        """
        # build an unbalanced test library
        generate_test_spectral_library(
            n_samples = {'test_target': 5, 'test_background': 10})
        materials = {
                    'target': [('test_target', '*')],
                    'background': [('test_background', '*')]}
        library = 'test_library'
        # make new MaterialCollection object
        result = MaterialCollection(
                    materials,
                    library,
                    'test',
                    load_existing= False,
                    balance_classes=False,
                    allow_out_of_bounds=True,
                    plot_profiles=False,
                    export_df=False)

        result.balance_class_sizes(random_state=1)
        # get the file list
        result_1 = result.main_df['Filepath']

        result.__del__(rmproj = True)
        # repeat and get the file list again
        result = MaterialCollection(
                    materials,
                    library,
                    'test',
                    load_existing= False,
                    balance_classes=False,
                    allow_out_of_bounds=True,
                    plot_profiles=False,
                    export_df=False)

        result.balance_class_sizes(random_state=1)
        # get the file list
        result_2 = result.main_df['Filepath']

        with self.subTest('Repeatable random sampling'):
            pd.testing.assert_series_equal(result_1, result_2)

        result.__del__(rmproj = True)
        # check that non-specified random sampling does not give same list
        result = MaterialCollection(
                    materials,
                    library,
                    'test',
                    load_existing= False,
                    balance_classes=False,
                    allow_out_of_bounds=True,
                    plot_profiles=False,
                    export_df=False)

        result.balance_class_sizes()
        # get the file list
        result_3 = result.main_df['Filepath']

        with self.subTest('Uncontrolled random sampling'):
            self.assertFalse(result_3.equals(result_1))

        result.__del__(rmproj = True)

        # clean up test spectral library
        delete_test_spectral_library()

    def test_channel_mask(self):
        """Testing channel_mask function.
        """
        generate_test_spectral_library(
                        out_of_bounds=True, n_samples = 1)
        materials = {
                    'target': [('test_target', '*')],
                    'background': [('test_background', '*')]}
        library = 'test_library'
        # make new MaterialCollection object
        test_matcol = MaterialCollection(
                    materials,
                    library,
                    'test',
                    load_existing= False,
                    balance_classes=False,
                    allow_out_of_bounds=True,
                    plot_profiles=False,
                    export_df=False)
        # get test instrument
        test_inst_name = build_test_instrument(use_config_spectral_range=True)
        test_inst = Instrument(test_inst_name, project_name='test')
        # get locations of channel masks
        test_refl = test_matcol.main_df.loc[:,SAMPLE_RES['wvl_min']:].to_numpy()
        result = list(test_matcol.channel_mask(test_refl, test_inst)[0])
        # expect all channels with wavelength greater less than cut_off in test
        cut_off = 0.7*SAMPLE_RES['wvl_max']
        expected = list((test_inst.cwls() < (cut_off - cut_off/20)).values)

        self.assertListEqual(result, expected)

        test_matcol.__del__(rmproj = True)

        # clean up test spectral library and instrument
        delete_test_spectral_library()
        delete_test_instrument(test_inst)

    def test_sample(self):
        """Testing the sample function.
        """
        generate_test_spectral_library(
                flat_target=0.5, flat_background=0.5, n_samples = 1)
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
        test_inst_name = build_test_instrument(use_config_spectral_range=True)
        test_inst = Instrument(test_inst_name, project_name='test')

        result = test_matcol.sample(test_inst)[0]
        expected = np.full(len(test_inst.cwls()), 0.5)
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

        test_matcol.__del__(rmproj = True)

        # clean up test spectral library and instrument
        delete_test_spectral_library()
        delete_test_instrument(test_inst)

    def test_get_refl_df(self):
        """Testing get_refl_df function for category and mineral selection.
        """
        test_data = generate_test_spectral_library(n_samples=5)
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

        with self.subTest('access category'):
            target_df = test_matcol.get_refl_df(category = 'target')
            result = target_df.index.to_list()
            expected = [test_data[idx]['Data ID'] for idx in range(0,5)]
            self.assertListEqual(result, expected)

        with self.subTest('access mineral species'):
            target_df = test_matcol.get_refl_df(mineral_name = 'test_target')
            result = target_df.index.to_list()
            expected = [test_data[idx]['Data ID'] for idx in range(0,5)]
            self.assertListEqual(result, expected)

        with self.subTest('access category & mineral species'):
            target_df = test_matcol.get_refl_df(category = 'target',
                            mineral_name = 'test_target')
            result = target_df.index.to_list()
            expected = [test_data[idx]['Data ID'] for idx in range(0,5)]
            self.assertListEqual(result, expected)

        with self.subTest('access all'):
            all_df = test_matcol.get_refl_df()
            result = all_df.index.to_list()
            expected = [entry['Data ID'] for entry in test_data]
            self.assertListEqual(result, expected)

        test_matcol.__del__(rmproj = True)

        # clean up test spectral library
        delete_test_spectral_library()

if __name__ == '__main__':        
    unittest.main(verbosity=2)