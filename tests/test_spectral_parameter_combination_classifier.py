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
from sptk.config import PLT_FRMT
from sptk.material_collection import MaterialCollection
from sptk.instrument import Instrument
from sptk.observation import Observation
from sptk.spectral_parameters import SpectralParameters
from sptk.spectral_parameter_combination_classifier import  \
                        SpectralParameterCombinationClassifier as SPCClassifier
from test_instrument import build_test_instrument
from test_material_collection import generate_test_spectral_library

def generate_test_objects(
        n_samples: int = 1,
        flat: bool = False,
        just_channels: bool=False,
        noise: bool=False):
    # build test data
    if flat:
        test_data = generate_test_spectral_library(
                                    flat_target=0.75,
                                    flat_background=0.25, n_samples=n_samples)
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
    # add noise if flat profiles
    if noise:
        test_obs.add_noise(0.01, n_samples, 'thermal')
    # make the spectral parameters
    test_sps = SpectralParameters(
                test_obs,
                load_existing=False)
    if not just_channels:
        test_sps.compute_spectral_parameters()
    return test_data, test_matcol, test_inst, test_obs, test_sps

class TestSpectralParameterCombinationClassifier(unittest.TestCase):
    """Class for testing the SpectralParameterCombinationClassifier class
    of the spectral_parameters module.
    """
    def test_spc_id_df(self):
        """Testing the get_spc_id_df static method.
        """
        k_combinations = 2
        test_sp_list = ['R_400_500', 'BD_600_700', 'S_500_600']
        result = SPCClassifier.spc_sps_df(test_sp_list, k_combinations)
        exp_spc_list = {
                'sp_0': [
                    'R_400_500',
                    'R_400_500',
                    'R_400_500',
                    'BD_600_700',
                    'BD_600_700',
                    'S_500_600'],
                'sp_1': [
                    'R_400_500',
                    'BD_600_700',
                    'S_500_600',
                    'BD_600_700',
                    'S_500_600',
                    'S_500_600'
                ]}
        exp_spc_ids = [0, 1, 2, 3, 4, 5]
        expected = pd.DataFrame(data = exp_spc_list, index = exp_spc_ids)
        expected.index.rename('spc_id', inplace=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_spc_fltr_id_df(self):
        """Testing the spc_fltr_id_df static method.

        Note - fails because of problem comparing series of frozensets
        using the pandas testing method.
        """
        k_combinations = 2
        test_sp_list = ['R_400_500', 'BD_600_700', 'S_500_600']
        spc_sps_df = SPCClassifier.spc_sps_df(test_sp_list, k_combinations)
        test_filter_list = ['F01, F02', 'F03, F04', 'F02, F03']
        test_sp_filters = pd.Series(data=test_filter_list, index=test_sp_list)
        result = SPCClassifier.spc_fltr_id_df(spc_sps_df, test_sp_filters)
        exp_spc_filter_list = {
                'fltrs_0': [
                    'F01, F02',
                    'F01, F02',
                    'F01, F02',
                    'F03, F04',
                    'F03, F04',
                    'F02, F03'],
                'fltrs_1': [
                    'F01, F02',
                    'F03, F04',
                    'F02, F03',
                    'F03, F04',
                    'F02, F03',
                    'F02, F03'
                ]}
        exp_spc_ids = [0, 1, 2, 3, 4, 5]
        expected = pd.DataFrame(data = exp_spc_filter_list, index = exp_spc_ids)
        expected.index.rename('spc_id', inplace=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_spc_uniq_fltrs(self):
        """Testing the spc_uniq_fltrs static method.
        """
        k_combinations = 2
        test_sp_list = ['R_400_500', 'BD_600_700', 'S_500_600']
        spc_sps_df = SPCClassifier.spc_sps_df(test_sp_list, k_combinations)
        test_filter_list = ['F01, F02', 'F03, F04', 'F02, F03']
        test_sp_fltrs = pd.Series(data=test_filter_list, index=test_sp_list)
        spc_fltr_id_df = SPCClassifier.spc_fltr_id_df(spc_sps_df,test_sp_fltrs)
        result = SPCClassifier.spc_uniq_fltrs(spc_fltr_id_df)

        exp_spc_filter_list = [
                    {'F01', 'F02'},
                    {'F01', 'F02', 'F03', 'F04'},
                    {'F01', 'F02', 'F03'},
                    {'F03', 'F04'},
                    {'F02', 'F03', 'F04'},
                    {'F02', 'F03'}
                ]
        exp_spc_filter_list = [frozenset(i) for i in exp_spc_filter_list]
        exp_spc_ids = [0, 1, 2, 3, 4, 5]
        expected = pd.Series(
                        data=exp_spc_filter_list,
                        index=exp_spc_ids,
                        name='uniq_fltrs',
                        dtype='category')
        expected.index.rename('spc_id', inplace=True)
        # pd.testing.assert_series_equal(result, expected)
        self.assertListEqual(list(result.to_numpy()), list(expected.to_numpy()))

    def test_spc_fc_ids(self):
        """Testing the spc_fc_id static method.
        """
        k_combinations = 2
        test_sp_list = ['R_400_500', 'BD_600_700', 'S_500_400']
        spc_sps_df = SPCClassifier.spc_sps_df(test_sp_list, k_combinations)
        test_filter_list = ['F01, F02', 'F03, F04', 'F02, F01']
        test_sp_fltrs = pd.Series(data=test_filter_list, index=test_sp_list)
        spc_fltr_id_df = SPCClassifier.spc_fltr_id_df(spc_sps_df,test_sp_fltrs)
        result = SPCClassifier.spc_uniq_fltrs(spc_fltr_id_df)

        exp_spc_filter_list = [
                    {'F01', 'F02'},
                    {'F01', 'F02', 'F03', 'F04'},
                    {'F01', 'F02'},
                    {'F03', 'F04'},
                    {'F01', 'F02', 'F03', 'F04'},
                    {'F01', 'F02'}
                ]
        exp_spc_filter_list = [frozenset(i) for i in exp_spc_filter_list]
        exp_spc_ids = [0, 1, 2, 3, 4, 5]
        expected = pd.Series(
                        data=exp_spc_filter_list,
                        index=exp_spc_ids,
                        name='uniq_fltrs',
                        dtype='category')
        expected.index.rename('spc_id', inplace=True)
        self.assertListEqual(list(result.to_numpy()), list(expected.to_numpy()))

    def test_spc_n_uniq_fltrs(self):
        """Testing the spc_n_uniq_fltrs static method.
        """
        k_combinations = 2
        test_sp_list = ['R_400_500', 'BD_600_700', 'S_500_600']
        spc_sps_df = SPCClassifier.spc_sps_df(test_sp_list, k_combinations)
        test_filter_list = ['F01, F02', 'F03, F04', 'F02, F03']
        test_sp_fltrs = pd.Series(data=test_filter_list, index=test_sp_list)
        spc_fltr_id_df = SPCClassifier.spc_fltr_id_df(spc_sps_df,test_sp_fltrs)
        result = SPCClassifier.spc_n_uniq_fltrs(spc_fltr_id_df)

        expected_n_uniq_fltrs = [2, 4, 3, 2, 3, 2]
        expected_spc_ids = [0, 1, 2, 3, 4, 5]
        expected = pd.Series(
                        data=expected_n_uniq_fltrs,
                        index=expected_spc_ids,
                        name='n_uniq_fltrs')
        expected.index.rename('spc_id', inplace=True)
        self.assertListEqual(list(result.to_numpy()), list(expected.to_numpy()))

    def test_build_new_spc_frame(self):
        """Testing the build_new_spc_frame static method.
        """
        k_combinations = 2
        test_sp_list = ['R_400_500', 'BD_600_700', 'S_500_600']
        test_filter_list = ['F01, F02', 'F03, F04', 'F02, F03']
        test_sp_filters = pd.Series(data=test_filter_list, index=test_sp_list)
        result = SPCClassifier.build_new_spc_frame(
                    sp_list=test_sp_list,
                    sp_filters=test_sp_filters,
                    k_combinations=k_combinations)

        expected_lists = {
                'sp_0': [
                    'R_400_500',
                    'R_400_500',
                    'R_400_500',
                    'BD_600_700',
                    'BD_600_700',
                    'S_500_600'],
                'sp_1': [
                    'R_400_500',
                    'BD_600_700',
                    'S_500_600',
                    'BD_600_700',
                    'S_500_600',
                    'S_500_600'
                ],
                'fltrs_0': [
                    'F01, F02',
                    'F01, F02',
                    'F01, F02',
                    'F03, F04',
                    'F03, F04',
                    'F02, F03'],
                'fltrs_1': [
                    'F01, F02',
                    'F03, F04',
                    'F02, F03',
                    'F03, F04',
                    'F02, F03',
                    'F02, F03'
                ],
                'n_uniq_fltrs': [2, 4, 3, 2, 3, 2],
                'uniq_fltrs': [
                    frozenset({'F01', 'F02'}),
                    frozenset({'F01', 'F02', 'F03', 'F04'}),
                    frozenset({'F01', 'F02', 'F03'}),
                    frozenset({'F03', 'F04'}),
                    frozenset({'F02', 'F03', 'F04'}),
                    frozenset({'F02', 'F03'})
                ]}

        expected = pd.DataFrame(data=expected_lists)
        expected.index.rename('spc_id', inplace=True)
        expected['uniq_fltrs'] = expected['uniq_fltrs'].astype('category')

        for col in expected.columns:
            with self.subTest(col=col):
                self.assertListEqual(result[col].tolist(), expected[col].tolist())

    def test_init_new(self):
        """Testing the init class for a new SPCClassifier object.
        """
        _, _, _, _, test_sps = generate_test_objects()
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)

        # k_combinations
        with self.subTest('k_combinations'):
            expected = 2
            result = test_spc.k_combinations
            self.assertEqual(result, expected)
        # sp_list
        with self.subTest('sp_list'):
            expected = test_sps.sp_list
            result = test_spc.sp_list
            self.assertEqual(result, expected)
        # sp_filters
        with self.subTest('sp_filters'):
            expected = test_sps.sp_filters
            result = test_spc.sp_filters
            pd.testing.assert_series_equal(result, expected)
        # main_df
        # ***fails because of frozenset comparison issue***
        # with self.subTest('main_df'):
        #     result = test_spc.main_df
        #     expected = SPCClassifier.build_new_spc_frame(test_spc.sp_list,
        #                                 test_spc.sp_filters, k_combinations)
        #     pd.testing.assert_frame_equal(result, expected)


    def test_flat_sps_list(self):
        """Testing the flat_sps_list function.
        """
        _, _, _, _, test_sps = generate_test_objects(1, True, True, False)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        flat_sps_list = test_spc.flat_sps_list()

        i=0
        for c in test_spc.spc_ids():
            for k in range(0, k_combinations):
                with self.subTest(f'spc_id: {c}, component {k}'):
                    expected = test_spc.spc_sps().loc[c][k]
                    result = flat_sps_list[i]
                    self.assertEqual(result, expected)
                i+=1

    def test_stack_k_combinations(self):
        """Testing the stack_k_combinations function.
        """
        # test input - 2D array of values
        #   - channels only, 2 samples, pair combinations
        _, _, _, _, test_sps = generate_test_objects(5, True, True, True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        k_stack, _ = test_spc.stack_k_combinations(test_sps.main_df)

        for c in test_spc.spc_ids():
            for n, sample_id in enumerate(test_sps.main_df.index):
                for k in range(0, k_combinations):
                    with self.subTest(f'spc_id:{c}, sample {n}, component {k}'):
                        result = k_stack[c][n][k]
                        sp = test_spc.spc_sps().loc[c][k]
                        expected = test_sps.main_df[sp][sample_id]
                        self.assertEqual(result, expected)

    def test_fit_lda(self):
        """Testing the fit_lda function.
        """
        _, _, _, _, test_sps = generate_test_objects(10, True, True, True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)

        result = test_spc.fit_lda(test_sps)
        # just check columns, assuming numerical action of functions
        # has been tested by test_lda.py
        expected = ['lda_score', 'lda_a_0_1', 'lda_a_1_1',
                                            'lda_boundary_1', 'lda_tg_gt_bg_1']
        self.assertListEqual(result.columns.to_list(), expected)
        # difficult to validate because different runs may rank spectral
        # parameters differently due to noise

    def test_rank_spcs(self):
        """Testing the rank_spcs function.
        """
        _, _, _, _, test_sps = generate_test_objects(10, True, False, True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        test_spc.fit_lda(test_sps)
        ranked_df = test_spc.rank_spcs(metric='lda_score', scope='all-data')
        # just check that the rank column has been added and has correct length
        result = ranked_df['rank'].values
        expected = np.arange(1, len(test_spc.spc_ids())+1)
        np.testing.assert_array_equal(result, expected)
        # difficult to validate because different runs may rank spectral
        # parameters differently due to noise

    def test_plot_spc(self):
        """Testing the plot_spc function.
        """
        _, _, _, _, test_sps = generate_test_objects(10, True, False, True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        test_spc.fit_lda(test_sps)
        test_spc.rank_spcs(metric='lda_score', scope='all-data')
        # plot top results
        test_spc_id = 0 # single sp
        test_spc.plot_sp_combo(test_spc_id, test_sps)
        test_spc_id = 1 # pair sp
        test_spc.plot_sp_combo(test_spc_id, test_sps)

    def test_plot_spc_lda(self):
        """Testing the plot_spc_lda function.
        """
        _, _, _, _, test_sps = generate_test_objects(10, True, False, True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        test_spc.fit_lda(test_sps)
        test_spc.rank_spcs(metric='lda_score', scope='all-data')
        # plot top results
        # plot spc's for comparison
        test_spc_id = 0 # single sp
        test_spc.plot_sp_combo(test_spc_id, test_sps)
        test_spc_id = 1 # pair sp
        test_spc.plot_sp_combo(test_spc_id, test_sps)

        test_spc_id = 0 # single sp
        test_spc.plot_spc_lda(test_spc_id, test_sps)
        test_spc_id = 1 # pair sp
        test_spc.plot_spc_lda(test_spc_id, test_sps)

    def test_binary_classifier(self):
        """Testing the binary_classifier function.
        """
        _, _, _, _, test_sps = generate_test_objects(10, True, False, True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        test_spc.fit_lda(test_sps)
        test_spc.rank_spcs(metric='lda_score', scope='all-data')
        predictions_bool = test_spc.binary_classifier(test_sps)
        print(predictions_bool)
        # this is difficult to validate, because of predictive nature
        # but we should be able to have some idea of which parameters
        # will classify which materials in which way

    def test_labelled_predictions(self):
        """Testing the labelled_predictions function.
        """
        _, _, _, _, test_sps = generate_test_objects(10, True, False, True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        test_spc.fit_lda(test_sps)
        test_spc.rank_spcs(metric='lda_score', scope='all-data')
        predicted = test_spc.binary_classifier(test_sps)
        result = test_spc.labelled_predictions(test_sps, predicted)
        print(result)
        # also difficult to validate

    def test_confusion_matrix(self):
        """Testing the confusion_matrix static method.
        """
        # test confusion matrix
        predicted_labels = np.array([True, True, False, False])
        actual_labels = np.array([True, False, False, True])
        tp, tn, fp, fn = SPCClassifier.confusion_matrix(
                                predicted_labels,
                                actual_labels,
                                summed=False)
        with self.subTest('true-positive'):
            result = tp
            expected = np.array([True, False, False, False])
            np.testing.assert_array_equal(result, expected)
        with self.subTest('true-negative'):
            result = tn
            expected = np.array([False, False, True, False])
            np.testing.assert_array_equal(result, expected)
        with self.subTest('false-positive'):
            result = fp
            expected = np.array([False, True, False, False])
            np.testing.assert_array_equal(result, expected)
        with self.subTest('false-negative'):
            result = fn
            expected = np.array([False, False, False, True])
            np.testing.assert_array_equal(result, expected)

        tp, tn, fp, fn = SPCClassifier.confusion_matrix(
                                predicted_labels,
                                actual_labels,
                                summed=True)
        with self.subTest('true-positive summed'):
            result = tp
            expected = 1
            np.testing.assert_array_equal(result, expected)
        with self.subTest('true-negative summed'):
            result = tn
            expected = 1
            np.testing.assert_array_equal(result, expected)
        with self.subTest('false-positive summed'):
            result = fp
            expected = 1
            np.testing.assert_array_equal(result, expected)
        with self.subTest('false-negative summed'):
            result = fn
            expected = 1
            np.testing.assert_array_equal(result, expected)

    def test_accuracy_metrics(self):
        """Testing the accuracy_metrics static method
        """
        predicted_labels = np.array([True, True, False, False])
        actual_labels = np.array([True, False, False, True])
        tp, tn, fp, fn = SPCClassifier.confusion_matrix(
                                predicted_labels,
                                actual_labels,
                                summed=True)
        acc, ppv, tpr, tnr = SPCClassifier.accuracy_metrics(tp, tn, fp, fn)
        with self.subTest('accuracy'):
            result = acc
            expected = (1 + 1) / (1 + 1 + 1 + 1)
            np.testing.assert_array_equal(result, expected)
        with self.subTest('precision'):
            result = ppv
            expected = 1 / (1+1)
            np.testing.assert_array_equal(result, expected)
        with self.subTest('sensitivity'):
            result = tpr
            expected = 1 / (1+1)
            np.testing.assert_array_equal(result, expected)
        with self.subTest('specificity'):
            result = tnr
            expected = 1 / (1+1)
            np.testing.assert_array_equal(result, expected)

    def test_binary_classifier_accuracy(self):
        """Testing the binary_classifier_accuracy function.
        """
        _, _, _, _, test_sps = generate_test_objects(
                                    n_samples=10,
                                    just_channels=True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        test_spc.fit_lda(test_sps)
        test_spc.rank_spcs(metric='lda_score', scope='all-data')
        predictions_bool = test_spc.binary_classifier(test_sps)
        cat_list = test_sps.cat_list()
        result = test_spc.binary_classifier_accuracy(predictions_bool, cat_list)
        print(result)
        # also difficult to validate

    def test_plot_roc(self):
        """Testing the plot_roc function.
        """
        _, _, _, _, test_sps = generate_test_objects(
                                    n_samples=10,
                                    flat=False,
                                    just_channels=False,
                                    noise=True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        training, testing = test_sps.train_test_random_split()
        test_spc.fit_lda(training)

        predictions = test_spc.binary_classifier(testing)
        cat_list = testing.main_df.Category
        test_spc.binary_classifier_accuracy(predictions, cat_list)
        test_spc.rank_spcs(metric='lda_acc_1', scope='all-data')
        test_spc.plot_roc(noisey=True)

        out_file = Path(test_spc.object_dir,
                                'analysis', 'roc_scatter_noisey').with_suffix(PLT_FRMT)
        self.assertTrue(os.path.isfile(out_file))

    def test_plot_metrics_vs_rank(self):
        """Testing the plot_metrics_vs_rank function.
        """
        _, _, _, _, test_sps = generate_test_objects(
                                    n_samples=10,
                                    flat=False,
                                    just_channels=False,
                                    noise=True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        training, testing = test_sps.train_test_random_split()
        test_spc.fit_lda(training)

        predictions = test_spc.binary_classifier(testing)
        test_spc.binary_classifier_accuracy(predictions, testing.cat_list())
        test_spc.rank_spcs(metric='lda_acc_1', scope='all-data')
        test_spc.plot_metrics_vs_rank(testtrain='test')

        out_file = Path(test_spc.object_dir,
                                'analysis', 'mean_metrics_vs_rank_test').with_suffix(PLT_FRMT)
        self.assertTrue(os.path.isfile(out_file))

    def test_plot_metric_vs_metric(self):
        """Testing the plot_metric_vs_metric function.
        """
        _, _, _, _, test_sps = generate_test_objects(
                                    n_samples=10,
                                    flat=False,
                                    just_channels=False,
                                    noise=True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        training, testing = test_sps.train_test_random_split()
        test_spc.fit_lda(training)

        predictions = test_spc.binary_classifier(testing)
        test_spc.binary_classifier_accuracy(predictions, testing.cat_list())
        test_spc.rank_spcs(metric='lda_acc_1', scope='all-data')
        test_spc.plot_metric_vs_metric(
                        ('lda_score', 'Fisher Ratio'),
                        ('lda_acc_1', 'Accuracy'))

        out_file = Path(test_spc.object_dir,
                                'analysis', 'lda_acc_1_vs_lda_score').with_suffix(PLT_FRMT)
        self.assertTrue(os.path.isfile(out_file))

    def test_all(self):
        """Testing the complete process of training and validating a
        classifier
        """
        _, _, _, _, test_sps = generate_test_objects(
                                    n_samples=10,
                                    flat=False,
                                    just_channels=False,
                                    noise=True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        training, testing = test_sps.train_test_random_split()
        test_spc.fit_lda(training)

        predictions = test_spc.binary_classifier(testing)
        test_spc.binary_classifier_accuracy(predictions, testing.cat_list())
        test_spc.rank_spcs(metric='lda_acc_1', scope='all-data')
        # test_spc.plot_top_ranks(training, 5, scope='train')
        # test_spc.plot_top_ranks(testing, 5, scope='test')
        test_spc.plot_roc(noisey=True)
        test_spc.plot_metrics_vs_rank()
        test_spc.plot_metric_vs_metric(
                        ('lda_score', 'Fisher Ratio'),
                        ('lda_acc_1', 'Accuracy'))
        test_spc.export_df()
        print(test_spc.main_df)

    def test_fit_lda_repeat_holdout(self):
        """Testing the fit_lda_repeat_holdout function.
        """
        _, _, _, _, test_sps = generate_test_objects(
                                    n_samples=10,
                                    flat=False,
                                    just_channels=False,
                                    noise=True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        stats_spc, trials_spc = test_spc.fit_lda_repeat_holdout(test_sps, export_df=True)
        test_spc.rank_spcs(metric='lda_acc_1', scope='all-data')


    def test_fit_lda_repeat_holdout_ranking(self):
        """Testing the fit_lda_repeat_holdout function.
        """
        _, _, _, _, test_sps = generate_test_objects(
                                    n_samples=10,
                                    flat=False,
                                    just_channels=False,
                                    noise=True)
        k_combinations = 2
        test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
        test_spc.fit_lda_repeat_holdout(test_sps, export_df=False)
        test_spc.fit_lda(test_sps)
        test_spc.merge_mean_accuracy()
        test_spc.rank_spcs(metric='lda_acc_1', scope='all-data')

        test_spc.plot_roc(noisey=True)
        test_spc.plot_metrics_vs_rank()
        test_spc.plot_metric_vs_metric(
                        ('lda_score', 'Fisher Ratio'),
                        ('lda_acc_1', 'Mean Accuracy'))
        test_spc.export_df('all')

    # def test_plot_channel_distribution(self):
    #     """Testing the plot_channel_distribution function.
    #     """
    #     _, _, _, _, test_sps = generate_test_objects(
    #                         n_samples=10,
    #                         flat=False,
    #                         just_channels=False,
    #                         noise=True)
    #     k_combinations = 2
    #     test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
    #     training, testing = test_sps.train_test_random_split()
    #     test_spc.fit_lda(training)

    #     predictions = test_spc.binary_classifier(testing)
    #     test_spc.binary_classifier_accuracy(predictions, testing.cat_list())
    #     test_spc.rank_spcs(metric='lda_acc_1', scope='all-data')
    #     n_ids = float(len(test_spc.spc_ids()))
    #     test_spc.plot_channel_distribution(top_ns = (20,int(n_ids/100),int(n_ids/10)))

    # def test_plot_n_uniq_fltr_distribution(self):
    #     """Testing the plot_n_uniq_fltr_distribution function.
    #     """
    #     _, _, _, _, test_sps = generate_test_objects(
    #                         n_samples=10,
    #                         flat=False,
    #                         just_channels=False,
    #                         noise=True)
    #     k_combinations = 2
    #     test_spc = SPCClassifier(test_sps, k_combinations, load_existing=False)
    #     training, testing = test_sps.train_test_random_split()
    #     test_spc.fit_lda(training)

    #     predictions = test_spc.binary_classifier(testing)
    #     test_spc.binary_classifier_accuracy(predictions, testing.cat_list())
    #     test_spc.rank_spcs(metric='lda_acc_1', scope='all-data')
    #     test_spc.plot_n_uniq_fltr_distribution()


if __name__ == '__main__':
    unittest.main(verbosity=2)