"""SpectralParameterCombinationClassifier Class

Hosts methods for performing Linear Discriminant Analysis on a SpectralParameter
dataset, training an LDA classifier.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 07-10-2022
"""

import itertools
import os
from pathlib import Path
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import sptk.config as cfg
from sptk.config import build_project_directory as build_pd
import sptk.linear_discriminant_analysis as lda
from sptk.spectral_parameters import SpectralParameters


class SpectralParameterCombinationClassifier():
    """Class for linearly combining spectral parameters, and fitting and
    evaluating Linear Discriminant classifiers to each combination.
    """

    def __init__(self,
        spectral_parameters: SpectralParameters,
        k_combinations: int,
        load_existing: bool=cfg.LOAD_EXISTING) -> None:
        """Constructor of SpectralParameterCombinationClassifier.

        :param spectral_parameters: spectral parameters object hosting the
            list of spectral parameters to combine.
        :type spectral_parameters: SpectralParameters
        :param k_combinations: num of spectral parameters in each combination,
            i.e. the 'k' in 'n-choose-k'
        :type k_combinations: int
        :param load_existing: instruct to use or overwrite existing directories
            and files of the same project_name, defaults to config.py setting.
        :type load_existing: bool, optional
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            print("Initialising SpectralParameterCombinationClassifier")

        SPCClassifier = SpectralParameterCombinationClassifier

        p_dir, p_name = build_pd(spectral_parameters.project_name,
                                                            'spc_classifier')
        self.project_dir = p_dir
        self.project_name = p_name
        self.object_dir = Path(self.project_dir / 'spc_classifier')

        self.k_combinations = k_combinations

        self.sp_list = spectral_parameters.sp_list
        self.sp_filters = spectral_parameters.sp_filters

        self.classes = spectral_parameters.main_df.Category.cat.categories

        if load_existing:
            existing_pkl_path = Path(self.object_dir,'spc_classifier.pkl')
            file_exists = os.path.isfile(existing_pkl_path)
            if file_exists:
                print("Loading existing SpectralParameterCombinationClassifier"\
                      f" DataFrame for {self.project_name}")
                self.main_df = pd.read_pickle(existing_pkl_path)
            else:
                print("No existing DataFrame")
                print("Building new SpectralParameterCombinationClassifier"\
                      f" DataFrame for {self.project_name}")
                self.main_df = SPCClassifier.build_new_spc_frame(
                                    self.sp_list,
                                    self.sp_filters,
                                    self.k_combinations)
        else:
            print("Building new SpectralParameterCombinationClassifier"\
                        f" DataFrame for {self.project_name}")
            self.main_df = SPCClassifier.build_new_spc_frame(
                                    self.sp_list,
                                    self.sp_filters,
                                    self.k_combinations)

        self.stats_df = None
        self.trials_df = None
        self.rank_metric = None
        self.rank_scope = None

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"SpectralParameterCombinationClassifier initialised in \
                 {toc - tic:0.4f} seconds.")

    @staticmethod
    def build_new_spc_frame(
        sp_list: List,
        sp_filters: pd.Series,
        k_combinations: int) -> pd.DataFrame:
        """Construct Spectral Parameter Combination list from input spectral
        parameter list.

        :param sp_list: Spectral Parameter list to find combinations from.
        :type sp_list: List
        :param sp_filters: Filters used in spectral parameter list
        :type sp_filters: pd.Series
        :param k_combinations: num of spectral parameters in each combination,
            i.e. the 'k' in 'n-choose-k'
        :type k_combinations: int
        :return: spectral parameter combination dataframe
        :rtype: pd.DataFrame
        """
        SPCClassifier = SpectralParameterCombinationClassifier
        # get combination list with assigned IDs
        spc_sps = SPCClassifier.spc_sps_df(sp_list, k_combinations)
        # get filters used in combination list
        spc_fltr_ids = SPCClassifier.spc_fltr_id_df(spc_sps, sp_filters)
        # get number of unique filters used in spcs
        spc_n_uniq_fltrs = SPCClassifier.spc_n_uniq_fltrs(spc_fltr_ids)
        # get unique filters in spcs
        spc_uniq_fltrs = SPCClassifier.spc_uniq_fltrs(spc_fltr_ids)
        # concat dataframe
        main_df = pd.concat([spc_sps,
                             spc_fltr_ids,
                             spc_n_uniq_fltrs,
                             spc_uniq_fltrs
                             ], axis=1)
        return main_df

    # """Dataframe Access"""

    def spc_ids(self) -> pd.Index:
        """Get the spectral parameter combination IDs

        :return: spectral parameter combination ids
        :rtype: pd.Index
        """
        return self.main_df.index

    def spc_sps(self) -> pd.DataFrame:
        """Get the components of the spectral parameter combinations

        :return: spectral parameters composing each spectral parameter
            combination
        :rtype: pd.DataFrame
        """
        combi_cols = ['sp_'+str(feat) for feat in range(0,self.k_combinations)]
        spc_sps = self.main_df[combi_cols]
        return spc_sps

    def n_uniq_fltrs(self) -> pd.Series:
        """Get the number of unique filters composing each spectral parameter
        combination.

        :return: number of unique filters composing each spectral parameter
            combination.
        :rtype: pd.Series
        """
        n_uniq_fltrs = self.main_df['n_uniq_fltrs']
        return n_uniq_fltrs

    def uniq_sp_list(self, spc_id: int) -> List:
        """To handle cases of one spectral parameter combined with itself in
        a spectral parameter combination,
        get the unique spectral parameter label.

        :param spc_id: spectral parameter combination ID
        :type spc_id: int
        :return: unique spectral parameters in spc_id
        :rtype: List
        """
        all_lbls = self.main_df.loc[spc_id].index
        sp_lbls = [lbl for lbl in all_lbls if 'sp_' in lbl]
        sp_list = self.main_df[sp_lbls].loc[spc_id]
        # get the unique spectral parameters
        uniq_sp_list = sp_list.drop_duplicates().tolist()
        return uniq_sp_list

    def projections(self, lda_ax: int=1) -> pd.DataFrame:
        """Get the projection coefficients for the learned LDA models for
        each spectral parameter combination.

        :param lda_ax: LDA axis, if more than 1, defaults to 1
        :type lda_ax: int, optional
        :return: projection coefficients for each spectral parameter combination
        :rtype: pd.DataFrame
        """
        # get the number of classes
        n_c = len(self.classes)
        # get the number of LDA dimensions
        k_c = self.k_combinations
        if lda_ax is None:
            feat_list = ['lda_a_'+str(feat)+'_'+str(i+1)
                             for i in range(0,n_c-1) for feat in range(0,k_c)]
        else:
            feat_list = ['lda_a_'+str(feat)+'_'+str(lda_ax)
                                                    for feat in range(0,k_c)]
        return self.main_df[feat_list]

    def lda_models(self, lda_ax: int=1) -> pd.DataFrame:
        """Get the complete learned LDA models for each spectral parameter
        combination

        :param lda_ax: LDA axis, if more than 1, defaults to 1
        :type lda_ax: int, optional
        :return: LDA model for each spectral parameter combination
        :rtype: pd.DataFrame
        """
        # get the number of classes
        n_c = len(self.classes)
        # get the number of LDA dimensions
        k_c = self.k_combinations
        if lda_ax is None:
            feat_list = ['lda_a_'+str(feat)+'_'+str(i+1)
                             for i in range(0,n_c-1) for feat in range(0,k_c)]
        else:
            feat_list = ['lda_a_'+str(feat)+'_'+str(lda_ax)
                                                    for feat in range(0,k_c)]
        feat_list.append(['lda_boundary_'+str(lda_ax),
                          'lda_tg_gt_bg_'+str(lda_ax)])
        return self.main_df[feat_list]

    @staticmethod
    def spc_sps_df(sp_list: List[str], k_combinations: int) -> pd.DataFrame:
        """Get list of k_combinations (e.g. pairs) of spectral parameters.

        :param sp_list: spectral parameters to use
        :type sp_list: List[str]
        :param k_combinations: num of spectral parameters in each combination,
            i.e. the 'k' in 'n-choose-k'
        :type k_combinations: int
        :return: list of k_combinations of spectral parameters
        :rtype: pd.DataFrame
        """
        # get list of k_combinations of spectral parameters
        combi = itertools.combinations_with_replacement(sp_list, k_combinations)
        combi_list = list(combi)
        # organise data for collection into dataframe
        combi_cols = ['sp_'+str(feat) for feat in range(0,k_combinations)]
        spc_sps_df = pd.DataFrame(
                        data=combi_list,
                        columns=combi_cols,
                        dtype=str)
        spc_sps_df.index.rename('spc_id', inplace=True)
        return spc_sps_df

    @staticmethod
    def spc_fltr_id_df(
        spc_sps_df: pd.DataFrame,
        sp_filters: pd.Series) -> pd.DataFrame:
        """Get a list (DataFrame) of filter IDs used in the given list of
        spectral parameter combinations.

        :param spc_sps_df: spectral parameter combination ID dataframe
        :type spc_sps_df: pd.DataFrame
        :return: spectral parameter combination filter ID dataframe
        :rtype: pd.DataFrame
        """
        filter_dict = sp_filters.to_dict()
        spc_fltr_id_df = spc_sps_df.replace(filter_dict)
        filter_cols = ['fltrs_'+str(feat) for
                                feat in range(0,len(spc_fltr_id_df.columns))]
        spc_fltr_id_df.columns = filter_cols
        return spc_fltr_id_df

    @staticmethod
    def get_spc_fltr_lists(spc_fltr_id_df: pd.DataFrame) -> pd.Series:
        """Get a combined list of filters contributing to each spectral
        parameter combination.

        :param spc_fltr_id_df: spectral parameter combination filter IDs
        :type spc_fltr_id_df: pd.DataFrame
        :return: filter contributing to each spectral parameter combination
        :rtype: pd.Series
        """
        spc_fltr_lists = spc_fltr_id_df.apply(', '.join, axis=1)
        spc_fltr_lists = spc_fltr_lists.str.split(', ', expand=False)
        return spc_fltr_lists

    @staticmethod
    def spc_n_uniq_fltrs(spc_fltr_id_df: pd.DataFrame) -> pd.Series:
        """Count the number of unique filters in each spectral parameter
        combination. Return in a series indexed by spc_id.

        :param spc_fltr_id_df: spectral parameter combination filter IDs
        :type spc_fltr_id_df: pd.DataFrame
        :return: number of unique filters in each spectral parameter combination
        :rtype: pd.Series
        """
        # join filter lists of each spectral parameter in combination
        SPCC = SpectralParameterCombinationClassifier
        spc_fltr_lists = SPCC.get_spc_fltr_lists(spc_fltr_id_df)
        # count the unique elements in joint filter lists
        spc_n_uniq_fltrs = spc_fltr_lists.apply(lambda x: len(set(x)))
        spc_n_uniq_fltrs.rename('n_uniq_fltrs', inplace=True)
        return spc_n_uniq_fltrs

    @staticmethod
    def spc_uniq_fltrs(spc_fltr_id_df: pd.DataFrame) -> pd.Series:
        """Get the unique filters in each spectral parameter combination.
        Return in a series indexed by spc_id. Note we don't care about order,
        so this should just be a set.

        :param spc_fltr_id_df: spectral parameter combination filter IDs
        :type spc_fltr_id_df: pd.DataFrame
        :return: list of unique filters in each spectral parameter combination
        :rtype: pd.Series
        """
        SPCC = SpectralParameterCombinationClassifier
        spc_fltr_lists = SPCC.get_spc_fltr_lists(spc_fltr_id_df)

        uniq = lambda a: frozenset(pd.Series(a).drop_duplicates())
        uniq_sets = [uniq(filter_list) for filter_list in spc_fltr_lists]
        spc_uniq_fltrs = pd.Series(
                            data=uniq_sets,
                            index=spc_fltr_id_df.index,
                            name='uniq_fltrs',
                            dtype='category')
        return spc_uniq_fltrs

    def stack_k_combinations(self,
            sp_df: pd.DataFrame) -> \
             Tuple[np.array, Tuple[List[int], List[str]]]:
        """Reshape the 2D array of spectral parameter values,
        [n_spectral_parameters x n_samples], into a 3D array,
        [n_combinations x n_samples x k_combinations].
        This prepares the data for vectorised LDA operations.

        :param sp_df: spectral parameters dataframe
        :type sp_df: pd.DataFrame
        :return: 3D array of spectral parameter values, with dimensions:
                [n_combinations x n_samples x k_combinations],
                with accompanying lists of element IDs for each dimension.
        :rtype: Tuple[np.array, Tuple[List[int], List[str], List[str]]]
        """
        # use combination list to build a 3D numpy array from the dataframe
        flat_obs_np = sp_df[self.flat_sps_list()].T.to_numpy()
        columns = [] # set inner array
        for i in np.arange(self.k_combinations):
            columns.append(flat_obs_np[i::self.k_combinations, :])
        columns = tuple(columns)
        k_stack = np.stack(columns, axis=-1)

        sample_id_order = sp_df.index.to_list()
        combination_order = self.spc_sps().columns.to_list()

        return k_stack, (sample_id_order, combination_order)

    def flat_sps_list(self) -> np.array:
        """Get a flattened list of the spectral parameter combinations, e.g.
                sp_0    sp_1
        spc_id
        0       R450    R450
        1       R450    R500
        ->
        [R450, R450, R450, R500]
        Allows for access of spectral_parameters data in parallel.

        :return: flatten list of spectral parameter combinations.
        :rtype: np.array
        """
        flat_sps_list = self.spc_sps().values.flatten()
        return flat_sps_list

    # """LDA Fitting"""

    def fit_lda(self,
        spectral_parameters: SpectralParameters) -> pd.DataFrame:
        """Perform Linear Discriminant Analysis on Spectral Parameter
        Combinations, and return in a DataFrame:
         - lda_score: the Separation Score evaluating the separation of
                labelled classes in the dataset for the combination.
         - lda_a_0_1, lda_a_1_1: the Linear Discriminant projection coordinates
                that maximise the Separation Score for the combination.
         - lda_boundary_1: the decision boundary along the LDA projection that
                separates the labelled classes for the combination.
         - lda_target>boundary_1: determines if the target class is above or
                below the decision boundary for the combination.

        :return: _description_
        :rtype: pd.DataFrame
        """
        # reformat data into a n_spc X n_samples x k_combinations array
        k_stack, _ = self.stack_k_combinations(spectral_parameters.main_df)
        cat_list = spectral_parameters.main_df.Category

        # perform LDA on the 3D array
        k_c = self.k_combinations
        wcsm = lda.within_class_scatter_matrix(k_stack, cat_list)
        bcsm = lda.between_class_scatter_matrix(k_stack, cat_list)
        categories = self.classes
        n_classes = len(categories)
        # get singular_spcs
        singular_spcs = (self.main_df['sp_0'] == self.main_df['sp_1']).to_numpy()
        projections, _, _ = lda.projection_matrix(wcsm, bcsm, n_classes, singular_spcs)
        score = lda.fisher_ratio(projections, bcsm, wcsm)

        # put projections into spectral parameter combination dataframe (spc_df)
        lda_df = pd.Series(data=score, index=self.spc_ids(), name='lda_score')
        for i in range(0,n_classes-1):
            # note this is setup for n-category case,
            # but code is otherwise restricted to 2 categories
            proj_cols = ['lda_a_'+str(f)+'_'+str(i+1) for f in range(0,k_c)]
            projection_df = pd.DataFrame(
                                data=projections[:,:,i],
                                index=self.spc_ids(),
                                columns=proj_cols)
            lda_df = pd.concat([lda_df, projection_df], axis=1)

        # find boundaries for each LDA
        for i in range(0, n_classes-1):# each lda axis
            boundary, tg_gt_bg = lda.compute_lda_boundary(
                                            projections[:,:,i],
                                            k_stack,
                                            cat_list)
            boundary_df = pd.DataFrame(
                                data={
                                   'lda_boundary_'+str(i+1): boundary,
                                    'lda_tg_gt_bg_'+str(i+1): tg_gt_bg
                                },
                                index=self.spc_ids())
            lda_df = pd.concat([lda_df, boundary_df], axis=1)

        # update main df
        # if column names exist already, overwrite these when concating
        if set(lda_df.columns.to_list()).issubset(set(self.main_df.columns.to_list())):
            self.main_df[lda_df.columns] = lda_df
        else:
            self.main_df = pd.concat([self.main_df, lda_df], axis=1)

        return lda_df

    def fit_lda_repeat_holdout(self,
            spectral_parameters: SpectralParameters,
            k_trials: int=10,
            seed: int=None,
            load_existing: bool=cfg.LOAD_EXISTING,
            export_df: bool=cfg.EXPORT_DF) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run k repeat trials of performing LDA on the dataset, and evaluating
        the accuracy of the fitted LDA classifiers.

        :param spectral_parameters: Evaluated Spectral Parameters for each entry
        :type spectral_parameters: SpectralParameters
        :param k_trials: Number of repeat trials, defaults to 10, max is 2**14
        :type k_trials: int, optional
        :param seed: Set a seed for reproducibility, defaults to None
        :type seed: int, optional
        :param load_existing: Access existing dataset, defaults to cfg.LOAD_EXISTING
        :type load_existing: bool, optional
        :param export_df: Export the trial results, defaults to cfg.EXPORT_DF
        :type export_df: bool, optional
        :return: The statistics and trial results dataframes
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """

        # check that k_trials is in range
        if k_trials > 2**14:
            raise ValueError(f'k_trials must be less than 2**14, not {k_trials}')

        # check for existing
        while load_existing:
            # check for existing
            table_dir = Path(self.object_dir / 'tables')
            trials_file = Path(table_dir, 'spc_trials.csv')
            stats_file = Path(table_dir, 'spc_stats.csv')
            try:
                self.trials_df = pd.read_csv(trials_file, header=[0,1], skiprows=[2], index_col=[0])
                if 'rank' in self.trials_df.columns.levels[0]:
                    self.trials_df.drop('rank', axis=1, inplace=True)
            except FileNotFoundError:
                load_existing = False
                break
            try:
                self.stats_df = pd.read_csv(stats_file, header=[0,1], skiprows=[2], index_col=[0])
            except FileNotFoundError:
                self.stats_from_trials(self.trials_df, export_df=export_df)
            self.stats_df.index.names = ['spc_id']
            self.trials_df.index.names = ['spc_id']
            self.trials_df.rename(mapper=int, axis=1, level=1, inplace=True)
            # check that number of trials matches
            loaded_k_trials = len(self.trials_df.columns.levels[1])
            if loaded_k_trials != k_trials:
                raise ValueError(f'Loaded data ({loaded_k_trials}) does not"\
                            " match requested number of trials ({k_trials})')
            print(f'Loaded existing {k_trials} LDA fitting trial results...')
            return self.stats_df, self.trials_df

        if cfg.TIME_IT:
            print(f'Running {k_trials} LDA fitting trials...')
            tic = time.perf_counter()

        # initialise storage for trial data
        lda_score = []
        a_0 = [] # currently 2 class case only
        a_1 = [] # currently 2 class case only
        acc = [] # accuracy
        ppv = [] # positive predictive value
        tpr = [] # true positive rate
        tnr = [] # true negative rate
        fpr = [] # false positive rate

        # average the results of k_trials repeat training trials
        trial_seed_seq = np.random.default_rng(seed).choice(2**14, size=2**14, replace=False)
        for i in range(0, k_trials):
            print(f'Running trial {i} of {k_trials}...')
            trial_seed = trial_seed_seq[i]
            train_sps, test_sps = spectral_parameters.train_test_random_split(
                                                            test_size=0.2,
                                                            seed=trial_seed,
                                                            balance_test=True)
            self.fit_lda(train_sps)
            predictions = self.binary_classifier(test_sps)
            self.binary_classifier_accuracy(predictions, test_sps.cat_list())
            # add train+test results to list of results.
            lda_score.append(self.main_df['lda_score'].to_list())
            a_0.append(self.main_df['lda_a_0_1'].to_list())
            a_1.append(self.main_df['lda_a_1_1'].to_list())
            acc.append(self.main_df['lda_acc_1'].to_list())
            ppv.append(self.main_df['lda_ppv_1'].to_list())
            tpr.append(self.main_df['lda_tpr_1'].to_list())
            tnr.append(self.main_df['lda_tnr_1'].to_list())
            fpr.append(self.main_df['lda_fpr_1'].to_list())

        # compute statistics on results
        trials = np.arange(0, k_trials)+1
        metrics = ['lda_score', 'lda_a_0_1', 'lda_a_1_1',
                # 'lda_boundary_1', 'lda_tg_gt_bg_1',
                'lda_acc_1', 'lda_ppv_1', 'lda_tpr_1', 'lda_tnr_1', 'lda_fpr_1']
        multiindex = pd.MultiIndex.from_product(
                        [metrics, trials],
                        names=['metric', 'trial'])
        data = np.concatenate([lda_score, a_0, a_1,
                                    # boundary, tg_gt_bg,
                                        acc, ppv, tpr, tnr, fpr]).T
        trials_df = pd.DataFrame(
                    data=data,
                    columns=multiindex,
                    index=self.main_df.index)

        stats_df = self.stats_from_trials(trials_df)

        self.trials_df = trials_df
        self.stats_df = stats_df

        # self.export_df(df_type='trials')

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Repeat trials complete in {toc - tic:0.4f} seconds.")

        return stats_df, trials_df

    def stats_from_trials(self,
            trials_df: pd.DataFrame,
            export_df: bool=cfg.EXPORT_DF) -> pd.DataFrame:
        """Compute the statistics on the trials

        :param trials_df: repeat-holdout trials dataset
        :type trials_df: pd.DataFrame
        :return: statistics dataframe
        :rtype: pd.DataFrame
        """
        mean_spc = trials_df.groupby(level='metric', axis=1).mean()
        var_spc = trials_df.groupby(level='metric', axis=1).std()**2
        stats_df = pd.concat([mean_spc, var_spc], keys=['mean', 'var'], axis=1)
        stats_df = stats_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
        metrics = trials_df.columns.get_level_values('metric').unique()
        stats_df = stats_df[metrics] # preserve order of headings
        self.stats_df = stats_df
        if export_df:
            self.export_df(df_type='stats')

        return stats_df

    # """LDA Predictions"""

    def binary_classifier(self,
            spectral_parameters: SpectralParameters) -> pd.DataFrame:
        """Classify the dataset with all spectral parameter combinations
        in parallel. Limited to the 2-class (binary) case.
        Return results in boolean logic tables.

        :param spectral_parameters: dataset in spectral parameter format
        :type spc_df: SpectralParameters
        :return:
            - table giving predicted labels, represented by boolean logic,
              for each spectral parameter combination
        :rtype: pd.DataFrame
        """
        if len(self.classes) > 2:
            raise ValueError("Binary classifier only valid for 2 class case.")
        projected_df = self.project_dataset(spectral_parameters.main_df)
        projected_df.drop('Category', axis=1, inplace=True)

        # evaluate projected data against decision boundaries for all spc's in
        # parallel
        boundaries = self.main_df['lda_boundary_'+str(1)].copy()
        target_vs_background = self.main_df['lda_tg_gt_bg_'+str(1)]
        gt = projected_df.gt(boundaries)
        le = projected_df.le(boundaries)
        tg = target_vs_background.to_numpy()
        predictions_bool = np.where(tg, gt, le)

        predictions_df = pd.DataFrame(
                                data=predictions_bool,
                                index = spectral_parameters.main_df.index,
                                columns=self.spc_ids())

        return predictions_df

    def cat2bool_dict(self) -> dict:
        """Provide mapping of category labels to boolean labels.
        Valid for the two class case only.

        :return: mapping of category labels to True/False
        :rtype: dict
        """
        if len(self.classes) > 2:
            raise ValueError("Binary classifier only valid for 2 class case.")
        bool_labels = {self.classes[0]: False, self.classes[1]: True}
        return bool_labels

    def categories_as_bools(self, cat_list: pd.Series) -> np.array:
        """Converts the category labels of a dataset to boolean, according
        to the cat2bool dictionary.
        Also drops 'category' dtype.
        Preserves order.

        :param cat_df: category of each data entry in the dataset
        :type cat_df: pd.Series
        :return: boolean representation of category in each data entry in the
            dataset.
        :rtype: np.array
        """
        if len(self.classes) > 2:
            raise ValueError("Binary classifier only valid for 2 class case.")
        # record actual labels in Boolean representation
        cat2bool = self.cat2bool_dict()
        cats_as_bools = cat_list.replace(cat2bool).to_numpy().reshape(-1,1)
        return cats_as_bools

    def project_dataset(self,
            sp_df: pd.DataFrame,
            spc_id: int=None) -> pd.DataFrame:
        """Project complete dataset for all spectral parameter combinations
        in parallel.

        :param sp_df: spectral parameters dataframe
        :type sp_df: pd.DataFrame
        :return: data projected onto the LDA axis of each spectral parameter
            combination.
        :rtype: pd.DataFrame
        """
        n_c = len(self.classes)
        projection = self.projections(lda_ax=n_c-1).to_numpy()
        k_stack, _ = self.stack_k_combinations(sp_df)
        projected_data = lda.project_data(k_stack, projection)
        cat_list = sp_df.Category
        projected_df = pd.DataFrame(
                            data=projected_data,
                            index=sp_df.index,
                            columns=self.spc_ids())
        projected_df = pd.concat([cat_list, projected_df], axis=1)
        if spc_id is not None:
            projected_df = projected_df[['Category', spc_id]]
        return projected_df

    def labelled_predictions(self,
            spectral_parameters: SpectralParameters,
            predictions_bool: pd.DataFrame) -> pd.DataFrame:
        """Convert Boolean prediction table with category labels, for more
        convenient visual inspection of prediction results.
        Not critical to functionality.

        :param predictions_bool: table giving predicted labels, represented by
            boolean logic, for each spectral parameter combination
        :type predictions_bool: pd.DataFrame
        :return: table giving the predicted label of each sample, for each
            spectral parameter combination
        :rtype: pd.DataFrame
        """
        cat_list = spectral_parameters.main_df.Category
        data_id = spectral_parameters.main_df.index
        index = pd.MultiIndex.from_arrays(
                                [data_id, cat_list],
                                names=['Data ID', 'Actual Category'])
        bool_labels = self.cat2bool_dict()
        inv_bool_labels = {bool: label for label, bool in bool_labels.items()}
        predictions_df = pd.DataFrame(data=predictions_bool.to_numpy(), index=index)
        predictions_df.replace(inv_bool_labels, inplace=True)
        return predictions_df

    def binary_classifier_accuracy(self,
            predictions_bool: pd.DataFrame,
            cat_list: pd.Series,
            apply: bool=True,
            ) -> pd.DataFrame:
        """Compute the classification accuracy for the given LDA classifiers,
        passed via the spectral parameter combination dataframe.

        :param predictions_bool: table giving predicted labels, represented by
            boolean logic, for each spectral parameter combination
        :type predictions_bool: pd.DataFrame
        :param cat_list: category of each entry of the dataset
        :type cat_list: pd.Categories
        :param apply: add the results to the main dataframe, defaults to False
        :type apply: bool, optional
        :return: table of accuracy metrics for each spectral parameter
            combination
        :rtype: pd.DataFrame
        """
        # compute confusion matrix
        actual_bool = self.categories_as_bools(cat_list)
        tp, tn, fp, fn = self.confusion_matrix(predictions_bool.to_numpy(), actual_bool)

        acc, ppv, tpr, tnr = self.accuracy_metrics(tp, tn, fp, fn)


        # add results to spectral parameter combination dataframe
        data = {'lda_acc_'+str(1): acc.T,
         'lda_ppv_'+str(1): ppv.T,
         'lda_tpr_'+str(1): tpr.T,
         'lda_tnr_'+str(1): tnr.T,
         'lda_fpr_'+str(1): 1 - tnr.T}

        acc_df = pd.DataFrame(data=data, index=self.spc_ids())

        # if column names exist already, overwrite these when concating
        if apply:
            if set(acc_df.columns.to_list()).issubset(set(self.main_df.columns.to_list())):
                self.main_df[acc_df.columns] = acc_df
            else:
                self.main_df = pd.concat([self.main_df, acc_df], axis=1)

        return acc_df

    @staticmethod
    def confusion_matrix(
            predicted: np.array,
            actual: np.array,
            summed: bool = True)->Tuple[np.array, np.array, np.array, np.array]:
        """Compute a table of the confusion matrix metrics, for each spectral
        parameter combination.

        :param predicted: table giving predicted labels, represented by
            boolean logic, for each spectral parameter combination
        :type predicted: np.array
        :param actual: vector giving the booleans of the actual labels
        :type actual: np.array
        :param sum: if true, return the metrics totalled over all samples,
            otherwise return tables giving the result for each sample,
            defaults to True
        :type sum: bool, optional
        :return: vectors giving accuracy metrics for each spectral parameter
            combination. Metrics are:
                - true-positives
                - true-negatives
                - false-postives
                - false-negatives
        :rtype: Tuple[np.array, np.array, np.array, np.array]
        """
        # true-positive or true-negative boolean list
        tptn = predicted == actual
        # compute confusion matrix
        # True-Positive: predicted * actual * tptn
        tp = np.array(predicted * actual * tptn, dtype=int)
        # True-Negative: ~predicted * ~actual * tptn
        tn = np.array((-1*predicted+1) * (-1*actual+1) * tptn, dtype=int)
        # False Positive: predicted * ~actual * ~tptn
        fp = np.array(predicted * (-1*actual+1) * (-1*tptn+1), dtype=int)
        # False Negative: ~predicted * actual * ~tptn
        fn = np.array((-1*predicted+1) * actual * (-1*tptn+1),dtype=int)
        if summed: # tally the total number for each metric
            tp = tp.sum(axis=0)
            tn = tn.sum(axis=0)
            fp = fp.sum(axis=0)
            fn = fn.sum(axis=0)
        return tp, tn, fp, fn

    @staticmethod
    def accuracy_metrics(
            tp: np.array, tn: np.array,
            fp: np.array, fn: np.array) ->  \
                Tuple[np.array, np.array, np.array, np.array]:
        """Compute the accuracy metrics of:
            - accuracy
            - precision
            - sensitivity
            - specificity
        from the true-positive, true-negative, false-positive and false-
        negative arrays.

        :param tp: true-positive
        :type tp: np.array
        :param tn: true-negative
        :type tn: np.array
        :param fp: false-positive
        :type fp: np.array
        :param fn: false-negative
        :type fn: np.array
        :return: accuracy metrics
        :rtype: Tuple[np.array, np.array, np.array, np.array, np.array]
        """
        zero_init = np.zeros_like(tp+tn, dtype=float)
        accuracy = np.divide(
                        tp + tn, (tp + tn + fp + fn),
                        out=zero_init, where=(tp + tn + fp + fn)!=0.0)
        zero_init = np.zeros_like(tp, dtype=float)
        precision = np.divide(
                        tp, (tp + fp),
                        out=zero_init, where=(tp + fp)!=0.0)
        zero_init = np.zeros_like(tp, dtype=float)
        sensitivity = np.divide(
                        tp, (tp + fn),
                        out=zero_init, where=(tp + fn)!=0.0)
        zero_init = np.zeros_like(tn, dtype=float)
        specificity = np.divide(
                        tn, (tn + fp),
                        out=zero_init, where=(tn + fp)!=0.0)
        return accuracy, precision, sensitivity, specificity

    # """Ranking and Sorting"""

    def rank_spcs(self, metric: str, scope: str) -> pd.DataFrame:
        """Rank the spectral parameter combinations of the classifier
        according to the given metric of either lda_score or accuracy, or
        both of these.
        If the stats and trials dataframes are populated, sync these.

        :param metric: Choice of 'lda_score' or 'accuracy' to sort over
        :type metric: str
        :param scope: Choice of 'all-data', 'mean' or 'var', to select whether
            the metric has been computed over all-data, or repeat-holdout.
        :type scope: str
        :return: the ranked spectral parameter combination dataframe
        :rtype: pd.DataFrame
        """
        if scope in ('mean', 'var'):
            try:
                rank_df = self.stats_df[metric][scope].copy()
            except ValueError as exc:
                raise ValueError(f'{metric} has not been evaluated yet') from exc
        elif scope == 'all-data':
            try:
                rank_df = self.main_df[metric].copy()
            except ValueError as exc:
                raise ValueError(f'{metric} has not been evaluated yet') from exc
        else:
            raise ValueError(f'Scope of {scope} not recognised.')

        rank = rank_df.rank(ascending=False, method='min')
        percentile = rank_df.rank(pct=True, method='max')*100

        self.main_df = self.main_df.loc[rank_df.index]

        try:
            self.main_df.insert(0, f'pct_{metric}_{scope}', percentile)
            self.main_df.insert(0, f'rank_{metric}_{scope}', rank)
        except ValueError:
            self.main_df[f'pct_{metric}_{scope}'] = percentile
            self.main_df[f'rank_{metric}_{scope}'] = rank


        if self.stats_df is not None:
            try:
                self.stats_df.insert(0, f'pct_{metric}_{scope}', percentile)
                self.stats_df.insert(0, f'rank_{metric}_{scope}', rank)
            except ValueError:
                self.stats_df[f'pct_{metric}_{scope}'] = percentile
                self.stats_df[f'rank_{metric}_{scope}'] = rank
        if self.trials_df is not None:
            try:
                self.trials_df.insert(0, f'pct_{metric}_{scope}', percentile)
                self.trials_df.insert(0, f'rank_{metric}_{scope}', rank)
            except ValueError:
                self.trials_df[f'pct_{metric}_{scope}'] = percentile
                self.trials_df[f'rank_{metric}_{scope}'] = rank

    def de_rank(self):
        """Removes the rank from tables, and re-orders to SPC ID
        """
        if 'rank' not in self.main_df.columns:
            print('Rank has not yet been applied.')
            return

        self.main_df.drop('rank', axis=1, inplace=True)
        self.main_df.sort_index(inplace=True)

        if self.stats_df is not None:
            self.stats_df.drop('rank', axis=1, inplace=True)
            self.stats_df.sort_index(inplace=True)

        if self.trials_df is not None:
            self.trials_df.drop('rank', axis=1, inplace=True)
            self.trials_df.sort_index(inplace=True)

    def top_ranks(self,
            metric: str,
            scope: str,
            top_n: int=5,
            n_uniq_fltrs: int=None) -> List:
        """Get a list of the top N ranking spectral parameter combination IDs
        for each specified number of unique filter channels.

        :param metric: The metric to use for ranking: lda_score for Fisher Ratio
            or lda_acc_1 for Classification Accuracy
        :type metric: str
        :param scope: The scope of the metric to use for ranking: mean for the
            mean of the metric over the holdout repeats, or all-data for the
            metric computed on the whole dataset
        :type scope: str
        :param top_n: number of top ranks to plot, defaults to 5
        :type top_n: int, optional
        :param n_uniq_fltrs: _description_, defaults to None
        :type n_uniq_fltrs: int, optional
        :param scope: _description_, defaults to 'all'
        :type scope: str, optional
        """
        r_m_s = f'rank_{metric}_{scope}'
        if n_uniq_fltrs is None:
            # get the spc_id and metric and scope ranks, order and filters, get spc_ids
            top_ranks = self.main_df[r_m_s].sort_values(
                                    ascending=True)[:top_n].index.to_list()
            # check for multiple equal top ranks
            eq_top_rank_ids = self.main_df[self.main_df[r_m_s] ==
                                                    self.main_df[r_m_s].max()]
            n_top_ranks = len(eq_top_rank_ids)
            if n_top_ranks > 1:
                print(f"WARNING: {n_top_ranks} equal top ranked SPCs")
            eq_top_ranks = eq_top_rank_ids.index.to_list()
        else:
            n_u_f_df = self.n_uniq_fltrs()
            top_ranks = []
            eq_top_ranks = {}
            for n_uniq_fltr in n_uniq_fltrs:
                n_u_f_df = self.main_df[self.main_df['n_uniq_fltrs'] == n_uniq_fltr]
                top_ranks = top_ranks + n_u_f_df[r_m_s].sort_values(
                                        ascending=True)[:top_n].index.to_list()
                # check for multiple equal top ranks
                eq_top_rank_ids = n_u_f_df[n_u_f_df[r_m_s] == n_u_f_df[r_m_s].min()].index.to_list()
                n_top_ranks = len(eq_top_rank_ids)
                if n_top_ranks > 1:
                    print(f"WARNING: {n_top_ranks} equal top ranked SPCs for NUC = {n_uniq_fltr}")
                eq_top_ranks[n_uniq_fltr] = eq_top_rank_ids
        return top_ranks, eq_top_ranks

    def top_spc_per_nuc(self,
            metric: str,
            scope: str,
            count_equal: bool=False,
            export_table: bool=True) -> pd.DataFrame:
        """Get top SPC IDs for each number of unique filters, when ranked
        by the given metric over the given scope. Optionally include
        equal valued ranks.

        :param metric: The metric to use for ranking: lda_score for Fisher Ratio
            or lda_acc_1 for Classification Accuracy
        :type metric: str
        :param scope: The scope of the metric to use for ranking: mean for the
            mean of the metric over the holdout repeats, or all-data for the
            metric computed on the whole dataset
        :type scope: str
        :param count_equal: Option for counting equal valued SPCs for the given
            metric and scope, defaults to False
        :type count_equal: bool, optional
        :param export_table: Option for saving the results, defaults to True
        :type export_table: bool, optional
        :return: Results table
        :rtype: pd.DataFrame
        """

        top_ranks, top_eq_ranks = self.top_ranks(
                metric=metric,
                scope=scope,
                top_n=1,
                n_uniq_fltrs=[1,2,3,4,5,6])

        if count_equal:
            spc_ids = sum([top_rank for top_rank in top_eq_ranks.values()], [])
        else:
            spc_ids = top_ranks

        top_spcs = self.main_df.loc[spc_ids].sort_values('n_uniq_fltrs')
        top_nucs = self.main_df.loc[top_spcs.index]['n_uniq_fltrs']
        top_ufc = self.main_df.loc[top_spcs.index]['uniq_fltrs']
        top_sps = self.main_df.loc[top_spcs.index][['sp_0','sp_1']]
        top_pct = self.main_df.loc[top_spcs.index][f'pct_{metric}_{scope}']

        if scope == 'mean':
            top_scores = self.stats_df.loc[top_spcs.index][(metric, scope)]
        elif scope == 'all-data':
            top_scores = self.main_df.loc[top_spcs.index][metric]

        top_spc_per_nuc = pd.concat([
                                top_scores,
                                top_nucs,
                                top_ufc,
                                top_sps,
                                top_pct], axis=1)
        top_spc_per_nuc.rename(columns={(metric, scope): metric,
                        f'pct_{metric}_{scope}': f'pct_{metric}'}, inplace=True)

        if metric == 'lda_score':
            if scope == 'mean':
                top_spc_per_nuc['scope'] = '$FR_{\mu}$'
            elif scope == 'all-data':
                top_spc_per_nuc['scope'] = '$FR_{D}$'
        elif metric == 'lda_acc_1':
            if scope == 'mean':
                top_spc_per_nuc['scope'] = '$ACC_{\mu}$'
            elif scope == 'all-data':
                top_spc_per_nuc['scope'] = '$ACC_{D}$'

        if export_table:
            # print/export the SPCs for each NUC
            table_dir = Path(self.object_dir / 'tables')
            table_dir.mkdir(parents=True, exist_ok=True)
            csv_out_file = Path(table_dir, f'top_{metric}_{scope}_per_nuc.csv')
            top_spc_per_nuc.to_csv(csv_out_file)

        return top_spc_per_nuc

    #  """Plotting"""

    @staticmethod
    def stretch_axis(
            start_lims: Tuple[float, float],
            stretch: float) -> Tuple[float, float]:
        """Stretch the given axis limits by the given amount

        :param start_lims: initial axis limits
        :type start_lims: Tuple[float, float]
        :param stretch: stretch factor
        :type stretch: float
        :return: stretched axis limits
        :rtype: Tuple[float, float]
        """
        lo_lim = start_lims[0] - stretch*(start_lims[1] - start_lims[0])
        hi_lim = start_lims[1] + stretch*(start_lims[1] - start_lims[0])
        stretched_lims = (lo_lim, hi_lim)
        return stretched_lims

    @staticmethod
    def draw_discriminants(
            ax: plt.Axes,
            projection: np.array) -> Tuple[np.array, np.array]:
        """Get x and y coordinates for drawing the discriminant functions on the
        the spectral parameters plot.

        :param ax: plot ax
        :type ax: plt.Axes
        :param projection: projection coefficients of the LDA model
        :type projection:
        :return: _description_
        :rtype: Tuple[np.array, np.array]
        """
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        x_mid = np.mean(x_lim)
        y_mid = np.mean(y_lim)
        # set offset of projection line to cut middle of plot
        a_0 = projection[0]
        a_1 = projection[1]
        c = y_mid - (a_1 / a_0) * x_mid
        x_line = [ min(x_lim), max(x_lim) ]
        y_line = [ (a_1 / a_0) * min(x_lim) + c, (a_1 / a_0) * max(x_lim) + c]
        return x_line, y_line

    @staticmethod
    def draw_boundary(
            ax: plt.Axes,
            projection: np.array, boundary) -> Tuple[np.array, np.array]:
        """Get x and y coordinates for drawing the decision boundary on the
        the spectral parameters plot.

        :param ax: plot ax
        :type ax: plt.Axes
        :param projection: projection coefficients of the LDA model
        :type projection:
        :return: _description_
        :rtype: Tuple[np.array, np.array]
        """
        x_lim = ax.get_xlim()
        a_0 = projection[0]
        a_1 = projection[1]
        x_b = np.linspace(min(x_lim), max(x_lim), 100)
        y_b = (boundary / a_1) - (a_0 / a_1)*x_b
        return x_b, y_b

    def sp_combi_plot_note(self,
            spc_id: int,
            spectral_parameters: SpectralParameters,
            metric: str,
            scope: str) -> Tuple[str, str]:
        """Build spectral parameter combination plot title and annotation

        :param spc_id: spectral parameter combination ID
        :type spc_id: int
        :param spectral_parameters: spectral parameters object hosting the
            list of spectral parameters to combine.
        :type spectral_parameters: SpectralParameters
        :param scope: the scope of the dataset - i.e. test, train, or all,
            defaults to 'all'
        :type scope: str, optional
        :return: plot annotation
        :rtype: str
        """
        spc_df = self.main_df

        try:
            rank = int(spc_df[f'rank_{metric}_{scope}'][spc_id])
            pct = spc_df[f'pct_{metric}_{scope}'][spc_id]
        except KeyError:
            rank = 'NA'
            pct = 'NA'
        n_u_f = spc_df['n_uniq_fltrs'][spc_id]
        uniq_sp_list = self.uniq_sp_list(spc_id)
        sp_str_list = '_v_'.join(uniq_sp_list)
        filter_list = spectral_parameters.sp_filters[uniq_sp_list]
        filter_str_list = ', '.join(filter_list)
        try:
            acc = spc_df['lda_acc_1'][spc_id]
            note = f'Rank {rank} ({pct:.1} P.R.) by {metric}-{scope},'\
                    f' Accuracy {acc:.3}, '\
                    f' #Filters {n_u_f}: {filter_str_list},'\
                    f' SPs: {sp_str_list}'
        except KeyError:
            acc = None
            note = f'Rank {rank} ({pct} P.R.) by {metric}-{scope},'\
                    f' #Filters {n_u_f}: '\
                    f' {filter_str_list}, SPs: {sp_str_list}'

        filename = f'nuc_{n_u_f}_rank_{metric}-{scope}_{rank}_{sp_str_list}'

        title = f'ID {spc_id}'

        return title, filename, note

    def plot_sp_combo(self,
            spc_id: int,
            spectral_parameters: SpectralParameters,
            metric: str=None,
            scope: str=None):
        """Plot data for the given single spectral parameter combination, and
        the projected data onto the fitted LDA projection.
        Also indicate decision boundary for the LDA classifier.

        :param spc_id: spectral parameter combination ID
        :type spc_id: int
        :param spectral_parameters: spectral parameters object hosting the
            list of spectral parameters to combine.
        :type spectral_parameters: SpectralParameters
        :param scope: the scope of the dataset - i.e. test, train, or all,
            defaults to 'all'
        :type scope: str, optional
        """

        # export the data plotted
        self.export_sp_combo(spc_id, spectral_parameters, metric=metric, scope=scope)

        SPCClassifier = SpectralParameterCombinationClassifier
        spc_df = self.main_df

        # get spectral parameter details
        uniq_sp_list = self.uniq_sp_list(spc_id)
        n_uniq_sps = len(uniq_sp_list)

        # get spectral parameter values
        sp_df = spectral_parameters.main_df
        sp_data = sp_df[uniq_sp_list + ['Category']] # need category list also

        # Set the plot title
        title, filename, _ = self.sp_combi_plot_note(
                                        spc_id,
                                        spectral_parameters,
                                        metric,
                                        scope)

        # set up seaborn
        sns.set_style("white")
        sns.set_style("ticks")
        sns.set_context(
                'paper',
                rc={'figure.dpi': cfg.DPI,
                'figure.autolayout': True})

        # single spectral parameter - histogram
        if n_uniq_sps == 1:
            x_feat = spc_df['sp_0'][spc_id]
            _, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            g = sns.histplot(
                    data=sp_data,
                    x=x_feat,
                    hue='Category',
                    palette = 'colorblind',
                    bins=25,
                    ax=ax)
            ax.set_title(title+' SPC Space', fontsize=cfg.TITLE_S)
            ax.set_xlabel(ax.get_xlabel(),fontsize=cfg.LABEL_S)
            ax.set_ylabel(ax.get_ylabel(), fontsize=cfg.LABEL_S)
            boundary = spc_df['lda_boundary_1'][spc_id]
            ax.axvline(boundary, ls='--', color='k')
            plt.setp(ax.get_legend().get_texts(), fontsize=cfg.LEGEND_S)
            plt.setp(ax.get_legend().get_title(), fontsize=cfg.LEGEND_S)

        # pair of unique spectral parameters - scatterplot
        elif n_uniq_sps == 2:
            x_feat = spc_df['sp_0'][spc_id]
            y_feat = spc_df['sp_1'][spc_id]

            g = sns.jointplot(
                    data=sp_data,
                    x=x_feat,
                    y=y_feat,
                    hue='Category',
                    height=cfg.FIG_SIZE[0],
                    palette='colorblind')
            g.fig.suptitle(title+' SPC Space', fontsize=cfg.TITLE_S)
            g.ax_joint.set_xlabel(x_feat, fontsize=cfg.LABEL_S)
            g.ax_joint.set_ylabel(y_feat, fontsize=cfg.LABEL_S)

            # overplot the discriminant function(s)
            ax = g.ax_joint
            x_lim = SPCClassifier.stretch_axis(ax.get_xlim(), 0.2)
            y_lim = SPCClassifier.stretch_axis(ax.get_ylim(), 0.2)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            projection = self.projections().loc[spc_id].to_numpy()
            x_line, y_line = SPCClassifier.draw_discriminants(ax, projection)
            ax.plot(
                x_line,
                y_line,
                color='k',
                label = 'projection')

            # overplot the decision boundary
            boundary = spc_df['lda_boundary_1'][spc_id]
            x_b, y_b = SPCClassifier.draw_boundary(ax, projection, boundary)
            ax.plot(x_b, y_b, color='k', linestyle='--', label = 'boundary')

            handles, labels = g.ax_joint.get_legend_handles_labels()
            g.ax_joint.legend(
                handles=handles,
                labels=labels,
                fontsize=cfg.LEGEND_S,
                loc='upper left')

            # text giving the SPC LDA parameters
            a_x = spc_df['lda_a_0_1'][spc_id]
            a_y = spc_df['lda_a_1_1'][spc_id]
            b = spc_df['lda_boundary_1'][spc_id]
            tg_gt_bg = spc_df['lda_tg_gt_bg_1'][spc_id]
            text = f"LDA=${a_x:.3}x {a_y:+.3}y$ \n $b= {b:.3}$ \n tg$>$bg= {tg_gt_bg}"
            ax.text(1,0.95,text,
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    fontsize=cfg.LEGEND_S,
                    fontdict={'family': 'serif'},
                    bbox=dict(facecolor='white', alpha=0.8))

        # Pair-plots of scatterplots of pairs of unique spectral parameters
        elif n_uniq_sps > 2:
            raise ValueError('Only pair combinations supported so far.')

        plt.tight_layout()
        plot_dir = Path(self.object_dir / 'spc_plots')
        plot_dir.mkdir(parents=True, exist_ok=True)
        feature_output_file = Path(plot_dir, filename).with_suffix(cfg.PLT_FRMT)
        plt.savefig(feature_output_file, dpi=cfg.DPI)

    def plot_spc_lda(self,
            spc_id: int,
            spectral_parameters: SpectralParameters,
            metric: str=None,
            scope: str=None):
        """Plot the given SPC in LDA Space - i.e. with the data from each
        component spectral parameter projected onto the LDA axis, according
        to the fitted LDA model.

        :param spc_id: Spectral Parameter Combination Identifier
        :type spc_id: int
        :param spectral_parameters: spectral parameters object hosting the
            evaluated list of spectral parameters to combine.
        :type spectral_parameters: SpectralParameters
        :param metric: , defaults to None
        :type metric: str, optional
        :param scope: _description_, defaults to None
        :type scope: str, optional
        :raises ValueError: _description_
        :raises ValueError: _description_
        """

        # export the data in the LDA space
        self.export_spc_lda(spc_id, spectral_parameters, metric=metric, scope=scope)

        # get the lda features
        n_ldas = len(self.classes) - 1

        sp_data = spectral_parameters.main_df

        # project the data for the given LDA features
        projected_df = self.project_dataset(sp_data, spc_id=spc_id)
        projected_df.rename(columns={spc_id: 'LDA'}, inplace=True)

        # Set the plot title
        title, filename, _ = self.sp_combi_plot_note(
                                        spc_id,
                                        spectral_parameters,
                                        metric,
                                        scope)

        sns.set_style('white')
        sns.set_style('ticks')
        sns.set_context(
                'paper',
                rc={'figure.dpi': cfg.DPI,
                'figure.autolayout': True})

        # 2-Class, 1 LDA Axis case - histogram plot
        if n_ldas == 1:

            # set x-axis label
            sp_x = self.main_df['sp_0'][spc_id]
            sp_y = self.main_df['sp_1'][spc_id]
            if sp_x == sp_y:
                x_feat = sp_x
            else:
                x_feat = f'LDA({sp_x}, {sp_y})'

            fig, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            g = sns.histplot(
                    data=projected_df,
                    x='LDA',
                    hue='Category',
                    palette='colorblind',
                    bins=25,
                    ax=ax)
            ax.set_xlabel(x_feat, fontsize=cfg.LABEL_S)
            ax.set_ylabel('Count', fontsize=cfg.LABEL_S)
            ax.set_title(title+' LDA Space', fontsize=cfg.TITLE_S)
            # add the bounary line
            boundary = self.main_df['lda_boundary_1'][spc_id]

            g.axvline(boundary, ls='--', color='k')
            sns.move_legend(g, loc='upper left')

            plt.setp(ax.get_legend().get_texts(), fontsize=cfg.LABEL_S)
            ax.get_legend().set_title('')

            # add text giving the FR mean, var, rank and the ACC mean, var and rank
            if 'rank_lda_score_mean' in self.main_df.columns:
                fr_mu_rank = int(self.main_df['rank_lda_score_mean'][spc_id])
                fr_mu = self.stats_df[('lda_score','mean')][spc_id]
                fr_sig = np.sqrt(self.stats_df[('lda_score','var')][spc_id])
                acc_mu_rank = int(self.main_df['rank_lda_acc_1_mean'][spc_id])
                acc_mu = self.stats_df[('lda_acc_1','mean')][spc_id]
                acc_sig = np.sqrt(self.stats_df[('lda_acc_1','var')][spc_id])
                text = f'Rank by $FR_{{\mu}}$: #{fr_mu_rank} \n $FR_{{\mu}}$: {fr_mu:.3} $\pm$ {fr_sig:.2} \n Rank by $ACC_{{\mu}}$: #{acc_mu_rank} \n $ACC_{{\mu}}$: {acc_mu:.3} $\pm$ {acc_sig:.2} \n'
                ax.text(0.97,0.95,text,
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform=ax.transAxes,
                        fontsize=cfg.LEGEND_S,
                        fontdict={'family': 'serif'},
                        bbox=dict(facecolor='white', alpha=0.8))

        # >2-Class, 2 LDA Axes case - joint plot the two LDA axes
        elif n_ldas == 2:
            raise ValueError('Multi-class case not yet supported')
            # g = sns.jointplot(
            #         data=projected_df,
            #         x='LDA1',
            #         y='LDA2',
            #         hue='Category',
            #         height=11.5*cfg.CM,
            #         palette='colorblind')
            # g.fig.suptitle(lda_title)

        # >2-Class, >2 LDA Axes case - joint plot pair combinations of the LDAs
        elif n_ldas > 2:
            raise ValueError('Multi-class case not yet supported')
            # lda_labels = [f'LDA{i}' for i in range(1, n_ldas+1)]
            # g = sns.pairplot(
            #         data=projected_df,
            #         x_vars=lda_labels,
            #         y_vars=lda_labels,
            #         hue='Category',
            #         palette = 'colorblind',
            #         height=4*cfg.CM,
            #         aspect=1,
            #         corner=True)
            # g.fig.suptitle(lda_title)

        plt.tight_layout()
        plot_dir = Path(self.object_dir / 'lda_plots')
        plot_dir.mkdir(parents=True, exist_ok=True)
        lda_output_file = Path(plot_dir, filename+'_LDA').with_suffix(cfg.PLT_FRMT)
        fig.savefig(lda_output_file, dpi=cfg.DPI)

    def plot_top_ranks(self,
            spectral_parameters: object,
            metric: str,
            scope: str,
            top_n: int=5,
            n_uniq_fltrs: int=None) -> List:
        """Plot the top N ranking spectral parameter combinations

        :param top_n: number of top ranks to plot, defaults to 5
        :type top_n: int, optional
        """
        top_ranks, _ = self.top_ranks(
                metric=metric,
                scope=scope,
                top_n=top_n,
                n_uniq_fltrs=n_uniq_fltrs)
        for spc_id in top_ranks:
            # choose only spc in top n ranks
            self.plot_sp_combo(spc_id, spectral_parameters, metric=metric, scope=scope)
            self.plot_spc_lda(spc_id, spectral_parameters, metric=metric, scope=scope)

        top_rank_stats = self.stats_df.loc[top_ranks][['rank_lda_score_mean',
                                                       'lda_score',
                                                       'rank_lda_acc_1_mean',
                                                       'lda_acc_1']]

        top_rank_sps = self.main_df.loc[top_ranks][['sp_0','sp_1',
                                                    'fltrs_0','fltrs_1',
                                                    'n_uniq_fltrs',
                                                    'uniq_fltrs',
                                                    'lda_a_0_1',
                                                    'lda_a_1_1',
                                                    'lda_boundary_1',
                                                    'lda_tg_gt_bg_1']]

        return top_rank_stats, top_rank_sps

    def plot_top_metric_per_nuc(self, top_metric: pd.DataFrame, metric: str):
        """Plot the top SPC scores of the given metric for each Number of Unique
        Channels (NUC).

        :param top_metric: DataFrame of the top SPC for each NUC
        :type top_metric: pd.DataFrame
        :param metric: Choice of 'lda_score' or 'accuracy' to sort over
        :type metric: str
        """
        col = sns.color_palette('colorblind')
        d_col = col[2]
        mu_col = col[3]
        _, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)

        top_nucs = top_metric['n_uniq_fltrs'].drop_duplicates().sort_values()

        pct_99_d = self.main_df[metric].quantile(q=0.99)
        pct_90_d = self.main_df[metric].quantile(q=0.90)
        pct_75_d = self.main_df[metric].quantile(q=0.75)
        pct_50_d = self.main_df[metric].quantile(q=0.50)

        pct_99_mu = self.stats_df[(metric, 'mean')].quantile(q=0.99)
        pct_90_mu = self.stats_df[(metric, 'mean')].quantile(q=0.90)
        pct_75_mu = self.stats_df[(metric, 'mean')].quantile(q=0.75)
        pct_50_mu = self.stats_df[(metric, 'mean')].quantile(q=0.50)

        # draw top percentile lines - as grid, labelled on second axis
        ax.axhline(pct_99_d, color=d_col, linewidth=0.5)
        ax.axhline(pct_90_d, color=d_col, linewidth=0.5)
        ax.axhline(pct_75_d, color=d_col, linewidth=0.5)
        ax.axhline(pct_50_d, color=d_col, linewidth=0.5)
        # draw top percentile lines - as grid, labelled on second axis
        ax.axhline(pct_99_mu, color=mu_col, linewidth=0.5)
        ax.axhline(pct_90_mu, color=mu_col, linewidth=0.5)
        ax.axhline(pct_75_mu, color=mu_col, linewidth=0.5)
        ax.axhline(pct_50_mu, color=mu_col, linewidth=0.5)

        col = sns.color_palette('colorblind')
        sns.lineplot(
            x='n_uniq_fltrs',
            y=metric,
            data=top_metric,
            markers=True,
            hue='scope',
            style='scope',
            palette=[d_col, mu_col],
            markersize=5
        )
        ax.set_xlabel('Number of Unique Channels',fontsize=cfg.LABEL_S)

        if metric == 'lda_score':
            ax.set_title('Top $FR$ for each NUC', fontsize=cfg.TITLE_S)
            ax.set_ylabel('Fisher Ratio', fontsize=cfg.LABEL_S)
        elif metric == 'lda_acc_1':
            ax.set_title('Top $ACC$ for each NUC', fontsize=cfg.TITLE_S)
            ax.set_ylabel('Accuracy', fontsize=cfg.LABEL_S)
        ax.set_xticks(top_nucs.values)
        ax.margins(x=0.2)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:], labels=labels[:],fontsize=cfg.LEGEND_S)

        ax.text(6.9, pct_99_d, '$P_{99}$', fontsize=cfg.LEGEND_S,
                va='bottom', ha='right', color=d_col)
        ax.text(6.9, pct_90_d, '$D_{9}$', fontsize=cfg.LEGEND_S,
                va='bottom', ha='right', color=d_col)
        ax.text(6.9, pct_75_d, '$Q_{3}$', fontsize=cfg.LEGEND_S,
                va='bottom', ha='right', color=d_col)
        ax.text(6.9, pct_50_d, '$M$', fontsize=cfg.LEGEND_S,
                va='bottom', ha='right',color=d_col)

        ax.text(0.1, pct_99_mu, '$P_{99}$', fontsize=cfg.LEGEND_S,
                va='top', ha='left', color=mu_col)
        ax.text(0.1, pct_90_mu, '$D_{9}$', fontsize=cfg.LEGEND_S,
                va='top',  ha='left',color=mu_col)
        ax.text(0.1, pct_75_mu, '$Q_{3}$', fontsize=cfg.LEGEND_S,
                va='top',  ha='left',color=mu_col)
        ax.text(0.1, pct_50_mu, '$M$', fontsize=cfg.LEGEND_S,
                va='top',  ha='left', color=mu_col)

        plt.tight_layout()
        plot_dir = Path(self.object_dir / 'analysis')
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(plot_dir, f'top_{metric}_nuc').with_suffix('.png')
        plt.savefig(output_file, dpi=cfg.DPI)

    def plot_roc(self,
            noisey: bool=False):
        """Plot the Reciever Operator Characteristic for the LDA classifier of
        each spectral parameter combination, producing a scatter-plot indicating
        the relationship between assigned ranking and classifier success across
        all spectral parameter combinations.

        :param noisey: indicates if the spc_df has been evalauted using a noisy
            data set, defaults to False
        :type noisey: bool, optional
        """
        fig, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)

        if self.stats_df is not None:
            x_lbl = ('lda_fpr_1', 'mean')
            y_lbl = ('lda_tpr_1', 'mean')
            hue_lbl = ('lda_acc_1', 'mean')
            cbar_lbl = 'Mean Accuracy'
            rankd_df = self.stats_df.sort_values(hue_lbl, ascending=False)
        else:
            x_lbl = 'lda_fpr_1'
            y_lbl = 'lda_tpr_1'
            hue_lbl = 'lda_acc_1'
            cbar_lbl = 'Accuracy'
            rankd_df = self.main_df.sort_values(('lda_acc_1'), ascending=False)

        norm = plt.Normalize(rankd_df[hue_lbl].min(), rankd_df[hue_lbl].max())
        smap = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        smap.set_array([])

        sns.despine(offset = 10)
        roc = sns.scatterplot(
            data=rankd_df,
            x=rankd_df[x_lbl],
            y=rankd_df[y_lbl],
            hue=rankd_df[hue_lbl],
            palette='viridis',
            s=10,
            linewidth=0.0,
            alpha=0.7,
            hue_order=rankd_df[hue_lbl].to_list(),
            ax=ax)
        # overplot densities
        sns.kdeplot(
            data=rankd_df,
            x=rankd_df[x_lbl],
            y=rankd_df[y_lbl],
            cmap="magma",
            fill=False,
            levels=6,
            bw_adjust=0.75,
            cut=1,
            ax=ax)
        roc.plot([0,1],[0,1], '--', color = 'r')
        roc.set_xlabel('1 - Mean Specificity (FPR)')
        roc.set_ylabel('Mean Sensitivity (TPR)')
        ax.get_legend().remove()
        cbar = ax.figure.colorbar(smap, label=cbar_lbl)
        cbar.ax.tick_params(labelsize=cfg.LEGEND_S)

        # export
        plot_dir = Path(self.object_dir / 'analysis')
        plot_dir.mkdir(parents=True, exist_ok=True)
        if noisey:
            roc.set_title('Noisey ROC Scatterplot')
            output_file = Path(plot_dir,
                            'roc_scatter_noisey').with_suffix(cfg.PLT_FRMT)
        else:
            roc.set_title('ROC Scatterplot')
            output_file = Path(plot_dir,
                            'roc_scatter').with_suffix(cfg.PLT_FRMT)
        fig.savefig(output_file,bbox_inches='tight', dpi=cfg.DPI)

    def plot_mean_cv(self,
            metric: str,
            noisey: bool=False):
        """Plot the mean-CV (coefficient fo variation) relationship of the
        given metric, where the metric is either the accuracy or the Fisher
        Ratio. Whichever metric that is not selected is used to give the
        hue.

        :param metric: accuracy or Fisher Ratio
        :type metric: str
        :param noisey: indicates if the spc_df has been evalauted using a noisy
            data set, defaults to False
        :type noisey: bool, optional
        """

        fig, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)

        if metric == 'accuracy':
            mean_metric = 'lda_acc_1'
            cv_metric = 'lda_acc_1'
            mean_label = 'Accuracy'
            cv_label = 'Accuracy'
            hue_metric = 'lda_score'
            hue_label = 'Fisher Ratio'
            asc = True
        elif metric == 'fisher_ratio':
            mean_metric = 'lda_score'
            cv_metric = 'lda_score'
            mean_label = 'Fisher Ratio'
            cv_label = 'Fisher Ratio'
            hue_metric = 'lda_acc_1'
            hue_label = 'Accuracy'
            asc = False
        elif metric == 'accuracy_vs_fisher ratio':
            mean_metric = 'lda_acc_1'
            cv_metric = 'lda_score'
            mean_label = 'Accuracy'
            cv_label = 'Fisher Ratio'
            hue_metric = 'lda_score'
            hue_label = 'Fisher Ratio'
            asc = True
        elif metric == 'fisher_ratio_vs_accuracy':
            mean_metric = 'lda_score'
            cv_metric = 'lda_acc_1'
            mean_label = 'Fisher Ratio'
            cv_label = 'Accuracy'
            hue_metric = 'lda_acc_1'
            hue_label = 'Accuracy'
            asc = False
        else:
            raise ValueError('Metric not recognised.'\
                              'Please use either "accuracy" or "fisher_ratio".')

        rankd_df = self.stats_df.sort_values((hue_metric,'mean'), ascending=asc)

        norm = plt.Normalize(rankd_df[(hue_metric,'mean')].min(),
                                            rankd_df[(hue_metric,'mean')].max())
        smap = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        smap.set_array([])

        sns.despine(offset = 10)
        mvp = sns.scatterplot(
            data=rankd_df,
            x=rankd_df[(cv_metric, 'var')]**(1.0/2) / rankd_df[(cv_metric, 'mean')],
            y=rankd_df[(mean_metric, 'mean')],
            hue=rankd_df[(hue_metric, 'mean')],
            palette='viridis',
            s=10,
            linewidth=0.0,
            alpha=0.7,
            hue_order=rankd_df[(hue_metric,'mean')].to_list(),
            ax=ax)
        # # overplot densities
        # sns.kdeplot(
        #     data=rankd_df,
        #     x=rankd_df[('lda_fpr_1', 'mean')],
        #     y=rankd_df[('lda_tpr_1', 'mean')],
        #     cmap="magma",
        #     shade=False,
        #     levels=6,
        #     ax=ax)
        mvp.set_xlabel('Coefficient of Variation of '+cv_label)
        mvp.set_ylabel('Mean '+mean_label)
        mvp.set_xscale('log')
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.get_legend().remove()
        cbar = ax.figure.colorbar(smap, label='Mean '+ hue_label)
        cbar.ax.tick_params(labelsize=cfg.LEGEND_S)

        # export
        plot_dir = Path(self.object_dir / 'analysis')
        plot_dir.mkdir(parents=True, exist_ok=True)
        if noisey:
            mvp.set_title(f'Noisey Mean vs. Coeff. of Variation of {cv_label}')
            output_file = Path(plot_dir,
                            'mean_cv_'+metric+'_noisey').with_suffix(cfg.PLT_FRMT)
        else:
            mvp.set_title(f'Mean vs. Coeff. of Variation of {cv_label}')
            output_file = Path(plot_dir,
                            'mean_cv_'+metric).with_suffix(cfg.PLT_FRMT)
        fig.savefig(output_file,bbox_inches='tight', dpi=cfg.DPI)

    def metric_vs_metric_regession(self,
            metric1: Tuple[str, str],
            metric2: Tuple[str, str]) -> float:
        """Compute the regression between the two given metrics

        :param metric1: First metric
        :type metric1: Tuple[str, str]
        :param metric2: Second metric
        :type metric2: Tuple[str, str]
        :return: R^2 correlation coefficient
        :rtype: float
        """

        if self.stats_df is not None:
            x = self.stats_df[metric1[0]]['mean']
            y = self.stats_df[metric2[0]]['mean']
            y_err = self.stats_df[metric2[0]]['var']
            weights = 1/y_err.to_numpy()
        else:
            x = self.stats_df[metric1[0]]['mean']
            y = self.stats_df[metric2[0]]['mean']
            weights = None

        # allow to apply a function to each metric
        if len(metric1) == 3:
            x = x.apply(metric1[2])
        if len(metric2) == 3:
            y = y.apply(metric2[2])

        # using statsmodel
        y_sm = y.to_numpy()
        x_sm = x.to_numpy()
        p_sm = x_sm.argsort()
        x_sm = x_sm[p_sm]
        y_sm = y_sm[p_sm]

        if self.stats_df is not None:
            weights = weights[p_sm]

        x_sm = sm.add_constant(x_sm)
        wls_model = sm.WLS(y_sm,x_sm, weights=weights)
        sm_fit = wls_model.fit()
        # coeffs = sm_fit.params
        # std_err = sm_fit.bse
        r_sqr = sm_fit.rsquared

        return r_sqr

    def plot_metric_vs_metric(self,
            metric1: Tuple[str, str],
            metric2: Tuple[str, str]):
        """Plot one spectral parameter combination LDA classifier metric against
        another, producing a joint density plot showing the distribution of
        values across all spectral parameter combinations.

        :param metric1: x-axis classifier parameter/metric and label
        :type metric1: Tuple[str, str]
        :param metric2: y-axis classifier parameter/metric and label
        :type metric2: Tuple[str, str]
        """
        sns.set_style('white')
        sns.set_style('ticks')

        if self.stats_df is not None:
            x = self.stats_df[metric1[0]]['mean']
            y = self.stats_df[metric2[0]]['mean']
            # y_err = self.stats_df[metric2[0]]['var']
        else:
            x = self.main_df[metric1[0]]
            y = self.main_df[metric2[0]]

        # allow to apply a function to each of the metrics
        if len(metric1) == 3:
            x = x.apply(metric1[2])
            x_label = metric1[0]+'_'+str(metric1[2])
        else:
            x_label = metric1[0]
        if len(metric2) == 3:
            y = y.apply(metric2[2])
            y_label = metric2[0]+'_'+str(metric2[2])
        else:
            y_label = metric2[0]

        mm_plt = sns.jointplot(
            data=self.main_df,
            x=x,
            y=y,
            kind='scatter',
            s=10,
            linewidth=0,
            alpha=0.7,
            marker='.',
            # ratio=3,
            marginal_kws=dict(bins=50),
            height=cfg.FIG_SIZE[1])
        mm_plt.plot_joint(sns.kdeplot, cmap="magma", fill=False,
                cut=1, gridsize=100, levels=6)
        mm_plt.ax_joint.set_xlabel(metric1[1], fontsize=cfg.LABEL_S)
        mm_plt.ax_joint.set_ylabel(metric2[1], fontsize=cfg.LABEL_S)
        mm_plt.fig.tight_layout()

        cfg.DPI = 600
        cfg.PLT_FRMT = '.png' # '.png' or '.pdf'

        # export
        plot_dir = Path(self.object_dir / 'analysis')
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(plot_dir,
            y_label+'_vs_'+x_label).with_suffix(cfg.PLT_FRMT)
        mm_plt.fig.savefig(output_file,bbox_inches='tight', dpi=cfg.DPI)

    def plot_metric_vs_trials(self, top_n: int = 20):
        """Plot the given metric against trial number. Only plot the top_n
        LDA classifier combinations.

        :param top_n: number of top spectral parameter combinations to show in
            plot, defaults to 20
        :type top_n: int, optional
        """
        fig, ax = plt.subplots(nrows=2, ncols=2,
                                sharex=True,
                                dpi=cfg.DPI,
                                figsize=(2*cfg.FIG_SIZE[0], 2*cfg.FIG_SIZE[1]))

        self.de_rank()
        self.rank_spcs(metric='lda_score', scope='mean')

        lda_scores = self.trials_df['lda_score']
        cmlt_lda_ave = lda_scores.rolling(len(lda_scores.columns),
                                          min_periods=2,axis=1).mean()
        metric_data = cmlt_lda_ave[0:top_n].T
        metric_data.plot(
                legend=False,
                colormap='viridis_r', ax=ax[0][0])
        ax[0][0].set_xlabel('Trials', fontsize=cfg.LABEL_S)
        ax[0][0].set_ylabel('Mean', fontsize=cfg.LABEL_S)
        ax[0][0].set_title('Fisher Ratio', fontsize=cfg.TITLE_S)
        ax[0][0].grid(True)
        plt.tight_layout()

        cmlt_lda_std = lda_scores.rolling(len(lda_scores.columns),
                                          min_periods=2,axis=1).std()
        metric_data = cmlt_lda_std[0:top_n].T
        metric_data.plot(
                legend=False,
                colormap='viridis_r', ax=ax[1][0])
        ax[1][0].set_xlabel('Trials', fontsize=cfg.LABEL_S)
        ax[1][0].set_ylabel('Std. Dev.', fontsize=cfg.LABEL_S)
        ax[1][0].grid(True)
        plt.tight_layout()

        self.de_rank()
        self.rank_spcs(metric='lda_acc_1', scope='mean')

        lda_accs = self.trials_df['lda_acc_1']
        cmlt_acc_ave = lda_accs.rolling(len(lda_accs.columns),
                                        min_periods=2,axis=1).mean()
        metric_data = cmlt_acc_ave[0:top_n].T
        metric_data.plot(
                legend=False,
                colormap='viridis_r', ax=ax[0][1])
        plt.tight_layout()
        ax[0][1].set_xlabel('Trials', fontsize=cfg.LABEL_S)
        ax[0][1].set_title('Accuracy', fontsize=cfg.TITLE_S)
        ax[0][1].grid(True)

        cmlt_acc_std = lda_accs.rolling(len(lda_accs.columns),
                                        min_periods=2,axis=1).std()
        metric_data = cmlt_acc_std[0:top_n].T
        metric_data.plot(
                xlabel='Trials',
                legend=False,
                colormap='viridis_r', ax=ax[1][1])
        ax[1][1].set_xlabel('Trials', fontsize=cfg.LABEL_S)
        ax[1][1].grid(True)

        fig.suptitle('Accuracy and Fisher Ratio vs Trials',
                     fontsize=cfg.TITLE_S)
        plt.tight_layout()

        # export
        plot_dir = Path(self.object_dir / 'analysis')
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(plot_dir,
                           'lda_score'+'_vs_trials').with_suffix(cfg.PLT_FRMT)
        plt.savefig(output_file,bbox_inches='tight', dpi=cfg.DPI)

    def plot_lda_score_stats(self) -> None:
        """Plot the all-data LDA score against the LDA mean and standard
        deviation
        """
        x = self.main_df['lda_score']
        y = self.stats_df['lda_score']['mean']
        y_err = self.stats_df['lda_score']['var'].apply(np.sqrt)

        # using statsmodel
        y_sm = y.to_numpy()
        x_sm = x.to_numpy()
        p_sm = x_sm.argsort()
        x_sm = x_sm[p_sm]
        y_sm = y_sm[p_sm]
        weights = 1/y_err.to_numpy()[p_sm]

        # concordance correlation coefficient
        sxy = np.sum((x_sm - x_sm.mean())*(y_sm - y_sm.mean()))/x_sm.shape[0]
        ccc = 2*sxy / (np.var(x_sm) + np.var(y_sm) + (x_sm.mean() - y_sm.mean())**2)

        x_sm = sm.add_constant(x_sm)
        wls_model = sm.WLS(y_sm,x_sm, weights=weights)
        sm_fit = wls_model.fit()
        coeffs = sm_fit.params
        std_err = sm_fit.bse
        r_sqr = sm_fit.rsquared

        _, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        ax.errorbar(
            x, y, y_err,
            linestyle='',
            marker='.',
            ecolor='k',
            alpha=0.7,
            elinewidth=0.4,
            capsize=0.6,
            zorder=1)
        ax.plot(
            x,y,
            marker='.',
            c='w',
            linestyle='',
            markersize=5,
            markevery=10,
            alpha=0.01,
            zorder=2)
        ax.plot(
            x_sm[:,1], x_sm[:,1],
            'r:',
            zorder=3,
            label=fr'$\rho_c$ = {ccc:.4f}')
        ax.plot(
            x_sm[:,1],
            x_sm @ coeffs,
            linewidth=1,
            label=f'$r^2$ = {r_sqr:.4f}', zorder=4)
        ax.fill_between(x_sm[:,1],
                        x_sm@(coeffs - std_err), x_sm @ (coeffs + std_err))
        ax.set_xlabel('All-Data Fisher Ratio', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Repeat-Holdout Fisher Ratio', fontsize=cfg.LABEL_S)
        ax.set_title('All-Data vs Average Fisher Ratio', fontsize=cfg.TITLE_S)
        ax.legend()
        plt.setp(ax.get_legend().get_texts(), fontsize=cfg.LEGEND_S)

        ax.grid(True)

        # export
        plot_dir = Path(self.object_dir / 'analysis')
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(plot_dir,
                           'final_vs_repeatholdout_lda_score'
                           ).with_suffix(cfg.PLT_FRMT)
        plt.savefig(output_file,bbox_inches='tight', dpi=cfg.DPI)

        return {'r^2': r_sqr, 'ccc': ccc}

    def plot_accuracy_stats(self, all_data_acc: pd.DataFrame) -> None:
        """Plot the all-data accuracy against the accuracy mean and standard
        deviation.

        :param all_data_acc: accuracy results from the all-data accuracy test
        :type all_data_acc: pd.DataFrame
        """
        x = all_data_acc['lda_acc_1']
        y = self.stats_df['lda_acc_1']['mean']
        y_err = self.stats_df['lda_acc_1']['var'].apply(np.sqrt)

        # using statsmodel
        y_sm = y.to_numpy()
        x_sm = x.to_numpy()
        p_sm = x_sm.argsort()
        x_sm = x_sm[p_sm]
        y_sm = y_sm[p_sm]
        weights = 1/y_err.to_numpy()[p_sm]

        # concordance correlation coefficient
        sxy = np.sum((x_sm - x_sm.mean())*(y_sm - y_sm.mean()))/x_sm.shape[0]
        ccc = 2*sxy / (
                np.var(x_sm) + np.var(y_sm) + (x_sm.mean() - y_sm.mean())**2)

        x_sm = sm.add_constant(x_sm)

        _, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        ax.errorbar(
            x,y,y_err,
            linestyle='',
            marker='.',
            markersize=5,
            ecolor='k',
            alpha=0.7,
            elinewidth=0.4,
            capsize=0.6,
            zorder=1)
        ax.plot(
            x,y,
            marker='.',
            c='w',
            linestyle='',
            markersize=5,
            markevery=10,
            alpha=0.01,
            zorder=2)
        ax.plot(
            x_sm[:,1], x_sm[:,1],
            'r:',
            zorder=3,
            label=fr'$\rho_c$ = {ccc:.4f}')

        fit_success = True
        try:
            wls_model = sm.WLS(y_sm,x_sm, weights=weights)
        except np.linalg.LinAlgError:
            fit_success = False
        try:
            sm_fit = wls_model.fit()
        except np.linalg.LinAlgError:
            fit_success = False
        if fit_success:
            coeffs = sm_fit.params
            std_err = sm_fit.bse
            r_sqr = sm_fit.rsquared
            ax.plot(
                x_sm[:,1], x_sm @ coeffs,
                linewidth=1,
                label=f'$r^2$ = {r_sqr:.4f}', zorder=4)
            ax.fill_between(x_sm[:,1],
                            x_sm @ (coeffs - std_err), x_sm @ (coeffs + std_err))
        else:
            r_sqr = None

        ax.set_xlabel('All-Data Accuracy', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Repeat-Holdout Accuracy', fontsize=cfg.LABEL_S)
        ax.set_title('All-Data vs Average Accuracy', fontsize=cfg.TITLE_S)
        ax.grid(True)
        ax.legend()
        plt.setp(ax.get_legend().get_texts(), fontsize=cfg.LEGEND_S)

        # export
        plot_dir = Path(self.object_dir / 'analysis')
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(plot_dir,
                           'final_vs_repeatholdout_accuracy'
                           ).with_suffix(cfg.PLT_FRMT)
        plt.savefig(output_file,bbox_inches='tight', dpi=cfg.DPI)

        return {'r^2': r_sqr, 'ccc': ccc}

    #"""Export the DataFrames"""
    def export_sp_combo(self,
            spc_id: int,
            spectral_parameters: SpectralParameters,
            metric: str=None,
            scope: str=None):
        """Export data for the given single spectral parameter combination, and
        the projected data onto the fitted LDA projection.
        Also indicate decision boundary for the LDA classifier.

        :param spc_id: spectral parameter combination ID
        :type spc_id: int
        :param spectral_parameters: spectral parameters object hosting the
            list of spectral parameters to combine.
        :type spectral_parameters: SpectralParameters
        :param scope: the scope of the dataset - i.e. test, train, or all,
            defaults to 'all'
        :type scope: str, optional
        """
        # get spectral parameter details
        uniq_sp_list = self.uniq_sp_list(spc_id)

        # get spectral parameter values
        sp_df = spectral_parameters.main_df
        sp_data = sp_df[uniq_sp_list + ['Category']] # need category list also

        # Set the export file title (mirroring the plot title format)
        _, filename, _ = self.sp_combi_plot_note(
                                        spc_id,
                                        spectral_parameters,
                                        metric,
                                        scope)

        plot_dir = Path(self.object_dir / 'spc_plots')
        plot_dir.mkdir(parents=True, exist_ok=True)
        feature_output_file = Path(plot_dir,
                                   filename+'_data').with_suffix('.csv')
        sp_data.to_csv(feature_output_file)

    def export_spc_lda(self,
            spc_id: int,
            spectral_parameters: SpectralParameters,
            metric: str=None,
            scope: str=None):
        """Export the linear combination of the evaluated spectral parameters
        for each entry of the material collection for the fitted linear
        discriminant classifier.

        :param spc_id: Spectral Parameter Combination Identifier
        :type spc_id: int
        :param spectral_parameters: spectral parameters object hosting the
            evaluated list of spectral parameters to combine.
        :type spectral_parameters: SpectralParameters
        :param metric: Metric used to rank the spc_id, defaults to None
        :type metric: str, optional
        :param scope: Scope used to rank the spc_id, defaults to None
        :type scope: str, optional
        """

        sp_data = spectral_parameters.main_df

        # project the data for the given LDA features
        projected_df = self.project_dataset(sp_data, spc_id=spc_id)
        projected_df.rename(columns={spc_id: 'LDA'}, inplace=True)

        # Set the plot title
        _, filename, _ = self.sp_combi_plot_note(
                                        spc_id,
                                        spectral_parameters,
                                        metric,
                                        scope)

        plt.tight_layout()
        plot_dir = Path(self.object_dir / 'lda_plots')
        plot_dir.mkdir(parents=True, exist_ok=True)
        lda_output_file = Path(plot_dir,
                               filename+'_LDA_data').with_suffix('.csv')
        projected_df.to_csv(lda_output_file)

    def export_df(self, df_type: str='main'):
        """Export the dataframe to csv file, and pickle.

        :param pkl_only: Only output a pkl file of the DataFrame
        :type pkl_only: bool, optional
        """

        table_dir = Path(self.object_dir / 'tables')
        table_dir.mkdir(parents=True, exist_ok=True)

        if df_type in ('main', 'all'):
            pkl_file = Path(self.object_dir, 'spc_scores.pkl')
            self.main_df.to_pickle(pkl_file)
            csv_out_file = Path(table_dir, 'spc_scores.csv')
            self.main_df.to_csv(csv_out_file)
        if df_type  in ('stats', 'all'):
            csv_out_file = Path(table_dir, 'spc_stats.csv')
            self.stats_df.to_csv(csv_out_file)
        if df_type == 'trials':
            print('Exporting all repeat trial data. This may take a while...')
            csv_out_file = Path(table_dir, 'spc_trials.csv')
            self.trials_df.to_csv(csv_out_file)

        print('SpectralParameterCombinationClassifier export complete.')

    def export_spc_list(self) -> None:
        """Export the list of spectral parameter combinations,
        giving SPC ID, constitent spectral parameters, constituent filters,
        and number of unique filters used.
        """

        table_dir = Path(self.object_dir / 'tables')
        table_dir.mkdir(parents=True, exist_ok=True)

        nuc_uniq_f = self.main_df['uniq_fltrs'].to_numpy()
        nuc_uniq_f = np.array([', '.join(list(uniq_f)) for uniq_f in nuc_uniq_f])
        uniq_fltrs = pd.Series(
            data=nuc_uniq_f,
            index=self.main_df.index,
            name='uniq_fltrs')
        spc_list = pd.concat([
            self.spc_sps(),
            uniq_fltrs,
            self.main_df['n_uniq_fltrs']],
            axis=1)

        csv_out_file = Path(table_dir, 'spc_list.csv')
        spc_list.to_csv(csv_out_file)

    def export_complete_df(self) -> pd.DataFrame:
        """Export the complete table of SPC results
        """

        # SPC Details
        nuc_uniq_f = self.main_df['uniq_fltrs'].to_numpy()
        nuc_uniq_f = np.array([', '.join(list(uniq_f)) for uniq_f in nuc_uniq_f])
        uniq_fltrs = pd.Series(
            data=nuc_uniq_f,
            index=self.main_df.index,
            name='uniq_fltrs')
        spc_df = pd.concat([self.main_df[[
                                'sp_0',
                                'sp_1',
                                'lda_a_0_1',
                                'lda_a_1_1',
                                'lda_boundary_1',
                                'lda_tg_gt_bg_1',
                                'n_uniq_fltrs']],
                                  uniq_fltrs], axis=1)
        spc_df.columns =[
            'sp_x',
            'sp_y',
            'a_x',
            'a_y',
            'boundary',
            'tg_gt_bg',
            'NUC',
            'UC']

        # Fisher Ratio results
        fr_df = pd.concat([
            self.main_df['lda_score'],
            self.main_df['rank_lda_score_all-data'],
            self.main_df['pct_lda_score_all-data'],
            self.stats_df[('lda_score', 'mean')],
            self.stats_df[('lda_score', 'var')],
            self.main_df['rank_lda_score_mean'],
            self.main_df['pct_lda_score_mean']
            ], axis=1)
        fr_df.columns = [
            'FR_D',
            'FR_D Rank',
            'FR_D P.R.',
            'FR_mu',
            'FR_var',
            'FR_mu Rank',
            'FR_mu P.R.']

        # Classification Accuracy results
        acc_df = pd.concat([
            self.main_df['lda_acc_1'],
            self.main_df['rank_lda_acc_1_all-data'],
            self.main_df['pct_lda_acc_1_all-data'],
            self.main_df['lda_fpr_1'],
            self.main_df['lda_fpr_1'],
            self.stats_df[('lda_acc_1', 'mean')],
            self.stats_df[('lda_acc_1', 'var')],
            self.main_df['rank_lda_acc_1_mean'],
            self.main_df['pct_lda_acc_1_mean'],
            self.stats_df[('lda_tpr_1', 'mean')],
            self.stats_df[('lda_tpr_1', 'var')],
            self.stats_df[('lda_fpr_1', 'mean')],
            self.stats_df[('lda_fpr_1', 'var')]
            ], axis=1)
        acc_df.columns = [
            'ACC_D',
            'ACC_D Rank',
            'ACC_D P.R.',
            'TPR_D',
            'FPR_D',
            'ACC_mu',
            'ACC_var',
            'ACC_mu Rank',
            'ACC_mu P.R.',
            'TPR_mu',
            'TPR_var',
            'FPR_mu',
            'FPR_var']

        complete_df = pd.concat([spc_df, fr_df, acc_df], axis=1)

        table_dir = Path(self.object_dir / 'tables')
        table_dir.mkdir(parents=True, exist_ok=True)

        csv_out_file = Path(table_dir, 'complete_results.csv')
        complete_df.to_csv(csv_out_file)

        return complete_df
