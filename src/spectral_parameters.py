"""SpectralParameters Class

Hosts spectral parameters computed from the reflectance values of an Observation
object.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 07-10-2022
"""
import copy
import itertools
import os
from pathlib import Path
from  shutil import rmtree
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import sptk.config as cfg
from sptk.config import build_project_directory as build_pd
from sptk.observation import Observation

SP_CODES = {
    'channel': 'R(?!_)',
    'ratio': 'R_',
    'slope': 'S_',
    'band_depth': 'BD_',
    'shoulder_height': 'SH_'
}

class SpectralParameters():
    """Class to host spectral parameters computed from an observation object.
    """

    def __init__(self,
            observation: Observation,
            load_existing: bool=cfg.LOAD_EXISTING,
            export_df: bool=cfg.EXPORT_DF
            ) -> None:
        """Initialise the list of spectral parameters to compute, using
        the spectral channels of the Observation object.

        :param observation: Observation object, holding isntrument-sampled
            material collection.
        :type observation: Observation
        :param load_existing: instruct to use or overwrite existing directories
            and files of the same project_name, defaults to cfg.LOAD_EXISTING
        :type load_existing: bool, optional
        :param export_df: Flag for exporting the list of spectral parameters
            after initialisation, defaults to cfg.EXPORT_DF
        :type export_df: bool, optional
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()

        self.observation = observation
        self.material_collection = observation.material_collection
        self.instrument = observation.instrument

        p_dir, p_name = build_pd(self.material_collection.project_name,
                                                        'spectral_parameters')
        self.project_dir = p_dir
        self.project_name = p_name
        self.object_dir = Path(self.project_dir / 'spectral_parameters')

        if load_existing:
            existing_pkl_path = Path(self.object_dir, 'spectral_parameters.pkl')
            file_exists = os.path.isfile(existing_pkl_path)
            if file_exists:
                print("Loading existing Spectral Parameters DataFrame...")
                self.main_df = pd.read_pickle(existing_pkl_path)
                self.sp_list = copy.deepcopy(
                        self.main_df.loc[:,'Spectral Parameters':]
                        .columns.drop('Spectral Parameters')).to_list()
                self.chnl_lbls = self.parse_sp_lbls(self.sp_list, 'channel')
                self.ratio_lbls = self.parse_sp_lbls(self.sp_list, 'ratio')
                self.slope_lbls = self.parse_sp_lbls(self.sp_list, 'slope')
                self.band_depth_lbls = self.parse_sp_lbls(self.sp_list,
                                                                'band_depth')
                self.shoulder_height_lbls = self.parse_sp_lbls(self.sp_list,
                                                            'shoulder_height')
                sp_filters_file = Path(self.object_dir, 'filter_ids.pkl')
                self.sp_filters = pd.read_pickle(sp_filters_file)
            else:
                print("No existing DataFrame,"\
                       "building new Spectral Parameters for"\
                          f"{self.project_name}")
                # import main_df from observation, relabel wvls with chnl_lbls
                self.main_df = copy.deepcopy(observation.main_df)
                new_cols = dict(zip(observation.wvls, observation.chnl_lbls))
                self.main_df.rename(columns=new_cols, inplace=True)
                self.main_df.rename(
                                columns={'Reflectance': 'Spectral Parameters'},
                                inplace=True)
                self.sp_list = copy.deepcopy(observation.chnl_lbls)
                self.chnl_lbls = copy.deepcopy(observation.chnl_lbls)
                self.ratio_lbls = None
                self.slope_lbls = None
                self.band_depth_lbls = None
                self.shoulder_height_lbls = None
                self.sp_filters = pd.Series(
                            data=self.observation.instrument.filter_ids,
                            index=self.sp_list)
        else:
            print("Building new Spectral Parameters DataFrame"\
                   f"for {self.project_name}")
            self.main_df = copy.deepcopy(observation.main_df)
            new_cols = dict(zip(observation.wvls, observation.chnl_lbls))
            self.main_df.rename(columns=new_cols, inplace=True)
            self.main_df.rename(
                            columns={'Reflectance': 'Spectral Parameters'},
                            inplace=True)
            self.sp_list = copy.deepcopy(observation.chnl_lbls)
            self.chnl_lbls = copy.deepcopy(observation.chnl_lbls)
            self.ratio_lbls = None
            self.slope_lbls = None
            self.band_depth_lbls = None
            self.shoulder_height_lbls = None
            self.sp_filters = pd.Series(
                            data=self.observation.instrument.filter_ids,
                            index=self.sp_list)
        if export_df:
            self.export_main_df()

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Spectral Parameters built in {toc - tic:0.4f} seconds.")

    def cat_list(self) -> pd.Series:
        """Get the category of each entry, as a Series

        :return: The category of each entry of the material collection
        :rtype: pd.Series
        """
        cat_list = self.main_df.Category
        return cat_list

    def __del__(self, rmdir: bool = False, rmproj: bool = False) -> None:
        """SpectralParameter Destructor object - optionally deletes
        spectral_parameters directory and/or entire project directory

        :param rmdir: instruct removal of spectral_parameters directory,
            defaults to False
        :type rmdir: bool, optional
        :param rmproj: instruct removal of entire project directory,
            defaults to False
        :type rmproj: bool, optional
        """
        name = self.project_name
        if rmdir:
            print(f"Deleting {name} SpectralParameter directory...")
            try:
                rmtree(Path(self.project_dir, 'spectral_parameters'))
                print(f"{name} SpectralParameter directory deleted.")
            except FileNotFoundError:
                print(f"No {name} SpectralParameter directory to delete.")
        if rmproj:
            print(f"Deleting {name} directory...")
            try:
                rmtree(Path(self.project_dir))
                print(f"{name} directory deleted.")
            except FileNotFoundError:
                print(f"No {name} directory to delete.")

    @property
    def categories(self) -> List:
        """The list of unique categories used, in order, of the entries

        :return: The list of unique categories used, in order, of the entries
        :rtype: List
        """
        return self.main_df.Category.unique().to_list()

    @staticmethod
    def parse_sp_lbls(sp_list: List, sp_type: str) -> List:
        """Parse a list of spectral parameters from an input file to lists
        of each type of spectral parameter.

        :param sp_list: list of spectral parameters in imported data
        :type sp_list: List
        :param sp_type: type of spectral parameter to make list from
        :type sp_type: str
        :return: list of spectral parameters of given type
        :rtype: List
        """
        try:
            sp_code = SP_CODES[sp_type]
        except ValueError as exc:
            raise ValueError(f'{sp_type} spec. param. not recognised') from exc
        sp_list = pd.Series(sp_list)
        sp_cols = sp_list.str.contains(sp_code, regex=True)
        sp_type_list = sp_list[sp_cols].to_list()
        # if empty, set to None
        if sp_type_list == []:
            sp_type_list = None
        return sp_type_list

    def compute_spectral_parameters(self,
            scope: str='all',
            export_df: bool=cfg.EXPORT_DF) -> None:
        """Compute the spectral parameters for the given set, for all possible
        valid channel permutations.

        Scope can be 'all', or specified as a subset, e.g. 'ratio', 'slope' etc.

        :param scope: the spectral parameter type to execute, defaults to 'all'
        :type scope: str, optional
        :param export_df: Flag for exporting the evaluated spectral parameters,
          defaults to cfg.EXPORT_DF
        :type export_df: bool, optional
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            print('Computing '+scope+' spectral parameters...')
        if scope in ('all', 'ratio'):
            self.compute_ratio_permutations()
        if scope in ('all', 'slope'):
            self.compute_slope_permutations()
        if scope in ('all', 'band_depth'):
            self.compute_band_depth_permutations()
        if scope in ('all', 'shoulder_height'):
            self.compute_shoulder_height_permutations()
        # update sp_list
        self.sp_list = copy.deepcopy(
                        self.main_df.loc[:,'Spectral Parameters':]
                        .columns.drop('Spectral Parameters')).to_list()
        if export_df:
            self.export_main_df()

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Spectral Parameters computed in {toc - tic:0.4f} seconds.")

    def compute_ratio_permutations(self,
            chnl_lbls: List[str]=None) -> pd.DataFrame:
        """Compute all Ratio Spectral Parameters for given Channels

        :param chnl_lbls: List of channels to use, defaults to all
        :type chnl_lbls: List[str], optional
        :return: Ordering of band permutations
        :rtype: pd.DataFrame
        """
        b1_lst = []
        b2_lst = []
        if chnl_lbls is None:
            chnl_lbls = self.chnl_lbls
        for a,b in itertools.permutations(chnl_lbls, 2):
            b1_lst.append(a)
            b2_lst.append(b)
        self.compute_ratio([b1_lst, b2_lst])
        return pd.DataFrame(data={'b1': b1_lst, 'b2':b2_lst})

    def compute_ratio(self, chnls: Tuple[List[str],...]) -> None:
        """Compute the Ratio Spectral Parameter with numerators and denominators
        given by the list of bands.

        :param chnls: ordered lists of numerator and denominator channels
        :type chnls: Tuple[List[str],...]
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            print("Computing "+str(len(chnls[0]))+" Ratio spectral parameters:")
        # get data for the channels in the lists
        b_df, b_cwls, b_filters = self.get_channel_data(chnls)
        # Ratio operation
        sp_data = np.divide(b_df[0].to_numpy(),b_df[1].to_numpy())
        # Construct list of spectral parameter labels
        sp_pfx = ['R'] * len(b_filters[0]) # shorthand prefix for 'Ratio'
        sp_lbls = [a+'_'+str(b)+'_'+str(c)
            for a,b,c in zip(sp_pfx, b_cwls[0], b_cwls[1])]
        # Construst list of filter id's used
        sp_filters = [a+', '+b for a,b in zip(b_filters[0], b_filters[1])]
        # Append to the main dataframe
        self.ratio_lbls = sp_lbls
        self.append_spectral_parameter(sp_data, sp_lbls, sp_filters)

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"{str(len(chnls[0]))} Ratio spectral parameters computed"\
                                                f"in {toc - tic:0.4f} seconds.")

    def compute_slope_permutations(self,
            chnl_lbls: List[str]=None) -> pd.DataFrame:
        """Compute all Slope Spectral Parameters for given Channels

        :param chnl_lbls: List of bands to use, defaults to all
        :type chnl_lbls: List[str], optional
        :return: Ordering of channel permutations
        :rtype: pd.DataFrame
        """
        b1 = []
        b2 = []
        if chnl_lbls is None:
            chnl_lbls = self.chnl_lbls
        for a,b in itertools.permutations(chnl_lbls, 2):
            # only use if a < b
            a_wvl = self.observation.get_cwl(a)
            b_wvl = self.observation.get_cwl(b)
            if a_wvl < b_wvl:
                b1.append(a)
                b2.append(b)
        self.compute_slope([b1, b2])
        return pd.DataFrame(data={'b1': b1, 'b2':b2})

    def compute_slope(self, chnls: Tuple[List[str],...]):
        """Compute the Slope Spectral Parameter with short and long wavelength
        points given by the list of channels.

        :param chnls: ordered lists of short and long wavelength bands
        :type chnls: Tuple[List[str],...]
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            print("Computing "+str(len(chnls[0]))+" Slope spectral parameters:")
        # get data for the bands in the lists
        b_df, b_cwls, b_filters = self.get_channel_data(chnls)
        # Slope operation
        sp_data = np.divide(b_df[0].to_numpy() - b_df[1].to_numpy(),
                                                        (b_cwls[0] - b_cwls[1]))
        # Construct list of spectral parameter labels
        sp_pfx = ['S'] * len(b_filters[0]) # shorthand for 'Slope'
        sp_lbls = [a+'_'+str(b)+'_'+str(c)
            for a,b,c in zip(sp_pfx, b_cwls[0], b_cwls[1])]
        # Construst list of filter id's used
        sp_filters = [a+', '+b for a,b in zip(b_filters[0], b_filters[1])]
        # Append to the main dataframe
        self.slope_lbls = sp_lbls
        self.append_spectral_parameter(sp_data, sp_lbls, sp_filters)

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"{str(len(chnls[0]))} Slope spectral parameters computed \
                                                in {toc - tic:0.4f} seconds.")

    def compute_band_depth_permutations(self,
            chnl_lbls: List[str]=None) -> pd.DataFrame:
        """Compute all Band Depth Spectral Parameters for given Bands

        :param chnl_lbls: List of channels to use, defaults to all
        :type chnl_lbls: List[str], optional
        :return: Ordering of band permutations
        :rtype: pd.DataFrame
        """
        b1 = []
        b2 = []
        b3 = []
        if chnl_lbls is None:
            chnl_lbls = self.chnl_lbls
        for a,b,c in itertools.permutations(chnl_lbls, 3):
            # only use if a < b < c
            a_wvl = int(a.replace('R',''))
            b_wvl = int(b.replace('R',''))
            c_wvl = int(c.replace('R',''))
            if (a_wvl < b_wvl) & (b_wvl < c_wvl):
                b1.append(a)
                b2.append(b)
                b3.append(c)
        self.compute_band_depth([b1, b2, b3])
        return pd.DataFrame(data={'b1': b1, 'b2':b2, 'b3':b3})

    def compute_band_depth(self, chnls: Tuple[List[str],...]):
        """Compute the Band Depth Spectral Parameter with short-wing,
        band-centre, and long-wing wavelength points given by the list of
        channels.

        :param chnls: ordered lists of short-wings, band-centres, and long-wingS
        :type chnls: Tuple[List[str],...]
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            n_sps = str(len(chnls[0]))
            print(f"Computing {n_sps} Band-Depth spectral parameters:")
        # get data for the bands in the lists
        b_df, b_cwls, b_filters = self.get_channel_data(chnls)
        # Band Depth operation
        b = (b_cwls[1] - b_cwls[0]) / (b_cwls[2] - b_cwls[0])
        a = 1 - b
        sp_data = 1.0 - (b_df[1].to_numpy() /
                                (a*b_df[0].to_numpy() + b*b_df[2].to_numpy()))
        # Construct list of spectral parameter labels
        sp_pfx = ['BD'] * len(b_filters[0]) # shorthand for 'band_depth'
        sp_lbls = [a+'_'+str(b)+'_'+str(c)+'_'+str(d)
            for a,b,c,d in zip(sp_pfx, b_cwls[0], b_cwls[1], b_cwls[2])]
        # Construst list of filter id's used
        sp_filters = [a+', '+b+', '+c for a,b,c in
                                zip(b_filters[0], b_filters[1], b_filters[2])]
        # Append to the main dataframe
        self.band_depth_lbls = sp_lbls
        self.append_spectral_parameter(sp_data, sp_lbls, sp_filters)

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"{str(len(chnls[0]))} Band-Depth spectral parameters \
                                        computed in {toc - tic:0.4f} seconds.")

    def compute_shoulder_height_permutations(self,
            chnl_lbls: List[str]=None) -> pd.DataFrame:
        """Compute all Shoulder Height Spectral Parameters for given chnls

        :param chnl_lbls: List of chnls to use, defaults to all
        :type chnl_lbls: List[str], optional
        :return: Ordering of chnl permutations
        :rtype: pd.DataFrame
        """
        b1 = []
        b2 = []
        b3 = []
        if chnl_lbls is None:
            chnl_lbls = self.chnl_lbls
        for a,b,c in itertools.permutations(chnl_lbls, 3):
            # only use if a < b < c
            a_wvl = int(a.replace('R',''))
            b_wvl = int(b.replace('R',''))
            c_wvl = int(c.replace('R',''))
            if (a_wvl < b_wvl) & (b_wvl < c_wvl):
                b1.append(a)
                b2.append(b)
                b3.append(c)
        self.compute_shoulder_height([b1, b2, b3])
        return pd.DataFrame(data={'b1': b1, 'b2':b2, 'b3':b3})

    def compute_shoulder_height(self, chnls: Tuple[List[str],...]):
        """Compute the Shoulder Height Spectral Parameter with short-wing,
        band-centre, and long-wing wavelength points given by the list of bands.

        :param chnls: ordered lists of short-wings, band-centres, and long-wings
        :type chnls: Tuple[List[str],...]
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            n_sps = str(len(chnls[0]))
            print(f"Computing {n_sps} Shoulder-Height spectral parameters:")
        # get data for the chnls in the lists
        b_df, b_cwls, b_filters = self.get_channel_data(chnls)
        # Shoulder Height operation
        b = (b_cwls[1] - b_cwls[0]) / (b_cwls[2] - b_cwls[0])
        a = 1 - b
        sp_data = 1.0 - np.divide((a*b_df[0].to_numpy() + b*b_df[2].to_numpy()),
                                                            b_df[1].to_numpy())
        # Construct list of spectral parameter labels
        sp_pfx = ['SH'] * len(b_filters[0]) # shorthand for 'shoulder_height'
        sp_lbls = [a+'_'+str(b)+'_'+str(c)+'_'+str(d)
            for a,b,c,d in zip(sp_pfx, b_cwls[0], b_cwls[1], b_cwls[2])]
        # Construst list of filter id's used
        sp_filters = [a+', '+b+', '+c for a,b,c in
                                zip(b_filters[0], b_filters[1], b_filters[2])]
        # Append to the main dataframe
        self.shoulder_height_lbls = sp_lbls
        self.append_spectral_parameter(sp_data, sp_lbls, sp_filters)

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"{n_sps} Shoulder-Height spectral parameters " \
                                    f"computed in {toc - tic:0.4f} seconds.")

    def get_channel_data(self,
            chnls: List[str]) -> Tuple[List[pd.DataFrame],
                List[np.array], List[List]]:
        """For a list of channels, get lists of the channel reflectances,
        centre-wavelengths, and filter-ids.

        :param chnls: ordered list of channels to return data on
        :type chnls: List[str]
        :return: reflectance data, centre-wavelengths, and filter ids
        :rtype: Tuple[List[pd.DataFrame], List[np.array], List[List]]
        """
        chnl_dfs = []
        chnl_cwls = []
        chnl_filters = []
        for chnl in chnls:
            chnl_df = self.main_df[chnl]
            cwl = self.observation.get_cwl(chnl)
            filter_id = self.instrument.get_filter_ids(cwl)
            chnl_dfs.append(chnl_df)
            chnl_filters.append(filter_id)
            chnl_cwls.append(cwl)
        return chnl_dfs, chnl_cwls, chnl_filters

    def append_spectral_parameter(
            self,
            sp_data: np.array,
            sp_lbls: List[str],
            sp_filters: List[str]):
        """Format the spectral parameter data into a DataFrame and append to
        the main DataFrame.
        Also add to the list of spectral parameters and filter ids

        :param sp_data: data for each spectral parameter
        :type sp_data: np.array
        :param sp_lbls: label of each spectral parameter
        :type sp_lbls: List[str]
        :param sp_filters: filters used in each spectral parameter
        :type sp_filters: List[str]
        """
        sp_df = pd.DataFrame(
                    data=sp_data,
                    columns=sp_lbls,
                    index=self.main_df.index)
        # overwrite existing spectral parameter entries of the same column names
        if set(sp_df.columns).issubset(self.main_df.columns):
            self.main_df.drop(sp_df.columns, axis=1, inplace=True)
        # add the spectral parameters to the main dataframe
        self.main_df = pd.concat([self.main_df,sp_df], axis=1)
        # add the spectral parameters to the sp_list
        self.sp_list.append(sp_lbls)
        # add the filters to the filter_id series
        sp_filter_series = pd.Series(data=sp_filters, index = sp_lbls)
        self.sp_filters = pd.concat([self.sp_filters, sp_filter_series])

    def train_test_random_split(self,
            test_size: float = 0.2,
            seed: int=None,
            balance_test: bool=False,
            balance_train: bool=False) -> Tuple[object, object]:
        """Randomly split the SpectralParameters dataset into training and
        testing subsets, according to the given test size percentage, with
         stratification, and optional balancing of the test and train datasets.

        Memory warning: Method puts the training and testing data subset in new
        SpectralParameters objects. This comprehensively records the data-split,
        but can use large amounts of memory.

        :param test_size: The percentage of the data to assign to the test
            dataset, defaults to 0.2
        :type test_size: float, optional
        :param seed: seed for the random number generator, for experiment
            control, defaults to None
        :type seed: int, optional
        :param balance_test: Flag for balancing the test dataset, defaults to
            False
        :type balance_test: bool, optional
        :param balance_train: Flag for balancing the train dataset, defaults to
            False
        :type balance_train: bool, optional
        :return: the train and test data subsets in Observation objects
        :rtype: Tuple['Observation', 'Observation']
        """
        # use scikitlearn to split the dataset
        train_df, test_df = train_test_split(
                                self.main_df,
                                test_size=test_size,
                                stratify=self.main_df['Category'],
                                random_state=seed)

        train_sps = copy.deepcopy(self)
        train_sps.main_df = train_df # overwrite with the split dataset
        if balance_train:
            train_sps.balance_class_sizes(random_state=seed)

        test_sps = copy.deepcopy(self)
        test_sps.main_df = test_df # overwrite with the split dataset
        if balance_test:
            test_sps.balance_class_sizes(random_state=seed)

        return [train_sps, test_sps]

    def balance_class_sizes(self, random_state: int = None):
        """Checks for class balance, and randomly removes samples so that all
        classes are of the same size, matching that of the smallest class.

        :param random_state: set random number seed for reproducibility,
            defaults to None
        :type random_state: int, optional
        """
        if self.main_df.Category.value_counts().is_unique:
            print('Balancing class sizes...')
            class_n_dict = self.main_df.Category.value_counts().to_dict()
            # for each class that is not the smallest class
            min_class = min(class_n_dict, key=class_n_dict.get)
            for cat in self.categories:
                if cat is min_class:
                    continue
                # number of samples to remove
                n_r = class_n_dict[cat] - class_n_dict[min_class]
                # randomly select n_r samples
                to_drop = self.main_df[self.main_df.Category == cat].sample(
                    n=n_r, random_state=random_state).index
                # remove the selected samples
                self.main_df.drop(to_drop, inplace=True)

    def export_main_df(self):
        """Export the dataframe to csv file, and pickle.
        """
        pkl_file = Path(self.object_dir, 'spectral_parameters.pkl')
        self.main_df.to_pickle(pkl_file)
        pkl_file = Path(self.object_dir, 'filter_ids.pkl')
        self.sp_filters.to_pickle(pkl_file)

        table_dir = Path(self.object_dir / 'tables')
        table_dir.mkdir(parents=True, exist_ok=True)
        csv_out_file = Path(table_dir, 'spectral_parameters.csv')
        self.main_df.transpose().to_csv(csv_out_file)

        print('SpectralParameters export complete.')

    # """
    # Plotting Spectral Parameters
    # """

    def gridplot_sp_histograms(self):
        """Plot and Organise Spectral Parameter Histograms into common grids.
        Also export a caption for each figure, populated with relevant
        information.
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
        print("Plotting Spectral Parameters...")
        if self.chnl_lbls is not None:
            self.gridplot_chnl_histograms()
        if self.ratio_lbls is not None:
            self.gridplot_ratio_histograms()
        if self.slope_lbls is not None:
            self.gridplot_slope_histograms()
        if self.band_depth_lbls is not None:
            self.gridplot_band_depth_histograms()
        if self.shoulder_height_lbls is not None:
            self.gridplot_shoulder_height_histograms()
        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Spectral Parameters plotted in {toc - tic:0.4f} seconds.")

    def gridplot_chnl_histograms(self):
        """Plot the single filter channels, with fixed x-axis of range 0 - 1,
        in a square grid, or otherwise a near-square grid with width greater
        than height.
        """
        # Get the chnl data and reorganise to 3 columns:
        # Category, Spectral Parameter, Value
        chnl_df = self.main_df[self.chnl_lbls + ['Category']]
        chnl_df = chnl_df.reset_index(drop=True)
        chnl_df_mlt = pd.melt(chnl_df,
                              id_vars='Category', var_name='spectral_parameter')
        # make a FacetGrid object for the plotting
        plt.rcParams.update({'font.size': 8})
        g = sns.FacetGrid(
                chnl_df_mlt,
                col='spectral_parameter',
                hue='Category',
                col_wrap=3,
                height=4*cfg.CM,
                aspect=1,
                legend_out=True)
        # get bin positions from dataset
        data_for_hist = [
            chnl_df_mlt.loc[chnl_df_mlt['Category']==cat, 'value'].to_numpy()
                                for cat in self.material_collection.categories]
        bins = np.histogram(np.hstack(data_for_hist), bins=20)[1] #get bin edges
        g.map_dataframe(sns.histplot, x="value", bins=bins, common_bins=True)
        g.add_legend()
        g.set_axis_labels('Reflectance', 'Count')
        g.set_titles(col_template="{col_name}")
        # calculate amount to adjust the figure by to accomodate the title
        height = g.fig.get_size_inches()[1] # get the figure size
        title_pad = 2.0 * 8.0 * 1.0/72.0 # 2x title-fontsizextitle size of 1/72"
        scale = 1.0 + title_pad / height
        g.fig.suptitle('Band Spectral Parameters', fontsize='medium', y=scale)

        # save / export
        Path(self.project_dir / 'observation' / 'feature_histograms'
            ).mkdir(parents=True, exist_ok=True)
        output_file = Path(self.project_dir / 'observation' /
                           'feature_histograms', 'chnls').with_suffix('.pdf')
        plt.savefig(output_file,bbox_inches='tight')

    def gridplot_ratio_histograms(self):
        """Plot the ratio spectral parameters in an N_F x N_F grid
        one axis gives denominator, the other gives the numerator
        """
        # Get the Ratio data and reorganise to 3 columns:
        # Category, Spectral Parameter, Value
        ratio_df = self.main_df[self.ratio_lbls + ['Category']]
        ratio_df = ratio_df.reset_index(drop=True)
        ratio_df_mlt = pd.melt(
            ratio_df, id_vars='Category', var_name='spectral_parameter')
        plt.rcParams.update({'font.size': 8})
        # make a FacetGrid object for the plotting
        g = sns.FacetGrid(
                ratio_df_mlt,
                col='spectral_parameter',
                hue='Category',
                col_wrap=len(self.chnl_lbls) - 1,
                sharex=False,
                height=4*cfg.CM,
                aspect=1,
                legend_out=True)
        # get bin positions from dataset
        data_for_hist = [
            ratio_df_mlt.loc[ratio_df_mlt['Category']==cat, 'value'].to_numpy()
                                for cat in self.material_collection.categories]
        bins = np.histogram(np.hstack(data_for_hist), bins=20)[1] #get bin edges
        g.map_dataframe(sns.histplot, x="value", bins=bins, common_bins=True)
        g.add_legend()
        g.set_axis_labels('Ratio', 'Count')
        g.set_titles(col_template="{col_name}")
        # calculate amount to adjust the figure by to accomodate the title
        height = g.fig.get_size_inches()[1] # get the figure size
        title_pad = 2.0 * 8.0 * 1.0/72.0 # 2xtitle-fontsize x title size 1/72"
        scale = 1.0 + title_pad / height
        g.fig.suptitle('Ratio Spectral Parameters', size='medium', y=scale)
        # save / export
        Path(self.project_dir / 'observation' / 'feature_histograms'
            ).mkdir(parents=True, exist_ok=True)
        output_file = Path(self.project_dir / 'observation' /
                           'feature_histograms', 'ratios').with_suffix('.pdf')
        plt.savefig(output_file,bbox_inches='tight')

    def gridplot_slope_histograms(self):
        """Plot the slope spectral parameters in an N_F x N_F grid
        one axis gives denominator, the other gives the numerator
        """
        # Get the Slope data and reorganise to 3 columns:
        # Category, Spectral Parameter, Value
        slope_df = self.main_df[self.slope_lbls + ['Category']]
        slope_df = slope_df.reset_index(drop=True)
        slope_df_mlt = pd.melt(
            slope_df, id_vars='Category', var_name='spectral_parameter')
        plt.rcParams.update({'font.size': 8})
        # make a FacetGrid object for the plotting
        g = sns.FacetGrid(
                slope_df_mlt,
                col='spectral_parameter',
                hue='Category',
                col_wrap=len(self.chnl_lbls) - 1,
                sharex=False,
                height=4*cfg.CM,
                aspect=1,
                legend_out=True)
        # get bin positions from dataset
        data_for_hist = [
            slope_df_mlt.loc[slope_df_mlt['Category']==cat, 'value'].to_numpy()
                                for cat in self.material_collection.categories]
        bins = np.histogram(np.hstack(data_for_hist), bins=20)[1] #get bin edges
        g.map_dataframe(sns.histplot, x="value", bins=bins, common_bins=True)
        g.add_legend()
        g.set_axis_labels('Slope', 'Count')
        g.set_titles(col_template="{col_name}")
        # calculate amount to adjust the figure by to accomodate the title
        height = g.fig.get_size_inches()[1] # get the figure size
        title_pad = 2.0 * 8.0 * 1.0/72.0 # 2x title-fontsize x title size 1/72"
        scale = 1.0 + title_pad / height
        g.fig.suptitle('Slope Spectral Parameters', size='medium', y=scale)
        # save / export
        Path(self.project_dir / 'observation' / 'feature_histograms'
            ).mkdir(parents=True, exist_ok=True)
        output_file = Path(self.project_dir / 'observation' /
                           'feature_histograms', 'slopes').with_suffix('.pdf')
        plt.savefig(output_file,bbox_inches='tight')

    def gridplot_band_depth_histograms(self):
        """Plot the band depth spectral parameters in a sequence of N_F grids,
        each as a lower triangular N_F x N_F grid,
        where each step in the sequence gives the centre of the band depth,
        and the vertical gives the leading band and the horizontal gives the
        trailing band
        """
        # get the band_depth labels
        bd_lbls = np.array(self.band_depth_lbls)
        # get central wavelength
        ce_wvls = np.array([bd_lbl.split('_')[2] for bd_lbl in bd_lbls])
        # for each central wavelength
        ce_wvl_lst = np.unique(ce_wvls)
        plt.rcParams.update({'font.size': 8})
        for ce_wvl in ce_wvl_lst:
            # get the list of band_depth labels for this ce_wvl
            ce_wvl_band_depths = bd_lbls[ce_wvls==ce_wvl]
            # Get the data and reorganise to 3 columns:
            # Category, Spectral Parameter, Value
            bd_df = self.main_df[ce_wvl_band_depths.tolist() + ['Category']]
            bd_df = bd_df.reset_index(drop=True)
            bd_df_mlt = pd.melt(
                bd_df, id_vars='Category', var_name='spectral_parameter')
            # get number of wavelengths greater than the ce_wvl
            width = len(ce_wvl_lst[ce_wvl_lst > ce_wvl]) + 1
            g = sns.FacetGrid(
                    bd_df_mlt,
                    col='spectral_parameter',
                    hue='Category',
                    col_wrap=width,
                    sharex=False,
                    height=4*cfg.CM,
                    aspect=1,
                    legend_out=True)
            # get bin positions from dataset
            hist_data = [
                bd_df_mlt.loc[bd_df_mlt['Category']==cat, 'value'].to_numpy()
                                for cat in self.material_collection.categories]
            bins = np.histogram(np.hstack(hist_data), bins=20)[1]#get bin edges
            g.map_dataframe(sns.histplot, x="value", bins=bins,common_bins=True)
            g.add_legend()
            g.set_axis_labels('Band Depth', 'Count')
            g.set_titles(col_template="{col_name}")
            # calculate amount to adjust the figure by to accomodate the title
            height = g.fig.get_size_inches()[1] # get the figure size
            title_pad = 2.0 * 8.0 * 1.0/72.0 #2xtitle-fontsize x title size 1/72
            scale = 1.0 + title_pad / height
            g.fig.suptitle(ce_wvl+' nm Band Depth Spectral Parameters',
                           size='medium', y=scale)
            # save / export
            Path(self.project_dir / 'observation' / 'feature_histograms'
                ).mkdir(parents=True, exist_ok=True)
            output_file = Path(self.project_dir / 'observation' /
                'feature_histograms', 'band_depth_'+ce_wvl).with_suffix('.pdf')
            plt.savefig(output_file,bbox_inches='tight')

    def gridplot_shoulder_height_histograms(self):
        """Plot the shoulder height spectral parameters in a sequence of N_F
        grids, each as a lower triangular N_F x N_F grid,
        where each step in the sequence gives the centre of the band depth,
        and one axis gives the leading band and the other gives the trailing
        band
        """
        # get the shoulder_height labels
        sh_lbls = np.array(self.shoulder_height_lbls)
        # get central wavelength
        ce_wvls = np.array([sh_lbl.split('_')[2] for sh_lbl in sh_lbls])
        # for each central wavelength
        ce_wvl_lst = np.unique(ce_wvls)
        plt.rcParams.update({'font.size': 8})
        for ce_wvl in ce_wvl_lst:
            # get the list of shoulder_height labels for this ce_wvl
            ce_wvl_shoulder_heights = sh_lbls[ce_wvls==ce_wvl]
            # Get the data and reorganise to 3 columns:
            # Category, Spectral Parameter, Value
            sh_df = self.main_df[ce_wvl_shoulder_heights.tolist()+['Category']]
            sh_df = sh_df.reset_index(drop=True)
            sh_df_mlt = pd.melt(sh_df, id_vars='Category',
                                var_name='spectral_parameter')
            # get number of wavelengths greater than the ce_wvl
            width = len(ce_wvl_lst[ce_wvl_lst > ce_wvl]) + 1
            # make facet grid to host the plots
            g = sns.FacetGrid(
                    sh_df_mlt,
                    col='spectral_parameter',
                    hue='Category',
                    col_wrap=width,
                    sharex=False,
                    height=4*cfg.CM,
                    aspect=1,
                    legend_out=True)
            # get bin positions from dataset
            hist_data = [
                sh_df_mlt.loc[sh_df_mlt['Category']==cat, 'value'].to_numpy()
                                for cat in self.material_collection.categories]
            bins = np.histogram(np.hstack(hist_data), bins=20)[1] #get bin edges
            g.map_dataframe(sns.histplot, x="value", bins=bins,common_bins=True)
            g.add_legend()
            g.set_axis_labels('Shoulder Height', 'Count')
            g.set_titles(col_template="{col_name}")
            # calculate amount to adjust the figure by to accomodate the title
            height = g.fig.get_size_inches()[1] # get the figure size
            title_pad = 2.0 * 8.0 * 1.0/72.0 # 2xtitle-fontsize x title size
            scale = 1.0 + title_pad / height
            g.fig.suptitle(ce_wvl+' nm Shoulder Height Spectral Parameters',
                           size='medium', y=scale)
            # save / export
            Path(self.project_dir / 'observation' / 'feature_histograms'
                ).mkdir(parents=True, exist_ok=True)
            output_file = Path(self.project_dir / 'observation' /
                               'feature_histograms',
                            'shoulder_height_'+ce_wvl).with_suffix('.pdf')
            plt.savefig(output_file,bbox_inches='tight')
