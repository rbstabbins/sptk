"""Material Collection Class

Hosts material reflectance data and auxilary information in a Pandas DataFrame

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 29-04-2021
"""

import errno
import os
import glob
import time
from pathlib import Path
from  shutil import rmtree
from typing import Dict, List, Tuple, Literal
import click
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import spectral.io.envi as envi
import sptk.config as cfg
from sptk.instrument import Instrument
from sptk.spectral_library_analyser import SpectralLibraryAnalyser

# metadata labels used in the MaterialCollection
HEADER_LIST = [
    'Sample ID', # the identifier for the physical sample observed
    'Species',   # the mineral species (or e.g. 'patch' name of a colour chart)
    'Subgroup',  # the mineral subgroup, following Mindat.org classification
    'Group',     # the mineral group, following Mindat.org classification
    'Library',   # the name of the spectral library from the '../data' directory
    'Sample Description', # a description of the sample, copied from entry
    'Date Added', # the date the sample was added to the spectral library
    'Viewing Geometry', # incidence (i) and emission (e) angles of the measurement
    'Other Information',  # misc
    'Composition', # chemical formula of the sample, or expected for the mineral species
    'Resolution', # the bandwidth of the spectrometer used to measure the sample
    'Grain Size', # grain size information
    'Locality', # the location where the sample was collected
    'Database of Origin' # the database from which the sample was sourced
    ] 

class MaterialCollection():
    """Hosts material reflectance data and auxilary information in a DataFrame.
    """

    def __init__(
            self,
            material_dict: dict,
            spectral_library: str,
            project_name: str,
            balance_classes: bool=True,
            random_bias_seed: int = None,
            allow_out_of_bounds: bool=False,
            load_existing: bool = cfg.LOAD_EXISTING,
            plot_profiles: bool = cfg.PLOT_PROFILES,
            export_df: bool = cfg.EXPORT_DF) -> None:
        """Constructor for MaterialCollection class

        :param material_dict: dictionary mapping category labels to mineral species
            or group and filenames or wildcards.
            e.g.: material_dict = {
                        'category_1': [('species_1', '*')], # all files in species_1
                        'category_2': [('species_2', filename), # only filename in species_2
                                    ('group_1', '*')] # all species and files in group_1
                                }
        :type material_dict: dict
        :param spectral_library: name of spectral library to draw samples from
        :type spectral_library: str
        :param project_name: name of project and directory
        :type project_name: str
        :param balance_classes: instruct whether to balance category sizes by
            randomly removing samples until category sizes are equal, defaults to
            True.
        :type balance_classes: bool, optional
        :param random_bias_seed: only valid if balance_classes is True; set to
            an integer for reproducible balancing, set to None for random
            balancing, defaults to None.
        :type random_bias_seed: int, optional
        :param allow_out_of_bounds: Instruct whether to include entries with
            wavelengths that do not cover the full range, defaults to False.
        :type allow_out_of_bounds: bool, optional
        :param load_existing: instruct to use or overwrite existing directories
            and files of the same project_name, defaults to config.py setting.
        :type load_existing: bool, optional
        :param plot_profiles: plot material profiles to project directory,
            defaults to False
        :type plot_profiles: bool, optional
        :param export_df: export DataFrames to project directory, defaults to
            False
        :type export_df: bool, optional
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()

        # set MaterialCollection attributes
        self.project_dir, self.project_name = cfg.build_project_directory(
                                            project_name, 'material_collection')
        self.object_dir = Path(self.project_dir / 'material_collection')
        self.spectral_library = spectral_library
        self.material_file_dict = MaterialCollection.parse_materials(
                                            material_dict, spectral_library)
        self.categories = list(self.material_file_dict.keys())
        self.wvls = cfg.WVLS
        self.allow_out_of_bounds = allow_out_of_bounds
        self.header_list = HEADER_LIST

        self.noisy = False # generally not used for a Material Collection

        # find the existing pickle file of the MaterialCollection DataFrame,
        # or make a new one from the material file dictionary
        if load_existing:
            existing_pkl_path = Path(
                    self.object_dir,
                    'material_collection').with_suffix('.pkl')
            file_exists = os.path.isfile(existing_pkl_path)
            if file_exists:
                print(f"Loading existing {project_name} MaterialCollection DataFrame")
                self.main_df = pd.read_pickle(existing_pkl_path)
            else:
                print(f"No existing {project_name} MaterialCollection DataFrame")
                print("Building new MaterialCollection")
                self.main_df = self.build_new_material_collection()
        else:
            print(f"Building new {project_name} MaterialCollection DF")
            self.main_df = self.build_new_material_collection()

        # apply class balancing if instructed
        if balance_classes:
            self.balance_class_sizes(random_state = random_bias_seed)

        # plot profiles and export DataFrames if instructed
        if plot_profiles:
            plotter = SpectralLibraryAnalyser(self)
            plotter.plot_profiles()
        if export_df:
            self.export_main_df()

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Material Collection built in {toc - tic:0.4f} seconds.")

    def __del__(self, rmdir: bool = False, rmproj: bool = False) -> None:
        """MaterialCollection Destructor object - optionally deletes
        material_collection directory and/or entire project directory

        :param rmdir: instruct removal of material_collection directory,
            defaults to False
        :type rmdir: bool, optional
        :param rmproj: instruct removal of entire project directory,
            defaults to False
        :type rmproj: bool, optional
        """
        name = self.project_name
        if rmdir:
            print(f"Deleting {name} MaterialCollection directory...")
            try:
                rmtree(Path(self.project_dir, 'material_collection'))
                print(f"{name} MaterialCollection directory deleted.")
            except FileNotFoundError:
                print(f"No {name} MaterialCollection directory to delete.")
        if rmproj:
            print(f"Deleting {name} directory...")
            try:
                rmtree(Path(self.project_dir))
                print(f"{name} directory deleted.")
            except FileNotFoundError:
                print(f"No {name} directory to delete.")

    @staticmethod
    def parse_materials(material_dict: dict, spectral_library: str) -> dict:
        """Parses the 'material_dict' dictionary into a dictionary of categories
        and entry filepaths, for look-up in the specified spectral library.

        :param material_dict: dictionary mapping category labels to mineral species
            or group and filenames or wildcards.
            e.g.: material_dict = {
                        'category_1': [('species_1', '*')], # all files in species_1
                        'category_2': [('species_2', filename), # only filename in species_2
                                    ('group_1', '*')] # all species and files in group_1
                                }
        :type material_dict: dict
        :param spectral_library: name of spectral library to draw samples from,
            defaults to None
        :type spectral_library: str, optional
        :return: Dictionary mapping category labels to lists of entry filepaths
        :rtype: dict
        """
        # set and check directory paths
        data_dir = cfg.DATA_DIRECTORY
        data_library = Path(data_dir, 'spectral_library', spectral_library) # note - no longer allows None-type spectral library        

        data_library_exists = os.path.isdir(cfg.resolve_path(data_library, root='data'))
        if not data_library_exists:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(data_library))

        # mapping the library
        groups = {os.path.basename(dir): dir 
                for dir in glob.glob(str(data_library)+'*/*', recursive=True)}
        subgroups = {os.path.basename(dir): dir 
                for dir in glob.glob(str(data_library)+'*/*/*', recursive=True)}
        species = {os.path.basename(dir): dir 
                for dir in glob.glob(str(data_library)+'*/*/*/*', recursive=True)}
        
        # initialise the filename dictionary
        file_dict = {}
        for cat in material_dict.keys():
            # for each category make new dict entry
            labels = material_dict[cat]
            
            # encase in list if singular entry
            label_is_singular = isinstance(labels, tuple)
            if label_is_singular:
                labels = [labels]

            # initialise list of labels for this category
            cat_labels = []
            # build list of filepaths for entries in this label and category
            for label in labels: 
                label_dirname = label[0]
                label_file_specifier = label[1]

                # search for the label_dirname in the directory tree                
                if label_dirname in species.keys():   
                    label_dir = species[label_dirname]                 
                elif label_dirname in subgroups.keys():
                    label_dir = subgroups[label_dirname]
                elif label_dirname in groups.keys():
                    label_dir = groups[label_dirname]

                else:
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), str(label_dirname))
                
                # get the filepaths for the label_file_specifier
                # if singular entry (e.g. '*'), make into a list
                if not isinstance(label_file_specifier, list):
                    label_file_specifier = [label_file_specifier]
                # add '.csv' to the end of the specifier
                label_filenames = [f + 'csv' for f in label_file_specifier]

                # search for each of the label filenames in the directories
                # and subdirectories under the label_dir
                for label_filename in label_filenames:
                    file_list = glob.glob(
                                        '**/'+label_filename, # '**' searches recursively
                                        root_dir=label_dir, 
                                        recursive=True)                  
                    file_list = [Path(label_dir) / file for file in file_list] # stitch the full path together
                    cat_labels = cat_labels + file_list

            file_dict[cat] = sorted(cat_labels)

        return file_dict

    def build_new_material_collection(self) -> pd.DataFrame:
        """Initialises and populates a new MaterialCollection DataFrame, and
        optionally balances the categories, produces plots, and exports to csv.

        Material files with out-of-range wavelengths will have values set to
        None in the DataFrame.
        Spectral reflectance is interpolated to wavelength resolution set in
        config.py.
        """
        main_df = MaterialCollection.init_frame(
            self.categories, self.material_file_dict)
        filepaths = main_df['Filepath'].tolist()
        last_cat = None
        last_group = None
        last_subgroup = None
        last_species = None
        with click.progressbar(filepaths) as load_bar:
            print('---------------------------------')
            print('Category')
            print("└──species|subgroup|group|")
            print("   └──data_id status")
            for filepath in load_bar:  # for each file, load_material
                new_entry = MaterialCollection.load_material(filepath)
                # extract data id, species, subgroup, group and library
                path_parts = list(Path(filepath).parts)
                data_id = path_parts[-1]
                path_list = list(path_parts)
                # get to the directory tree
                while path_list.pop(0) != self.spectral_library: pass

                if path_list[0] != data_id:
                    group = path_list.pop(0)
                else:
                    group = None
                if path_list[0] != data_id:
                    subgroup = path_list.pop(0)
                else:
                    subgroup = None
                if path_list[0] != data_id:
                    species = path_list.pop(0)
                else:
                    species = None
                cat = main_df[main_df.Filepath == filepath].Category.to_numpy()[0]
                if cat != last_cat:
                    print('---------------------------------')
                    print(f'{cat.title()}')
                if species != last_species:
                    print(f"└──-{species}|{subgroup}|{group}|")
                if not self.allow_out_of_bounds:
                    has_nan = np.isnan(np.sum(new_entry[cfg.WVLS].to_numpy()))
                    if has_nan:
                        print(f"   └──-{data_id} does not span wavelength range, removing...")
                        main_df.drop(
                            main_df[main_df.Filepath == filepath].index,
                            inplace=True)
                    else:
                        print(f"   └──-{data_id} loaded")
                        main_df.loc[main_df.Filepath == filepath,
                                        new_entry.columns] = new_entry.values
                else:
                    print(f"   └──-{data_id} loaded")
                    main_df.loc[main_df.Filepath == filepath,
                                    new_entry.columns] = new_entry.values
                last_cat = cat
                last_group = group
                last_subgroup = subgroup
                last_species = species
        # add the group, subgroup, species to the main_df
        # put the Data ID as the index
        main_df.set_index('Data ID', inplace=True)
        print(f'Loading {len(main_df.index)} of entries complete.')
        return main_df

    @staticmethod
    def init_frame(
            categories: List,
            material_file_dict: Dict,
            ) -> pd.DataFrame:
        """Initialise the Material Collection DataFrame with the list of entry
        filepaths and associated Categories, with pre-defined feature labels.

        DataFrame format:
         - Columns: material properties and reflectance wavelengths
         - Rows: material samples

        :param categories: categories of the material collection
        :type categorires: List
        :param material_file_dict: dictionary mapping categories to materials
            and entry filepaths in the spectral library
        :type material_file_dict: Dict
        :return: Initiliased Material Collection DataFrame.
        :rtype: pd.DataFrame
        """
        filepaths = []
        material_categories = []
        # build lists of material files, and associated categories
        for category in categories:
            cat_filepaths = material_file_dict[category]
            for material_filepath in cat_filepaths:
                filepaths.append(material_filepath)
                material_categories.append(category)
        # build null list as placeholder for feature column list
        null_list = [''] * len(filepaths)
        empt_df = pd.DataFrame({  # put lists in dataframe
            'Filepath': pd.Series(filepaths, dtype='str'),
            'Category': pd.Series(material_categories, dtype='category'),
            'Data ID': pd.Series(null_list, dtype='str'),
            'Sample ID': pd.Series(null_list, dtype='str'),
            'Species': pd.Series(null_list, dtype='str'), # renamed from Mineral Name
            'Subgroup': pd.Series(null_list, dtype='str'), # new entry
            'Group': pd.Series(null_list, dtype='str'), # new entry
            'Library': pd.Series(null_list, dtype='str'), # new entry - replaces Database of Origin
            'Sample Description': pd.Series(null_list, dtype='str'),
            'Date Added': pd.Series(null_list, dtype='str'),
            'Viewing Geometry': pd.Series(null_list, dtype='str'),
            'Other Information': pd.Series(null_list, dtype='str'),
            # 'Formula': pd.Series(null_list, dtype='str'), # merge with Composition
            'Composition': pd.Series(null_list, dtype='str'),
            'Resolution': pd.Series(null_list, dtype='str'),
            'Grain Size': pd.Series(null_list, dtype='str'),
            'Locality': pd.Series(null_list, dtype='str'),
            'Database of Origin': pd.Series(null_list, dtype='str'),
            'Reflectance': pd.Series(['---']*len(filepaths), dtype='str')})
        n_entries = sum([len(files) for files in material_file_dict.values()])
        refl_idx = pd.RangeIndex(stop=n_entries)
        refl_df = pd.DataFrame(
                np.nan, index=refl_idx, columns=cfg.WVLS)  # reflectance DF
        empt_df = pd.concat([empt_df, refl_df], axis=1)
        return empt_df

    @staticmethod
    def load_material(filepath: str) -> pd.DataFrame:
        """Access material file in the sptk spectral library, and parse data
        into DataFrame suitable for appending to master dataframe.

        notation: 'mtrl' is 'material'

        :param material_filepath: filepath location of the material
        :type material_filepath: str
        :return: material reflectance data and metadata
        :rtype: pd.DataFrame
        """
        resolved_filepath = cfg.resolve_path(filepath, root='data')
        mtrl_in = pd.read_csv(resolved_filepath,
                              header=None,
                              index_col=0,
                              engine='c',
                              on_bad_lines='skip')

        # resample the wavelengths to the specified range
        mtrl_refl_in = mtrl_in.loc['Wavelength':]
        mtrl_refl_in = mtrl_refl_in.drop(mtrl_refl_in.index[0])
        wvls_in = mtrl_refl_in.index.to_numpy().astype('float64')

        # perform interpolation; put NaN in bad values
        refl_in = mtrl_refl_in.to_numpy().astype('float64')        
        refl_in = np.reshape(refl_in, len(refl_in))
        refl_func = interp1d(wvls_in, refl_in, bounds_error=False)
        refl_interp = refl_func(cfg.WVLS)
        refl_series = pd.Series(data=refl_interp, index=cfg.WVLS)

        # parse the material header information
        hdr = mtrl_in.loc[:'Wavelength']

        # replace 'Sample Name' with 'Mineral Name'
        hdr = hdr.rename({'Sample Name': 'Mineral Name'})
        # replace Mineral Name with Species
        hdr = hdr.rename({'Mineral Name': 'Species'})

        # replace 'Formula' with 'Composition'
        hdr = hdr.rename({'Formula': 'Composition'})

        # extract data ID, species, subgroup, group and library
        path_parts = list(Path(filepath).parts)
        data_id = path_parts[-1].split('.')[0]
        path_list = list(path_parts)
        # get to the directory tree
        while path_list.pop(0) != 'spectral_library': pass
        spectral_library = path_list.pop(0)

        # library/data_id
        if len(path_list) == 1:
            group = None
            subgroup = None
            species = None
        # library/species/data_id
        elif len(path_list) == 2:
            group = None
            subgroup = None
            species = path_list.pop(0)
        # library/group/species/data_id
        elif len(path_list) == 3:
            group = path_list.pop(0)
            subgroup = None
            species = path_list.pop(0)
        # library/group/subgroup/species/data_id
        elif len(path_list) == 4:
            group = path_list.pop(0)
            subgroup = path_list.pop(0)
            species = path_list.pop(0)

        # set the Species to the directory name
        hdr.loc['Species'] = species

        # try:
        #     # if no 'Data ID' label, try 'Sample ID'
        #     hdr.loc['Data ID']
        # except KeyError:

        #  always use species + filename as Data ID
        # no don't do this - but make a label for it when plotting...
        hdr.loc['Data ID'] = species.capitalize() + ' ' +data_id
        # should we be taking sample id from filename?

        # handling of new Grain Size and Grain Size Description entries to VISOR
        # discards Grain Size in favour of 'old' style Grain Size Description,
        # but keeps label of 'Grain Size'.
        # TODO: Problem - new ViSOR file formats use '(X, Y)' to define range of
        # grain sizes, so we have a comma in a field, causing the Pandas csv
        # parser to fail. Current hack fix is to just skip the bad lines, so
        # that the reflectance data still loads correctly.
        # To fix properly, we need a way to define the information within the
        # brackets as a single field.
        # If the parser fails, need to try to wrap the '(X, Y)' grain
        # information in "" quotes, and read again using quotchar = "". Where
        # the entries for the file have '(...)', convert to '"(...)"' and retry.

        props = hdr.index
        if 'Grain Size' in props and 'Grain Size Description' in props:
            hdr.loc['Grain Size'] = hdr.loc['Grain Size Description']
        elif 'Grain Size' not in props and 'Grain Size Description' in props:
            hdr.loc['Grain Size'] = hdr.loc['Grain Size Description']

        # get header data that matches specified HEADER_LIST
        header_indx = pd.Index(['Data ID'] + HEADER_LIST)
        mtrl_hdr = hdr[hdr.index.isin(header_indx)].squeeze()
        # add HEADER_LIST features that are not in the imported headings list
        missing_header = pd.Series(
            data=[None]*len(header_indx[~header_indx.isin(mtrl_hdr.index)]),
            index=header_indx[~header_indx.isin(mtrl_hdr.index)],
            dtype=np.float64)
        mtrl_hdr = pd.concat([mtrl_hdr, missing_header])
        mtrl_hdr = mtrl_hdr.reindex(header_indx)  # apply the original headings
        mtrl_hdr = mtrl_hdr.str.strip() # remove leading/trailing white space
        mtrl_hdr['Species'] = mtrl_hdr['Species'].lower()
        mtrl_hdr['Subgroup'] = subgroup
        mtrl_hdr['Group'] = group
        mtrl_hdr['Library'] = spectral_library

        # prepare data for appending to main dataframe
        mtrl_df = pd.concat([mtrl_hdr, refl_series]).to_frame().transpose()

        return mtrl_df

    def balance_class_sizes(self, random_state: int = None):
        """Checks for class balance, and randomly removes samples so that all
        categories are of the same size, matching that of the smallest category.

        If a Bayesian method of calculating the decision boundary were
        implemented, then this step wouldn't be necessary.

        :param random_state: set random number seed for reproducibility,
            defaults to None
        :type random_state: int, optional
        """
        if self.main_df.Category.value_counts().is_unique:
            print('Balancing class sizes...')
            cat_n_dict = self.main_df.Category.value_counts().to_dict()
            # for each category that is not the smallest category
            min_cat = min(cat_n_dict, key=cat_n_dict.get)
            for cat in self.categories:
                if cat is min_cat:
                    continue
                # number of samples to remove
                n_r = cat_n_dict[cat] - cat_n_dict[min_cat]
                # randomly select n_r samples
                to_drop = self.main_df[self.main_df.Category == cat].sample(
                    n=n_r, random_state=random_state).index
                # remove the selected samples
                self.main_df.drop(to_drop, inplace=True)

    def balance_mineral_groups(self, random_state: int=None):
        """Checks for class and mineral group balance. First balances mineral
        groups, then balances categories by stratified undersampling of the
        mineral groups of the larger category size.

        :param random_state: set random number seed for reproducibility,
            defaults to None
        :type random_state: int, optional
        """
        # for each category
        for cat in self.categories:
            cat_df = self.main_df[self.main_df.Category == cat]
            mnrl_n = cat_df.groupby('Mineral Name')['Mineral Name'].count()
            mnrl_r = mnrl_n - mnrl_n.min() # n entries to remove
            for mnrl in mnrl_r.index:
                to_drop = cat_df[cat_df['Mineral Name'] == mnrl].sample(
                                n=mnrl_r[mnrl], random_state=random_state).index
                # remove the selected samples
                self.main_df.drop(to_drop, inplace=True)
        # check difference between category sizes
        if self.main_df.Category.value_counts().is_unique:
            print('Balancing class sizes...')
            cat_n_dict = self.main_df.Category.value_counts().to_dict()
            # get the smallest category
            min_cat = min(cat_n_dict, key=cat_n_dict.get)
            # for each category that is not the smallest category:
            for cat in self.categories:
                if cat is min_cat:
                    continue
                # number of samples to remove from this category is the
                # difference between the number of samples this category and the
                # number of samples in the smallest category
                n_r = cat_n_dict[cat] - cat_n_dict[min_cat]

                # get the number of samples in each mineral group of this category
                cat_df = self.main_df[self.main_df.Category == cat]
                mnrl_n = cat_df.groupby('Mineral Name')['Mineral Name'].count()

                # number of samples to remove from each mineral group
                n_m_r = n_r // len(mnrl_n)
                # remainder of samples to remove from the category
                n_r_r = n_r % len(mnrl_n)

                # remove number of samples from each mineral group
                for mnrl in mnrl_n.index:
                    # randomly select n_m_r samples
                    to_drop = cat_df[cat_df['Mineral Name'] == mnrl].sample(
                        n=n_m_r, random_state=random_state).index
                    # remove the selected samples
                    self.main_df.drop(to_drop, inplace=True)
                    # update the cat_df to reflect the removal
                    cat_df = self.main_df[self.main_df.Category == cat]
                # remove remainder from this category
                if n_r_r != 0:
                    to_drop = cat_df.sample(
                            n=n_r_r, random_state=random_state).index
                        # remove the selected samples
                    self.main_df.drop(to_drop, inplace=True)

    def print_category_entries(self, print_minerals: bool=False) -> None:
        """Print the number of entries in each category, and optionally
        print the number of entries in each mineral group.

        :param print_minerals: print no. of entries in mineral groups,
            defaults to False
        :type print_minerals: bool, optional
        """
        for cat in self.categories:
            cats = self.main_df[self.main_df.Category == cat]['Sample ID']
            print(f"Samples in {cat} class: {len(cats)}")
            print(f"Unique sample sources in {cat} class: {len(cats.unique())}")
            if print_minerals:
                mnrls = self.main_df[self.main_df.Category==cat]['Mineral Name']
                for mnrl in mnrls.unique():
                    n_mnrls = len(mnrls[mnrls == mnrl])
                    print(f"  Samples in {mnrl} mineral group: {n_mnrls}")

    def sample(self, instrument: Instrument) -> np.array:
        """Sample the material collection with a given instrument.

        :param instrument: the instrument used to perform spectral sampling
        :type instrument: Instrument object
        :return: matrix of sample results
        :rtype: np.array
        """
        print('Sampling the Material Collection with the Instrument...')
        if cfg.TIME_IT:
            tic = time.perf_counter()
        # prepare material collection for sampling
        mat_refl = self.get_refl_df().to_numpy()
        if self.allow_out_of_bounds:
            # log locations of NaN entries
            channel_mask = self.channel_mask(mat_refl, instrument)
            # set NaN locations to last valid value  
            for i in range(0, len(mat_refl)):
                #  one line at a time version
                mat_refl[i] = np.interp(self.wvls, self.wvls[~np.isnan(mat_refl[i])], mat_refl[i][~np.isnan(mat_refl[i])])
        # prepare instrument transmission for performing the sampling
        cam_trans = instrument.get_trans_df().T.to_numpy()
        # compute matrix mult. of reflectance against channel transmission
        product = np.matmul(mat_refl, cam_trans)
        # compute sum of filter transmissions over wavelength dimension
        trans_sum = cam_trans.sum(axis=0)
        # reformat sum to match matrix product dimensions
        trans_sum = np.tile(trans_sum, [product.shape[0], 1])
        # normalise matrix multiplication product by filter summation
        sample_matrix = np.divide(product, trans_sum)
        if self.allow_out_of_bounds:
            # apply NaN mask
            sample_matrix[~channel_mask] = np.NaN
        print('Sampling complete.')
        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Observation sampled in {toc - tic:0.4f} seconds.")
        return sample_matrix

    def channel_mask(self,
            mat_refl: np.array,
            instrument: Instrument) -> np.array:
        """Produces an array of masks, labelling channels with NaN entries,
        so that these channels can be neglected when sampling with the
        instrument in the case where allow_out_of_bounds is true.

        :param mat_refl: material collection reflectance array
        :type mat_refl: np.array
        :param instrument: Instrument object
        :type instrument: Instrument
        :return: Boolean array of channels to include for each entry
        :rtype: np.array
        """
        min_wvl = []
        max_wvl = []
        # for i in range(0,len(mat_refl)):
        #     idx = np.where(~np.isnan(mat_refl[i]))[0]
        #     try:
        #         min_nan = idx[0]
        #         min_wvl.append(self.wvls[min_nan])
        #     except IndexError:
        #         min_wvl.append(self.wvls[0])
        #     try:
        #         max_nan = idx[-1]
        #         max_wvl.append(self.wvls[max_nan])
        #     except IndexError:
        #         max_wvl.append(self.wvls[0])

        for i, _ in enumerate(mat_refl):
            idx = np.where(~np.isnan(mat_refl[i]))[0]
            try:
                min_nan = idx[0]
                min_wvl.append(self.wvls[min_nan])
            except IndexError:
                min_wvl.append(self.wvls[0])
            try:
                max_nan = idx[-1]
                max_wvl.append(self.wvls[max_nan])
            except IndexError:
                max_wvl.append(self.wvls[0])

        min_chnls = (instrument.cwls() - instrument.fwhms()).to_numpy()
        max_chnls = (instrument.cwls() + instrument.fwhms()).to_numpy()

        # broadcast min_chnls, max_chnls, min_wvl & max_wvl to match dimensions
        min_wvl = np.tile(min_wvl, (len(min_chnls), 1)).T
        max_wvl = np.tile(max_wvl, (len(max_chnls), 1)).T
        min_chnls = np.tile(min_chnls, (len(mat_refl), 1))
        max_chnls = np.tile(max_chnls, (len(mat_refl), 1))

        min_mask = np.less(min_wvl, min_chnls)
        max_mask = np.greater(max_wvl, max_chnls)

        channel_mask = min_mask * max_mask

        return channel_mask

    def get_main_df(self) -> pd.DataFrame:
        """Return a copy of the main dataframe.

        :return: copy of the main material collection DataFrame
        :rtype: pd.DataFrame
        """
        return self.main_df.copy()

    def set_refl_data(self, new_refl_data: pd.DataFrame):
        """Set the reflectance data array according to data
        given in an array, with columns labeled by wavelength

        :param new_refl_data: new reflectance data
        :type new_refl_data: pd.DataFrame
        """
        # access the columns given in the new reflectance DataFrame
        new_refl_data.index = self.main_df.index
        self.main_df[self.wvls] = new_refl_data[self.wvls]

    def get_refl_df(self,
            category: str = None,
            library: str = None,
            group: str = None,
            subgroup: str = None,            
            species: str = None,
            sample_id: str = None) -> pd.DataFrame:
        """Return a copy of the reflectance dataframe subset of the array.
        Allows for selection of data from specific category and mineral type.

        :param category: categorical subset of data, defaults to None
        :type category: str, optional
        :param mineral_name: mineral subset of data, defaults to None
        :type mineral_name: str, optional
        :return: reflectance data only of the material collection
        :rtype: pd.DataFrame
        """
        subset_df = self.get_subset_df(
                            category, 
                            library,
                            group,
                            subgroup,
                            species,
                            sample_id)
        refl_df = subset_df.loc[:, cfg.SAMPLE_RES['wvl_min']:]
        return refl_df

    def get_subset_df(self,
            category: str = None,
            library: str = None,
            group: str = None,
            subgroup: str = None,            
            species: str = None,
            sample_id: str = None) -> pd.DataFrame:
        """Return a subset of the dataframe, according to selection
        from specific category and species.

        :param category: categorical subset of data, defaults to None
        :type category: str, optional
        :param species: species subset of data, defaults to None
        :type mineral_name: str, optional
        :return: subset of dataframe according to category and mineral
        :rtype: pd.DataFrame
        """
        if (category is not None) and (species is not None):
            cat_mnrl_selection = (self.main_df.Category == category) & (
                    self.main_df['Species'] == species)
            subset_df = self.main_df.loc[cat_mnrl_selection,:]
        elif category is not None:
            cat_selection = self.main_df.Category == category
            subset_df = self.main_df.loc[cat_selection,:]
        elif species is not None:
            mnrl_selection = self.main_df['Species'] == species
            subset_df = self.main_df.loc[mnrl_selection,:]
        else:
            subset_df = self.main_df
        return subset_df

    def get_hdr_df(self,
            category: str = None,
            species: str = None) -> pd.DataFrame:
        """Return a copy of the header dataframe subset of the array.
        Allows for selection of data from specific category and species.

        :param category: categorical subset of data, defaults to None
        :type category: str, optional
        :param species: species subset of data, defaults to None
        :type species: str, optional
        :return: material collection header data
        :rtype: pd.DataFrame
        """
        subset_df = self.get_subset_df(category, species)
        hdr_df = subset_df.loc[:, self.header_list]
        return hdr_df

    def get_cat_df(self,
            category: str = None,
            species: str = None) -> pd.DataFrame:
        """Returns a copy of the Categories of the array.
        Allows for selection of data from specific category and species.

        :param category: categorical subset of data, defaults to None
        :type category: str, optional
        :param species: species of data, defaults to None
        :type species: str, optional
        :return: category labels
        :rtype: pd.DataFrame
        """
        subset_df = self.get_subset_df(category, species)
        cat_df = pd.DataFrame(subset_df.Category)
        return cat_df
    
    def get_label_df(self,
            category: str = None,
            species: str = None) -> pd.DataFrame:
        """Returns a copy of the Labels (Library, Group, Subgroup, Species) of 
        the array. Allows for selection of data from specific category and species.

        :param category: categorical subset of data, defaults to None
        :type category: str, optional
        :param species: species of data, defaults to None
        :type species: str, optional
        :return: label data
        :rtype: pd.DataFrame
        """
        subset_df = self.get_subset_df(category, species)
        label_df = subset_df.loc[:, ['Library', 'Group', 'Subgroup', 'Species', 'Sample ID']].copy()
        return label_df

    def get_mineral_list(self,
            category: str=None,
            unique: bool=False) -> List:
        """Get the list of mineral types in the DataFrame, and give the
        list of unique mineral types if requested

        :param category: give mineral list belonging to category only, defaults
            to None
        :type category: str, optional
        :param unique: request unique mineral names only, defaults to False
        :type unique: bool, optional
        :return: all mineral names or unique mineral names in dataframe
        :rtype: List
        """
        if category is not None:
            cat_selection = self.main_df.Category == category
            mineral_list = self.main_df.loc[cat_selection, 'Mineral Name']
        else:
            mineral_list = self.main_df['Mineral Name']
        if unique:
            mineral_list = mineral_list.unique()
        return mineral_list

    def export_main_df(self):
        """Export the dataframe to csv and pickle file formats."""

        if cfg.TIME_IT:
            tic = time.perf_counter()
        print('Exporting the Material Collection to CSV and Pickle formats...')
        table_path = Path(self.object_dir / 'tables')
        table_path.mkdir(parents=True, exist_ok=True)
        csv_out_file = Path(table_path, 'material_collection.csv')
        self.main_df.transpose().to_csv(csv_out_file)
        pkl_out_file = Path(self.object_dir, 'material_collection.pkl')
        self.main_df.to_pickle(pkl_out_file)
        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Material Collection exported in {toc - tic:0.4f} seconds.")

    def export_main_df_2_envi(self,
                              retain_dir_tree: bool=False):
        """Export the dataframe to ENVI sli/hdr spectral library format.

        :param retain_dir_tree: retain the directory tree structure of the
            spectral library, defaults to False
        :type retain_dir_tree: bool, optional
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
        print('Exporting the Material Collection to ENVI sli/hdr format...')
        envi_lib_path = Path(self.object_dir / 'envi_sli')
        envi_lib_path.mkdir(parents=True, exist_ok=True)

        if retain_dir_tree:
            # get list of mineral groups
            categories = self.categories
            for cat in categories:
                # make a subdirectory
                cat_path = Path(envi_lib_path / cat)
                cat_path.mkdir(parents=True, exist_ok=True)
                # get the subset of the dataframe
                cat_df = self.get_subset_df(cat)
                # for each mineral name in the category
                mnrls = cat_df['Mineral Name'].unique()
                for mnrl in mnrls:
                    mnrl_path = Path(cat_path / mnrl)
                    mnrl_path.mkdir(parents=True, exist_ok=True)
                    refl_df = self.get_refl_df(category=cat,
                                                            mineral_name=mnrl)
                    cat_df = self.get_cat_df(category=cat,
                                                            mineral_name=mnrl)
                    cat_mnrl_df = pd.concat([refl_df, cat_df], axis=1)
                    # mnrl_df = self.get_subset_df(cat, mnrl)

                    # preparing the sli/hdr files
                    header = {
                        'wavelength': self.wvls,
                        'fwhm:': np.empty(len(self.wvls)).fill(3.0), # typically this information is poorly supplied, so let's estimate with 3 nm for high-resolution spectral library data.
                        'spectra names': cat_mnrl_df.index.to_list(),
                        'wavelength units': 'nm',
                        'description': f'{cat} {mnrl} {self.spectral_library} reflectance data converted from ViSOR format',
                    }
                    
                    spectra = self.get_refl_df(cat, mnrl).to_numpy()

                    # write the sli/hdr files
                    this_mnrl_sli = envi.SpectralLibrary(data=spectra, header=header)

                    this_mnrl_sli.save(str(Path(mnrl_path, mnrl)))

                    # upgrade - plot and save the library profiles alongside these.
                    plotter = SpectralLibraryAnalyser(self)
                    plotter.render_profile_plot(data_df=cat_mnrl_df, cat=cat, mnrl=mnrl, out_dir=mnrl_path)
        else:
            # export entire library to single sli file
            header = {
                'wavelength': self.wvls,
                'fwhm:': np.empty(len(self.wvls)).fill(3.0), # typically this information is poorly supplied, so let's estimate with 3 nm for high-resolution spectral library data.
                'spectra names': self.main_df.index.to_list(),
                'wavelength units': 'nm'
            }
            
            spectra = self.get_refl_df().to_numpy()

            # write the sli/hdr files
            this_lib_sli = envi.SpectralLibrary(data=spectra, header=header)

            this_lib_sli.save(str(envi_lib_path / self.spectral_library))

            plotter = SpectralLibraryAnalyser(self)
            refl_df = self.get_refl_df()                                                    
            cat_df = self.get_cat_df()
            cat_mnrl_df = pd.concat([refl_df, cat_df], axis=1)
            plotter.render_profile_plot(data_df=cat_mnrl_df, out_dir=envi_lib_path)

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Material Collection exported in {toc - tic:0.4f} seconds.")


    def plot_profiles(self, 
                    stacked: bool=False,
                    spectrogram: bool=False,
                    scope: str='all', 
                    groupby: str='Category',
                    pad_factor: float=1/6, # adjust the padding factor for the plot
                    ci: bool=False
                    ) -> plt.Axes:
        """Plot the profiles of the materials
        """
        plotter = SpectralLibraryAnalyser(self)
        axes = plotter.plot_profiles(
                        scope=scope,
                        groupby=groupby,
                        stacked=stacked,
                        spectrogram=spectrogram,
                        pad_factor=pad_factor
                        )
        return axes
    
    def plot_spectrogram(self,
                         continuum_removed: bool=False) -> plt.Axes:
        """Plot the spectrogram of the materials
        """
        plotter = SpectralLibraryAnalyser(self)
        axes = plotter.plot_spectrogram(continuum_removed=continuum_removed)
        return axes
    
    def render_colour(self, 
            illuminant: Literal['D65', 'A', 'C', 'D50', 'D55', 'D75']='D65'
            ) -> Tuple[pd.DataFrame, plt.figure]:
        """Render the colour of the material collection under the specified
        illuminant.
        """
        plotter = SpectralLibraryAnalyser(self)
        colour_spectra_obj = plotter.compute_colour(illuminant)        
        fig = plotter.render_colour(colour_spectra_obj)
        return colour_spectra_obj, fig
