"""Spectral Library Analyser Class

Performs analysis on the given Spectral Library,
visualising and measuring the locations and strengths of
spectral features, independent of instrument resolution.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 18-07-2022
"""
from pathlib import Path
from ast import literal_eval
import copy
import os
import time
from typing import Literal, Tuple, Union, List, Dict
import colour
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import scipy.interpolate as interpolate
import pysptools.spectro as spectro
import sptk.config as cfg

class SpectralLibraryAnalyser():
    """A Class for analysing the contents of a MaterialCollection or Observation
    spectral library.
    """

    def __init__(
            self,
            spectra_obj: object, # TODO fix this to use Union[MaterialCollection, Observation] (currently involves circular import)
            synthetic: bool=False,
            ) -> None:
        """Create a SpectralLibraryAnalyser object

        TODO figure out type hints for Union[MaterialCollection, Observation]
        without circular imports

        :param spectra_obj: Material Collection or Observation to analyse
        :type spectra_obj: Union[MaterialCollection, Observation]
        :param synthetic: _description_, defaults to False
        :type synthetic: bool, optional
        :raises ValueError: _description_
        """
        try:
            self.obj_type = os.path.basename(spectra_obj.object_dir)
        except ValueError as exc:
            raise ValueError('Spectra object type not recognised') from exc

        self.wvls = spectra_obj.wvls
        self.project_name = spectra_obj.project_name
        self.synthetic = synthetic
        if self.synthetic:
            synthetic_tag = 'synthetic'
        else:
            synthetic_tag = 'original'
        self.project_dir = Path(
                                spectra_obj.project_dir,
                                self.obj_type,
                                'analysis',
                                synthetic_tag)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.spectra_obj = spectra_obj
        self.spectra_obj_cr = None
        # check if the feature table has already been made
        band_file = Path(self.project_dir, 'band_info', 'band_info.csv')
        if os.path.isfile(band_file):
            self.band_info = pd.read_csv(band_file, converters={
                    'band_centres': literal_eval,
                    'band_depths': literal_eval,
                    'band_widths': literal_eval,
                    'band_areas': literal_eval})
            print(f'Band Info loaded from {band_file}')
        else:
            self.band_info = pd.DataFrame()

    @staticmethod
    def stack_spectra(
            data_df: pd.DataFrame, 
            wvls: List, 
            noisy: bool=False,
            offset: pd.Series=None, 
            pad_factor: float=1/6) -> pd.DataFrame:
        """Stack the spectra of the dataframe for plotting.

        :param data_df: Reflectance data to be stacked
        :type data_df: pd.DataFrame
        :return: Stacked Reflectance data
        :rtype: pd.DataFrame
        """        
        # apply offsets to reflectance

        # sort categories by alphabetical order
        # get groupby from data_df
        groupby = data_df.columns[-1]
        data_df = data_df.sort_values(by=[groupby, 'Data ID'], ascending=False)

        # reset index
        data_df = data_df.reset_index(drop=True)

        # if ci, then get mean values
        if noisy:
            base_df = data_df.groupby(['Root Data ID'], sort=False).mean(numeric_only=True) # note order of Root Data ID is not preserved on groupby
            base_df = base_df.reset_index()
        else:
            base_df = data_df
        
                
        if offset is None:
            # get minima
            minima = base_df[wvls].min(axis=1)
            maxima = base_df[wvls].max(axis=1)
            spec_range = maxima - minima
            max_range = (spec_range).max()

            # offset the reflectance
            padding =  max_range * pad_factor
            spec_range[spec_range < padding] = padding
            offset = - minima + (spec_range+padding).cumsum().shift(periods=1, fill_value = 0) + padding/2            

            # now have to expand the offset to the full length of the data_df
            if noisy:
                n_repeats = int(len(data_df) / len(base_df))
                offset = offset.repeat(n_repeats)
                offset.reset_index(drop=True, inplace=True)

            # insert extra space where new group starts
            for i in range(1, len(data_df)):
                if data_df[groupby][i] != data_df[groupby][i-1]:
                    offset.loc[i:] += padding  
        else:
            max_range = 0 # hack              
    
        data_df[wvls] = (data_df[wvls].T + offset).T

        # get the last finite reflectance value of each row
        level = (data_df[wvls].T.apply(lambda x: x[x.notnull()].values[-1])).tolist()  
        # for some case here, the offset is resulting in negative 'level' values
        if min(level) < 0:
            print('Negative level values detected')
        # now have to expand the offset to the full length of the data_df
        if noisy:
            n_repeats = int(len(data_df) / len(base_df))
            level = (data_df.groupby(['Root Data ID'], sort=False).mean(numeric_only=True).T.apply(lambda x: x[x.notnull()].values[-1]))
            level.sort_values(inplace=True)
            level = level.repeat(n_repeats).to_list()
        # get the max reflectance value of each row - this is ok to account for noise
        groupby_level = (data_df[wvls].T.apply(lambda x: x[x.notnull()].max())).tolist()
        # get the min reflectance value of each row
        groupby_min = (data_df[wvls].T.apply(lambda x: x[x.notnull()].min())).tolist()

        # if the level is nearest max, va is upper, else lower
        level2max = abs(np.array(level) - np.array(groupby_level))
        level2min = abs(np.array(level) - np.array(groupby_min))
        level2mid = abs(np.array(level) - (np.array(groupby_min) + np.array(groupby_level))/2)
        # va = ['top' if level2max[i] < level2min[i] else 'bottom' for i in range(len(level))]
        # set va to center is nearest mid, top if nearest max, bottom if nearest min
        va = ['center' if level2mid[i] < level2max[i] and level2mid[i] < level2min[i] else 'top' if level2max[i] < level2min[i] else 'bottom' for i in range(len(level))]

        if noisy:
            label = (data_df['Root Data ID']).tolist()
            # only include first instance of unique labels, set all others to ''
            label = [l if l not in label[:i] else '' for i, l in enumerate(label)]
        else:
            label = (data_df['Data ID']).tolist()
        # replace space with '\n' in the label
        label = [l.replace(' ', '\n') for l in label]
        maxima = data_df[wvls].max(axis=1)

        new_level = level.copy()
        new_label = label.copy()
        new_va = va.copy()
   
        i = 1
        c = 1
        while c < len(data_df): # this part is not accounting for noise
            if data_df[groupby][c] != data_df[groupby][c-1]:
                # insert the groupby label at the start of the new group
                # get font heigh in data units                
                new_level.insert(i, groupby_level[c-1]) #level[c-1] + max_range/4)
                new_label.insert(i, data_df[groupby][c-1])
                new_va.insert(i, 'bottom')
                # now the i counter is out of sync, so make it in sync
                i += 1
            i += 1
            c +=1
        # fix missing groupby label at the end
        new_level.append(groupby_level[-1])
        new_label.append(data_df[groupby].iloc[-1])
        new_va.append('bottom')

        annotations = pd.DataFrame(data={'level':new_level, 'label':new_label, 'va':new_va})        

        return data_df, offset, annotations

    def setup_plot(self, 
                   dataframe: pd.DataFrame,
                   groupby: str, 
                   stacked: bool=False,
                   subfig: plt.figure=None) -> Tuple[plt.figure, plt.Axes]:
        """Setup a profile plot figure and axes.

        :param dataframe: the data that will be plotted
        :type dataframe: pd.DataFrame
        :param groupby: _description_
        :type groupby: str
        :param stacked: _description_, defaults to False
        :type stacked: bool, optional
        :return: _description_
        :rtype: Tuple[plt.figure, plt.Axes]
        """  

        # problem - this aspect ratio is for an axes object, 
        # including the legend, or for a stacked figure.
        width_factor = 1 # + 0.3
        height_factor = 1 #1.1 # allow for legend at bottom
        if stacked:
            # stretch the vertical axis of the plot
            # assume that ~6 entries will fit per height_factor of 1 
            if 'Root Data ID' in dataframe.columns:
               height_factor = max(len(dataframe['Root Data ID'].unique()) / 6, height_factor)               
            else:
                height_factor = max(len(dataframe) / 6, height_factor)
            width_factor = width_factor # * 1.2
        else:
            # extend the bottom of the figure to accomodate the legend
            # get number of entries in the legend, determined by groupby
            labels = dataframe[groupby].unique().tolist()
            # get estimate of cumulative length in inches of horizontal 
            # legend, if as a single row
            label_len_chars = np.array([len(label) for label in labels]).cumsum()
            # assume char width is 0.5 char height, handle line is 4 chars
            label_len_inches = (4 + label_len_chars)*cfg.LEGEND_S/72/2
            # all for estimate of row length to be 1/2 figure width
            n_rows = 1 + (label_len_inches[-1]) // (cfg.FIG_SIZE[0]/2)
            n_cols = len(labels) // n_rows
            # give 0.1 inch per row of legend, + offset
            height_factor = height_factor + n_rows*0.1 

        # limit the height to 3
        height_factor = min(height_factor, 3)

        fig_size = (width_factor*cfg.FIG_SIZE[0], height_factor*cfg.FIG_SIZE[1])
        if subfig is not None:
            # set the subfigure size
            ax = subfig.add_subplot()
            fig = subfig
        else:
            fig, ax = plt.subplots(figsize=fig_size, dpi=cfg.DPI, layout='constrained')

        # set up the plot
        sns.set_context("paper")
        # use futura font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Futura'

        return fig, ax

    def render_profile_plot(self,
            data_df: pd.DataFrame,
            ax: plt.Axes,
            scope: Literal[
                'all',
                'libraries', # one plot for each library used
                'categories', # one plot for each category
                'groups', # one plot for each group
                'subgroups', # one plot for each subgroup
                'species', # one plot for each species
                'samples', # one plot for each sample
                str # for specific category, group, subgroup, species, or sample
                ]='all',        
            groupby: Literal[
                'Library', # hue/style by library
                'Category', # hue/style by category
                'Group', # hue/style by group
                'Subgroup', # hue/style by subgroup
                'Species', # hue/style by species
                'Sample ID', # hue/style by Sample ID
                'Data ID' # hue/style by Data ID
                ]='Category',    
            stacked: bool=False,
            pad_factor: float=1/6,
            ci: bool=False,            
            hires_under: bool=False,
            out_dir: Union[bool, str]=False) -> None:
        """Method for producing the plot itself, according to given DataFrame,
        scope and groupby keywords.

        :param data_df: Material Collection or Observation scope/groupby 
            DataFrame subset.
        :type data_df: pd.DataFrame
        :param ax: Axes object to plot on
        :type ax: plt.Axes
        :param scope: Indicates subplot groupings, defaults to 'all'
        :type scope: str, optional
        :param groupby: Indicates hue/style groupings, defaults to 'Category'
        :type groupby: str, optional
        :param stacked: Plot stacked spectra, defaults to False
        :type stacked: bool, optional
        :param pad_factor: Adjust the padding factor for stacked spectra, 
            defaults to 1/6
        :type pad_factor: float, optional
        :param ci: Plot mean spectra with confidence interval if dataset is 
            noisy, defaults to False
        :type ci: bool, optional
        :param hires_under: for Observation under-plot the laboratory spectra, 
                            defaults to False
        :type hires_under: bool, optional
        """

        # if scope is a list, concat to a single string
        if isinstance(scope, list):
            scope = ', '.join(scope)

        # set title and filename
        if scope == 'all':
            title = 'All Entries by ' + groupby
            filename = f'all_entries_by_{groupby.lower()}_profile_plot'
        else:
            title = scope.title()
            filename = f'{scope}_by_{groupby.lower()}'

        if stacked:
            filename = filename + '_stacked'

        if self.spectra_obj.continuum_removed:
            title = title + '\n Continuum Removed'
            filename = filename + '_continuum_removed'

        if self.spectra_obj.noisy:
            with_noise = True
            title = title + ' with Noise'
            filename = filename + '_with_noise'
        else:
            with_noise = False

        if self.obj_type == 'observation':
            inst = self.spectra_obj.instrument.name            
            title = inst.title() + ' Sampled ' + title
            filename = inst + '_sampled_' + filename

        # prepare data frame for plotting
        data_ids = data_df.index
        data_df = data_df.reset_index()

        # offset specta for plotting
        if stacked:
            data_df, offset, annotations = SpectralLibraryAnalyser.stack_spectra(data_df, self.spectra_obj.wvls, noisy=with_noise, pad_factor=pad_factor)
            
        # long form version of plotting, to aggregate data
        if with_noise:
            data_df =pd.melt(data_df, id_vars=['Data ID', groupby, 'Root Data ID'])
        else:
            data_df =pd.melt(data_df, id_vars=['Data ID',groupby])

        # y_max = max([1.0, data_df.value.max()])

        # set plot properties
        hue_flag = groupby

        if self.obj_type == 'observation':
            marker_flag = True
        else:
            marker_flag = False    

        n_channels = len(data_df['variable'].unique())
        if n_channels > 24:
            marker_flag = False  

        if ci:
            errobar = 'sd'
            units=None
            estimator='mean'
        else:
            units = 'Data ID'
            errobar = None
            estimator=None

        sns.lineplot(
            data=data_df,
            x='variable',
            y='value',
            hue=hue_flag,
            style=hue_flag,
            markeredgewidth=0.0,
            markers=marker_flag,
            units=units,
            estimator=estimator,
            errorbar=errobar,
            lw=0.5, 
            ax=ax)

        # format axes
        # set axes limits according to spectarl object spectral range
        # ax.set_xlim(self.spectra_obj.wvls[0]-10, self.spectra_obj.wvls[-1]+10)
        ax.set_xlim(cfg.SAMPLE_RES['wvl_min']-10, cfg.SAMPLE_RES['wvl_max']+10)
        ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        if stacked:
            
            # get font ehight in y-axis data units
            inv = ax.transData.inverted()
            fig_dpi = cfg.DPI  # Dots per inch of the figure
            font_height_display = (cfg.LEGEND_S+1) / 72 * fig_dpi  # Convert points to pixels (1 point = 1/72 inch)            
            _, font_height_data = inv.transform((0, font_height_display)) - inv.transform((0, 0)) # Convert from display (pixel) coordinates to data coordinates
            
            ax.set_ylim(0, annotations['level'].max()+font_height_data)
            ax.set_ylabel('Stacked Reflectance', fontsize=cfg.LABEL_S)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        else:
            ax.set_ylabel('Reflectance', fontsize=cfg.LABEL_S)
        sns.despine(ax=ax, right=True, top=True)
        # Set the font name for axis tick labels to be Arial
        for tick in ax.get_xticklabels():
            tick.set_fontname("Arial")
            tick.set_fontsize(cfg.LABEL_S)
        for tick in ax.get_yticklabels():
            tick.set_fontname("Arial")
            tick.set_fontsize(cfg.LABEL_S)
        # add minor grid lines at 50 nm intervals and major gridlines at 100 nm
        # or minor at 100 and major at 500, depending on spectral range
        spec_range = cfg.SAMPLE_RES['wvl_max'] - cfg.SAMPLE_RES['wvl_min']
        if spec_range <= 1000:
            ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(50))
            ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(100))
        elif spec_range <= 5000:
            ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(100))
            ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(500))
        else:
            ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(500))
            ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(1000))
        if stacked:
            ax.grid(True, which='major',axis='x', lw=0.6)
            ax.grid(True, which='minor',axis='x', lw=0.3)
        else:
            ax.grid(True, which='major',axis='both', lw=0.6)
            ax.grid(True, which='minor',axis='both', lw=0.3)

        # add IR active band locations
            

        # format legend and scope and groupby labels
        handles, labels = ax.get_legend_handles_labels()
        if stacked:
            # annotate the spectra with the Data ID
            # get legend labels
            for idx, annotation in annotations.iterrows():
                if annotation['label'] in labels:
                    # if a group label make text bigger and same colour as lines
                    ax.text(
                        self.spectra_obj.wvls[0], 
                        annotation['level'], 
                        annotation['label'].title(), 
                        # colour it the same as the line
                        color=handles[labels.index(annotation['label'])].get_color(),
                        fontsize=cfg.LEGEND_S+1, 
                        va=annotation['va'])                 
                else:
                    ax.text(
                        self.spectra_obj.wvls[-1]+20, 
                        annotation['level'], 
                        annotation['label'], 
                        fontsize=cfg.LEGEND_S, 
                        va=annotation['va']) 
            ax.legend().remove()
        else:
            labels = data_df[groupby].unique().tolist()
            # get estimate of cumulative length in inches of horizontal 
            # legend, if as a single row
            label_len_chars = np.array([len(label) for label in labels]).cumsum()
            # assume char width is 0.5 char height, handle line is 4 chars
            label_len_inches = (4 + label_len_chars)*cfg.LEGEND_S/72/2
            # all for estimate of row length to be 1/2 figure width
            n_rows = 1 + (label_len_inches[-1]) // (cfg.FIG_SIZE[0]/2)
            n_cols = len(labels) // n_rows
            # capitalise the labels
            if groupby != 'Sample ID':
                labels = [label.title() for label in labels]
            ax.legend(
                    handles,
                    labels,
                    loc="upper center", 
                    fontsize=cfg.LEGEND_S,
                    bbox_to_anchor=(0.5, -0.25),
                    # mode='expand',
                    ncol=n_cols,
                    frameon=True,
                    fancybox=False
                    )
        
        # Special treatment for Instrument Sampled spectra
        if self.obj_type == 'observation':
            if ci:
                title = title + ' Mean ± 1σ'
            # include the hi-res spectra that has been sampeld by the instrument
            if hires_under:
                # find the scope label for the given scope string
                matcol = self.spectra_obj.material_collection
                # if the observation is noisy, then we need to get the non-noisy list of data ids
                if self.spectra_obj.noisy:
                    orig_ids = self.spectra_obj.main_df['Root Data ID'][data_ids].unique()
                    subset_df = matcol.main_df.loc[orig_ids]
                else:
                    subset_df = matcol.main_df.loc[data_ids]
                refl_df = subset_df.loc[:, cfg.SAMPLE_RES['wvl_min']:]
                # get the category and group labels
                groupby_df = subset_df.loc[:, groupby]
                # incorporate error DF into this data for plotting
                hires_df = pd.concat([refl_df, groupby_df], axis=1)
                hires_df = hires_df.reset_index()
                if stacked:
                    if self.spectra_obj.noisy:
                        # reduce offset back to original data list
                        offset = pd.Series(offset.unique())
                    hires_df, _, _ = SpectralLibraryAnalyser.stack_spectra(hires_df, matcol.wvls, False, offset=offset, pad_factor=pad_factor)
                # long form version of plotting, to aggregate data
                hires_df =pd.melt(hires_df, id_vars=['Data ID', groupby]) 

                sns.lineplot(
                    data=hires_df,
                    x='variable',
                    y='value',
                    hue=hue_flag,
                    style=hue_flag,
                    markeredgewidth=0.0,
                    alpha=0.5,
                    units='Data ID',
                    estimator=None,
                    lw=0.5,
                    legend=False,
                    ax=ax)

        plt.title(title, fontsize=cfg.LABEL_S) # update - removing titles from plots

        return ax

    def render_waterfall(self,
            data_df: pd.DataFrame,
            ax: plt.Axes,
            scope: Literal[
                'all',
                'libraries', # one plot for each library used
                'categories', # one plot for each category
                'groups', # one plot for each group
                'subgroups', # one plot for each subgroup
                'species', # one plot for each species
                'samples', # one plot for each sample
                str # for specific category, group, subgroup, species, or sample
                ]='all',        
            groupby: Literal[
                'Library', # hue/style by library
                'Category', # hue/style by category
                'Group', # hue/style by group
                'Subgroup', # hue/style by subgroup
                'Species', # hue/style by species
                'Sample ID', # hue/style by Sample ID
                'Data ID' # hue/style by Data ID
                ]='Category',    
            ci: bool=False,            
            out_dir: Union[bool, str]=False) -> None:
        """Method for producing the plot itself, according to given DataFrame,
        scope and groupby keywords.

        :param data_df: Material Collection or Observation scope/groupby 
            DataFrame subset.
        :type data_df: pd.DataFrame
        :param ax: Axes object to plot on
        :type ax: plt.Axes
        :param scope: Indicates subplot groupings, defaults to 'all'
        :type scope: str, optional
        :param groupby: Indicates hue/style groupings, defaults to 'Category'
        :type groupby: str, optional
        :param stacked: Plot stacked spectra, defaults to False
        :type stacked: bool, optional
        :param pad_factor: Adjust the padding factor for stacked spectra, 
            defaults to 1/6
        :type pad_factor: float, optional
        :param ci: Plot mean spectra with confidence interval if dataset is 
            noisy, defaults to False
        :type ci: bool, optional
        :param hires_under: for Observation under-plot the laboratory spectra, 
                            defaults to False
        :type hires_under: bool, optional
        """

        # if scope is a list, concat to a single string
        if isinstance(scope, list):
            scope = ', '.join(scope)

        # set title and filename
        if scope == 'all':
            title = 'All Entries by ' + groupby
            filename = f'all_entries_by_{groupby.lower()}_profile_plot'
        else:
            title = scope.title()
            filename = f'{scope}_by_{groupby.lower()}'

        if self.spectra_obj.continuum_removed:
            title = title + '\n Continuum Removed'
            filename = filename + '_continuum_removed'
            colorbar_label = 'Band Depth'
        else:
            colorbar_label = 'Reflectance'

        if self.spectra_obj.noisy:
            with_noise = True
            title = title + ' with Noise'
            filename = filename + '_with_noise'
        else:
            with_noise = False

        if self.obj_type == 'observation':
            inst = self.spectra_obj.instrument.name            
            title = inst.title() + ' Sampled ' + title
            filename = inst + '_sampled_' + filename

        # prepare data frame for plotting
        data_df = data_df.sort_values(by=[groupby, 'Data ID'], ascending=False)
        data_ids = data_df.index
        data_df = data_df.reset_index()
            
        if with_noise:
            data = data_df.loc[:, self.spectra_obj.wvls].to_numpy() # do groupby in some way
        else:
            data = data_df.loc[:, self.spectra_obj.wvls].to_numpy()

        if self.spectra_obj.continuum_removed:
            data = 1.0 - data

        # set plot properties
        hue_flag = groupby

        if self.obj_type == 'observation':
            marker_flag = True
        else:
            marker_flag = False    

        n_channels = len(self.spectra_obj.wvls)
        if n_channels > 24:
            marker_flag = False  

        if ci:
            errobar = 'sd'
            units=None
            estimator='mean'
        else:
            units = 'Data ID'
            errobar = None
            estimator=None
        
        # get number of noisy samples
        if with_noise:
            eg_root_data_id = data_df['Root Data ID'].unique()[0]
            n_spacers = data_df[data_df['Root Data ID'] == eg_root_data_id].shape[0]
        else:
            n_spacers = 1
            
        # get locations of each groupby label
        groupby_ticks = []
        groupby_label_y = {}
        for g, grouping in enumerate(data_df[groupby].unique()):
            hi = max(data_df[data_df[groupby] == grouping].index) + g*n_spacers + 1
            # insert n_spacers rows of NaNs to separate the groups
            spacers = np.full((n_spacers, data.shape[1]), np.nan)
            data = np.insert(data, hi, spacers, axis=0)
            groupby_label_y[grouping] = hi
            groupby_ticks.append(hi)

        # get locations of each Data ID or Root Data ID
        id_ticks = []
        if with_noise:
            id_tag = 'Root Data ID'
        else:   
            id_tag = 'Data ID'
        ids = data_df[id_tag].unique()
        id_label_y = {}
        id_bounds = {}
        for id in ids:
            lo = min(data_df[data_df[id_tag] == id].index)
            hi = max(data_df[data_df[id_tag] == id].index) + 1
            grouping = data_df[data_df[id_tag] == id][groupby].iloc[0]
            grouping_n = list(data_df[groupby].unique()).index(grouping)
            lo = lo + grouping_n * n_spacers
            hi = hi + grouping_n * n_spacers
            id_bounds[id] = [lo, hi]
            id_label_y[id] = np.mean([lo, hi])
            id_ticks.append(lo)

        # set plot limits        
        wvl_lo = self.spectra_obj.wvls[0]
        wvl_hi = self.spectra_obj.wvls[-1]
        # set up a good figure size so that good # of samples are shown per cm.
        # A4 = 210 x 297 mm
        # minus 3 cm for border
        # fig size - width = 190 mm
        # fig size - height = 2mm * #samples
        height = 0.2*len(data)
        if height < 15:
            height = 15 # limit the minimum height to 15 cm
        cmap = plt.get_cmap('viridis')
        cmap.set_bad('black')

        # convert data from arbitrary gridding of the spectra to a regular grid
        # this moves the arbitrarily spaced channels of the given Instrument
        # to the regular gridding of the simulation - usually 1 nm.
        # get the number of samples
        if self.obj_type == 'observation':            
            n_lines = len(data)
            lines = np.arange(0, n_lines, 1)
            # create a regular grid
            wvls_out = np.arange(wvl_lo, wvl_hi, cfg.SAMPLE_RES['delta_wvl'])
            # create an out meshgrid
            wvls_out, lines_out = np.meshgrid(wvls_out, lines)
            # use regular interpolator to make use of rectilinear grid. i.e don't
            # interpolate across samples, only across wavelengths
            interp = interpolate.RegularGridInterpolator(
                (lines, self.spectra_obj.wvls),
                data)
            data = interp((lines_out, wvls_out))

        # if the data is from an Observation, then I want the interpolation to
        # reflect the band widths...

        # draw plot
        im = ax.imshow(
                data,
                aspect='auto',
                extent=[wvl_lo, wvl_hi, 0, len(data)],
                interpolation='nearest',
                origin='lower',
                cmap=cmap)
        
        wvl_lo = cfg.SAMPLE_RES['wvl_min']-10
        wvl_hi = cfg.SAMPLE_RES['wvl_max']+10

        # add Species labels
        for id in ids:
            # ax.annotate(
            #     '',
            #     xy=(wvl_hi + 50, id_bounds[id][0]),
            #     xytext=(wvl_hi + 50, id_bounds[id][1]+1),
            #     arrowprops=dict(arrowstyle='<|-|>', shrinkA=0, shrinkB=0),
            #     annotation_clip=False)
            # label = id.replace(' ', '\n')
            ax.annotate(
                id,
                xy=(wvl_hi + 70, id_label_y[id]),
                xytext=(wvl_hi + 70, id_label_y[id]),
                fontsize=cfg.LEGEND_S,
                rotation=0,
                ha='left',
                va='center',
                annotation_clip=False)

        # add category labels
        for grouping in data_df[groupby].unique():
            ax.annotate(
                grouping.title(),
                xy=(wvl_lo, groupby_label_y[grouping]),
                xytext=(wvl_lo, groupby_label_y[grouping]),
                # set fontsize
                fontsize=cfg.LEGEND_S,
                # set font colur
                color='white',
                rotation=0,
                ha='left',
                va='bottom',
                annotation_clip=False)

        # format axes
        # set axes limits according to spectarl object spectral range
        # ax.set_xlim(self.spectra_obj.wvls[0]-10, self.spectra_obj.wvls[-1]+10)
        ax.set_xlim(cfg.SAMPLE_RES['wvl_min']-10, cfg.SAMPLE_RES['wvl_max']+10)
        #set background of the axes plot area to be black, not white
        ax.set_facecolor('black')    
        ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        ax.set_ylabel('', fontsize=cfg.LABEL_S)

        # Set the font name for axis tick labels to be Arial
        for tick in ax.get_xticklabels():
            tick.set_fontname("Arial")
            tick.set_fontsize(cfg.LABEL_S)

        # add minor grid lines at 50 nm intervals and major gridlines at 100 nm
        # or minor at 100 and major at 500, depending on spectral range
        spec_range = cfg.SAMPLE_RES['wvl_max'] - cfg.SAMPLE_RES['wvl_min']
        if spec_range <= 1000:
            ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(50))
            ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(100))
        elif spec_range <= 5000:
            ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(100))
            ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(500))
        else:
            ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(500))
            ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(1000))

        # draw vertical lines at each observation instrument cwl
        if self.obj_type == 'observation':
            # if not a spectrometer
            if self.spectra_obj.instrument.filter_ids[0][0] != 'S':
                for fltr in self.spectra_obj.instrument.filter_ids:
                    cwl = self.spectra_obj.instrument.cwls()[fltr]
                    col = self.spectra_obj.instrument.filter_cols.loc[fltr].to_numpy()
                    ax.axvline(x=cwl, color=col, lw=0.6, ls='-')
                    # # draw dotted lines at ±fwhm
                    # fwhm = self.spectra_obj.instrument.fwhms()[fltr]
                    # ax.axvline(x=cwl+fwhm/2, color=col, lw=0.4, ls='--')
                    # ax.axvline(x=cwl-fwhm/2, color=col, lw=0.4, ls='--')

        # set ticks
        ax.tick_params(left = False, right=False) 
        # set top ticks to be both major and minor
        # ax.tick_params(which='both', axis='x', bottom=True, top=True)
        # ax.grid(which='minor', axis='y', lw=0.5)
        ax.set_yticks(id_ticks, minor=False)
        # ax.tick_params(right=True)
        ax.set_yticklabels([])
        # ax.grid(which='major', axis='y', lw=0.5, color='w')
        # ax.tick_params(right=True)

        # add colourbar
        divider = make_axes_locatable(ax)

        # get font height in y-axis data units
        font_height_inches = (cfg.LEGEND_S) / 72
        cax = divider.new_vertical(size=font_height_inches, pack_start = True)
        # get the figure
        fig = ax.get_figure()
        fig.add_axes(cax)

        cbar = plt.colorbar(im, cax = cax, orientation='horizontal')        
        # move label to right of colorbar
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label(colorbar_label, fontsize=cfg.LEGEND_S, ha='left', x=0, y=0)
        # set ticks to point inward
        cbar.ax.tick_params(direction='in')
        for tick in cbar.ax.get_xticklabels():
            tick.set_fontname("Arial")
            tick.set_fontsize(cfg.LEGEND_S)

        # add IR active band locations
            

        # # format legend and scope and groupby labels
        # handles, labels = ax.get_legend_handles_labels()
        # if stacked:
        #     # annotate the spectra with the Data ID
        #     # get legend labels
        #     for idx, annotation in annotations.iterrows():
        #         if annotation['label'] in labels:
        #             # if a group label make text bigger and same colour as lines
        #             ax.text(
        #                 self.spectra_obj.wvls[0], 
        #                 annotation['level'], 
        #                 annotation['label'].title(), 
        #                 # colour it the same as the line
        #                 color=handles[labels.index(annotation['label'])].get_color(),
        #                 fontsize=cfg.LEGEND_S+1, 
        #                 va=annotation['va'])                 
        #         else:
        #             ax.text(
        #                 self.spectra_obj.wvls[-1]+20, 
        #                 annotation['level'], 
        #                 annotation['label'], 
        #                 fontsize=cfg.LEGEND_S, 
        #                 va=annotation['va']) 
        #     ax.legend().remove()
        # else:
        #     labels = data_df[groupby].unique().tolist()
        #     # get estimate of cumulative length in inches of horizontal 
        #     # legend, if as a single row
        #     label_len_chars = np.array([len(label) for label in labels]).cumsum()
        #     # assume char width is 0.5 char height, handle line is 4 chars
        #     label_len_inches = (4 + label_len_chars)*cfg.LEGEND_S/72/2
        #     # all for estimate of row length to be 1/2 figure width
        #     n_rows = 1 + (label_len_inches[-1]) // (cfg.FIG_SIZE[0]/2)
        #     n_cols = len(labels) // n_rows
        #     # capitalise the labels
        #     if groupby != 'Sample ID':
        #         labels = [label.title() for label in labels]
        #     ax.legend(
        #             handles,
        #             labels,
        #             loc="upper center", 
        #             fontsize=cfg.LEGEND_S,
        #             bbox_to_anchor=(0.5, -0.25),
        #             # mode='expand',
        #             ncol=n_cols,
        #             frameon=True,
        #             fancybox=False
        #             )
        
        # # Special treatment for Instrument Sampled spectra
        # if self.obj_type == 'observation':
        #     if ci:
        #         title = title + ' Mean ± 1σ'
        #     # include the hi-res spectra that has been sampeld by the instrument
        #     if hires_under:
        #         # find the scope label for the given scope string
        #         matcol = self.spectra_obj.material_collection
        #         # if the observation is noisy, then we need to get the non-noisy list of data ids
        #         if self.spectra_obj.noisy:
        #             orig_ids = self.spectra_obj.main_df['Root Data ID'][data_ids].unique()
        #             subset_df = matcol.main_df.loc[orig_ids]
        #         else:
        #             subset_df = matcol.main_df.loc[data_ids]
        #         refl_df = subset_df.loc[:, cfg.SAMPLE_RES['wvl_min']:]
        #         # get the category and group labels
        #         groupby_df = subset_df.loc[:, groupby]
        #         # incorporate error DF into this data for plotting
        #         hires_df = pd.concat([refl_df, groupby_df], axis=1)
        #         hires_df = hires_df.reset_index()
        #         if stacked:
        #             if self.spectra_obj.noisy:
        #                 # reduce offset back to original data list
        #                 offset = pd.Series(offset.unique())
        #             hires_df, _, _ = SpectralLibraryAnalyser.stack_spectra(hires_df, matcol.wvls, False, offset=offset, pad_factor=pad_factor)
        #         # long form version of plotting, to aggregate data
        #         hires_df =pd.melt(hires_df, id_vars=['Data ID', groupby]) 

        #         sns.lineplot(
        #             data=hires_df,
        #             x='variable',
        #             y='value',
        #             hue=hue_flag,
        #             style=hue_flag,
        #             markeredgewidth=0.0,
        #             alpha=0.5,
        #             units='Data ID',
        #             estimator=None,
        #             lw=0.5,
        #             legend=False,
        #             ax=ax)

        ax.set_title(title, fontsize=cfg.LABEL_S) # update - removing titles from plots

        return ax

    def plot_profiles(self,
            scope: Literal[
                'all',
                'libraries', # one plot for each library used
                'categories', # one plot for each category
                'groups', # one plot for each group
                'subgroups', # one plot for each subgroup
                'species', # one plot for each species
                'samples', # one plot for each sample
                str # for specific category, group, subgroup, species, or sample
                ]='all',         
            groupby: Literal[
                'Library', # hue/style by library
                'Category', # hue/style by category
                'Group', # hue/style by group
                'Subgroup', # hue/style by subgroup
                'Species', # hue/style by species
                'Sample ID', # hue/style by Sample ID
                'Data ID' # hue/style by Data ID
                ]='Category',
            stacked: bool=False,
            waterfall: bool=False,
            pad_factor: float=1/6,
            ci: bool=False,
            hires_under: bool=False,
            out_dir: Union[bool, str]=False
            ) -> Tuple[plt.figure, plt.Axes]:
        """Plot the profiles of the materials of the spectral library
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            print('Plotting reflectance profiles of materials...')

        # map the scope to the correct column in the dataframe
        scope_dict = {
                'libraries': 'Library',
                'categories': 'Category',
                'groups': 'Group',
                'subgroups': 'Subgroup',
                'species': 'Species',
                'samples': 'Sample ID'            
            }

        axes = []

        if self.spectra_obj.noisy:
            with_noise = True
        else:
            with_noise = False

        if self.spectra_obj.continuum_removed:
            continuum_removed = True
        else:
            continuum_removed = False

        # Plot the entire material collection in one figure
        if isinstance(scope, str) and scope == 'all':

            subset_df = self.spectra_obj.main_df
            refl_df = subset_df.loc[:, self.spectra_obj.wvls[0]:]
            # get the category and group labels
            groupby_df = subset_df.loc[:, groupby]
            groupby_df = self.spectra_obj.main_df.loc[:, groupby]

            if with_noise:
                # add root id to the data_df
                rootid_df = self.spectra_obj.main_df.loc[:, 'Root Data ID']
                all_df = pd.concat([refl_df, rootid_df, groupby_df], axis=1)
            else:            
                all_df = pd.concat([refl_df, groupby_df], axis=1)
            
            # update to write figure here
            fig, ax = self.setup_plot(all_df, groupby, stacked)

            if waterfall:
                ax = self.render_waterfall(
                            all_df,
                            ax,
                            scope=scope,
                            groupby=groupby,
                            ci=ci)
            else:
                ax = self.render_profile_plot(
                            all_df,
                            ax,
                            scope=scope,
                            groupby=groupby,
                            stacked=stacked,
                            pad_factor=pad_factor,
                            hires_under=hires_under,
                            ci=ci)
            ax = [ax]
            
            fig, ax = self.export_plot(fig, ax, scope, groupby, stacked, waterfall, continuum_removed, with_noise, out_dir)

            if cfg.TIME_IT:
                toc = time.perf_counter()
                print(f"Reflectance profiles plotted in {toc - tic:0.4f} s.")

            return fig, ax
        
        # Plot each 'scope' collection in a separate plot
        elif isinstance(scope, str) and scope in scope_dict.keys():
            
            # get the list of unique values for the scope
            scope_label = scope_dict[scope]
            scope_list = self.spectra_obj.main_df[scope_label].unique().tolist() 

            # compute grid layout of subfigures
            n_subfigs = len(scope_list)
            max_fig_cols = 2
            max_fig_rows = 3
            n_cols = min(n_subfigs, max_fig_cols)
            tot_n_rows = int(np.ceil(n_subfigs/max_fig_cols))
            # compute number of figure pages needed
            n_figs = (tot_n_rows-1)//max_fig_rows + 1
            figs = []
            subfigs = []
            fig_rows = []
            for f in range(n_figs):
                if f < n_figs-1:
                    n_rows = max_fig_rows
                else:
                    n_rows = tot_n_rows - max_fig_rows*(n_figs-1)
                fig = plt.figure(layout='constrained', figsize=(n_cols*cfg.FIG_SIZE[0], n_rows*cfg.FIG_SIZE[1]), dpi=cfg.DPI)           
                subfig = fig.subfigures(n_rows, n_cols)
                figs.append(fig)
                subfigs.append(subfig)
                fig_rows.append(n_rows)

            for s, this_scope in enumerate(scope_list):
                subset_df = self.spectra_obj.main_df[self.spectra_obj.main_df[scope_label]==this_scope]
                refl_df = subset_df.loc[:, self.spectra_obj.wvls[0]:]
                # get the category and group labels
                groupby_df = subset_df.loc[:, groupby]

                if with_noise:
                    # add root id to the data_df
                    rootid_df = subset_df.loc[:, 'Root Data ID']
                    scope_df = pd.concat([refl_df, rootid_df, groupby_df], axis=1)
                else:
                    scope_df = pd.concat([refl_df, groupby_df], axis=1)
                
                sf = s // (max_fig_cols*max_fig_rows)

                # activate figure
                plt.figure(figs[sf].number)

                # update to write figure here
                row_n = s%fig_rows[sf]
                if sf == 0:
                    col_n = s//fig_rows[sf]
                else:
                    col_n = (s-(sf*fig_rows[sf-1]*n_cols))//fig_rows[sf]
                
                scope_fig, ax = self.setup_plot(
                    scope_df, 
                    groupby, 
                    stacked, 
                    subfig=subfigs[sf][row_n][col_n])
                                
                if waterfall:
                    ax = self.render_waterfall(
                                scope_df,
                                ax,
                                scope=this_scope,
                                groupby=groupby,
                                ci=ci)
                else:
                    ax = self.render_profile_plot(
                                scope_df,
                                ax,
                                scope=this_scope,
                                groupby=groupby,
                                stacked=stacked,
                                pad_factor=pad_factor,
                                hires_under=hires_under,
                                ci=ci)

                # add letter to the ax title. If letter is >Z, then wrap arround and use AA, AB, etc.
                if s < 26:
                    subtitle_char = chr(65 + s)
                else:
                    subtitle_char = chr(65 + (s//26 - 1)) + chr(65 + s%26)
                ax.set_title(f'{subtitle_char}. {this_scope.title()}')
                axes.append(ax)

                subfigs[sf][row_n][col_n] = scope_fig

            for f, fig in enumerate(figs):
                if n_figs > 1:
                    index = f'({f+1}/{len(figs)})'
                else:
                    index=None
                fig, axes = self.export_plot(fig, axes, scope, groupby, stacked, waterfall, continuum_removed, with_noise, index, out_dir)
            
            # show the subfigure
            plt.show()

            if cfg.TIME_IT:
                toc = time.perf_counter()
                print(f"Reflectance profiles plotted in {toc - tic:0.4f} s.")

            return axes
        
        # Plot the given scope collection in one figure
        else:
            if isinstance(scope, str):
                scope = [scope]

            subset_df = self.spectra_obj.main_df[self.spectra_obj.main_df.isin(scope).any(axis=1)]

            # subset_df = self.spectra_obj.main_df[self.spectra_obj.main_df[scope_label]==scope]
            refl_df = subset_df.loc[:, self.spectra_obj.wvls[0]:]
            # get the category and group labels
            groupby_df = subset_df.loc[:, groupby]

            if with_noise:
                # add root id to the data_df
                rootid_df = subset_df.loc[:, 'Root Data ID']
                all_df = pd.concat([refl_df, rootid_df, groupby_df], axis=1)
            else:
                all_df = pd.concat([refl_df, groupby_df], axis=1)

            # update to write figure here
            fig, ax = self.setup_plot(all_df, groupby, stacked)

            if waterfall:
                ax = self.render_waterfall(
                            all_df,
                            ax,
                            scope=scope,
                            groupby=groupby,
                            ci=ci)
            else:
                ax = self.render_profile_plot(
                            all_df,
                            ax,
                            scope=scope,
                            groupby=groupby,
                            stacked=stacked,
                            pad_factor=pad_factor,
                            hires_under=hires_under,
                            ci=ci)


            # ax = self.render_profile_plot(
            #             all_df,
            #             ax,
            #             scope=scope,
            #             groupby=groupby,
            #             stacked=stacked,
            #             pad_factor=pad_factor,
            #             hires_under=hires_under,
            #             ci=ci)
            # remove the given title
            ax.set_title('')
            axes.append(ax)

            ax = [ax]
            
            fig, ax = self.export_plot(fig, ax, scope, groupby, stacked, waterfall, continuum_removed, with_noise, out_dir)

            if cfg.TIME_IT:
                toc = time.perf_counter()
                print(f"Reflectance profiles plotted in {toc - tic:0.4f} s.")
            
            return ax

    def export_plot(self, 
                fig, 
                axes, 
                scope,
                groupby: str, 
                stacked: bool=False, 
                waterfall: bool=False,
                continuum_removed: bool=False,
                with_noise: bool=False, 
                index: str=None, # for saving multiple plots
                out_dir: Union[bool, str]=False) -> Tuple[plt.figure, plt.Axes]:
        """Export the profile plot figure to PDF and SVG formats.

        :param fig: the figure to be exported
        :type fig: plt.figure
        :param axes: the axes to be exported
        :type axes: plt.Axes
        :param groupby: _description_
        :type groupby: str
        :param stacked: _description_, defaults to False
        :type stacked: bool, optional
        :param with_noise: _description_, defaults to False
        :type with_noise: bool, optional
        :param out_dir: _description_, defaults to False
        :type out_dir: Union[bool, str], optional
        :return: _description_
        :rtype: Tuple[plt.figure, plt.Axes]
        """

        # do formatting of figure here

        # fig.tight_layout()

        # save figure
        if out_dir:
            out_dir = Path(out_dir)
        else:
            out_dir = Path(self.spectra_obj.object_dir / 'plots')
            out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(scope, list):
            scope = ', '.join(scope)


        # set figure filename and title
        if scope == 'all':
            #get title from figure
            title = axes[0].title.get_text()
            filename = f'all_entries_by_{groupby.lower()}'
        else:
            title = scope.title() + ' grouped by ' + groupby.title()
            filename = f'{scope}_by_{groupby.lower()}'

        if waterfall:
            filename = filename + '_waterfall'
        else:
            filename = filename + '_profile_plot'

        if stacked:
            filename = filename + '_stacked'

        if continuum_removed:
            title = title + '\n Continuum Removed'
            filename = filename + '_continuum_removed'

        if with_noise: # worry about this for observations
            title = title + ' with Noise'
            filename = filename + '_with_noise'

        if self.obj_type == 'observation':
            if scope != 'all':
                title = self.spectra_obj.instrument.name.title() + ' Sampled ' + title
            filename = self.spectra_obj.instrument.name + '_sampled_' + filename

        if scope != 'all':
            if index:
                    title = title + f' {index}'
            fig.suptitle(title, fontsize=cfg.TITLE_S)
        
        if index:
            index = index.replace('(', '').replace(')', '').replace('/', 'of')   
            filename = filename + f'_{index}' 
                
        # pdf output
        output_file = Path(out_dir, filename).with_suffix('.pdf')
        fig.savefig(output_file, bbox_inches='tight', pad_inches = 0, format='pdf')

        # svg output
        plt.rcParams['svg.fonttype'] = 'none'
        output_file = Path(out_dir, filename).with_suffix('.svg')
        fig.savefig(output_file, bbox_inches='tight', pad_inches = 0, format='svg')
    
        # force show of the plot?

        return fig, axes

    # """
    # waterfall Visualisation & Continuum Removal
    # """

    def remove_continuum(self, plot: bool=False):
        """Remove the continuum from all spectra, and overwrite the local copy
        of the spectra object reflectance data.
        """
        self.spectra_obj_cr = copy.deepcopy(self.spectra_obj)
        cat_s = self.spectra_obj_cr.get_cat_df()
        spectra = self.spectra_obj_cr.get_refl_df()
        # self.spectra_obj_cr.object_dir = Path(
        #                         self.project_dir,'continuum_removed') # don't change the object dir
        for index, spectrum in spectra.iterrows():
            # remove NaNs prior to analysis
            notnans = spectrum[~spectrum.isna()].index
            spectrum.index = self.wvls
            spectrum = spectrum.dropna()
            wvls = spectrum.index
            try:
                schq = spectro.SpectrumConvexHullQuotient(spectrum.tolist(),
                                                                 wvls.tolist())
                # TODO investigate: 1. why some entries are skipped,
                # TODO              2. why some hull fitting routines fail.
            except (ValueError, TypeError) as error:
                print(index)
                raise error
            if plot:
                path = Path(
                        self.spectra_obj_cr.object_dir,
                        'cr_algorithm_plots',
                        cat_s.loc[index].Category)
                path.mkdir(parents=True, exist_ok=True)
                schq.plot(path, index, suffix=None)
            # make sure that the correct wavelengths are added back in after
            spectra.loc[index][notnans] = schq.get_continuum_removed_spectrum()
        # rewrite spectra_obj object with continuum removed spectra
        data = spectra.reset_index()
        self.spectra_obj_cr.set_refl_data(data)
        self.spectra_obj_cr.continuum_removed = True
        return self.spectra_obj_cr

    def analyse_bands(self):
        """Find the centre-wavelengths, fwhms, depths and areas of distinct
        bands in each entry.
        Note: method is not complete
        Note: this is not a time sensitive operation, so have not attempted to
        parallelise.
        """

        print('Extracing Band Info...')
        cat_s = self.spectra_obj.get_cat_df()
        spectra = self.spectra_obj.get_refl_df()
        # prepare containers for feature information
        indices = []
        n_features = []
        band_centres = []
        band_depths = []
        band_widths = []
        band_areas = []
        for index, spectrum in spectra.iterrows():
            path = Path(self.project_dir, 'features', cat_s.loc[index].Category)
            path.mkdir(parents=True, exist_ok=True)
            # remove NaNs prior to analysis
            spectrum.index = self.wvls
            spectrum = spectrum.dropna()
            wvls = spectrum.index
            try:
                fea = spectro.FeaturesConvexHullQuotient(
                            spectrum.tolist(),
                            wvls.tolist(),
                            baseline=0.98)
                fea.plot(path, index, feature='all')
                n_feat = fea.get_number_of_kept_features()
                bcs = [fea.get_absorbtion_wavelength(f)
                                                    for f in range(0, n_feat)]
                bds = [1.0 - fea.get_absorbtion_depth(f)
                                                    for f in range(0, n_feat)]
                bws = [fea.get_full_width_at_half_maximum(f)
                                                    for f in range(0, n_feat)]
                bas = [fea.get_area(f) for f in range(0, n_feat)]

                # sort the lists by band centres (lowest to highest)
                bds = [x for _, x in sorted(zip(bcs, bds))]
                bws = [x for _, x in sorted(zip(bcs, bws))]
                bas = [x for _, x in sorted(zip(bcs, bas))]
            except TypeError:
                n_feat = []
                bcs = []
                bds = []
                bws = []
                bas = []

            # append to make list of lists
            indices.append(index)
            n_features.append([n_feat])
            band_centres.append(sorted(bcs))
            band_depths.append(bds)
            band_widths.append(bws)
            band_areas.append(bas)
            print(f"{index}: {n_feat} features")

        # put the feature information in a DataFrame
        df = pd.DataFrame(
                data = {
                    'n_features'  : n_features,
                    'band_centres': band_centres,
                    'band_depths' : band_depths,
                    'band_widths' : band_widths,
                    'band_areas'  : band_areas},
                index = indices)
        hdr_df = self.spectra_obj.get_hdr_df() # add metadata for entries

        # put this as a new feature of the object
        self.band_info = pd.concat([hdr_df, df], axis=1)

        return self.band_info

    def export_band_info(self):
        """Export band info to csv
        """

        path = Path(
            self.project_dir,
            'features', 'feature_table').with_suffix('.csv')
        self.band_info.sort_values(
            by=['Category', 'Species', 'Sample ID'],
            inplace=True,
            ignore_index = True)
        self.band_info.to_csv(path)

        # Order the data for export
        band_centre_info = [
            'Category',
            'Species',
            'band_centres',
            'band_widths',
            'band_depths',
            'band_areas']
        bc_df = self.band_info.loc[:,band_centre_info]

        bc_df['Band Centres (nm)'] = [
            ', '.join(map("{:.0f}".format, l)) for l in bc_df['band_centres']]
        bc_df.drop('band_centres', axis=1, inplace=True)
        bc_df['Band Widths (nm)'] = [
            ', '.join(map("{:.0f}".format, l)) for l in bc_df['band_widths']]
        bc_df.drop('band_widths', axis=1, inplace=True)
        bc_df['Band Depths'] = [
            ', '.join(map("{:.2f}".format, l)) for l in bc_df['band_depths']]
        bc_df.drop('band_depths', axis=1, inplace=True)
        bc_df['Band Areas'] = [
            ', '.join(map("{:.1f}".format, l)) for l in bc_df['band_areas']]
        bc_df.drop('band_areas', axis=1, inplace=True)

        path = Path(
            self.project_dir, 'features', 'feature_table').with_suffix('.tex')
        bc_df.to_latex(path, float_format="%.3f", longtable=True,
                       column_format='lll|p{3.5cm}|p{3.5cm}|p{3.5cm}|p{3.5cm}')

        # Use this script to generate latex table:
        # \documentclass[a4paper, landscape]{article}
        # \usepackage[a4paper,margin=1in,landscape]{geometry}
        # \usepackage{longtable}
        # \usepackage{booktabs}
        # \begin{document}
        # \include{feature_table}
        # \end{document}

        print('end')

    def synthesize_spectra_from_band_info(self) -> object:
        """Produce a new MaterialCollection with reflectance spectra synthesized
        from the band info extracted. The idea is to compare the visualisation
        of this synthesized spectra with the original input data.

        :return: A copy of the Material Collection with synthesized spectra
        :rtype: MaterialCollection
        """

        print('begin synthesizing band spectra')
        mat_synthetic = copy.deepcopy(self.spectra_obj)
        spectra = mat_synthetic.get_refl_df()
        for _, entry in self.band_info.iterrows():
            print(entry['Data ID'])
            spectrum = np.zeros(len(self.wvls))
            cwl = np.array(entry['band_centres'])
            fwhm = np.array(entry['band_widths'])
            depth = np.array(entry['band_depths'])
            if len(cwl) != 0:
                gauss = self.build_gauss_feature(cwl, fwhm, depth)
                spectrum+=gauss
            spectrum = -spectrum + 1
            spectra.loc[entry['Data ID']] = spectrum

        # overwrite reflectance with spectra array
        data = spectra.reset_index()
        mat_synthetic.set_refl_data(data)
        mat_synthetic.project_dir = Path(
                                self.spectra_obj.project_dir,
                                self.obj_type,
                                'analysis',
                                'synthetic')
        mat_synthetic.plot_material_profiles()
        # return new Spectral Library
        return mat_synthetic

    def build_gauss_feature(self,
            cwl: np.array,
            fwhm: np.array,
            depth: np.array) -> np.array:
        """Build Gaussian absorption feature profile according to the given cwl,
        fwhm, depths and wvls.

        :param cwl: Centre wavelength(s) (nm)
        :type cwl: np.array
        :param fwhm: Full-Width at Half Maximum(s) (nm)
        :type fwhm: np.array
        :param depth: Absorption feature depth(s)
        :type depth: np.array
        :returns: Gaussian absorption profile
        :rtype: np.array
        """
        sig = fwhm / 2.355482004503 # convert from fwhm to 1-sigma
        # vectorisation: extend cwls, sigs & wvls to match dimensions
        cwls = np.tile(cwl, [self.wvls.shape[0],1])
        sigs = np.tile(sig, [self.wvls.shape[0],1])
        depths = np.tile(depth, [self.wvls.shape[0],1])
        wvls = np.tile(self.wvls, [cwl.shape[0],1]).transpose()
        # compute the Gaussian profiles in parallel
        gauss = depths*np.exp(-np.power(wvls-cwls,2.)/(2*np.power(sigs, 2.)))
        gauss = np.sum(gauss, axis=1)
        # sum over the correct axis to get final profile
        return gauss
    
    # """
    # Colour Processing & Rendering
    # """

    def plot_filter_ids(self, 
            filter_ids: Tuple[str, str, str],
            ax: plt.Axes=None
            ) -> Tuple[plt.figure, plt.Axes]:
        """Plot the filter ids used for the false colour rendering.

        :param filter_ids: List of filter ids to plot
        :type filter_ids: List[str, str, str]        
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.Axes]
        """
        # ***plot the instrument profiles***        

        # get the filters
        transmission = self.spectra_obj.instrument.get_trans_df()
        wvls = self.spectra_obj.instrument.wvls

        title = filter_ids[0] + ' ' + filter_ids[1] + ' ' + filter_ids[2]

        if ax is None:
            fig, ax = plt.subplots(1, 1, 
                                figsize=(cfg.FIG_SIZE[0], cfg.FIG_SIZE[1]), 
                                dpi=cfg.DPI)
        else:
            fig = ax.get_figure()

        # handle CaSSIS synthetic BLU
        compute_sBLU = False
        if 'sBLU' in filter_ids:
            filter_ids.remove('sBLU')
            filter_ids.append('BLU')
            compute_sBLU = True
            
        cols = ['r', 'g', 'b']
        for filt in filter_ids:
            normed_filter_profile = transmission.loc[filt].to_numpy() / transmission.loc[filt].max()
            ax.plot(wvls, normed_filter_profile, color=cols.pop(0), label=filt, lw=0.8)
        ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Spectral Response', fontsize=cfg.LABEL_S)
        ax.legend(loc='upper right', fontsize=cfg.LEGEND_S)
        ax.set_title(title, fontsize=cfg.LABEL_S)

        if compute_sBLU:
            filter_ids.remove('BLU')
            filter_ids.append('sBLU')

        fig.tight_layout()

        return fig, ax

    def plot_cmfs(self, 
            cmf_label: Literal[colour.MSDS_CMFS.keys()]='CIE 1964 10 Degree Standard Observer',
            ax: plt.Axes=None
            ) -> Tuple[plt.figure, plt.Axes]:
        """Plot the colour matching functions for the given observer.

        :param cmf_label: Label of the colour matching functions, 
                            defaults to 'CIE 1964 10 Degree Standard Observer'
        :type cmf_label: str, optional
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.Axes]
        """
        # ***plot the colour matching functions ***        

        # get the cmfs
        cmfs = colour.MSDS_CMFS[cmf_label]
        extrap_params = colour.SpectralShape(cfg.SAMPLE_RES['wvl_min'], 
                                                cfg.SAMPLE_RES['wvl_max'], 
                                                cfg.SAMPLE_RES['delta_wvl'])
        profiles = cmfs.extrapolate(extrap_params).values
        wvls = cmfs.extrapolate(extrap_params).wavelengths
        labels = ['$\hat{x}$', '$\hat{y}$', '$\hat{z}$']

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(cfg.FIG_SIZE[0], cfg.FIG_SIZE[1]), dpi=cfg.DPI)
        else:
            fig = ax.get_figure()

        cols = ['r', 'g', 'b']
        for profile in profiles.transpose():
            ax.plot(wvls, profile, color=cols.pop(0), label=labels.pop(0), lw=0.8)
        ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Spectral Response', fontsize=cfg.LABEL_S)
        ax.legend(loc='upper right', fontsize=cfg.LEGEND_S)
        ax.set_title(cmf_label, fontsize=cfg.LABEL_S)

        # fig.tight_layout()

        return fig, ax
    
    def plot_illuminant(self,
            illuminant: Literal['D65', 'A', 'C', 'D50', 'D55', 'D75']='D65',
                        ) -> Tuple[plt.figure, plt.Axes]:
        """Plot the spectral power distribution of the given illuminant.

        :param illuminant: Illuminant to plot, defaults to 'D65'
        :type illuminant: str, optional
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.Axes]
        """
        # ***plot the illuminant profiles***
        # get the illuminant
        illum = colour.SDS_ILLUMINANTS[illuminant]

        fig, ax = plt.subplots(1, 1, figsize=(cfg.FIG_SIZE[0], cfg.FIG_SIZE[1]), dpi=cfg.DPI)
        ax.plot(illum.wavelengths, illum.values, lw=0.8)
        ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Spectral Power Distribution', fontsize=cfg.LABEL_S)
        ax.set_title(illuminant, fontsize=cfg.LABEL_S)

        fig.tight_layout()

        return fig, ax
        
    def compute_colour(self,
                conditions: Dict[str, str]={'illuminant': 'D65', 'cmfs': 'CIE 1964 10 Degree Standard Observer', 'label': 'CIE 1964 10 Degree Standard Observer D65'},
                normalise_rgb: bool=False
                    ) -> object:
        """Compute the colour of each entry in the Material Collection or 
        Observation according to the given illuminant.

        :param illuminant: illuminant to use to compute colour, defaults to 'D65'
        :type illuminant: str, optional
        :param normalise_rgb: indicate to normalise the RGB values to intentsity,
                            defaults to False
        :type normalise_rgb: bool, optional
        :return: Table of colours for each entry
        :rtype: pd.DataFrame
        """
        # deep copy the spectra object
        col_obj = copy.deepcopy(self.spectra_obj)
        
        # get the relfectance data from the original spectra object
        refl_df = self.spectra_obj.get_refl_df()
        
        # prepare the copied spectra object for colour rather than reflectance
        col_obj.main_df.drop(self.wvls, axis=1, inplace=True)
        col_obj.main_df.rename(columns={"Reflectance": "Colour"}, inplace=True)        
                
        # convert the reflectance data to colour-science spectral ditributions
        sds = colour.MultiSpectralDistributions(refl_df.T)
        # get the illuminant
        illum = colour.SDS_ILLUMINANTS[conditions['illuminant']]
        
        # convert the spectral distributions to XYZ space
        # set the integration method
        if self.obj_type == 'observation':
            method = 'Integration' # assume discrete non-continuous spectra
        else:
            method = 'ASTM E308' # assume high-resolution continuous spectra
        
        # set the colour matching functions
        cmfs=colour.colorimetry.MSDS_CMFS[conditions['cmfs']]

        # convert from spectra to XYZ tristimulus values
        # Note: if spectral range of observed (instrument) data is inside the
        # range of the cmfs, then the observed spectra will be extrapolated
        # in accordance with CIE 15:2004 and CIE 167:2005, i.e. by assuming that
        # all values out of the range have the same value as the nearest value.
        xyz = colour.sd_to_XYZ(
                                sds, 
                                cmfs=cmfs,
                                illuminant=illum, 
                                k=1/100, 
                                method=method)

        # convert the XYZ values to sRGB        
        illum_ccs = colour.CCS_ILLUMINANTS[conditions['cmfs']][conditions['illuminant']]
        rgb = colour.XYZ_to_sRGB(xyz, illum_ccs)
        rgb = np.clip(rgb, 0, 1)

        if normalise_rgb:
            rgb[:,0] = rgb[:,0] / np.sum(rgb, axis=1)
            rgb[:,1] = rgb[:,1] / np.sum(rgb, axis=1)
            rgb[:,2] = rgb[:,2] / np.sum(rgb, axis=1)
            rgb = np.clip(rgb, 0, 1)
            # recompute XYZ values
            xyz = colour.sRGB_to_XYZ(rgb, illum_ccs)

        # append the XYZ values to the new colour df
        col_obj.main_df['X'] = xyz[:,0]
        col_obj.main_df['Y'] = xyz[:,1]
        col_obj.main_df['Z'] = xyz[:,2]

        col_obj.main_df['R'] = rgb[:,0]
        col_obj.main_df['G'] = rgb[:,1]
        col_obj.main_df['B'] = rgb[:,2]

        # convert to chromaticity coordinates
        xyY = colour.XYZ_to_xyY(xyz)

        col_obj.main_df['x'] = xyY[:,0]
        col_obj.main_df['y'] = xyY[:,1]

        # convert to Lab
        Lab = colour.XYZ_to_Lab(xyz, illum_ccs)

        col_obj.main_df['L*'] = Lab[:,0]
        col_obj.main_df['a*'] = Lab[:,1]
        col_obj.main_df['b*'] = Lab[:,2]

        # convert to LCh
        LCh = colour.Lab_to_LCHab(Lab)
        # col_obj.main_df['L*'] = LCh[:,0]
        col_obj.main_df['C*'] = LCh[:,1]
        col_obj.main_df['h'] = LCh[:,2]
        
        # assign the new colour object df to the spectra object
        # collect the colour space columns under the cmf_label multiindex
        # why drop the other columns?? The information is not going to be 
        # renamed, so it keeps it tidy. Only problem might occur when other 
        # colour-spaces are added...
        col_obj.main_df = col_obj.main_df.loc[:, 'Colour':]
        # drop 'Colour' column
        col_obj.main_df.drop('Colour', axis=1, inplace=True)
        colour_conditions = conditions['label']
        col_obj.main_df.columns = pd.MultiIndex.from_product([[colour_conditions], col_obj.main_df.columns])

        self.spectra_obj.colour_df = col_obj.main_df

        return col_obj

    def compute_false_colour(self,
            conditions: Dict,
            normalise_rgb: bool=False,
            normalise_spectrum: bool=False,
            normalise_L: bool=False,
            overwrite: bool=False) -> object:
        """Compute the false colour of each entry in the given Observation for
        the given instrument filter IDs.

        :param conditions: Dictionary of conditions for the false colour computation
        :type conditions: Dict
        :param normalise_rgb: indicate to normalise the RGB to intensity,
                            defaults to False
        :type normalise_rgb: bool, optional
        :param normalise_spectrum: indicate to normalise the spectra to intensity,
                            defaults to False
        :type normalise_spectrum: bool, optional
        :return: Observation duplicate of false colours for each entry
        :rtype: Observation
        """

        filter_ids = conditions['filter_ids']

        # special case for CaSSIS synthetic RGB image:
        compute_sBLU = False
        if 'sBLU' in filter_ids:
            filter_ids.remove('sBLU')
            filter_ids.append('BLU')
            compute_sBLU = True 

        filter_labels= conditions['label']
        # deep copy the spectra object
        false_col_obj = copy.deepcopy(self.spectra_obj)

        # get the reflectance data for the given filter IDs
        # get the cwls for the filters
        inst_info = self.spectra_obj.instrument.get_metrics()
        cwls = inst_info.loc[filter_ids].cwl.to_list()
        refl_df = self.spectra_obj.get_refl_df()

        if normalise_spectrum:
            norm_refl_df = refl_df.divide(refl_df.max(axis=1),axis=0)
            rgb = norm_refl_df[cwls].to_numpy()    
        else:
            rgb = refl_df[cwls].to_numpy()

        rgb = np.clip(rgb, 0, 1)

        if compute_sBLU:
            # set I/F -> DN conversion factors for CaSSIS from Pommerol et al. 2022
            C_BLU = 2.793E-8
            C_PAN = 1.481E-8
            r_h2 = 1.52**2
            t_exp = 0.6E-3
            rgb[:,2] = 2*rgb[:,0]/(C_PAN*r_h2/t_exp) - 0.3*rgb[:,1]/(C_BLU*r_h2/t_exp)
            rgb[:,1] = rgb[:,1]/(C_BLU*r_h2/t_exp)
            rgb[:,0] = rgb[:,0]/(C_PAN*r_h2/t_exp)
            rgb = rgb/rgb.max()
            filter_ids.remove('BLU')
            filter_ids.append('sBLU')
        rgb = np.clip(rgb, 0, 1)        

        if normalise_rgb:
            rgb[:,0] = rgb[:,0] / np.sum(rgb, axis=1)
            rgb[:,1] = rgb[:,1] / np.sum(rgb, axis=1)
            rgb[:,2] = rgb[:,2] / np.sum(rgb, axis=1)
            rgb = np.clip(rgb, 0, 1)

        # convert RGB back to XYZ and add to object
        XYZ = colour.RGB_to_XYZ(rgb, 'sRGB')

        # convert to Lab
        Lab = colour.XYZ_to_Lab(XYZ)
        
        if normalise_L:
            Lab[:,0] = Lab[:,0]*0.0 + 100.0
            XYZ = colour.Lab_to_XYZ(Lab)
            rgb = colour.XYZ_to_RGB(XYZ, 'sRGB')
            rgb = np.clip(rgb, 0, 1)

        # convert XYZ to xyY and add to object
        xyY = colour.XYZ_to_xyY(XYZ)

        # convert to LCh
        LCh = colour.Lab_to_LCHab(Lab)

        # set the RGB values for the false colours
        false_col_obj.main_df.drop(self.wvls, axis=1, inplace=True)
        false_col_obj.main_df.rename(columns={"Reflectance": "Colour"}, inplace=True)
        false_col_obj.main_df['R'] = rgb[:,0]
        false_col_obj.main_df['G'] = rgb[:,1]
        false_col_obj.main_df['B'] = rgb[:,2]

        # append the XYZ values to the new colour df
        false_col_obj.main_df['X'] = XYZ[:,0]
        false_col_obj.main_df['Y'] = XYZ[:,1]
        false_col_obj.main_df['Z'] = XYZ[:,2]

        # append the xyY values to the new colour df
        false_col_obj.main_df['x'] = xyY[:,0]
        false_col_obj.main_df['y'] = xyY[:,1]
        false_col_obj.main_df['Y'] = xyY[:,2]

        # append the Lab values to the new colour df
        false_col_obj.main_df['L*'] = Lab[:,0]
        false_col_obj.main_df['a*'] = Lab[:,1]
        false_col_obj.main_df['b*'] = Lab[:,2]

        # append the LCh values to the new colour df
        # col_obj.main_df['L*'] = LCh[:,0]
        false_col_obj.main_df['C*'] = LCh[:,1]
        false_col_obj.main_df['h'] = LCh[:,2]

        # assign the new colour object df to the spectra object
        # collect the colour space columns under the cmf_label multiindex
        false_col_obj.main_df = false_col_obj.main_df.loc[:, 'Colour':]
        # drop 'Colour' column
        false_col_obj.main_df.drop('Colour', axis=1, inplace=True)        
        false_col_obj.main_df.columns = pd.MultiIndex.from_product([[filter_labels], false_col_obj.main_df.columns])

        # if colour_df exists, append this df
        if hasattr(self.spectra_obj, 'colour_df'):    
            levels = self.spectra_obj.colour_df.columns.get_level_values(0).unique()
            if filter_labels not in levels:
                self.spectra_obj.colour_df = pd.concat([self.spectra_obj.colour_df, false_col_obj.main_df], axis=1)
            elif overwrite:
                self.spectra_obj.colour_df[filter_labels] = false_col_obj.main_df
        else:
            self.spectra_obj.colour_df = false_col_obj.main_df

        return false_col_obj
    
    def render_colour_contact_sheet(self,
            conditions: Dict,
            scope: Literal[
                'all',
                'libraries', # one plot for each library used
                'categories', # one plot for each category
                'groups', # one plot for each group
                'subgroups', # one plot for each subgroup
                'species', # one plot for each species
                'samples', # one plot for each sample
                str # for specific category, group, subgroup, species, or sample
                ]='all',         
            groupby: Literal[
                'Library', # hue/style by library
                'Category', # hue/style by category
                'Group', # hue/style by group
                'Subgroup', # hue/style by subgroup
                'Species', # hue/style by species
                'Sample ID', # hue/style by Sample ID
                'Data ID' # hue/style by Data ID
                ]='Category',
            srgb_compare: Union[bool, object]=False
            ) -> Tuple[plt.figure, plt.axes]:
        """Render the colour of each entry in the spectral library according
        to the given computed colour coordinates.
        Arrange on A4 portrait sheets.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :param srgb_compare: Indicate if sRGB comparison is to be made against 
            the provided MaterialCollection (object), defaults to False
        :type srgb_compare: Union[bool, object], optional
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        
        # title_sfx = conditions
        title_sfx = conditions['label']
        
        # if compare, load the comparison material collection colour df
        if srgb_compare and self.obj_type == 'observation':
            # TODO handle noise
            srgb_obj = srgb_compare.material_collection
            srgb_rgb = srgb_obj.colour_df[['R', 'G', 'B']].to_numpy()
            srgb_xyY = srgb_obj.colour_df[['x', 'y', 'Y']].to_numpy()
            srgb_cats = srgb_obj.colour_df['Category']
            title_sfx = f"sRGB vs. {title_sfx}"

        # ***Configure Page Layout(s) and Matplotlib Figure(s)***
        # Parse through the complete spectral library to count
        # the number of pages needed and the distribution of the categories
        # and mineral groups over the columns and rows of each page.

        # define page settings
        N_rows = 8 # max number of rows allowed for 1 page  (1 fig)
        N_cols = 6 # max number of columns allowed for 1 page (1 fig)
        spacing = 1.0 # space in inches between rows and columns

        # initiate category counter dicts
        cat_ns = {}     # dict of entries in each category        
        cat_cols = {}   # dict of columns needed for each category
        cat_rows = {}   # dict of rows needed for each category
        t_rows = 0 # counter of total number of rows needed for all categories

        # populate category counters
        
        # handle noise
        if self.spectra_obj.noisy:
            data_df = self.spectra_obj.noiseless_df
            index = data_df.index
            cats = data_df['Category'][index]
        else:
            index = self.spectra_obj.main_df.index
            cats = self.spectra_obj.main_df['Category'][index]
        
        for cat in cats.unique():
            cat_n = len(cats[cats == cat]) # number of entries in given category
            cat_ns[cat] = cat_n # dict lookup of # entries in category
            if cat_n >= N_cols: # if there are more entries than columns...
                cat_cols[cat] = N_cols  # ...set number of columns to N_cols
                 # compute the number of rows needed given the fixed # columns
                cat_rows[cat] = int(np.ceil(cat_n / N_cols))
            else: # otherwise the number of columns is the number of entries
                cat_cols[cat] = cat_n 
                cat_rows[cat] = 1 # and the number of rows is 1                        
            t_rows += cat_rows[cat] # running total of rows needed for all categories

        N_pages = 1 + (t_rows-1) // N_rows # number of pages needed to plot all
        
        # Build a map of distribution of categories and entries across the pages
        page_cats = {} # dict of pages with categories and entry indices
        page = 0
        cat_list = cats.unique().to_list()
        cat = cat_list.pop()
        cat_rows_left = cat_rows[cat]
        i = 0 # initialise the first index of the category
        f = 0 # initialise the last index of the category
        page_cats[page] = {} # initialise the first page cat dictionary
        page_rows = 0
        
        rows_used = 0  # count the used rows of the page
        # if it hits N_rows, then move to next page
        while rows_used < t_rows:  # stop when every row is rendered                     
            if cat_rows_left + page_rows < N_rows: 
                # if the rest of the category fits on the page,
                # add the category to the page
                f = i + cat_rows_left*N_cols - 1
                page_cats[page][cat] = (i,f)
                page_rows += cat_rows_left
                rows_used += cat_rows_left
                if f < i:                    
                    raise ValueError(f'Contact sheet counting error for p. {page} cat. {cat}: f < i')
                if len(cat_list) > 0:
                    cat = cat_list.pop()
                    cat_rows_left = cat_rows[cat]
                    i = 0                
            elif cat_rows_left + page_rows == N_rows: 
                # if the rest of the category fills the page,
                # add the category to the page
                f = i + cat_rows_left*N_cols - 1
                page_cats[page][cat] = (i,f)
                page_rows += cat_rows_left
                rows_used += cat_rows_left
                if f < i:
                    raise ValueError(f'Contact sheet counting error for p. {page} cat. {cat}: f < i')
                if len(cat_list) > 0:
                    cat = cat_list.pop()
                    cat_rows_left = cat_rows[cat]
                    i = 0
                if rows_used < t_rows:
                    page += 1
                    page_rows = 0
                    page_cats[page] = {}
            elif cat_rows_left + page_rows > N_rows: 
                # if the rest of the category does not fit on the page,
                # only use rows up to total of N_rows
                cat_r = N_rows - page_rows # the number of rows available
                f = i + cat_r*N_cols - 1
                cat_rows_left -= cat_r
                rows_used += cat_r
                page_rows += cat_r
                page_cats[page][cat] = (i,f)
                if f < i:
                    raise ValueError(f'Contact sheet counting error for p. {page} cat. {cat}: f < i')
                page += 1
                page_rows = 0
                page_cats[page] = {}
                i = f + 1

        # ***Render the Colour Contact Sheet according to the above mapping***
        figs = []
        axes = []
        for page in np.arange(N_pages):
            
            # *** Formatting the page of the figure ***
            
            # get the categories on the page
            cats_on_page = list(page_cats[page].keys())

            # get the total number of rows used on the page
            rows_used = 0
            for cat in cats_on_page:    
                page_cat_i = page_cats[page][cat][0]
                page_cat_f = page_cats[page][cat][1]
                cat_rows_used = int((page_cat_f - page_cat_i + 1) / N_cols)
                rows_used += cat_rows_used
            
            # draw figure on page
            fig = plt.figure(
                figsize=(N_cols*spacing, rows_used*spacing), 
                dpi=cfg.DPI, 
                layout='compressed')
            spec = fig.add_gridspec(rows_used,1) # use gridspec to handle multi-page plots

            # *** Drawing the figure on the page ***

            r = 0 # initialise the row counter
            for c, cat in enumerate(cats_on_page):
                
                # use scatterplot to distribute entries evenly over the N_cols 
                # and N_rows of the grid.

                # get the index of the samples in this category    
                i = page_cats[page][cat][0]
                f = page_cats[page][cat][1]
                if self.spectra_obj.noisy:
                    colour_df = self.spectra_obj.colour_df[conditions['label']]
                    labeled_colour_df = pd.concat([self.spectra_obj.get_hdr_df(), colour_df], axis=1)
                    data_df = self.spectra_obj.colour_df.groupby('Root Data ID', level=0).mean(numeric_only=True)
                    cat_index = self.spectra_obj.noiseless_df['Category'] == cat
                    cat_df = data_df[cat_index]
                else:
                    cat_df = self.spectra_obj.colour_df[self.spectra_obj.main_df['Category'] == cat]                
                cat_rgb = cat_df[conditions['label']][['R', 'G', 'B']].iloc[i:f+1]
                
                # get the number of rows used by this category
                cat_r = int((f - i + 1) / N_cols)

                # add a subplot for the category
                ax_c = fig.add_subplot(spec[r:r+cat_r, :], adjustable='box')

                r += cat_r

                # get the index of the comparison samples in this category
                if srgb_compare and self.obj_type == 'observation':
                    srgb_obj = srgb_compare.material_collection
                    srgb_cat_df = srgb_obj.colour_df[srgb_obj.colour_df['Category'] == cat]
                    srgb_cat_rgb = srgb_cat_df[['R', 'G', 'B']].iloc[i:f+1]                

                for i, entry in enumerate(cat_rgb.index):
                    
                    x = i % cat_cols[cat] + 0.5
                    y = np.ceil(i // cat_cols[cat]) + 0.5

                    if srgb_compare and self.obj_type == 'observation':
                        srgb_col = srgb_cat_rgb.loc[entry].to_numpy()
                        ax_c.scatter(x-0.15,y,
                                        color=srgb_col,
                                        s=200,
                                        edgecolor='black')
                        col = cat_rgb.loc[entry].to_numpy()
                        ax_c.scatter(x+0.15,y,
                                        color=col,
                                        s=200,
                                        edgecolor='black')
                    else:
                        col = cat_rgb.loc[entry].to_numpy()
                        ax_c.scatter(x,y,
                                        color=col,
                                        s=200,
                                        edgecolor='black')
                    
                    # annotate                                                            
                    entry = str(entry).replace('_', '\n') # turn underscore into carriage return
                    entry = str(entry).replace(' ', '\n') # get Species
                    entry = entry.title() 
                    ax_c.annotate(entry, (x, y), 
                                     (0,-1.5), 
                                     textcoords='offset fontsize', 
                                     fontsize=cfg.LEGEND_S, 
                                     ha='center', va='top')
                # remove the axes
                ax_c.set_xlim(0, cat_cols[cat], auto=False)
                ax_c.set_ylim(0, cat_r, auto=False)
                ax_c.invert_yaxis()
                ax_c.set_aspect('equal', adjustable='box', share=True)
                ax_c.axis('off')
                # set title
                ax_c.set_title(str.capitalize(cat), 
                               fontsize=cfg.TITLE_S, 
                               y = 1.0, 
                               verticalalignment= 'bottom', 
                               pad=-cfg.TITLE_S)            
            fig.suptitle(f'{self.spectra_obj.spectral_library} '+title_sfx, fontsize=cfg.TITLE_S)
            fig.tight_layout()

            # export page as pdf page
            contact_sheet_dir=Path(self.project_dir,'contact_sheet')            
            contact_sheet_dir.mkdir(parents=True, exist_ok=True)
            filename = f'{self.spectra_obj.spectral_library} {conditions["label"]} page_{page}'
            filepath=Path(contact_sheet_dir, filename).with_suffix('.pdf') # locked as PDF not PNG
            plt.savefig(filepath, bbox_inches='tight', pad_inches = 0.1)

            figs.append(fig)
            axes.append(ax_c)
        
        return figs, axes
    
    def render_rgb_cube(self,
            conditions: Dict,
            scope: Literal[
                'all',
                'libraries', # one plot for each library used
                'categories', # one plot for each category
                'groups', # one plot for each group
                'subgroups', # one plot for each subgroup
                'species', # one plot for each species
                'samples', # one plot for each sample
                str # for specific category, group, subgroup, species, or sample
                ]='all',         
            groupby: Literal[
                'Library', # hue/style by library
                'Category', # hue/style by category
                'Group', # hue/style by group
                'Subgroup', # hue/style by subgroup
                'Species', # hue/style by species
                'Sample ID', # hue/style by Sample ID
                'Data ID' # hue/style by Data ID
                ]='Category',
            ax: plt.Axes=None) -> Tuple[plt.figure, plt.axes]:
        """Render the RGB cube of the spectral library for the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions['label']
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        rgb = self.spectra_obj.colour_df[conditions['label']][['R', 'G', 'B']].to_numpy()

        # make a 3D plot of rgb array
        if ax is None:
            fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
        cat_codes = cats.astype('category').cat.codes
        cat_codes.index = cats.values
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>',
                        'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']
        for i, cat in enumerate(uniq_cats):
            cat_rgb = rgb[cats == cat]
            # plot each point in turn to give separate colour
            sym = syms_list[cat_codes.loc[cat].unique()[0]]
            for e in range(cat_rgb.shape[0]):
                mrkr, stem, base = ax.stem([cat_rgb[e,2]], 
                                            [cat_rgb[e,0]], 
                                            [cat_rgb[e,1]], 
                                            label=cat, 
                                            markerfmt=sym)                
                mrkr.set_markerfacecolor(cat_rgb[e,:])
                mrkr.set_markeredgecolor(cat_rgb[e,:])
                stem.set_color(cat_rgb[e,:])                
                base.set_color(cat_rgb[e,:])
        ax.set_xlabel('B', color='blue', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='x', colors='blue', labelsize=cfg.LEGEND_S)
        ax.set_ylabel('R', color='red', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='y', colors='red', labelsize=cfg.LEGEND_S)
        ax.set_zlabel('G', color='green', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='z', colors='green', labelsize=cfg.LEGEND_S)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)       
        ax.zaxis.labelpad=-3.5
        ax.invert_yaxis()
        ax.set_title(f'{title_sfx}\n RGB Cube', fontsize=cfg.LABEL_S)
        
        # fig.tight_layout()
        # export as pdf   

        return fig, ax
    
    def render_XYZ_cube(self,
            conditions: Dict,
            scope: Literal[
                'all',
                'libraries', # one plot for each library used
                'categories', # one plot for each category
                'groups', # one plot for each group
                'subgroups', # one plot for each subgroup
                'species', # one plot for each species
                'samples', # one plot for each sample
                str # for specific category, group, subgroup, species, or sample
                ]='all',         
            groupby: Literal[
                'Library', # hue/style by library
                'Category', # hue/style by category
                'Group', # hue/style by group
                'Subgroup', # hue/style by subgroup
                'Species', # hue/style by species
                'Sample ID', # hue/style by Sample ID
                'Data ID' # hue/style by Data ID
                ]='Category',
            ax: plt.Axes=None) -> Tuple[plt.figure, plt.axes]:
        """Render the XYZ cube of the spectral library for the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions['label']
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        XYZ = self.spectra_obj.colour_df[conditions['label']][['X', 'Y', 'Z']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions['label']][['R', 'G', 'B']].to_numpy()

        # make a 3D plot of XYZ array
        if ax is None:
            fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
        cat_codes = cats.astype('category').cat.codes
        cat_codes.index = cats.values
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>',
                        'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']
        for i, cat in enumerate(uniq_cats):
            cat_XYZ = XYZ[cats == cat]
            cat_rgb = rgb[cats == cat]
            sym = syms_list[cat_codes.loc[cat].unique()[0]]
            ax.scatter(cat_XYZ[:,0], cat_XYZ[:,1], cat_XYZ[:,2], 
                        c=cat_rgb, 
                        depthshade=True, 
                        marker=sym, 
                        alpha=0.9,
                        label=cat)
        ax.set_xlabel('X', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='x', labelsize=cfg.LEGEND_S)
        ax.set_ylabel('Y', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='y', labelsize=cfg.LEGEND_S)
        ax.set_zlabel('Z', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='z', labelsize=cfg.LEGEND_S)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)   
        ax.zaxis.labelpad=-3.5     
        ax.invert_yaxis()
        ax.set_title(f'{title_sfx}\n XYZ Cube', fontsize=cfg.LABEL_S)
        
        fig.tight_layout()
        # export as pdf    

        return fig, ax
    
    def render_xyY_cube(self,
            conditions: Dict,
            scope: Literal[
                'all',
                'libraries', # one plot for each library used
                'categories', # one plot for each category
                'groups', # one plot for each group
                'subgroups', # one plot for each subgroup
                'species', # one plot for each species
                'samples', # one plot for each sample
                str # for specific category, group, subgroup, species, or sample
                ]='all',         
            groupby: Literal[
                'Library', # hue/style by library
                'Category', # hue/style by category
                'Group', # hue/style by group
                'Subgroup', # hue/style by subgroup
                'Species', # hue/style by species
                'Sample ID', # hue/style by Sample ID
                'Data ID' # hue/style by Data ID
                ]='Category',
            ax: plt.Axes=None) -> Tuple[plt.figure, plt.axes]:
        """Render the xyY cube of the spectral library for the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions['label']
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        xyY = self.spectra_obj.colour_df[conditions['label']][['x', 'y', 'Y']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions['label']][['R', 'G', 'B']].to_numpy()

        # make a 3D plot of xyY array
        if ax is None:
            fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
        cat_codes = cats.astype('category').cat.codes
        cat_codes.index = cats.values
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>',
                        'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']
        for i, cat in enumerate(uniq_cats):
            cat_xyY = xyY[cats == cat]
            cat_rgb = rgb[cats == cat]
            sym = syms_list[cat_codes.loc[cat].unique()[0]]
            for e in range(cat_xyY.shape[0]):
                markerline, stemlines, baseline = ax.stem([cat_xyY[e,0]], [cat_xyY[e,1]], [cat_xyY[e,2]], label=cat, markerfmt=sym)                
                markerline.set_markerfacecolor(cat_rgb[e,:])
                markerline.set_markeredgecolor(cat_rgb[e,:])
                stemlines.set_color(cat_rgb[e,:])                
                baseline.set_color(cat_rgb[e,:])
        ax.set_xlabel('x', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='x', labelsize=cfg.LEGEND_S)
        ax.set_ylabel('y', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='y', labelsize=cfg.LEGEND_S)
        ax.set_zlabel('Y', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='z', labelsize=cfg.LEGEND_S)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)  
        ax.zaxis.labelpad=-3.5      
        # ax.invert_yaxis()
        ax.set_title(f'{title_sfx}\n xyY Cube', fontsize=cfg.LABEL_S)
        
        fig.tight_layout()
        # export as pdf    

        return fig, ax

    def render_xy_chromaticity_diagram(self,
            scope: Literal[
                'all',
                'libraries', # one plot for each library used
                'categories', # one plot for each category
                'groups', # one plot for each group
                'subgroups', # one plot for each subgroup
                'species', # one plot for each species
                'samples', # one plot for each sample
                str # for specific category, group, subgroup, species, or sample
                ]='all',         
            groupby: Literal[
                'Library', # hue/style by library
                'Category', # hue/style by category
                'Group', # hue/style by group
                'Subgroup', # hue/style by subgroup
                'Species', # hue/style by species
                'Sample ID', # hue/style by Sample ID
                'Data ID' # hue/style by Data ID
                ]='Category',
            conditions: Dict,
            ax: plt.Axes=None) -> Tuple[plt.figure, plt.axes]:
        """Render the xy chromaticity diagram of the spectral library for 
        the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """

        # *** Plotting in chromaticity space ***

        title_sfx = conditions['label']
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        xyY = self.spectra_obj.colour_df[conditions['label']][['x', 'y', 'Y']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions['label']][['R', 'G', 'B']].to_numpy()

        if ax is None:        
            fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(
                cmfs=colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer'],
                show=False, 
                show_spectral_locus=False,
                show_diagram_colours=True,
                transparent_background=False,            
                figsize=(cfg.FIG_SIZE[0], cfg.FIG_SIZE[1]),
                dpi=cfg.DPI
                )
            # set figure size
            fig.set_size_inches(cfg.FIG_SIZE[0], cfg.FIG_SIZE[1])
        else:
            fig = ax.get_figure()
 
        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        # cat to integer
        uniq_cats = cats.unique()
        cat_codes = cats.astype('category').cat.codes
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>', 
                     'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']

        min_list = index.to_list()
        for i, min in enumerate(min_list):
           
            sym = syms_list[cat_codes.loc[min]]
            swatch_name = f"{min}".capitalize()
                        
            xy = xyY[i, 0:2]
            x, y = xy
            ax.plot(x, y, 
                    f"{sym}", color=list(rgb[i]), 
                    label=swatch_name, 
                    markeredgewidth=0.4,
                    markersize=4,
                    markeredgecolor='white')
        
        # plot the sRGB space in the chromaticity diagram
        sRGB_ps = colour.RGB_COLOURSPACES['sRGB'].primaries
        ax.plot(sRGB_ps[:,0],sRGB_ps[:,1], color='k', marker='', label='sRGB', linewidth=0.5)
        ax.plot(sRGB_ps[[2,0],0],sRGB_ps[[2,0],1], color='k', marker='', linewidth=0.5) 

        # if number of entries is >20, just add legend for categories, as a white symbol
        handles, labels = ax.get_legend_handles_labels()

        if len(labels) > 20:
            labels = uniq_cats.categories.to_list()
            # set the handle to the category symbol
            handles = [mpl.lines.Line2D([0], [0], 
                                color='w', markeredgecolor='k', 
                                marker=syms_list[i], markersize=6, 
                                label=uniq_cats.categories.to_list()[i]) for i in np.arange(len(uniq_cats))]

        else:
            # insert category labels into the legend        
            for cat in uniq_cats:
                cat_index = cats[cats == cat].index
                cat_index = labels.index(cat_index[0])
                labels.insert(cat_index, cat)
                handles.insert(cat_index, mpl.lines.Line2D([0], [0], 
                                    color='w', marker='o', markersize=6, label=cat))
        
        ax.legend(handles, labels, loc='upper right', fontsize=cfg.LEGEND_S-2)
        
        # set axes font sizes
        ax.set_xlabel('CIE $x$', fontsize=cfg.LABEL_S)
        ax.set_ylabel('CIE $y$', fontsize=cfg.LABEL_S)

        # set axes tick label font size
        ax.tick_params(axis='both', labelsize=cfg.LABEL_S)

        # set x and y limits
        ax.set_xlim(0, 0.95)
        ax.set_ylim(0, 0.85)

        # reset title
        ax.set_title(f'{title_sfx}\n CIE 1931 Chromaticity Diagram', fontsize=cfg.LABEL_S)

        # set figure size
        fig.set_size_inches(1.5*cfg.FIG_SIZE[0], 1.5*cfg.FIG_SIZE[1])

        # set DPI
        fig.set_dpi(cfg.DPI)

        # fig.tight_layout()      

        return fig, ax

    def render_Lab_cube(self,        
            conditions: Dict,
            scope: Literal[
                    'all',
                    'libraries', # one plot for each library used
                    'categories', # one plot for each category
                    'groups', # one plot for each group
                    'subgroups', # one plot for each subgroup
                    'species', # one plot for each species
                    'samples', # one plot for each sample
                    str # for specific category, group, subgroup, species, or sample
                    ]='all',         
            groupby: Literal[
                    'Library', # hue/style by library
                    'Category', # hue/style by category
                    'Group', # hue/style by group
                    'Subgroup', # hue/style by subgroup
                    'Species', # hue/style by species
                    'Sample ID', # hue/style by Sample ID
                    'Data ID' # hue/style by Data ID
                    ]='Category',
            ax: plt.Axes=None) -> Tuple[plt.figure, plt.axes]:
        """Render the L*a*b* cube of the spectral library for the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions['label']
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        Lab = self.spectra_obj.colour_df[conditions['label']][['L*', 'a*', 'b*']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions['label']][['R', 'G', 'B']].to_numpy()

        # make a 3D plot of Lab array
        if ax is None:
            fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
        cat_codes = cats.astype('category').cat.codes
        cat_codes.index = cats.values
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>',
                        'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']
        for i, cat in enumerate(uniq_cats):
            cat_Lab = Lab[cats == cat]
            cat_rgb = rgb[cats == cat]
            sym = syms_list[cat_codes.loc[cat].unique()[0]]
            for e in range(cat_Lab.shape[0]):
                markerline, stemlines, baseline = ax.stem([cat_Lab[e,1]], [cat_Lab[e,2]], [cat_Lab[e,0]], label=cat, markerfmt=sym)                
                markerline.set_markerfacecolor(cat_rgb[e,:])
                markerline.set_markeredgecolor(cat_rgb[e,:])
                stemlines.set_color(cat_rgb[e,:])                
                baseline.set_color(cat_rgb[e,:])
            ax.scatter(cat_Lab[:,1], cat_Lab[:,2], cat_Lab[:,0], 
                        c=cat_rgb, 
                        depthshade=True, 
                        marker=syms_list[i], 
                        alpha=0.9,
                        label=cat)
        ax.set_xlabel('a*', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='x', labelsize=cfg.LEGEND_S)
        ax.set_ylabel('b*', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='y', labelsize=cfg.LEGEND_S)
        ax.set_zlabel('L*', fontsize=cfg.LEGEND_S)
        ax.tick_params(axis='z', labelsize=cfg.LEGEND_S)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(0, 100)  
        ax.zaxis.labelpad=-3.5              
        ax.set_title(f'{title_sfx}\n CIE L*a*b* Cube', fontsize=cfg.LABEL_S)
        fig.tight_layout()
        # export as pdf    

        # ab plot
        fig_ab = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        ax_ab = fig_ab.add_subplot(111)

        # add grid
        ax_ab.set_axisbelow(True)
        ax_ab.grid(True)

        for i, cat in enumerate(uniq_cats):
            cat_Lab = Lab[cats == cat]
            cat_rgb = rgb[cats == cat]
            ax_ab.scatter(cat_Lab[:,1], cat_Lab[:,2], 
                        c=cat_rgb,
                        marker=syms_list[i], label=cat)
        ax_ab.set_xlabel('a*', fontsize=cfg.LEGEND_S)
        ax_ab.tick_params(axis='x', labelsize=cfg.LEGEND_S)
        ax_ab.set_ylabel('b*', fontsize=cfg.LEGEND_S)
        ax_ab.tick_params(axis='y', labelsize=cfg.LEGEND_S)
        ax_ab.set_xlim(-100, 100)
        ax_ab.set_ylim(-100, 100)
        ax_ab.set_title(f'{title_sfx}\n CIE a*b* Plane', fontsize=cfg.LABEL_S)
        
        fig_ab.tight_layout()        
        # export as pdf

        return fig, ax

    def render_Chab_plane(self,        
            conditions: Dict,
            scope: Literal[
                    'all',
                    'libraries', # one plot for each library used
                    'categories', # one plot for each category
                    'groups', # one plot for each group
                    'subgroups', # one plot for each subgroup
                    'species', # one plot for each species
                    'samples', # one plot for each sample
                    str # for specific category, group, subgroup, species, or sample
                ]='all',         
            groupby: Literal[
                    'Library', # hue/style by library
                    'Category', # hue/style by category
                    'Group', # hue/style by group
                    'Subgroup', # hue/style by subgroup
                    'Species', # hue/style by species
                    'Sample ID', # hue/style by Sample ID
                    'Data ID' # hue/style by Data ID
                ]='Category',
            ax: plt.Axes=None) -> Tuple[plt.figure, plt.axes]:
        """Render the C*h(ab) plane of the spectral library for the given conditions.
        Note that no method is available AFAIK for rendring a polar cyclindrical
        plot of the LCh space. So only plotting 2D C*h polar plane.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions['label']
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        LCh = self.spectra_obj.colour_df[conditions['label']][['L*', 'C*', 'h']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions['label']][['R', 'G', 'B']].to_numpy()


        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
        cat_codes = cats.astype('category').cat.codes
        cat_codes.index = cats.values
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>',
                        'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']
        
        # Disable LCh cube plot - cannot plot polar cylindrical
        # # make a 3D plot of rgb array
        # fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        # ax = fig.add_subplot(111, projection='3d')
        # for i, cat in enumerate(uniq_cats):
        #     cat_LCh = LCh[cats == cat]
        #     cat_rgb = rgb[cats == cat]
        #     ax.scatter(cat_LCh[:,1], cat_LCh[:,2], cat_LCh[:,0], 
        #                 c=cat_rgb, 
        #                 depthshade=True, 
        #                 marker=syms_list[i], 
        #                 alpha=0.9,
        #                 label=cat)
        # ax.set_xlabel('C*', fontsize=cfg.LEGEND_S)
        # ax.tick_params(axis='x', labelsize=cfg.LEGEND_S)
        # ax.set_ylabel('h*', fontsize=cfg.LEGEND_S)
        # ax.tick_params(axis='y', labelsize=cfg.LEGEND_S)
        # ax.set_zlabel('L*', fontsize=cfg.LEGEND_S)
        # ax.tick_params(axis='z', labelsize=cfg.LEGEND_S)
        # ax.set_xlim(0, 100)
        # ax.set_ylim(0, 360)
        # ax.set_zlim(0, 100)  
        # ax.zaxis.labelpad=-3.5      
        # ax.invert_yaxis()
        # ax.set_title(f'{title_sfx}\n L*a*b* Cube', fontsize=cfg.LABEL_S)
        # fig.tight_layout()
        # # export as pdf    

        # Ch plot
        if ax is None:
            fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            ax = fig.add_subplot(111, projection='polar')
        else:
            fig = ax.get_figure()

        # add grid
        ax.set_axisbelow(True)
        ax.grid(True)        

        for i, cat in enumerate(uniq_cats):
            cat_LCh = LCh[cats == cat]
            cat_rgb = rgb[cats == cat]
            sym = syms_list[cat_codes.loc[cat].unique()[0]]
            ax.scatter(np.deg2rad(cat_LCh[:,2]), cat_LCh[:,1], 
                        c=cat_rgb,
                        marker=sym, label=cat)
            ax.set_rmax(100)  
            # set radial axis font size
            ax.tick_params(axis='x', labelsize=cfg.LEGEND_S)
            ax.tick_params(axis='y', labelsize=cfg.LEGEND_S)
        ax.set_title(fr"{title_sfx}""\n CIE L*C*h(ab) Chroma "rf"($r$) hue ($\theta$) Plane", fontsize=cfg.LABEL_S)
        
        # fig.tight_layout()        
        # export as pdf

        return fig, ax

    def render_colour(self, 
            conditions: Dict,
            scope: Literal[
                    'all',
                    'libraries', # one plot for each library used
                    'categories', # one plot for each category
                    'groups', # one plot for each group
                    'subgroups', # one plot for each subgroup
                    'species', # one plot for each species
                    'samples', # one plot for each sample
                    str # for specific category, group, subgroup, species, or sample
                    ]='all',         
            groupby: Literal[
                    'Library', # hue/style by library
                    'Category', # hue/style by category
                    'Group', # hue/style by group
                    'Subgroup', # hue/style by subgroup
                    'Species', # hue/style by species
                    'Sample ID', # hue/style by Sample ID
                    'Data ID' # hue/style by Data ID
                ]='Category',
            colour_contact_sheet: bool=True,
            sampling_profiles: bool=True,
            rgb_cube: bool=True,
            XYZ_cube: bool=False,
            xyY_cube: bool=False,
            xy_chromaticity_diagram: bool=True,
            Lab_cube: bool=False,
            Chab_plane: bool=True
            ) -> Tuple[plt.figure, plt.axes]:
        """Render the colour of each spectrum in the spectral library according
        to the given computed colour coordinates.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :param colour_contact_sheet: Indicate if a colour contact sheet is to be rendered, defaults to True
        :type colour_contact_sheet: bool, optional
        :param sampling_profiles: Indicate if sampling profiles are to be rendered, defaults to True
        :type sampling_profiles: bool, optional
        :param rgb_cube: Indicate if the RGB cube is to be rendered, defaults to True
        :type rgb_cube: bool, optional
        :param XYZ_cube: Indicate if the XYZ cube is to be rendered, defaults to False
        :type XYZ_cube: bool, optional
        :param xyY_cube: Indicate if the xyY cube is to be rendered, defaults to False
        :type xyY_cube: bool, optional
        :param xy_chromaticity_diagram: Indicate if the xy chromaticity diagram is to be rendered, defaults to False
        :type xy_chromaticity_diagram: bool, optional
        :param Lab_cube: Indicate if the Lab cube is to be rendered, defaults to False
        :type Lab_cube: bool, optional
        :param Chab_plane: Indicate if the Ch(ab) plane is to be rendered, defaults to True
        :type Chab_plane: bool, optional
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """

        # ----------------------------------------------------------------------
        # make the colour contact sheet separately
        # ----------------------------------------------------------------------
        if colour_contact_sheet:
            # render the colour contact sheet
            figs_cs, axes_cs = self.render_colour_contact_sheet(conditions,
                                                                scope=scope,
                                                                groupby=groupby)
        
        # ----------------------------------------------------------------------
        # prepare the figure(s) of the colourspace plots
        # ----------------------------------------------------------------------

        # count the number of figures other than the contact sheet
        n_figs = sum([
            rgb_cube, 
            XYZ_cube, 
            xyY_cube, 
            xy_chromaticity_diagram, 
            Lab_cube, 
            Chab_plane])
        
        # set the number of rows and columns for the figure according to 
        # n_figures, with max of 2 columns
        n_rows = int(np.ceil(n_figs / 2))
        n_cols = min(n_figs, 2)

        # set the figure size
        fig_size = (cfg.FIG_SIZE[0]*n_cols, cfg.FIG_SIZE[1]*n_rows)

        # create the figure
        fig = plt.figure(figsize=fig_size, dpi=cfg.DPI, layout='compressed')

        # create grid for different subplots
        spec = mpl.gridspec.GridSpec(ncols=n_cols, nrows=n_rows,
                         width_ratios=[1, 1], wspace=0.1,
                         hspace=0.5, height_ratios=[1, 1])

        # ----------------------------------------------------------------------
        # call rendering functions for each plot type
        # ----------------------------------------------------------------------
        i = 0
        if sampling_profiles:
            ax = fig.add_subplot(spec[i])
            if 'filter_ids' in conditions.keys():
                fig_sp, ax_sp = self.plot_filter_ids(conditions['filter_ids'], ax=ax)
            else:
                fig_sp, ax_sp = self.plot_cmfs(conditions['cmfs'], ax=ax)
            fig_label = chr(ord('@')+i+1)
            ax.set_title(fig_label+'. Sampling Profiles', fontsize=cfg.LEGEND_S)            
            i += 1
        if rgb_cube:
            ax = fig.add_subplot(spec[i], projection='3d')
            fig_rgb, ax_rgb = self.render_rgb_cube(
                                            conditions, 
                                            scope, 
                                            groupby, 
                                            ax=ax)            
            fig_label = chr(ord('@')+i+1)
            ax.set_title(fig_label+'. RGB Cube', fontsize=cfg.LEGEND_S)        
            i += 1
        if XYZ_cube:
            ax = fig.add_subplot(spec[i], projection='3d')
            fig_XYZ, ax_XYZ = self.render_XYZ_cube(
                                            conditions, 
                                            scope, 
                                            groupby, 
                                            ax=ax)            
            fig_label = chr(ord('@')+i+1)
            ax.set_title(fig_label+'. CIE XYZ Cube', fontsize=cfg.LEGEND_S)        
            i += 1
        if xyY_cube:
            ax = fig.add_subplot(spec[i], projection='3d')
            fig_xyY, ax_xyY = self.render_xyY_cube(
                                            conditions, 
                                            scope, 
                                            groupby, 
                                            ax=ax)            
            fig_label = chr(ord('@')+i+1)
            ax.set_title(fig_label+'. CIE xyY Cube', fontsize=cfg.LEGEND_S)        
            i += 1
        if xy_chromaticity_diagram:
            ax = fig.add_subplot(spec[i])
            _, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(
                cmfs=colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer'],
                show=False, 
                show_spectral_locus=False,
                show_diagram_colours=True,
                transparent_background=False,            
                figsize=(1.5*cfg.FIG_SIZE[0], 1.5*cfg.FIG_SIZE[1]),
                dpi=cfg.DPI,
                axes=ax
                )
            fig_xy, ax_xy = self.render_xy_chromaticity_diagram(conditions, scope, groupby, ax=ax)    
            # remove title
            fig_label = chr(ord('@')+i+1)
            ax.set_title(fig_label+'. CIE Chromaticity Diagram', fontsize=cfg.LEGEND_S)        
            i += 1
        if Lab_cube:
            ax = fig.add_subplot(spec[i], projection='3d')
            fig_Lab, ax_Lab = self.render_Lab_cube( 
                                                conditions, 
                                                scope, 
                                                groupby, 
                                                ax=ax)
            fig_label = chr(ord('@')+i+1)
            ax.set_title(fig_label+'. CIE L\*a\*b\* Cube', fontsize=cfg.LEGEND_S)        
            i += 1
        if Chab_plane:
            ax = fig.add_subplot(spec[i], projection='polar')
            fig_Ch, ax_Ch = self.render_Chab_plane(
                                                conditions, 
                                                scope, 
                                                groupby, 
                                                ax=ax)
            fig_label = chr(ord('@')+i+1)
            ax.set_title(fig_label+'. CIE Chroma 'r'($r$) Hue ($\theta$) Plane', fontsize=cfg.LEGEND_S)        
            i += 1

        # ----------------------------------------------------------------------
        # final figure formatting
        # ----------------------------------------------------------------------
        # add title to plot
        fig.suptitle(f'{self.spectra_obj.spectral_library} '+conditions['label'], fontsize=cfg.TITLE_S)

        plt.gcf().set_size_inches(cfg.FIG_SIZE[0]*n_cols, cfg.FIG_SIZE[1]*n_rows)

        # constraint the layout of the subplots
        # fig.tight_layout()
        
        return figs_cs, axes_cs