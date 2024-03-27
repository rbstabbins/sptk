"""Observation Class

Hosts the reflectance sampled by a given instrument of a given material
collection.

Provides methods for computing RMSE of observation compared to original
spectral library data.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 29-04-2021
"""
import os
from pathlib import Path
from shutil import rmtree
import time
from typing import List, Union
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interpolate
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sptk.material_collection import MaterialCollection
from sptk.instrument import Instrument
from sptk.spectral_library_analyser import SpectralLibraryAnalyser
import sptk.config as cfg
from sptk.config import build_project_directory as build_pd

plt.rcParams['figure.figsize'] = cfg.FIG_SIZE

class Observation():
    """Observation Class
    Hosts the reflectance sampled by a given instrument of a given material
    collection.

    Provides methods for computing RMSE of observation compared to original
    spectral library data.
    """
    def __init__(self,
            material_collection: MaterialCollection,
            instrument: Instrument,
            load_existing: bool=cfg.LOAD_EXISTING,
            plot_profiles: bool=cfg.PLOT_PROFILES,
            export_df: bool=cfg.EXPORT_DF):
        """Constructor for Observation class

        :param material_collection: original material collection
        :type material_collection: MaterialCollection
        :param instrument: instrument that performed the sampling
        :type instrument: Instrument
        :param load_existing: instruct to use or overwrite existing directories
            and files of the same project_name, defaults to config.py setting.
        :type load_existing: bool, optional
        :param plot_profiles: plot profiles, defaults to True
        :type plot_profiles: bool, optional
        :param export_df: export DataFrames, defaults to False
        :type export_df: bool, optional
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()

        p_dir, p_name = build_pd(material_collection.project_name,'observation')
        self.project_dir = p_dir
        self.project_name = p_name
        self.object_dir = Path(self.project_dir / 'observation')

        self.material_collection = material_collection
        self.categories = self.material_collection.categories

        self.instrument = instrument
        self.wvls = self.instrument.cwls().to_numpy() # set to channel cwls
        self.chnl_lbls = ['R' + str(s) for s in self.wvls] #label as R[cwl]

        if load_existing:
            existing_pkl_path = Path(self.object_dir, 'observation.pkl')
            file_exists = os.path.isfile(existing_pkl_path)
            if file_exists:
                print("Loading existing Observation DataFrame...")
                self.main_df = pd.read_pickle(existing_pkl_path)
            else:
                print("No existing DataFrame, building new Observation \
                                                for % r..." % self.project_name)
                self.main_df = self.build_new_observation(
                            material_collection,
                            instrument,
                            self.wvls)
                # self.error_df = self.compute_uncertainty() # TODO handle mising SNR
        else:
            print("Building new Observation DataFrame  \
                                                for % r..." % self.project_name)
            self.main_df = self.build_new_observation(
                        material_collection,
                        instrument,
                        self.wvls)
            # self.error_df = self.compute_uncertainty() # TODO handle missing SNR

        if export_df:
            self.export_main_df()
        if plot_profiles:
            self.plot_profiles()
            self.statistics()

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Observation built in {toc - tic:0.4f} seconds.")

    def __del__(self, rmdir: bool = False, rmproj: bool = False) -> None:
        """Observation Destructor object - optionally deletes
        observation directory and/or entire project directory

        :param rmdir: instruct removal of observation directory,
            defaults to False
        :type rmdir: bool, optional
        :param rmproj: instruct removal of entire project directory,
            defaults to False
        :type rmproj: bool, optional
        """
        name = self.project_name
        if rmdir:
            print("Deleting % r Observation directory..." % name)
            try:
                rmtree(Path(self.project_dir, 'observation'))
                print("% r Observation directory deleted." % name)
            except FileNotFoundError:
                print("No % r Observation directory to delete." % name)
        if rmproj:
            print("Deleting % r directory..." % name)
            try:
                rmtree(Path(self.project_dir))
                print("% r directory deleted." % name)
            except FileNotFoundError:
                print("No % r directory to delete." % name)

    @staticmethod
    def build_new_observation(
            material_collection: MaterialCollection,
            instrument: Instrument,
            cwls: List) -> pd.DataFrame:
        """Build a new Observation object by sampling the given material
        collection with the given instrument.
        """
        sample_matrix = material_collection.sample(instrument)
        # put matrix into dataframe compatible with the material collection
        main_df = pd.DataFrame(
            data=sample_matrix,
            index=material_collection.main_df.index,
            columns=cwls)
        # include the material collection header and category data
        cat_df = material_collection.get_cat_df()
        hdr_df = material_collection.get_hdr_df()
        hdr_df['Reflectance'] = '---'
        main_df = pd.concat([cat_df, hdr_df, main_df], axis=1)

        return main_df

    def compute_uncertainty(self) -> pd.DataFrame:
        """Compute the uncertainty of the observation data, given the
        instrument SNR.

        :return: uncertainty of the observation data
        :rtype: pd.DataFrame
        """
        error_df = self.main_df.copy()
        snrs = pd.Series(index=self.instrument.main_df['cwl'].values, data=self.instrument.main_df['snr'].values)
        noise = error_df[self.wvls].to_numpy()/snrs.to_numpy()
        error_df[self.wvls] = noise
        error_df.rename(columns={'Reflectance':'Reflectance Std. Dev.'}, inplace=True)
        return error_df

    def add_noise(self,
            n_duplicates: int,            
            snr: float=None,
            apply: bool = True) -> pd.DataFrame:
        """Add n_duplicates of noisey entries to the sampled data,
        under assumption of Gaussian distribution of noise, given by 1-sigma
        argument.

        Note: key assumption is that each filter channel was captured optimally,
            i.e. with the same count level, and thus the same shot noise, and
            that the additional dark shot noise as a function of exposure time
            has a negilible difference between filter channels. Also assumes
            that the electron count of the exposed image is >>20, which is
            valid for typical natural illumination conditions, and thus the
            Poissonian shot noise can be approximated with a Gaussian.

        :param snr: instrument signal-to-noise ratio
        :type snr: float
        :param n_duplicates: number of noisy entries to add to the data
        :type n_duplicates: int
        :param noise_type: determine if dominant noise is shot or thermal,
            defaults to shot
        :type noise_type: str
        :param apply: apply the noise to the object main_df, defaults to True
        :type apply: bool
        :return: the main dataframe
        :type return: pd.DataFrame
        """
        # access the observation dataframe and make duplicates of each entry
        obs_df = pd.concat([self.main_df]*n_duplicates).sort_index()
        # apply noise to the duplicate entries  
        if snr is None:    
            snr = self.instrument.main_df['snr'].to_numpy()  
        noise = obs_df[self.wvls].to_numpy()/snr
        noise_array = np.random.normal(0.0, noise, obs_df[self.wvls].shape)

        obs_df[self.wvls] = obs_df[self.wvls] + noise_array # update dataframe
        obs_df[self.wvls].clip(lower = 0.0, inplace=True) # clip to range
        # update index for unique ids
        suffix = obs_df.groupby(level=0).cumcount().astype(str).replace('0','')
        obs_df.index = obs_df.index +'.'+ suffix
        obs_df.index.name = 'Data ID'

        if apply:
            self.main_df = obs_df # update the main_df

        return obs_df

    # """
    # RMSE Computation between Observation and Material Collection
    # """

    def compute_rmse(self) -> pd.Series:
        """Compute the RMSE between the Observation and the Material Collection
        by interpolating the Observation back to the Material Collection range

        :return: RMSE of each entry
        :rtype: pd.Series
        """
        obs_refl = self.resample_wavelengths()

        mat_refl = self.material_collection.get_refl_df()
        indices_match = (mat_refl.index).equals(self.main_df.index)
        if not indices_match:
            raise IndexError("MaterialCollection and Observation Indices don't \
                                                                        match")

        mat_refl = mat_refl.to_numpy()
        rmse_data = np.sqrt(np.nanmean((mat_refl - obs_refl)**2,axis=1))

        rmse = pd.Series(data=rmse_data, index=self.main_df.index, name='RMSE')

        # append rmse to the main df in the column to the left of 'Reflectance'
        rmse_loc = self.main_df.columns.get_loc('Reflectance')
        self.main_df.insert(rmse_loc, 'RMSE', rmse)

        return rmse

    def resample_wavelengths(self) -> np.array:
        """Resample the observation data to wavelength resolution of the
        material collection.

        Note that interpolation will fill out of bound values with NaN

        :return: reflectance data interpolated to high-resolution domain.
        :rtype: np.array
        """
        obs_refl = self.get_refl_df()
        mat_col_wvls = cfg.WVLS
        finterp = interpolate.interp1d(self.wvls, obs_refl, bounds_error=False)
        obs_refl_interp = finterp(mat_col_wvls)

        return obs_refl_interp

    def get_rmse_df(self, rmse: pd.Series) -> pd.DataFrame:
        """Returns the RMSE information in a dataframe
        gives error if RMSE has not been computed yet - suggests computing the
        RMSE also gives header information

        :param rmse: RMSE of each entry
        :type rmse: pd.Series
        :return: rmse information in dataframe format
        :rtype: pd.DataFrame
        """
        hdr_df = self.material_collection.get_hdr_df()
        cat_df = self.material_collection.get_cat_df()
        labelled_rmse_df = pd.concat([cat_df, hdr_df, rmse], axis=1)
        # add category

        return labelled_rmse_df

    def export_rmse_df(self, rmse: pd.Series) -> None:
        """Export the RMSE DataFrame

        :param rmse: RMSE of each entry
        :type rmse: pd.Series
        """
        # export the RMSE dataframe
        table_path = Path(self.object_dir / 'tables')
        table_path.mkdir(parents=True, exist_ok=True)
        csv_out_file = Path(table_path, 'rmse').with_suffix('.csv')
        rmse_df = self.get_rmse_df(rmse)
        rmse_df.transpose().to_csv(csv_out_file)

    def plot_rmse(self, rmse_df: pd.DataFrame) -> None:
        """Plot the RMSE of each material entry, grouped by category and
        material type.

        :param rmse_df: RMSE results DataFrame
        :type rmse_df: pd.DataFrame
        """
        rmse_df.sort_values(by=['Category', 'Mineral Name', 'RMSE'],
                            inplace=True, ignore_index=True)
        # put index as column number
        rmse_df.reset_index(inplace=True)

        # set NaN grain sizes to 'Unspecified'
        rmse_df['Grain Size'].fillna('Unspecified', inplace=True)

        # TODO - filtering of grain size information to give numerical values.
        # Builds on new use of Grain Size in the VISOR database

        # get locations of each category
        cat_ticks = []
        cat_label_y = {}
        cat_bounds = {}
        for cat in self.categories:
            cat_lo = min(rmse_df[rmse_df.Category == cat].index)
            cat_hi = max(rmse_df[rmse_df.Category == cat].index)
            cat_bounds[cat] = [cat_lo, cat_hi]
            cat_label_y[cat] = np.mean([cat_lo, cat_hi]) + 1
            cat_ticks.append(cat_lo)

        # get locations of each Mineral Name
        min_ticks = []
        mineral_names = rmse_df['Mineral Name'].unique()
        mineral_label_y = {}
        mineral_bounds = {}
        for mineral_name in mineral_names:
            min_lo = min(rmse_df[rmse_df['Mineral Name'] == mineral_name].index)
            min_hi = max(rmse_df[rmse_df['Mineral Name'] == mineral_name].index)
            mineral_bounds[mineral_name] = [min_lo, min_hi]
            mineral_label_y[mineral_name] = np.mean([min_lo, min_hi])
            min_ticks.append(min_lo)

        height = 0.2*len(rmse_df)
        if height < 15:
            height = 15 # limit the minimum height to 15 cm
        _, ax = plt.subplots(figsize=[15*cfg.CM, height*cfg.CM], dpi=cfg.DPI)
        cmap = plt.get_cmap('viridis_r')
        cmap.set_bad('black')

        # plot the rmse values
        sns.barplot(
            data=rmse_df,
            x='Sample ID',
            y='RMSE',
            hue='Grain Size',
            ax=ax, dodge=False)
        plt.xticks(rotation=90)

        # add mineral name labels
        rmse_hi = max(rmse_df.RMSE) * 1.03
        rmse_lo = max(rmse_df.RMSE) * 1.01

        for mineral_name in mineral_names:
            ax.annotate(
                '',
                xy=(mineral_bounds[mineral_name][0]-0.5, rmse_hi),
                xytext=(mineral_bounds[mineral_name][1]+0.5, rmse_hi),
                arrowprops=dict(arrowstyle='<|-|>', shrinkA=0, shrinkB=0),
                annotation_clip=False)
            ax.annotate(
                mineral_name,
                xy=(mineral_label_y[mineral_name], rmse_hi),
                xytext=(mineral_label_y[mineral_name], rmse_hi),
                rotation=45,
                ha='left',
                va='bottom',
                annotation_clip=False)

        # add category labels
        for cat in self.categories:
            ax.annotate(
                '',
                xy=(cat_bounds[cat][0]-0.5, rmse_lo),
                xytext=(cat_bounds[cat][1]+0.5, rmse_lo),
                arrowprops=dict(arrowstyle='<|-|>', shrinkA=0, shrinkB=0),
                annotation_clip=False)
            ax.annotate(
                cat,
                xy=(cat_label_y[cat], rmse_lo),
                xytext=(cat_label_y[cat], rmse_lo),
                rotation=45,
                ha='right',
                va='top',
                annotation_clip=False)

        sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
        sns.despine()
        plt.tight_layout()

        output_file = Path(self.object_dir / 'analysis',
                            'rmse').with_suffix(cfg.PLT_FRMT)
        plt.savefig(output_file,bbox_inches='tight')

    # """
    # Statistics Overview Functions
    # """

    def statistics(self):
        """Perform statistical analyses over the channels, including,
        correlation, PCA and LDA, and visualise these results.
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()

        print('Performing statistical analyses over all channels...')
        _ = self.channel_correlation()
        _,_ = self.channel_pca()
        _ = self.channel_lda()

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Statistical analyses complete in {toc - tic:0.4f} seconds.")

    def channel_correlation(self, plot_profiles: bool = cfg.PLOT_PROFILES):
        """Compute the correlation matrix between all channels, visualise this,
        and plot a pair-plot / grid-plot of scatterplots for correlations.
        """
        print('Computing channel correlations...')
        corr = self.main_df[self.wvls].corr()
        if plot_profiles:
            self.plot_correlations(corr)
        return corr

    def plot_correlations(self, corr_matrix: np.array):
        """Plot the correlation matrix of channels.

        :param corr_matrix: n_channel x n_channel correlation matrix
        :type corr_matrix: np.array
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            print('Plotting channel correlations...')

        if len(self.chnl_lbls) > 20:
            annotate = False
            lw=0
        else:
            annotate = True
            lw=0.5

        # plot the correlation matrix
        Path(self.object_dir / 'statistics' ).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        ax = sns.heatmap(
                corr_matrix,
                # vmin=-1, vmax=1, center=0,
                # cmap=sns.diverging_palette(20, 220, n=2000),
                cmap='viridis',
                square=True,
                annot=annotate,
                annot_kws={"fontsize": cfg.LABEL_S/2},
                linewidths=lw,
                cbar_kws={'label': 'Pearson Correlation Coeff.', "shrink": 0.8})
        ax.figure.axes[-1].yaxis.label.set_size(cfg.LABEL_S)
        ax.tick_params(axis="x", labelrotation=80,labelright=True)
        ax.set_xlabel('Channel Centre-Wavelengths (nm)', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Channel Centre-Wavelengths (nm)', fontsize=cfg.LABEL_S)
        ax.set_title(f'{len(self.chnl_lbls)} Channel Correlation Matrix',
                     fontsize=cfg.TITLE_S)
        fig.tight_layout()

        output_file = Path(self.object_dir / 'statistics',
                        'channel_correlation_matrix').with_suffix(cfg.PLT_FRMT)
        fig.savefig(output_file,bbox_inches='tight')

        # plot the correlation relationships in a grid plot - takes too long
        # grid_plot = sns.pairplot(
        #                 data=self.main_df,
        #                 vars = self.wvls,
        #                 hue = 'Category',
        #                 height=1.5*cfg.CM,
        #                 aspect=1,
        #                 corner=False,
        #                 plot_kws={"s": 5})
        # grid_plot.fig.suptitle(f'{len(self.chnl_lbls)} Channel Scatterplots')
        # grid_plot.set(xlim=(0,1), ylim=(0,1))
        # output_file = Path(self.object_dir / 'statistics',
        #                               'channel_pairplots').with_suffix('.pdf')
        # grid_plot.savefig(output_file,bbox_inches='tight')
        # if INLINE:
        #     plt.show()
        # else:
        #     plt.close(grid_plot.fig)

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Correlation plots complete in {toc - tic:0.4f} seconds.")

    def channel_pca(self, plot_profiles: bool=cfg.PLOT_PROFILES):
        """Perform Principal Component Analysis on all channels,
        and visualise the variance of the data across the PC's, and the
        distribution of the transformed data in the PCA basis.
        """
        print('Computing and plotting band principal components...')
        data = self.get_refl_df()

        pca = decomposition.PCA() # initiate the PCA decomposition object
        pca.fit(data) # fit the PCA model to the data
        # put PCA transformed data in DataFrame
        cols = list(map(str, np.arange(0, pca.n_components_)+1))
        pca_data_df = pd.DataFrame(
                        data=pca.transform(data),
                        index=self.main_df.index,
                        columns=cols)
        pca_data_df['Category'] = self.main_df['Category'].astype("category")

        if plot_profiles:
            self.plot_channel_pca(pca_data_df)
            self.plot_channel_pca_variance(pca)
            self.plot_channel_pca_weights(pca)

        return pca_data_df, pca

    def plot_channel_pca(self,
            pca_data_df: pd.DataFrame,
            dimension: int=None) -> None:
        """Plot the variance across the PC's, and the distribution of the
        transformed data in the PCA basis.

        :param pca_data_df: PCA transformed data
        :type pca_data_df: pd.DataFrame
        """

        # plot the transformed data on the first 3 Principal Components in 3D
        Path(self.object_dir / 'statistics' ).mkdir(parents=True, exist_ok=True)
        if dimension in set([None, 3]):
            fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('PCA 1', fontsize=cfg.LABEL_S)
            ax.set_ylabel('PCA 2', fontsize=cfg.LABEL_S)
            ax.set_zlabel('PCA 3', fontsize=cfg.LABEL_S)
            cat_cols = sns.color_palette()
            i=0
            for cat in self.categories:
                pc_cat_data = pca_data_df.loc[
                    pca_data_df['Category'].values == cat]
                ax.scatter(
                        pc_cat_data['1'].to_numpy(),
                        pc_cat_data['2'].to_numpy(),
                        pc_cat_data['3'].to_numpy(),
                        color=cat_cols[i],
                        label=cat)
                i+=1
            ax.legend()
            plt.title(f'3D PCA Transformation from {len(self.wvls)} Channels',
                      fontsize=cfg.TITLE_S)
            plt.setp(ax.get_legend().get_texts(), fontsize=cfg.LEGEND_S)
            plt.setp(ax.get_legend().get_title(), fontsize=cfg.LEGEND_S)
            fig.tight_layout()

            output_file = Path(self.object_dir /
                        'statistics','channel_pca_3D').with_suffix(cfg.PLT_FRMT)
            fig.savefig(output_file,bbox_inches='tight')

        # plot 2D case of PCA1 vs PCA2
        if dimension in set([None, 2]):
            fig, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            sns.scatterplot(
                data=pca_data_df,
                x='1',
                y='2',
                hue = 'Category',
                ax=ax)
            ax.set_xlabel('PCA 1', fontsize=cfg.LABEL_S)
            ax.set_ylabel('PCA 2', fontsize=cfg.LABEL_S)
            plt.title(f'2D PCA Transformation from {len(self.wvls)} Channels',
                      fontsize=cfg.TITLE_S)
            plt.setp(ax.get_legend().get_texts(), fontsize=cfg.LEGEND_S)
            plt.setp(ax.get_legend().get_title(), fontsize=cfg.LEGEND_S)
            fig.tight_layout()
            output_file = Path(self.object_dir / 'statistics',
                               'channel_pca_2D').with_suffix(cfg.PLT_FRMT)
            fig.savefig(output_file,bbox_inches='tight')

        # plot the 1D case of PCA1 only
        if dimension in set([None, 1]):
            fig, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            sns.histplot(data=pca_data_df, x='1',hue='Category', bins=25,ax=ax)
            ax.set_xlabel('PCA 1', fontsize=cfg.LABEL_S)
            ax.set_ylabel('Count', fontsize=cfg.LABEL_S)
            ax.set_title(f'1D PCA Transform from {len(self.wvls)} Channels',
                          fontsize=cfg.TITLE_S)
            plt.setp(ax.get_legend().get_texts(), fontsize=cfg.LEGEND_S)
            plt.setp(ax.get_legend().get_title(), fontsize=cfg.LEGEND_S)
            fig.tight_layout()
            output_file = Path(self.object_dir / 'statistics',
                               'channel_pca_1D').with_suffix(cfg.PLT_FRMT)
            fig.savefig(output_file,bbox_inches='tight')

    def plot_channel_pca_variance(self,
            pca: decomposition.PCA()) -> None:
        """Plot the explained variance ratios

        :param pca: PCA object
        :type pca: decomposition.PCA
        """
        fig, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        x_vals = np.arange(0, pca.n_components_)+1
        plt.bar(x_vals, pca.explained_variance_ratio_, log=True)
        plt.xticks(x_vals, x_vals)
        ax.tick_params(axis="x", labelrotation=80,labelright=True)
        ax.set_xlabel('Component', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Explained Variance Ratio (% of Total Variance)',
                      fontsize=cfg.LABEL_S)
        plt.title(f'PCA Explained Variance for {len(self.wvls)} Channels',
                  fontsize=cfg.TITLE_S)
        output_file = Path(self.object_dir, 'statistics',
                           'pca_weights').with_suffix(cfg.PLT_FRMT)
        fig.savefig(output_file,bbox_inches='tight')

    def plot_channel_pca_weights(self,
            pca: decomposition.PCA()) -> None:
        """Plot the 1st PCA component weightings

        :param pca: PCA object
        :type pca: decomposition.PCA
        """
        fig, ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        sns.barplot(x=self.wvls, y=pca.components_[0], color='tab:blue', ax=ax)
        ax.set_xlabel('Channel Centre Wavelengths (nm)', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Contribution to PCA1', fontsize=cfg.LABEL_S)
        ax.set_title('PCA 1 Projection Coefficients', fontsize=cfg.TITLE_S)
        ax.tick_params(axis="x", labelrotation=80,labelright=True)
        fig.tight_layout()

        output_file = Path(self.object_dir / 'statistics',
                           'pca1_coeffs').with_suffix(cfg.PLT_FRMT)
        fig.savefig(output_file,bbox_inches='tight')

    def channel_lda(self, plot_profiles: bool = cfg.PLOT_PROFILES) -> None:
        """Perform Linear Discriminant Analysis on the channels,
        and visualise the distribution of the transformed data in the LDA basis.
        """
        print('Computing and plotting band linear discriminants...')

        # perform LDA on the band dataset
        lda = LinearDiscriminantAnalysis()
        lda.fit(self.main_df[self.wvls], self.main_df['Category'].values)
        n_classes = len(lda.classes_)
        # put LDA transformed data in a DataFrame
        lda_data_df = pd.DataFrame(
                        data=lda.transform(self.main_df[self.wvls]),
                        index=self.main_df.index,
                        columns=list(map(str, np.arange(0, n_classes-1)+1)))
        lda_data_df['Category'] = self.main_df['Category'].astype("category")

        if plot_profiles:
            self.plot_channel_lda(lda_data_df, lda)

        return lda_data_df

    def plot_channel_lda(self,
            lda_data_df: pd.DataFrame,
            lda: LinearDiscriminantAnalysis) -> None:
        """Plot the distribution of the transformed data in the LDA basis.

        :param lda_data_df: LDA transformed data
        :type lda_data_df: pd.DataFrame
        :param lda: LDA object
        :type lda: LinearDiscriminantAnalysis
        """
        # plot the lDA results
        n_classes = len(lda.classes_)
        n_chnls = len(self.wvls)
        Path(self.object_dir / 'statistics').mkdir(parents=True, exist_ok=True)
        if n_classes == 2:
            # plot the 1st (and only) LDA axis projection
            lda_fig, lda_ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            sns.histplot(
                    data=lda_data_df,
                    x='1',
                    hue=lda_data_df['Category'],
                    bins=25,
                    ax=lda_ax)
            lda_ax.set_xlabel('LDA 1', fontsize=cfg.LABEL_S)
            lda_ax.set_ylabel('Count', fontsize=cfg.LABEL_S)
            lda_ax.set_title(f'LDA Transformation from {n_chnls} Channels',
                             fontsize=cfg.TITLE_S)
            plt.setp(lda_ax.get_legend().get_texts(), fontsize=cfg.LEGEND_S)
            plt.setp(lda_ax.get_legend().get_title(), fontsize=cfg.LEGEND_S)

            lda_fig.tight_layout()

            output_file = Path(self.object_dir / 'statistics',
                               'channel_lda').with_suffix(cfg.PLT_FRMT)
            plt.savefig(output_file,bbox_inches='tight')


            # plot the 1st (and only) LDA band weightings
            lda_w_fig,lda_w_ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            lda_w_ax = sns.barplot(
                            x=self.wvls,
                            y=lda.coef_[0],
                            color='tab:blue',
                            ax=lda_w_ax)
            lda_w_ax.set_xlabel('Channel Centre Wavelength (nm)',
                                 fontsize=cfg.LABEL_S)
            lda_w_ax.set_ylabel('Contribution to LDA1', fontsize=cfg.LABEL_S)
            lda_w_ax.set_title('LDA 1 Projection Coefficients',
                               fontsize=cfg.TITLE_S)
            lda_w_ax.tick_params(axis="x", labelrotation=80,labelright=True)
            lda_w_fig.tight_layout()

            output_file = Path(self.object_dir / 'statistics',
                               'lda1_coeffs').with_suffix(cfg.PLT_FRMT)
            lda_w_fig.savefig(output_file,bbox_inches='tight')
        elif n_classes == 3:
            lda_fig, lda_ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            lda_ax = sns.scatterplot(
                    data=lda_data_df,
                    x='1',
                    y='2',
                    hue='Category',
                    ax=lda_ax)
            lda_ax.set_xlabel('LDA 1', fontsize=cfg.LABEL_S)
            lda_ax.set_ylabel('LDA 2', fontsize=cfg.LABEL_S)
            lda_ax.set_title(f'LDA Transformation from {n_chnls} Channels',
                              fontsize=cfg.TITLE_S)
            lda_fig.tight_layout()

            output_file = Path(self.object_dir / 'statistics',
                               'lda_2D').with_suffix(cfg.PLT_FRMT)
            lda_fig.savefig(output_file,bbox_inches='tight')
        elif n_classes == 4:
            # TODO enter code for 3D plot
            print('do something')

    # """
    # Getter/Setter functions
    # """

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
            mineral_name: str = None) -> pd.DataFrame:
        """Return a copy of the reflectance dataframe subset of the array.
        Allows for selection of data from specific category and mineral type.

        :param category: categorical subset of data, defaults to None
        :type category: str, optional
        :param mineral_name: mineral subset of data, defaults to None
        :type mineral_name: str, optional
        :return: reflectance data only of the material collection
        :rtype: pd.DataFrame
        """
        subset_df = self.get_subset_df(category, mineral_name)
        refl_df = subset_df.loc[:, self.wvls[0]:]
        return refl_df

    def get_subset_df(self,
            category: str = None,
            mineral_name: str = None) -> pd.DataFrame:
        """Return a subset of the dataframe, according to selection
        from specific category and mineral type.

        :param category: categorical subset of data, defaults to None
        :type category: str, optional
        :param mineral_name: mineral subset of data, defaults to None
        :type mineral_name: str, optional
        :return: subset of dataframe according to category and mineral
        :rtype: pd.DataFrame
        """
        if (category is not None) and (mineral_name is not None):
            cat_mnrl_selection = (self.main_df.Category == category) & (
                    self.main_df['Mineral Name'] == mineral_name)
            subset_df = self.main_df.loc[cat_mnrl_selection,:]
        elif category is not None:
            cat_selection = self.main_df.Category == category
            subset_df = self.main_df.loc[cat_selection,:]
        elif mineral_name is not None:
            mnrl_selection = self.main_df['Mineral Name'] == mineral_name
            subset_df = self.main_df.loc[mnrl_selection,:]
        else:
            subset_df = self.main_df
        return subset_df

    def get_hdr_df(self,
            category: str = None,
            mineral_name: str = None) -> pd.DataFrame:
        """Return a copy of the header dataframe subset of the array.
        Allows for selection of data from specific category and mineral type.

        :param category: categorical subset of data, defaults to None
        :type category: str, optional
        :param mineral_name: mineral subset of data, defaults to None
        :type mineral_name: str, optional
        :return: material collection header data
        :rtype: pd.DataFrame
        """
        subset_df = self.get_subset_df(category, mineral_name)
        hdr_df = subset_df.loc[:, self.material_collection.header_list]
        return hdr_df

    def get_cat_df(self,
            category: str = None,
            mineral_name: str = None) -> pd.DataFrame:
        """Returns a copy of the Categories of the array.
        Allows for selection of data from specific category and mineral type.

        :param category: categorical subset of data, defaults to None
        :type category: str, optional
        :param mineral_name: mineral subset of data, defaults to None
        :type mineral_name: str, optional
        :return: category labels
        :rtype: pd.DataFrame
        """
        subset_df = self.get_subset_df(category, mineral_name)
        cat_df = pd.DataFrame(subset_df.Category)
        return cat_df

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

    def get_chnl_lbl(self, cwl: Union[str, List]) -> Union[str, List]:
        """Get channel label(s) for given centre-wavelength(s)

        :param cwl: centre-wavelength(s) to retrieve channel label(s) for
        :type cwl: Union[str, List]
        :return: channel label(s)
        :rtype: Union[str, List]
        """
        channel_dict = dict(zip(self.wvls, self.chnl_lbls))
        chnl_lbl = channel_dict[cwl]
        return chnl_lbl

    def get_cwl(self, chnl_lbl: List) -> List:
        """Get centre-wavelength(s) for given channel label(s)

        :param cwl: channel label(s) to retrieve for centre-wavelength(s)
        :type cwl: Union[str, List]
        :return: centre-wavelength(s)
        :rtype: Union[float, List]
        """
        cwl_series = pd.Series(data=self.wvls, index=self.chnl_lbls)
        if isinstance(chnl_lbl, str):
            chnl_lbl = [chnl_lbl] # incapsulate in list
        cwl = cwl_series[chnl_lbl].to_numpy()
        return cwl

    # """
    # Export functions
    # """

    def export_main_df(self):
        """Export the dataframe to csv file, and pickle.

        :param pkl_only: Only output a pkl file of the DataFrame
        :type pkl_only: bool, optional
        """
        print('Exporting the Observation Pickle format...')
        table_dir = Path(self.object_dir / 'tables')
        table_dir.mkdir(parents=True, exist_ok=True)

        pkl_file = Path(self.object_dir, 'observation.pkl')
        self.main_df.to_pickle(pkl_file)

        csv_out_file = Path(table_dir, 'observation.csv')
        self.main_df.transpose().to_csv(csv_out_file)

        print('Observation export complete.')

    def plot_profiles(self, categories_only: bool=False, ci: bool=False) -> List[plt.Axes]:
        """Plot the profiles of the materials as sampled by the instrument
        """
        plotter = SpectralLibraryAnalyser(self)
        axes = plotter.plot_profiles(categories_only=categories_only, ci=ci)
        return axes