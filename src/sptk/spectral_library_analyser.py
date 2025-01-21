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
from typing import Literal, Tuple, Union, List
import colour
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
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

    def plot_profiles(self,
            with_noise: bool=False,
            scope: str='all',
            categories_only: bool=False,
            ci: bool=False,
            hires_under: bool=False,
            out_dir: Union[bool, str]=False
            ) -> plt.Axes:
        """Plot the profiles of the materials of the spectral library
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            print('Plotting reflectance profiles of materials...')

        # get the reflectance data for all minerals and for each category
        refl_df = self.spectra_obj.get_refl_df()
        cat_df = self.spectra_obj.get_cat_df()

        # incorporate error DF into this data for plotting
        all_df = pd.concat([refl_df, cat_df], axis=1)
        
        ax = self.render_profile_plot(all_df,
                    scope=scope,
                    with_noise=with_noise,
                    ci=ci)
        axes = [ax]
        if categories_only:
            if cfg.TIME_IT:
                toc = time.perf_counter()
                print(f"Reflectance profiles plotted in {toc - tic:0.4f} s.")
            return ax

        # plot for each category and mineral name
        for cat in self.spectra_obj.categories:
            mineral_list = self.spectra_obj.get_mineral_list(cat, unique=True)
            for mnrl in mineral_list:
                # get data for category and mineral
                refl_df = self.spectra_obj.get_refl_df(category=cat,
                                                            mineral_name=mnrl)
                cat_df = self.spectra_obj.get_cat_df(category=cat,
                                                            mineral_name=mnrl)
                cat_mnrl_df = pd.concat([refl_df, cat_df], axis=1)
                ax = self.render_profile_plot(cat_mnrl_df,
                            cat=cat,
                            mnrl=mnrl,
                            scope=scope,
                            with_noise=with_noise,
                            ci=ci,
                            hires_under=hires_under,
                            out_dir=out_dir)
                axes.append(ax)
        
        # put all axes into a new figure

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Reflectance profiles plotted in {toc - tic:0.4f} s.")

        return axes

    def render_profile_plot(self,
            data_df: pd.DataFrame,
            cat: str='all',
            mnrl: str='entries',
            ci: bool=False,
            scope: str='all',
            with_noise: bool=False,
            hires_under: bool=False,
            out_dir: Union[bool, str]=False) -> None:
        """Method for producing the plot itself, according to given DataFrame,
        class, mineral name, and scope.

        :param data_df: Material Collection or Observation to DataFrame
        :type data_df: pd.DataFrame
        :param cat: Class label, defaults to 'all'
        :type cat: str, optional
        :param mnrl: Mineral Name, defaults to 'entries'
        :type mnrl: str, optional
        :param ci: Plot mean spectra with confidence interval, defaults to False
        :type ci: bool, optional
        :param scope: indicates specific elements or all data, defaults to 'all'
        :type scope: str, optional
        :param with_noise: Indicates if noise has been added, defaults to False
        :type with_noise: bool, optional
        :param hires_under: for Observation under-plot the laboratory spectra, 
                            defaults to False
        :type hires_under: bool, optional
        """

        data_df = data_df.reset_index()
        # long form version of plotting, to aggregate data
        data_df =pd.melt(data_df, id_vars=['Data ID','Category'])

        if out_dir:
            out_dir = Path(out_dir)
        else:
            out_dir = Path(self.spectra_obj.object_dir / 'plots')
            out_dir.mkdir(parents=True, exist_ok=True)

        if with_noise:
            sfx = '_with_noise'
        else:
            sfx = ''

        sns.set_context("paper")

        if cat != 'all':
            n_ids = len(data_df['Data ID'].unique())
            n_cols = -(-n_ids // 15)
        else:
            n_cols = 1            
        width_factor = 1 + 0.3 * n_cols
        fig_size = (width_factor*cfg.FIG_SIZE[0], cfg.FIG_SIZE[1])
        fig, ax = plt.subplots(figsize=fig_size, dpi=cfg.DPI)
        # y_max = max([1.0, data_df.value.max()])

        if cat == 'all':
            hue_flag = 'Category'
        else:
            hue_flag = 'Data ID'

        if self.obj_type == 'observation':
            marker_flag = True
        else:
            marker_flag = False

        if ci:            
            sns.lineplot(
                data=data_df,
                x='variable',
                y='value',
                hue=hue_flag,
                style=hue_flag,
                markeredgewidth=0.0,
                markers=marker_flag,
                errorbar='sd',
                lw=0.7,
                ax=ax)
        else:
            sns.lineplot(
                data=data_df,
                x='variable',
                y='value',
                hue=hue_flag,
                style=hue_flag,
                markeredgewidth=0.0,
                units='Data ID',
                estimator=None,
                lw=0.5, markers=marker_flag,
                ax=ax)

        ax.set_xlim(cfg.SAMPLE_RES['wvl_min']-10, cfg.SAMPLE_RES['wvl_max']+10)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
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
        ax.grid(True, which='major',axis='both', lw=0.6)
        ax.grid(True, which='minor',axis='both', lw=0.3)

        ax.legend(loc="center left", 
                  fontsize=cfg.LEGEND_S, 
                  bbox_to_anchor=(1.02, 0.5),
                  ncol=n_cols
                  )

        # plot title
        Cat = cat.capitalize()
        Mnrl = mnrl.capitalize()
        project_str = self.spectra_obj.project_name.replace('_', ' ')
        if self.obj_type == 'observation':
            leg_title = f'Class: {Cat}, Group: {mnrl} ({scope} data) - sampled'
            if ci:
                title = f'{self.spectra_obj.instrument.name} {Cat} {Mnrl} Mean ± 1σ'
            else:                
                title = f'{self.spectra_obj.instrument.name} {Cat} {Mnrl}'
            if hires_under:
                refl_df = self.spectra_obj.material_collection.get_refl_df(category=cat,
                                                            mineral_name=mnrl)
                hires_df = refl_df.reset_index()
                # long form version of plotting, to aggregate data
                hires_df =pd.melt(hires_df, id_vars=['Data ID'])
                sns.lineplot(
                    data=hires_df,
                    x='variable',
                    y='value',
                    hue=hue_flag,
                    style=hue_flag,
                    markeredgewidth=0.0,
                    alpha=0.3,
                    units='Data ID',
                    estimator=None,
                    lw=0.5,
                    legend=False,
                    ax=ax)
        else:
            leg_title = f'Class: {Cat}, Group: {Mnrl} ({scope} data)'
            title = f'Laboratory {Cat} {Mnrl}'
        plt.title(title, fontsize=cfg.LABEL_S) # update - removing titles from plots

        fig.tight_layout()

        # save figure
        project_str = self.spectra_obj.project_name
        if self.obj_type == 'observation':
            inst = self.spectra_obj.instrument.name
            filename = f'{project_str}_{inst}_{cat}_{mnrl}_{scope}'+sfx
        else:
            filename = f'{project_str}_{cat}_{mnrl}_{scope}'+sfx
        output_file = Path(out_dir, filename).with_suffix(cfg.PLT_FRMT)
        fig.savefig(output_file, bbox_inches='tight', pad_inches = 0)

        return ax

    # """
    # Spectrogram Visualisation & Continuum Removal
    # """

    def remove_continuum(self):
        """Remove the continuum from all spectra, and overwrite the local copy
        of the spectra object reflectance data.
        """
        self.spectra_obj_cr = copy.deepcopy(self.spectra_obj)
        cat_s = self.spectra_obj_cr.get_cat_df()
        spectra = self.spectra_obj_cr.get_refl_df()
        self.spectra_obj_cr.object_dir = Path(
                                self.project_dir,'continuum_removed')
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
        plotter = SpectralLibraryAnalyser(self.spectra_obj_cr)
        plotter.plot_profiles()
        return self.spectra_obj_cr.main_df

    def visualise_spectrogram(self, continuum_removed: bool=True):
        """Display the reflectance data in 2D density plots, with colour giving
        absorption depth.

        :param continuum_removed: indicate to use continuum_removed data,
                defaults to True
        :type continuum_removed: bool, optional
        """
        # get reflectance sorted by Category, Mineral Name, then Sample ID
        if continuum_removed:
            try:
                vis_df = self.spectra_obj_cr.main_df
            except AttributeError:
                self.remove_continuum()
                vis_df = self.spectra_obj_cr.main_df
        vis_df.sort_values(by=['Category', 'Mineral Name', 'Sample ID'],
                                            inplace=True, ignore_index = True)

        if self.obj_type == 'material_collection':
            title = 'High-Resolution Spectral Library'
            data = vis_df[self.wvls].to_numpy() - 1.0
        elif self.obj_type == 'observation':
            title=f'{self.spectra_obj.instrument.name} sampled Spectral Library'
            data = self.spectra_obj_cr.resample_wavelengths() - 1.0
        else:
            raise ValueError("spectral object type not recognised")

        # get locations of each category
        cat_ticks = []
        cat_label_y = {}
        cat_bounds = {}
        for cat in self.spectra_obj.categories:
            cat_lo = min(vis_df[vis_df.Category == cat].index)
            cat_hi = max(vis_df[vis_df.Category == cat].index)
            cat_bounds[cat] = [cat_lo, cat_hi]
            cat_label_y[cat] = np.mean([cat_lo, cat_hi]) + 1
            cat_ticks.append(cat_lo)

        # get locations of each Mineral Name
        min_ticks = []
        mineral_names = vis_df['Mineral Name'].unique()
        mineral_label_y = {}
        mineral_bounds = {}
        for mineral_name in mineral_names:
            min_lo = min(vis_df[vis_df['Mineral Name'] == mineral_name].index)
            min_hi = max(vis_df[vis_df['Mineral Name'] == mineral_name].index)
            mineral_bounds[mineral_name] = [min_lo, min_hi]
            mineral_label_y[mineral_name] = np.mean([min_lo, min_hi])
            min_ticks.append(min_lo)

        # set plot limits
        wvl_lo = cfg.SAMPLE_RES['wvl_min']
        wvl_hi = cfg.SAMPLE_RES['wvl_max']
        # set up a good figure size so that good # of samples are shown per cm.
        # A4 = 210 x 297 mm
        # minus 3 cm for border
        # fig size - width = 190 mm
        # fig size - height = 2mm * #samples
        height = 0.2*len(data)
        if height < 15:
            height = 15 # limit the minimum height to 15 cm
        fig, ax = plt.subplots(figsize=[15*cfg.CM, height*cfg.CM], dpi=cfg.DPI)
        cmap = plt.get_cmap('viridis_r')
        cmap.set_bad('black')

        # draw plot
        im = ax.imshow(
                data,
                aspect='auto',
                extent=[wvl_lo, wvl_hi, 0, len(data)],
                interpolation='nearest',
                origin='lower',
                cmap=cmap, vmin=-1.0, vmax=0.0)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_title(title)

        # add mineral name labels
        for mineral_name in mineral_names:
            ax.annotate(
                '',
                xy=(wvl_hi + 50, mineral_bounds[mineral_name][0]),
                xytext=(wvl_hi + 50, mineral_bounds[mineral_name][1]+1),
                arrowprops=dict(arrowstyle='<|-|>', shrinkA=0, shrinkB=0),
                annotation_clip=False)
            ax.annotate(
                mineral_name,
                xy=(wvl_hi + 70,mineral_label_y[mineral_name]),
                xytext=(wvl_hi + 70,mineral_label_y[mineral_name]),
                rotation=45,
                ha='left',
                va='bottom',
                annotation_clip=False)

        # add category labels
        for cat in self.spectra_obj.categories:
            ax.annotate(
                '',
                xy=(wvl_lo - 50, cat_bounds[cat][0]),
                xytext=(wvl_lo - 50, cat_bounds[cat][1]+1),
                arrowprops=dict(arrowstyle='<|-|>', shrinkA=0, shrinkB=0),
                annotation_clip=False)
            ax.annotate(
                cat,
                xy=(wvl_lo - 70,cat_label_y[cat]),
                xytext=(wvl_lo - 70,cat_label_y[cat]),
                rotation=45,
                ha='right',
                va='top',
                annotation_clip=False)

        # set ticks
        minor_ticks = np.arange(0, len(data)+1, 1)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', axis='y', lw=0.5)
        ax.set_yticks(min_ticks, minor=False)
        ax.tick_params(right=True)
        ax.set_yticklabels([])
        ax.grid(which='major', axis='y', lw=0.8, color='w')
        ax.tick_params(right=True)

        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size=0.5*cfg.CM, pad=0.6, pack_start = True)
        fig.add_axes(cax)

        cbar = plt.colorbar(im, cax = cax, orientation='horizontal', shrink=0.5)
        cbar.set_label('Band Depth')

        # export
        filepath=Path(self.project_dir,'spectrogram').with_suffix(cfg.PLT_FRMT)
        plt.savefig(filepath, bbox_inches='tight', pad_inches = 0.1)

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
            by=['Category', 'Mineral Name', 'Sample ID'],
            inplace=True,
            ignore_index = True)
        self.band_info.to_csv(path)

        # Order the data for export
        band_centre_info = [
            'Category',
            'Mineral Name',
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

        fig, ax = plt.subplots(1, 1, 
                               figsize=(cfg.FIG_SIZE[0], cfg.FIG_SIZE[1]), 
                               dpi=cfg.DPI)
        cols = ['r', 'g', 'b']
        for filt in filter_ids:
            normed_filter_profile = transmission.loc[filt].to_numpy() / transmission.loc[filt].max()
            ax.plot(wvls, normed_filter_profile, color=cols.pop(0), label=filt, lw=0.8)
        ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Spectral Response', fontsize=cfg.LABEL_S)
        ax.legend(loc='upper right', fontsize=cfg.LEGEND_S)
        ax.set_title(title, fontsize=cfg.LABEL_S)

        return fig, ax

    def plot_cmfs(self, 
            cmf_label: Literal[colour.MSDS_CMFS.keys()]='CIE 1964 10 Degree Standard Observer',
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

        fig, ax = plt.subplots(1, 1, figsize=(cfg.FIG_SIZE[0], cfg.FIG_SIZE[1]), dpi=cfg.DPI)
        cols = ['r', 'g', 'b']
        for profile in profiles.transpose():
            ax.plot(wvls, profile, color=cols.pop(0), label=labels.pop(0), lw=0.8)
        ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Spectral Response', fontsize=cfg.LABEL_S)
        ax.legend(loc='upper right', fontsize=cfg.LEGEND_S)
        ax.set_title(cmf_label, fontsize=cfg.LABEL_S)

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

        return fig, ax
        
    def compute_colour(self,
                illuminant: Literal['D65', 'A', 'C', 'D50', 'D55', 'D75']='D65',
                cmf_label: Literal[colour.MSDS_CMFS.keys()]='CIE 1964 10 Degree Standard Observer',
                       ) -> object:
        """Compute the colour of each entry in the Material Collection or 
        Observation according to the given illuminant.

        :param illuminant: illuminant to use to compute colour, defaults to 'D65'
        :type illuminant: str, optional
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
        illum = colour.SDS_ILLUMINANTS[illuminant]
        
        # convert the spectral distributions to XYZ space
        # set the integration method
        if self.obj_type == 'observation':
            method = 'Integration' # assume discrete non-continuous spectra
        else:
            method = 'ASTM E308' # assume high-resolution continuous spectra
        
        # set the colour matching functions
        cmfs=colour.colorimetry.MSDS_CMFS[cmf_label]

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

        # append the XYZ values to the new colour df
        col_obj.main_df['X'] = xyz[:,0]
        col_obj.main_df['Y'] = xyz[:,1]
        col_obj.main_df['Z'] = xyz[:,2]

        # convert the XYZ values to sRGB        
        illum_ccs = colour.CCS_ILLUMINANTS[cmf_label][illuminant]
        rgb = colour.XYZ_to_sRGB(xyz, illum_ccs)
        rgb = np.clip(rgb, 0, 1)

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
        col_obj.main_df = col_obj.main_df.loc[:, 'Colour':]
        # drop 'Colour' column
        col_obj.main_df.drop('Colour', axis=1, inplace=True)
        if self.obj_type == 'observation':
            colour_conditions = cmf_label+' '+illuminant+' '+self.spectra_obj.instrument.name
        else:
            colour_conditions = cmf_label+' '+illuminant
        col_obj.main_df.columns = pd.MultiIndex.from_product([[colour_conditions], col_obj.main_df.columns])

        self.spectra_obj.colour_df = col_obj.main_df

        return col_obj

    def compute_false_colour(self,
            filter_ids: Tuple[str, str, str],
            illuminant: str=None) -> object:
        """Compute the false colour of each entry in the given Observation for
        the given instrument filter IDs.

        :param filter_ids: Filter IDs for the instrument
        :type filter_ids: Tuple[str, str, str]
        :return: Observation duplicate of false colours for each entry
        :rtype: Observation
        """
        # special case for CaSSIS synthetic RGB image:
        compute_sBLU = False
        if 'sBLU' in filter_ids:
            filter_ids.remove('sBLU')
            filter_ids.append('BLU')
            compute_sBLU = True 

        filter_labels= filter_ids[0]+'-'+filter_ids[1]+'-'+filter_ids[2]
        # deep copy the spectra object
        false_col_obj = copy.deepcopy(self.spectra_obj)

        # get the reflectance data for the given filter IDs
        # get the cwls for the filters
        inst_info = self.spectra_obj.instrument.get_metrics()
        cwls = inst_info.loc[filter_ids].cwl.to_list()
        refl_df = self.spectra_obj.get_refl_df()
        rgb = refl_df[cwls].to_numpy()
        rgb = np.clip(rgb, 0, 1)

        if compute_sBLU:
            rgb[:,2] = 2*rgb[:,0] - 0.3*rgb[:,1]
            filter_ids.remove('BLU')
            filter_ids.append('sBLU')
        rgb = np.clip(rgb, 0, 1)
        filter_labels= filter_ids[0]+'-'+filter_ids[1]+'-'+filter_ids[2]


        # set the RGB values for the false colours
        false_col_obj.main_df.drop(self.wvls, axis=1, inplace=True)
        false_col_obj.main_df.rename(columns={"Reflectance": "Colour"}, inplace=True)
        false_col_obj.main_df['R'] = rgb[:,0]
        false_col_obj.main_df['G'] = rgb[:,1]
        false_col_obj.main_df['B'] = rgb[:,2]
        
        # convert RGB back to XYZ and add to object
        XYZ = colour.RGB_to_XYZ(rgb, 'sRGB')

        # append the XYZ values to the new colour df
        false_col_obj.main_df['X'] = XYZ[:,0]
        false_col_obj.main_df['Y'] = XYZ[:,1]
        false_col_obj.main_df['Z'] = XYZ[:,2]

        # convet XYZ to xyY and add to object
        xyY = colour.XYZ_to_xyY(XYZ)

        false_col_obj.main_df['x'] = xyY[:,0]
        false_col_obj.main_df['y'] = xyY[:,1]
        false_col_obj.main_df['Y'] = xyY[:,2]

        # convert to Lab
        Lab = colour.XYZ_to_Lab(XYZ)

        false_col_obj.main_df['L*'] = Lab[:,0]
        false_col_obj.main_df['a*'] = Lab[:,1]
        false_col_obj.main_df['b*'] = Lab[:,2]

        # convert to LCh
        LCh = colour.Lab_to_LCHab(Lab)
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
        else:
            self.spectra_obj.colour_df = false_col_obj.main_df

        return false_col_obj
    
    def render_colour_contact_sheet(self,
            conditions: str,
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
        
        # # load the spectral library        
        # index = self.spectra_obj.main_df.index

        # rgb = self.spectra_obj.colour_df[conditions][['R', 'G', 'B']].to_numpy()        

        # cats = self.spectra_obj.main_df['Category'][index]
        
        # title_sfx = conditions
        title_sfx = conditions
        
        # if compare, load the comparison material collection colour df
        if srgb_compare and self.obj_type == 'observation':
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
                cat_rows[cat] = 1 # and the numebr of rows is 1                        
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
                cat_df = self.spectra_obj.colour_df[self.spectra_obj.main_df['Category'] == cat]
                cat_rgb = cat_df[conditions][['R', 'G', 'B']].iloc[i:f+1]
                
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
                    min_name = self.spectra_obj.main_df.loc[entry]['Mineral Name']
                    entry = str(entry).replace('_', '\n') # turn underscore into carriage return
                    entry = str(entry).replace(' ', '\n') # get mineral name
                    entry = entry.title() 
                    entry = min_name.title() + '\n' + entry
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
            filepath=Path(contact_sheet_dir, f'{self.spectra_obj.spectral_library} {conditions} page_{page}').with_suffix('.pdf') # locked as PDF not PNG
            plt.savefig(filepath, bbox_inches='tight', pad_inches = 0.1)

            figs.append(fig)
            axes.append(ax_c)
        
        return figs, axes
    
    def render_rgb_cube(self,
                        conditions: str) -> Tuple[plt.figure, plt.axes]:
        """Render the RGB cube of the spectral library for the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        rgb = self.spectra_obj.colour_df[conditions][['R', 'G', 'B']].to_numpy()

        # make a 3D plot of rgb array
        fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        ax = fig.add_subplot(111, projection='3d')

        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>',
                        'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']
        for i, cat in enumerate(uniq_cats):
            cat_rgb = rgb[cats == cat]
            # plot each point in turn to give separate colour
            ax.scatter(cat_rgb[:,0], cat_rgb[:,1], cat_rgb[:,2], 
                        c=cat_rgb, 
                        depthshade=True, 
                        marker=syms_list[i], 
                        alpha=0.9,
                        label=cat)
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
        fig.tight_layout()
        # export as pdf   

        return fig, ax
    
    def render_XYZ_cube(self,
                        conditions: str) -> Tuple[plt.figure, plt.axes]:
        """Render the XYZ cube of the spectral library for the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        XYZ = self.spectra_obj.colour_df[conditions][['X', 'Y', 'Z']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions][['R', 'G', 'B']].to_numpy()

        # make a 3D plot of rgb array
        fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        ax = fig.add_subplot(111, projection='3d')

        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>',
                        'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']
        for i, cat in enumerate(uniq_cats):
            cat_XYZ = XYZ[cats == cat]
            cat_rgb = rgb[cats == cat]
            ax.scatter(cat_XYZ[:,0], cat_XYZ[:,1], cat_XYZ[:,2], 
                        c=cat_rgb, 
                        depthshade=True, 
                        marker=syms_list[i], 
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
                        conditions: str) -> Tuple[plt.figure, plt.axes]:
        """Render the xyY cube of the spectral library for the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        xyY = self.spectra_obj.colour_df[conditions][['x', 'y', 'Y']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions][['R', 'G', 'B']].to_numpy()

        # make a 3D plot of rgb array
        fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        ax = fig.add_subplot(111, projection='3d')

        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>',
                        'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']
        for i, cat in enumerate(uniq_cats):
            cat_xyY = xyY[cats == cat]
            cat_rgb = rgb[cats == cat]
            for e in range(cat_xyY.shape[0]):
                markerline, stemlines, baseline = ax.stem([cat_xyY[e,0]], [cat_xyY[e,1]], [cat_xyY[e,2]], label=cat, markerfmt=syms_list[i])                
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
                        conditions: str) -> Tuple[plt.figure, plt.axes]:
        """Render the xy chromaticity diagram of the spectral library for 
        the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """

        # *** Plotting in chromaticity space ***

        title_sfx = conditions
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        xyY = self.spectra_obj.colour_df[conditions][['x', 'y', 'Y']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions][['R', 'G', 'B']].to_numpy()
        
        fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(
            cmfs=colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer'],
            show=False, 
            show_spectral_locus=True,
            show_diagram_colours=True,
            transparent_background=False,            
            figsize=(cfg.FIG_SIZE[0], cfg.FIG_SIZE[1]),
            dpi=cfg.DPI
            )
    
        # set figure size
        fig.set_size_inches(cfg.FIG_SIZE[0], cfg.FIG_SIZE[1])
            
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
                    markeredgewidth=0.5,
                    markeredgecolor='white', markersize=4)
        
        # plot the sRGB space in the chromaticity diagram
        sRGB_ps = colour.RGB_COLOURSPACES['sRGB'].primaries
        ax.plot(sRGB_ps[:,0],sRGB_ps[:,1], color='k', marker='', label='sRGB')
        ax.plot(sRGB_ps[[2,0],0],sRGB_ps[[2,0],1], color='k', marker='') 

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
        
        ax.legend(handles, labels, loc='upper right', fontsize='x-small')
                
        # set figure size
        fig.set_size_inches(2*cfg.FIG_SIZE[0], 2*cfg.FIG_SIZE[1])   
        # set DPI
        fig.set_dpi(cfg.DPI)

        fig.tight_layout()      

        return fig, ax

    def render_Lab_cube(self,
                        conditions: str) -> Tuple[plt.figure, plt.axes]:
        """Render the L*a*b* cube of the spectral library for the given conditions.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        Lab = self.spectra_obj.colour_df[conditions][['L*', 'a*', 'b*']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions][['R', 'G', 'B']].to_numpy()

        # make a 3D plot of rgb array
        fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        ax = fig.add_subplot(111, projection='3d')

        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
        syms_list = ['o', 's', 'D', 'v', '^', '<', '>',
                        'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']
        for i, cat in enumerate(uniq_cats):
            cat_Lab = Lab[cats == cat]
            cat_rgb = rgb[cats == cat]
            for e in range(cat_Lab.shape[0]):
                markerline, stemlines, baseline = ax.stem([cat_Lab[e,1]], [cat_Lab[e,2]], [cat_Lab[e,0]], label=cat, markerfmt=syms_list[i])                
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
        ax.set_title(f'{title_sfx}\n L*a*b* Cube', fontsize=cfg.LABEL_S)
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
        ax_ab.set_title(f'{title_sfx} Lab ab Plane', fontsize=cfg.LABEL_S)
        fig_ab.tight_layout()        
        # export as pdf

        return fig, ax

    def render_Chab_plane(self,
                        conditions: str) -> Tuple[plt.figure, plt.axes]:
        """Render the C*h(ab) plane of the spectral library for the given conditions.
        Note that no method is available AFAIK for rendring a polar cyclindrical
        plot of the LCh space. So only plotting 2D C*h polar plane.

        :param conditions: Conditions used in the colour computation
        :type conditions: str
        :return: Figure and Axes of the plot
        :rtype: Tuple[plt.figure, plt.axes]
        """
        title_sfx = conditions
        
        # load the spectral library
        index = self.spectra_obj.main_df.index
        LCh = self.spectra_obj.colour_df[conditions][['L*', 'C*', 'h']].to_numpy()
        rgb = self.spectra_obj.colour_df[conditions][['R', 'G', 'B']].to_numpy()


        # separately plot each catgeory with a different marker
        cats = self.spectra_obj.main_df['Category'][index]
        uniq_cats = cats.unique()
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
        fig = plt.figure(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        ax = fig.add_subplot(111, projection='polar')

        # add grid
        ax.set_axisbelow(True)
        ax.grid(True)        

        for i, cat in enumerate(uniq_cats):
            cat_LCh = LCh[cats == cat]
            cat_rgb = rgb[cats == cat]
            ax.scatter(np.deg2rad(cat_LCh[:,2]), cat_LCh[:,1], 
                        c=cat_rgb,
                        marker=syms_list[i], label=cat)
            ax.set_rmax(100)  
            # set radial axis font size
            ax.tick_params(axis='x', labelsize=cfg.LEGEND_S)
            ax.tick_params(axis='y', labelsize=cfg.LEGEND_S)
        ax.set_title(fr"{title_sfx}""\n LCh Chroma "rf"($r$) hue ($\theta$) Plane", fontsize=cfg.LABEL_S)
        fig.tight_layout()        
        # export as pdf

        return fig, ax

    def render_colour(self,
            conditions: str,
            srgb_compare: Union[bool, object]=False) -> plt.figure:
        """Render the colour of each spectrum in the spectral library according
        to the given computed colour coordinates.

        :param colour_obj: Colour object with tables of RGB, XYZ and xyY values
        :type colour_obj: object, MaterialCollection or Observation
        :param srgb_compare: Indicate if sRGB comparison is to be made,
                defaults to False
        :type srgb_compare: Union[bool, object], optional
        :return: Table of colours for each spectrum, and figure of colours
        :rtype: pd.DataFrame, plt.figure
        """        
        # load the spectral library        
        index = self.spectra_obj.main_df.index

        rgb = self.spectra_obj.colour_df[conditions][['R', 'G', 'B']].to_numpy()
        xyY = self.spectra_obj.colour_df[conditions][['x', 'y', 'Y']].to_numpy()

        cats = self.spectra_obj.main_df['Category'][index]
        
        title_sfx = conditions

        # if compare, load the comparison material collection colour df
        if srgb_compare and self.obj_type == 'observation':
            srgb_obj = srgb_compare.material_collection
            srgb_rgb = srgb_obj.colour_df[['R', 'G', 'B']].to_numpy()
            srgb_xyY = srgb_obj.colour_df[['x', 'y', 'Y']].to_numpy()
            srgb_cats = srgb_obj.colour_df['Category']
            title_sfx = f"sRGB vs. {title_sfx}"

        # Plotting the RGB Cube
        # make a 3D plot of rgb array
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(rgb[:,0], rgb[:,1], rgb[:,2], c=rgb, s=100, depthshade=True)
        # ax.set_xlabel('R')
        ax.tick_params(axis='x', colors='red')
        # ax.set_ylabel('G')
        ax.tick_params(axis='y', colors='green')
        # ax.set_zlabel('B')
        ax.tick_params(axis='z', colors='blue')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)        
        ax.set_title(f'{title_sfx} RGB Cube')
        fig.tight_layout()
        # export as pdf

        # *** Plotting in chromaticity space ***

        # TODO plot the comparison xyY coordinates, with annotations showing change
        
        uniq_cats = cats.unique()
        fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(
            show=False, 
            show_spectral_locus=True,
            show_diagram_colours=True,
            transparent_background=False,            
            )
        
        # cat to integer
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
                    markeredgewidth=0.5,
                    markeredgecolor='white', markersize=4)
        
        # plot the sRGB space in the chromaticity diagram
        sRGB_ps = colour.RGB_COLOURSPACES['sRGB'].primaries
        ax.plot(sRGB_ps[:,0],sRGB_ps[:,1], color='k', marker='', label='sRGB')
        ax.plot(sRGB_ps[[2,0],0],sRGB_ps[[2,0],1], color='k', marker='') 

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
        
        ax.legend(handles, labels, loc='upper right', fontsize='x-small')
        
        fig.suptitle(f'{title_sfx}', fontsize='large')

        # set figure size
        fig.set_size_inches(2*cfg.FIG_SIZE[0], 2*cfg.FIG_SIZE[1])   
        # set DPI
        fig.set_dpi(cfg.DPI)

        fig.tight_layout()  
        
        return fig