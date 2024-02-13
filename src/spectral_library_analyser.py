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
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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
            spectra_obj: object,
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
            ci: bool=False):
        """Plot the profiles of the materials of the spectral library
        """
        if cfg.TIME_IT:
            tic = time.perf_counter()
            print('Plotting reflectance profiles of materials...')

        # get the reflectance data for all minerals and for each category
        refl_df = self.spectra_obj.get_refl_df()
        cat_df = self.spectra_obj.get_cat_df()
        all_df = pd.concat([refl_df, cat_df], axis=1)
        self.render_profile_plot(all_df,
                    scope=scope,
                    with_noise=with_noise,
                    ci=ci)

        if categories_only:
            if cfg.TIME_IT:
                toc = time.perf_counter()
                print(f"Reflectance profiles plotted in {toc - tic:0.4f} s.")
            return

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
                self.render_profile_plot(cat_mnrl_df,
                            cat=cat,
                            mnrl=mnrl,
                            scope=scope,
                            with_noise=with_noise,
                            ci=ci)

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Reflectance profiles plotted in {toc - tic:0.4f} s.")

    def render_profile_plot(self,
            data_df: pd.DataFrame,
            cat: str='all',
            mnrl: str='entries',
            ci: bool=False,
            scope: str='all',
            with_noise: bool=False) -> None:
        """Method for producing the plot itself, according to given DataFrame,
        class, mineral name, and scope.

        :param data_df: Material Collection or Observation to DataFrame
        :type data_df: pd.DataFrame
        :param cat: Class label, defaults to 'all'
        :type cat: str, optional
        :param mnrl: Mineral Name, defaults to 'entries'
        :type mnrl: str, optional
        :param ci: Plot mean spectra with condifence interval, defaults to False
        :type ci: bool, optional
        :param scope: indicates specific elements or all data, defaults to 'all'
        :type scope: str, optional
        :param with_noise: INdicates if noise has been added, defaults to False
        :type with_noise: bool, optional
        """

        data_df = data_df.reset_index()
        data_df =pd.melt(data_df, id_vars=['Data ID','Category'])

        out_dir = Path(self.spectra_obj.object_dir / 'plots')
        out_dir.mkdir(parents=True, exist_ok=True)

        if with_noise:
            sfx = '_with_noise'
        else:
            sfx = ''

        fig_size = (1.3*cfg.FIG_SIZE[0], cfg.FIG_SIZE[1])
        fig, ax = plt.subplots(figsize=fig_size, dpi=cfg.DPI)
        y_max = max([1.0, data_df.value.max()])

        if cat == 'all':
            hue_flag = 'Category'
        else:
            hue_flag = 'Data ID'

        if self.obj_type == 'observation':
            marker_flag = True
        else:
            marker_flag = False

        plt.rcParams.update({'font.size': 8, 'lines.markersize': 3})
        ax.set(
            xbound=(cfg.SAMPLE_RES['wvl_min']-10,cfg.SAMPLE_RES['wvl_max']+10),
            ybound=(-0.1, y_max+0.1),
            autoscale_on=False)

        if ci:
            # long form version of plotting, to aggregate
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

        ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        ax.set_ylabel('Reflectance', fontsize=cfg.LABEL_S)
        # add minor grid lines at 50 nm intervals and major gridlines
        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(True, which='major',axis='both', lw=0.6)
        ax.grid(True, which='minor',axis='both', lw=0.3)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax.legend(loc='upper right')
        plt.setp(ax.get_legend().get_texts(), fontsize=cfg.LEGEND_S)
        plt.setp(ax.get_legend().get_title(), fontsize=cfg.LEGEND_S)

        # plot title
        project_str = self.spectra_obj.project_name.replace('_', ' ')
        if self.obj_type == 'observation':
            leg_title = f'Class: {cat}, Group: {mnrl} ({scope} data) - sampled'
            if ci:
                title = 'Mean ± 1σ Sampled Spectral Library'
            else:
                title = 'Instrument Sampled Spectral Library'
        else:
            leg_title = f'Class: {cat}, Group: {mnrl} ({scope} data)'
            title = 'High-Resolution Spectral Library'
        plt.title(title, fontsize=cfg.TITLE_S)

        # save legend separately
        if cat != 'all':
            label_params = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            figl, axl = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            axl.axis(False)
            n_ids = len(data_df['Data ID'].unique())
            leg = axl.legend(*label_params, loc="center",
                bbox_to_anchor=(0.5, 0.5),
                ncol=-(-n_ids // 20),
                fontsize='xx-small')
            leg.set_title(leg_title, prop={'size': 'x-small'})

        fig.tight_layout()
        # if cat != 'all':
        #     try:
        #         figl.tight_layout()
        #     except UserWarning:
        #         axl.legend(*label_params, loc="center",
        #             bbox_to_anchor=(0.5, 0.5),
        #             ncol=-(-data_df.shape[1] // 35),
        #             fontsize=3.0)

        # save figure
        project_str = self.spectra_obj.project_name
        if self.obj_type == 'observation':
            inst = self.spectra_obj.instrument.name
            filename = f'{project_str}_{inst}_{cat}_{mnrl}_{scope}'+sfx
        else:
            filename = f'{project_str}_{cat}_{mnrl}_{scope}'+sfx
        output_file = Path(out_dir, filename).with_suffix(cfg.PLT_FRMT)
        fig.savefig(output_file)
        if cat != 'all':
            legend_file=Path(out_dir,filename+'_lgnd').with_suffix(cfg.PLT_FRMT)
            figl.savefig(legend_file)

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

    # """
    # Methods still in development
    # """

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
