"""Instrument and Instrument Builder Classes

Instrument Class:
Hosts the transmission data for the filters of spectral sampling instrument.

Instrument Builder Class:
Produces files listing the instrument sampling central wavelengths, FWHMs,
and channel labels.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 29-04-2021
"""
import os.path as osp
from pathlib import Path
import glob
from  shutil import rmtree
import time
from typing import Dict, List, Union
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
import sptk.config as cfg

class Instrument():
    """Hosts the transmission data for the channels of spectral sampling
    instrument."""

    def __init__(
            self,
            name: str,
            project_name: str = 'case',
            shape: str='gauss',
            load_existing: bool=cfg.LOAD_EXISTING,
            plot_profiles: bool = cfg.PLOT_PROFILES,
            export_df: bool = cfg.EXPORT_DF):
        """Constructor for Instrument

        Represents the spectral transmission of the channels of an instrument
        in a DataFrame.

        :param name: the instrument name, to be looked-up from local directory
        :type name: str
        :param project_name: name of project, defaults to 'case'
        :type project_name: str, optional
        :param plot_profiles: plot material profiles, defaults to PLOT_PROFILES
        :type plot_profiles: bool, optional
        :param export_df: export DataFrames, defaults to EXPORT_DF
        :type export_df: bool, optional
        """
        print('Building Instrument...')
        if cfg.TIME_IT:
            tic = time.perf_counter()

        p_dir, p_name = cfg.build_project_directory(project_name, 'instrument')
        self.project_dir = p_dir
        self.project_name = p_name
        self.object_dir = Path(self.project_dir / 'instrument')
        self.wvls = cfg.WVLS
        self.name = name

        if load_existing:
            # check if the instrument has data already in the project directory
            existing_pkl_path = Path(self.object_dir, 'instrument.pkl')
            if osp.isfile(existing_pkl_path):
                print("Loading existing DataFrame for % r..." % name)
                self.main_df = pd.read_pickle(existing_pkl_path)
            else:
                print('No existing DataFrame, building new for % r...' % name)
                self.build_new_instrument(shape, plot_profiles, export_df)
        else:
            print("Building new DataFrame for % r..." % name)
            self.build_new_instrument(shape, plot_profiles, export_df)

        self.filter_ids = self.main_df.index.to_list()

        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Instrument built in {toc - tic:0.4f} seconds.")

    def __del__(self, rmdir: bool = False, rmproj: bool = False) -> None:
        """Instrument Destructor method - optionally deletes
        instrument directory and/or entire project directory

        :param rmdir: instruct removal of instrument directory,
            defaults to False
        :type rmdir: bool, optional
        :param rmproj: instruct removal of entire project directory,
            defaults to False
        :type rmproj: bool, optional
        """
        if rmdir:
            name = self.name
            print(f"Deleting {name} Instrument directory...")
            try:
                rmtree(Path(self.project_dir, 'instrument'))
                print(f"{name} Instrument directory deleted.")
            except FileNotFoundError:
                print(f"No {name} Instrument directory to delete.")
        if rmproj:
            name = self.project_name
            print(f"Deleting {name} directory...")
            try:
                rmtree(Path(self.project_dir))
                print(f"{name} directory deleted.")
            except FileNotFoundError:
                print(f"No {name} directory to delete.")

    def build_new_instrument(self,
            shape: str='gauss',
            plot_profiles: bool = cfg.PLOT_PROFILES,
            export_df: bool = cfg.EXPORT_DF) -> None:
        """Build DataFrame and optionally export and produce plots.

        :param plot_profiles: plot material profiles, defaults to PLOT_PROFILES
        :type plot_profiles: bool, optional
        :param export_df: export DataFrames, defaults to EXPORT_DF
        :type export_df: bool, optional
        """
        inst_data = Instrument.read_instrument_data(self.name)
        self.main_df = Instrument.build_instrument_df(inst_data, shape=shape)
        # optionally produce plots and files of the instrument
        if plot_profiles:
            self.plot_filter_profiles()
        if export_df:
            self.export_main_df()

    @staticmethod
    def read_instrument_data(name: str) -> pd.DataFrame:
        """Find instrument data in the library instrument directory,
        and read into DataFrame

        :return: instrument cwl and fwhm data
        :rtype: pd.DataFrame
        """
        # get allowed instrument names from the list of existing instrument
        # files in the sptk distribution
        search_str = str(cfg.DATA_DIRECTORY / 'instruments' / '*csv')
        inst_files = glob.glob(search_str)
        allowed_names = [osp.splitext(osp.basename(i))[0] for i in inst_files]
        if name not in allowed_names:
            raise ValueError(f'{name} is not in the instrument directory.')
        inst_file = Path(cfg.DATA_DIRECTORY / 'instruments',
                                                name).with_suffix('.csv')
        inst_data = pd.read_csv(inst_file, index_col=0)
        return inst_data

    @staticmethod
    def build_gauss_filter(
            cwl: Union[np.array, float],
            fwhm: Union[np.array,float]) -> np.array:
        """Build Gaussian filter profiles according to the given cwl, fwhm and
        wvls, in parallel.

        :param cwl: Centre wavelength(s) (nm)
        :type cwl: np.array
        :param fwhm: Full-Width at Half Maximum(s) (nm)
        :type fwhm: np.array
        :returns: table of Gaussian transmission profile data
        :rtype: np.array
        """
        sig = fwhm / 2.355482004503 # convert from fwhm to 1-sigma
        # vectorisation: extend cwls, sigs & wvls to match dimensions
        cwls = np.tile(cwl, [cfg.WVLS.shape[0],1])
        sigs = np.tile(sig, [cfg.WVLS.shape[0],1])
        wvls = np.tile(cfg.WVLS, [cwls.shape[1],1]).transpose()
        # compute the Gaussian profiles in parallel
        gauss = np.exp(-np.power(wvls - cwls, 2.) / (2 * np.power(sigs, 2.)))
        return gauss.transpose()

    @staticmethod
    def build_tophat_filter(
            cwl: Union[np.array, float],
            fwhm: Union[np.array,float]) -> np.array:
        """Build Top-Hat filter profiles according to the given cwl, fwhm and
        wvls, in parallel.

        :param cwl: Centre wavelength(s) (nm)
        :type cwl: np.array
        :param fwhm: Full-Width at Half Maximum(s) (nm)
        :type fwhm: np.array
        :returns: table of Top-Hat transmission profile data
        :rtype: np.array
        """        
        # vectorisation: extend cwls, sigs & wvls to match dimensions
        cwls = np.tile(cwl, [cfg.WVLS.shape[0],1])
        lower = np.tile(cwl - fwhm/2, [cfg.WVLS.shape[0],1])
        upper = np.tile(cwl + fwhm/2, [cfg.WVLS.shape[0],1])
        wvls = np.tile(cfg.WVLS, [cwls.shape[1],1]).transpose()
        transmission = np.where((wvls > lower) & (wvls < upper), 1.0, 0.0) 
        return transmission.transpose()

    @staticmethod
    def build_instrument_df(inst_df: pd.DataFrame, shape: str='gauss') -> pd.DataFrame:
        """Builds instrument transmission profiles for filter cwls and fwhms
        using a Gaussian function, and returns in a DataFrame.

        DataFrame format:
         - Columns: filter
         - Rows: wavelengths (according to cfg.SAMPLE_RES)

        :param inst_df: Instrument filter names, cwls and fwhms
        :type inst_df: pd.DataFrame
        :returns: the instrument transmission table
        :rtype: pd.DataFrame
        """
        if 'cwl' in inst_df.columns:
            # build each filter in the instrument in parallel
            cwls = inst_df['cwl'].values
            fwhms = inst_df['fwhm'].values
            if shape == 'gauss':
                out = Instrument.build_gauss_filter(cwls, fwhms)
            elif shape == 'tophat':
                out = Instrument.build_tophat_filter(cwls, fwhms) 
            # initialise the dataframe according to contents
            init_df =pd.DataFrame(data=out,columns=cfg.WVLS,index=inst_df.index)
        else:
            out = inst_df.to_numpy().T

            # interpolate each channel to simulation wavelengths
            wvls_in = inst_df.index.to_numpy().astype('float64')
            trans_in = inst_df.to_numpy().astype('float64')
            trans_func = interp1d(wvls_in, trans_in.T, bounds_error=False)
            out = trans_func(cfg.WVLS)

            # initialise the dataframe according to contents
            init_df = pd.DataFrame(
                            data=out,
                            columns=cfg.WVLS,
                            index=inst_df.columns)
            init_df.index.rename('filter_id')

            # get CWl and FWHM values
            cwls = np.sum((out*cfg.WVLS),axis=1) / np.sum((out),axis=1)
            cwls = np.round(cwls)

            half_maxima = init_df.max(axis=1)/2
            limits = np.greater(out.T, half_maxima.to_numpy().T).T
            fwhms = np.zeros(len(half_maxima))
            for channel in range(len(half_maxima)):
                wvls_above_hmax = cfg.WVLS[limits[channel]]
                fwhms[channel] = wvls_above_hmax[-1] - wvls_above_hmax[0]

            inst_df = pd.DataFrame(cwls, dtype=int, index=init_df.index)
            inst_df.index.rename('filter_id', inplace=True)
            inst_df = inst_df.rename(columns={0:'cwl'})
            inst_df['fwhm'] = fwhms
            # if 'snr' in inst_df.columns:
            #     inst_df['snr'] = snrs

        # concat with cwl and fwhm information (via inst_df)
        main_df = pd.merge(
                    left=inst_df,
                    right=init_df,
                    left_index=True,
                    right_index=True,
                    how='outer')
        return main_df

    def get_trans_df(self) -> pd.DataFrame:
        """Return a copy of the transmission dataframe only

        :return: copy of the master instrument transmission DataFrame
        :rtype: pd.DataFrame
        """
        trans_df = self.main_df[cfg.WVLS] # fix this to get wavelength column names instead of hardcoding
        return trans_df.copy()

    def cwls(self) -> pd.Series:
        """Get the centre wavelengths of the instrument channels

        :return: instrument channel centre wavelengths
        :rtype: pd.Series
        """
        return self.main_df.cwl.copy(deep=True)

    def fwhms(self) -> pd.Series:
        """Get the centre wavelengths of the instrument channels

        :return: instrument channel centre wavelengths
        :rtype: pd.Series
        """
        return self.main_df.fwhm.copy(deep=True)

    def get_metrics(self) -> pd.DataFrame:
        """Return a copy of the filter metrics (cwl, fwhm) dataframe only

        :return: copy of the instrument cwl and fwhm DataFrame
        :rtype: pd.DataFrame
        """
        metric_df = self.main_df.iloc[:,:2]
        return metric_df.copy()

    def get_filter_ids(self,
            cwl: Union[float, int, np.array, List]) -> Union[str, List]:
        """Get filter id(s) for given centre-wavelength(s)

        :param cwl: centre-wavelength(s) to retrieve filter id(s) for
        :type cwl: Union[str, List]
        :return: channel label(s)
        :rtype: Union[str, List]
        """
        if isinstance(cwl, (int, float)):
            cwl = [cwl] # incapsulate in list
        filter_series = pd.Series(
                            data=self.cwls().index.to_list(),
                            index=self.cwls().to_list())
        filter_id = filter_series[cwl].to_list()
        return filter_id

    def plot_filter_profiles(self):
        """Plot all filter profiles
        """
        print('Plotting Instrument Transmission...')
        if 'snr' in self.main_df.columns:
            trans_df = pd.melt(self.main_df.reset_index(),
                                id_vars=['cwl', 'fwhm', 'snr','filter_id'])
        else:
            trans_df = pd.melt(self.main_df.reset_index(),
                                id_vars=['cwl', 'fwhm', 'filter_id'])
        fltr_ax_size = (1.3*cfg.FIG_SIZE[0], cfg.FIG_SIZE[1])
        fig, fltr_ax = plt.subplots(figsize=fltr_ax_size, dpi=cfg.DPI)
        plt.rcParams.update({'font.size': 8})
        fltr_ax.set(
            xbound=(cfg.SAMPLE_RES['wvl_min']-10, cfg.SAMPLE_RES['wvl_max']+10),
            ybound=(-0.05,1.15),
            autoscale_on=False)
        sns.lineplot(
            data=trans_df,
            x='variable',
            y='value',
            ax=fltr_ax,
            hue='cwl',
            palette='nipy_spectral',
            linewidth=0.6,
            legend="full")
        fltr_ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        fltr_ax.set_ylabel('Transmission', fontsize=cfg.LABEL_S)
        fltr_ax.set_title(
                f'Transmission Profiles ({self.name})', fontsize=cfg.TITLE_S)
        # add minor grid lines at 50 nm intervals and major gridlines
        fltr_ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        fltr_ax.grid(True, which='major',axis='both', lw=0.6)
        fltr_ax.grid(True, which='minor',axis='both', lw=0.3)
        label_params = fltr_ax.get_legend_handles_labels()

        if len(self.main_df['cwl']) > 12:
            fltr_ax.get_legend().remove()
        else:
            labels = (self.main_df['cwl'].astype('str') + '±'
                    + (self.main_df['fwhm']/2).astype('str')+' nm').to_list()
            n_ids = len(labels)
            new_label_params = (label_params[0], labels)
            fltr_ax.legend(*new_label_params,
                    loc='center left', bbox_to_anchor=(1, 0.5),
                    ncol=-(-n_ids // 20),
                    fontsize=cfg.LEGEND_S)
        # TODO colour code the filters in a more appropriate way
        for fltr_id in self.main_df.index.to_list():
            fltr_ax.annotate(
                fltr_id,
                (self.main_df.loc[fltr_id,'cwl'], 1.02),
                ha='left',
                annotation_clip=False,
                rotation=60)
        plt.tight_layout()
        output_file = Path(self.object_dir, self.name).with_suffix(cfg.PLT_FRMT)
        fig.savefig(output_file)

        if len(self.main_df['cwl']) > 12:
            # make table of cwl and fwhm
            figl, axl = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
            axl.axis(False)
            labels = (self.main_df.index.astype('str') + ', '
                        + self.main_df['cwl'].astype('str') + ' nm, ∆'
                        + self.main_df['fwhm'].astype('str')+' nm').to_list()
            n_ids = len(labels)
            new_label_params = (label_params[0], labels)
            axl.legend(*new_label_params, loc="center",
                    bbox_to_anchor=(0.5, 0.5),
                    ncol=-(-n_ids // 20),
                    fontsize=cfg.LEGEND_S)
            plt.tight_layout()
            output_file = Path(self.object_dir,
                                self.name+'_lgnd').with_suffix(cfg.PLT_FRMT)
            figl.savefig(output_file)

        print('Plots exported to '+str(Path(self.object_dir)))

    def export_main_df(self):
        """Export the Instrument Transmission to CSV and Pickle."""
        if cfg.TIME_IT:
            tic = time.perf_counter()
        print('Exporting the Instrument to CSV and Pickle formats...')
        out_dir = Path(self.object_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_out_file = out_dir / "instrument.csv"
        self.main_df.transpose().to_csv(csv_out_file)
        pkl_out_file = out_dir / "instrument.pkl"
        self.main_df.to_pickle(pkl_out_file)
        if cfg.TIME_IT:
            toc = time.perf_counter()
            print(f"Instrument exported in {toc - tic:0.4f} seconds.")

class InstrumentBuilder:
    """A Class for constructing a cwl and fwhm look-up table for an instrument
    of a given spectral range and resolution
    """
    def __init__(
            self,
            instrument_name: str,
            instrument_type: str,
            sampling: Union[int, str],
            resolution: float,
            spectral_range: List=None
        ) -> None:

        self.name = instrument_name
        self.instrument_type = instrument_type
        self.sampling = sampling
        self.resolution = resolution
        self.spectral_range = spectral_range

    def build_instrument(self) -> None:
        """Builds the instrument table and exports to file
        """
        if self.instrument_type == 'filter-wheel':
            inst_df = self.generate_filter_band_table()
        elif self.instrument_type == 'aotf':
            inst_df = self.generate_aotf_band_table()
        elif self.instrument_type == 'lvf':
            inst_df = self.generate_lvf_band_table()
        else:
            raise ValueError('Instrument Type not recognised.')
        self.export_instrument(inst_df)

    def generate_filter_band_table(self) -> pd.DataFrame:
        """For a Filter-Wheel type spectrometer,
        given a number of filter positions to occupy and a spectral range,
        generate a list of centre wavelengths distributed across the spectral
        range evenly, and compute FWHM's according to resolution, and format
        these into a table to be read by the Instrument
        class.

        :return: channel cwls and fwhms of instrument
        :rtype: pd.DataFrame
        """
        if self.instrument_type != 'filter-wheel':
            raise ValueError('Instrument type is not filter-wheel. \
            Please generate band table suitable for % r' % self.instrument_type)
        filter_ids = []

        if not isinstance(self.sampling,
                    (list, pd.core.series.Series,np.ndarray)):
            raise ValueError("Sampling type must be array-like giving CWLs of \
                                                            filters in wheel.")

        cwls = np.round(np.array(self.sampling)).astype(int)
        fwhms = cwls.astype(float) / self.resolution

        for idx in range(1,len(cwls)+1):
            filter_id = f'F{idx:02d}'
            filter_ids.append(filter_id)
        inst_df = pd.DataFrame(
                    data={'filter_id':filter_ids,
                                    'cwl': cwls,
                                    'fwhm': fwhms})
        return inst_df

    def generate_lvf_band_table(self) -> pd.DataFrame:
        """For a Linear Variable Filter type spectrometer,
        given a sampling condition and spectral resolving power, generate a list
        of centre wavelengths and full widths at half maximum, and format these
        into a table to be read by the instrument function of the spectral
        parameters toolkit

        :return: channel cwls and fwhms of instrument
        :rtype: pd.DataFrame
        """
        if self.instrument_type != 'lvf':
            raise ValueError('Instrument type is not lvf. \
            Please generate band table suitable for % r' % self.instrument_type)

        filter_ids = []
        cwls = []
        fwhms = []

        wvl_lo = self.spectral_range[0]
        wvl_hi = self.spectral_range[1]
        start_cwl = wvl_lo + (wvl_lo / self.resolution)
        end_cwl = wvl_hi - (wvl_hi / self.resolution)

        if self.sampling == 'nyquist':
            fwhm_si = 0.5
        elif self.sampling == 'critical':
            fwhm_si = 1.0
        elif self.sampling == 'undersampled':
            fwhm_si = 2.0
        elif self.sampling == 'hi-res':
            fwhm_si = 0.3
        else:
            raise ValueError('Sampling criteria not recognised')

        cwl = start_cwl
        i = 1
        while cwl <= end_cwl:
            cwls.append(cwl)
            fwhm = cwl / self.resolution
            fwhms.append(fwhm)
            filter_id = f'S{i:03d}'
            filter_ids.append(filter_id)
            cwl = cwl + (fwhm * fwhm_si)
            i+=1
        inst_df = pd.DataFrame(data={
                                    'filter_id':filter_ids,
                                    'cwl': cwls,
                                    'fwhm': fwhms})
        return inst_df

    def generate_aotf_band_table(self) -> pd.DataFrame:
        """For an Acoust-Optic Tunable Filter type spectrometer,
        given a sampling condition and spectral resolving power gradient with
        wavenumber, generate a list of centre wavelengths and full widths at
        half maximum, and format these into a table to be read by the instrument
        function of the spectral parameters toolkit

        :return: channel cwls and fwhms of instrument
        :rtype: pd.DataFrame
        """
        if self.instrument_type != 'aotf':
            raise ValueError('Instrument type is not aotf. \
            Please generate band table suitable for % r' % self.instrument_type)

        filter_ids = []
        cwls = []
        fwhms = []

        wvl_lo = self.spectral_range[0]
        wvl_hi = self.spectral_range[1]
        res = self.resolution
        start_cwl = wvl_lo + InstrumentBuilder.aotf_cwl_2_fwhm(wvl_lo, res)
        end_cwl = wvl_hi - InstrumentBuilder.aotf_cwl_2_fwhm(wvl_hi, res)

        if self.sampling == 'nyquist':
            fwhm_si = 0.5
        elif self.sampling == 'critical':
            fwhm_si = 1.0
        elif self.sampling == 'undersampled':
            fwhm_si = 2.0
        elif self.sampling == 'hi-res':
            fwhm_si = 0.3
        else:
            raise ValueError('Sampling criteria not recognised')

        cwl = start_cwl
        i = 1
        while cwl <= end_cwl:
            cwls.append(cwl)
            fwhm = InstrumentBuilder.aotf_cwl_2_fwhm(cwl, res)
            fwhms.append(fwhm)
            filter_id = f'S{i:03d}'
            filter_ids.append(filter_id)
            cwl = cwl + (fwhm * fwhm_si)
            i+=1
        inst_df = pd.DataFrame(data={
                        'filter_id':filter_ids,
                        'cwl': cwls,
                        'fwhm': fwhms})
        return inst_df

    def export_instrument(self, inst_df: pd.DataFrame) -> None:
        """Export the instrument table to csv file, and save in sptk directory

        :param inst_df: channel cwls and fwhms of instrument
        :type inst_df: pd.DataFrame
        """
        inst_dir = Path(cfg.DATA_DIRECTORY / 'instruments')
        filepath = Path(inst_dir, f'{self.name}').with_suffix('.csv')
        print(f"Exporting instrument to {filepath}...")
        inst_df.to_csv(filepath, index=False)

    @staticmethod
    def aotf_cwl_2_fwhm(
        cwl: Union[np.float, np.array],
        resolution_model: Dict,
        ) -> Union[np.float, np.array]:
        """Compute the fwhm (nm) given a cwl (nm)

        :param cwl: central wavelength (nm) to compute fwhm at
        :type cwl: Union[n.float, np.array]
        :param resolution_model: linear model for converting wavenumber to
            spectral resolving power
        :type resolution_per_wavenumber: dict
        :return: full-width at half-maximum(maxima), for central wavelength(s)
        :rtype: Union[np.float, np.array]
        """
        cwn = 1E7 / cwl # cm^-1
        resolution_per_wavenumber = resolution_model['m'] # 1/cm^-1
        offset = resolution_model['c']
        power = cwn * resolution_per_wavenumber + offset
        fwhm = cwl / power
        return fwhm
