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
import colour
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

        self.filter_cols = self.set_filter_cols()

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
    def build_cauchy_filter(
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
        gamma = fwhm / 2 # convert from fwhm to 1-gamma (hwhm)
        # vectorisation: extend cwls, sigs & wvls to match dimensions
        cwls = np.tile(cwl, [cfg.WVLS.shape[0],1])
        gammas = np.tile(gamma, [cfg.WVLS.shape[0],1])
        wvls = np.tile(cfg.WVLS, [cwls.shape[1],1]).transpose()
        # compute the Gaussian profiles in parallel
        cauchy = np.divide((np.power(gammas, 2.)), (np.power(wvls - cwls, 2.) + (np.power(gammas, 2.))))
        return cauchy.transpose()

    @staticmethod
    def build_instrument_df(
            inst_df: pd.DataFrame, 
            shape: str='gauss') -> pd.DataFrame:
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
            elif shape == 'cauchy':
                out = Instrument.build_cauchy_filter(cwls, fwhms) 
            else:
                raise ValueError('Filter shape not recognised.')
            # initialise the dataframe according to contents
            init_df =pd.DataFrame(data=out,columns=cfg.WVLS,index=inst_df.index)
        else:
            out = inst_df.to_numpy().T

            # interpolate each channel to simulation wavelengths
            wvls_in = inst_df.index.to_numpy().astype('float64')
            trans_in = inst_df.to_numpy().astype('float64')
            trans_func = interp1d(wvls_in, trans_in.T, bounds_error=False)
            out = trans_func(cfg.WVLS)
            # TODO check that the input wavelengths match or exceed the 
            # simulation wavelengths, and deal with it if not.
            # Quick fix - set all NaN values to 0
            out = np.where(np.isnan(out), 0, out)
            
            # Normalise to the overall max value (should this be for each channel??)
            out = out / out.max()

            # initialise the dataframe according to contents
            init_df = pd.DataFrame(
                            data=out,
                            columns=cfg.WVLS,
                            index=inst_df.columns)
            init_df.index.rename('filter_id')

            # get CWl and FWHM values
            enrgy_count = init_df * init_df.columns
            norm_enrgy_count = (enrgy_count.T / enrgy_count.max(axis=1)).T

            cwls = np.sum((norm_enrgy_count*cfg.WVLS),axis=1) / np.sum((norm_enrgy_count),axis=1)
            cwls = np.round(cwls)

            half_maxima = 0.5
            limits = np.greater(norm_enrgy_count.T, half_maxima).T.to_numpy()
            fwhms = np.zeros(len(cwls))
            for channel in range(len(cwls)):
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
        
        # set order by cwl
        main_df = main_df.sort_values(by='cwl')

        return main_df

    def set_filter_cols(self) -> pd.Series:
        """Set the filter colours of the instrument

        :return: filter colours
        :rtype: Series
        """
        
        # if not a spectrometer, get filter colours
        if self.filter_ids[0][0] != 'S':
            # set colours for the filters        
            norm_trans = self.get_trans_df().T
            sds = colour.MultiSpectralDistributions(norm_trans)    
            # normalise the spds
            # sds = sds / np.sum(sds, axis=1)[:,None]
                
            illum = colour.SDS_ILLUMINANTS['D65'] # use a D65 standard illuminant
            xyz = colour.sd_to_XYZ(sds, illuminant=illum, k=1.0)/100 # convert to XYZ space
            # xyz = xyz / np.sum(xyz, axis=1)[:,None]
            # xyz = xyz.clip(0,100) # clip to 0-1 range
            rgb = colour.XYZ_to_sRGB(xyz) # convert to sRGB    
            rgb = rgb.clip(0,1) # clip to 0-1 range
                
            # if np.any(rgb < 0):
            #     # We're not in the RGB gamut: approximate by desaturating
            #     w = - np.min(rgb, axis=0)
            #     rgb = rgb + w
            if not np.all(rgb==0):
                # Normalize the rgb vector
                rgb /= np.max(rgb)

            # just do central wavelength to XYZ
            # xyz = colour.wavelength_to_XYZ(self.cwls().to_numpy())
            # colour.plotting.plot_single_sd(sds[:,0])
            
            # make df of colours
            col_df = pd.DataFrame(rgb, columns=['r','g','b'], index=self.filter_ids)
        else:
            cwl_colours = 'husl'
            # get list of colours based on husl colour palette
            col_df = pd.DataFrame(
                    sns.color_palette(
                        cwl_colours, 
                        n_colors=len(self.filter_ids)), 
                    columns=['r','g','b'], 
                    index=self.filter_ids)
        return col_df

    def set_snr(self,
                snr: Union[float, np.ndarray, List, Dict, pd.Series]) -> None:
        """Set the signal-to-noise ratio of the instrument

        :param snr: signal-to-noise ratio(s) to set
        :type snr: Union[float, np.array, List, Dict, pd.Series]
        """
        if isinstance(snr, (float, int)):
            # set all channels to the same snr
            self.main_df['snr'] = snr
        elif isinstance(snr, (np.ndarray, list)):
            # set each channel to the corresponding snr
            self.main_df['snr'] = snr
        elif isinstance(snr, (Dict, pd.Series)):
            # set each channel to the corresponding snr
            self.main_df['snr'] = snr
        else:
            raise ValueError('SNR data type not recognised.')

        # put snr after fwhms in dataframe
        snr_col = self.main_df.pop('snr')
        self.main_df.insert(2, 'snr', snr_col)

    def set_opt_refl(self,
                    refl_opt: Union[float, np.ndarray, List, Dict, pd.Series]) -> None:
        """Set the reflectance value that the each channel of the instrument
        is optimised for - i.e. the value at which the defined SNR is acheived 
        at. Particularly important for framing spectral imagers like CaSSIS.

        :param refl_opt: optimal reflectance values
        :type refl_opt: Union[float, np.ndarray, List, Dict, pd.Series]
        """
        if isinstance(refl_opt, (float, int)):
            # set all channels to the same snr
            self.main_df['refl_opt'] = refl_opt
        elif isinstance(refl_opt, (np.ndarray, list)):
            # set each channel to the corresponding snr
            self.main_df['refl_opt'] = refl_opt
        elif isinstance(refl_opt, (Dict, pd.Series)):
            # set each channel to the corresponding snr
            self.main_df['refl_opt'] = refl_opt
        else:
            raise ValueError('Optimal Reflectance data type not recognised.')
        
        # put optimal refl after snr in dataframe
        refl_col = self.main_df.pop('refl_opt')
        self.main_df.insert(3, 'refl_opt', refl_col)

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

    def plot_filter_profiles(self, subfig: plt.figure=None):
        """Plot all filter profiles
        """
        print('Plotting Instrument Transmission...')

        # if 'snr' in self.main_df.columns:
        #     trans_df = pd.melt(self.main_df.reset_index(),
        #                         id_vars=['cwl', 'fwhm', 'snr','filter_id'])
        # else:
        #     trans_df = pd.melt(self.main_df.reset_index(),
        #                         id_vars=['cwl', 'fwhm', 'filter_id'])
            
        wvl_lo_idx = self.main_df.columns.get_loc(self.wvls[0])
        id_vars = self.main_df.iloc[:,:wvl_lo_idx].columns.to_list()
        id_vars.append(self.main_df.index.name)
        trans_df = pd.melt(self.main_df.reset_index(), id_vars=id_vars)

        if subfig is not None:
            fltr_ax = subfig.add_subplot()
            fig = subfig
        else:
            fltr_ax_size = (cfg.FIG_SIZE[0], 1.1*cfg.FIG_SIZE[1])
            fig, fltr_ax = plt.subplots(figsize=fltr_ax_size, dpi=cfg.DPI)

        # set up the plot
        sns.set_context("paper")
        # use futura font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Futura'
        sns.despine(ax=fltr_ax, right=True, top=True)
        # plt.rcParams.update({'font.size': 8})
        
        fltr_ax.set(
            xbound=(cfg.SAMPLE_RES['wvl_min']-10, cfg.SAMPLE_RES['wvl_max']+10),
            ybound=(-0.05,1.15),
            autoscale_on=False)        
        
        # get colour palette from the filter colours DataFrame
        cwl_colours = self.filter_cols.to_numpy()        

        # if instrument is a spectrometer, select subset of wavelengths
        # if filter ids start with S then it's a spectrometer
        if self.filter_ids[0][0] == 'S':
            # select first and last wavelengths, then middle
            sns.lineplot(
                data=trans_df,
                x='variable',
                y='value',
                ax=fltr_ax,
                hue='filter_id',
                alpha=0.3,
                palette="husl",
                linewidth=0.6,
                legend=False)

            cwls = self.cwls()[[0, len(self.cwls())//2, -1]].unique()
            trans_df = trans_df[trans_df['cwl'].isin(cwls)]
            filter_ids = trans_df.filter_id.unique()
            fwhms = trans_df.fwhm.unique()
            # downselect cwl_colours
            cwl_colours = cwl_colours[[0, len(cwl_colours)//2, -1]]
        else:
            cwls = self.cwls().to_numpy()
            filter_ids = self.filter_ids
            fwhms = self.fwhms().unique()
                
        sns.lineplot(
            data=trans_df,
            x='variable',
            y='value',
            ax=fltr_ax,
            hue='filter_id',
            palette=cwl_colours,
            linewidth=0.6,
            legend="full")
        
        fltr_ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        fltr_ax.set_ylabel('Transmission', fontsize=cfg.LABEL_S)
        fltr_ax.set_title(
                f'{self.name.title()} Transmission Profiles', fontsize=cfg.TITLE_S)
        
        # add minor grid lines at 50 nm intervals and major gridlines at 100 nm
        # or minor at 100 and major at 500, depending on spectral range
        spec_range = cfg.SAMPLE_RES['wvl_max'] - cfg.SAMPLE_RES['wvl_min']
        if spec_range <= 1000:
            fltr_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(50))
            fltr_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(100))
        elif spec_range <= 5000:
            fltr_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(100))
            fltr_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(500))
        else:
            fltr_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(500))
            fltr_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(1000))

        fltr_ax.grid(True, which='major',axis='both', lw=0.6)
        fltr_ax.grid(True, which='minor',axis='both', lw=0.3)

        # Set the font name for axis tick labels to be Comic Sans
        for tick in fltr_ax.get_xticklabels():
            tick.set_fontname("Arial")
            tick.set_fontsize(cfg.LABEL_S)
        for tick in fltr_ax.get_yticklabels():
            tick.set_fontname("Arial")
            tick.set_fontsize(cfg.LABEL_S)

        label_params = fltr_ax.get_legend_handles_labels()

        labels = ["%s\n%.0f nm\n±%.1f nm" % (filter_id, cwl, fwhm) for filter_id, cwl, fwhm in zip(filter_ids, cwls, fwhms)]

        # if self.filter_ids[0][0] == 'S':
        #     # insert '...' between labels
        #     labels.insert(1, '...')
        #     labels.insert(3, '...')
        #     blank = mpl.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',visible=False)
        #     label_params[0].insert(1, blank)
        #     label_params[0].insert(3, blank)

        n_ids = len(labels)
        new_label_params = (label_params[0], labels)

        # put the legend below the plot, and make horizontal
        fltr_ax.legend(*new_label_params,
                loc='upper center', bbox_to_anchor=(0.5, -0.25),
                ncols=n_ids, frameon=False,
                fontsize=cfg.LEGEND_S)
               
        if self.filter_ids[0][0] == 'S':
            # insert '...' between labels
            filter_ids = np.insert(filter_ids,1, '  ...')
            filter_ids = np.insert(filter_ids, 3, '  ...')            
            cwls = np.insert(cwls, 1, cwls[0]+(cwls[1]-cwls[0])/2) 
            cwls = np.insert(cwls, 3, cwls[2]+(cwls[3]-cwls[2])/2)

        for f, fltr_id in enumerate(filter_ids):
            fltr_ax.annotate(
                fltr_id,
                (cwls[f], 1.02),
                ha='left',
                annotation_clip=False,
                fontsize=cfg.LEGEND_S,
                rotation=60)
            
        if subfig is None:
            plt.tight_layout()
            output_file = Path(self.object_dir, self.name).with_suffix(cfg.PLT_FRMT)
            fig.savefig(output_file)
            print('Plots exported to '+str(Path(self.object_dir)))

        return fig, fltr_ax

    def plot_spectral_resolution(self, subfig: plt.figure=None):
        """Plot the spectral power resolution of the instrument as a function 
        of cwl
        """   

        print('Plotting Spectral Resolution...')

        if subfig is not None:
            res_ax = subfig.add_subplot()
            fig = subfig
        else:
            fig, res_ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)
        
        # set up the plot
        sns.set_context("paper")
        # use futura font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Futura'
        sns.despine(right=True, top=True)
        # plt.rcParams.update({'font.size': 8})                
        
        spectral_resolution = self.cwls() / self.fwhms() 

        res_ax.set(
            xbound=(cfg.SAMPLE_RES['wvl_min']-10, cfg.SAMPLE_RES['wvl_max']+10),
            ybound=(0, 1.1*np.max(spectral_resolution)),
            autoscale_on=False)        
        
        # get colour palette from the filter colours DataFrame
        cwl_colours = self.filter_cols.to_numpy()        
        res_ax.scatter(
            self.cwls(),
            spectral_resolution,
            marker='o',
            c=cwl_colours,
            zorder=1
        )

        res_ax.plot(
            self.cwls(),
            spectral_resolution,
            'k--',
            zorder=0
        )
        
        res_ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        res_ax.set_ylabel('Spectral Resolving Power', fontsize=cfg.LABEL_S)
        res_ax.set_title(
                f'{self.name.title()} Spectral Resolution', fontsize=cfg.TITLE_S)
        
        # add minor grid lines at 50 nm intervals and major gridlines at 100 nm
        # or minor at 100 and major at 500, depending on spectral range
        spec_range = cfg.SAMPLE_RES['wvl_max'] - cfg.SAMPLE_RES['wvl_min']
        if spec_range <= 1000:
            res_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(50))
            res_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(100))
        elif spec_range <= 5000:
            res_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(100))
            res_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(500))
        else:
            res_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(500))
            res_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(1000))

        res_ax.grid(True, which='major',axis='both', lw=0.6)
        res_ax.grid(True, which='minor',axis='both', lw=0.3)

        # Set the font name for axis tick labels to be Comic Sans
        for tick in res_ax.get_xticklabels():
            tick.set_fontname("Arial")
        for tick in res_ax.get_yticklabels():
            tick.set_fontname("Arial")

        if subfig is None:        
            plt.tight_layout()
            output_file = Path(self.object_dir, self.name+'_spectral_resolution').with_suffix(cfg.PLT_FRMT)
            fig.savefig(output_file)
            print('Plots exported to '+str(Path(self.object_dir)))

        return fig, res_ax

    def plot_fwhm(self, subfig: plt.figure=None):
        """Plot the full-width at half-maximum of the instrument as a function
        of cwl
        """

        print('Plotting FWHM...')
        # set up the plot
        # use futura font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Futura'
        sns.despine(right=True, top=True)

        if subfig is not None:
            fwhm_ax = subfig.add_subplot()
            fig = subfig
        else:
            fig, fwhm_ax = plt.subplots(figsize=cfg.FIG_SIZE, dpi=cfg.DPI)

        fwhms = self.fwhms()

        fwhm_ax.set(
            xbound=(cfg.SAMPLE_RES['wvl_min']-10, cfg.SAMPLE_RES['wvl_max']+10),
            ybound=(0, 1.1*np.max(fwhms)),
            autoscale_on=False)

        # get colour palette from the filter colours DataFrame
        cwl_colours = self.filter_cols.to_numpy()        
        fwhm_ax.scatter(
            self.cwls(),
            fwhms,
            marker='o',
            c=cwl_colours,
            zorder=1
        )

        fwhm_ax.plot(
            self.cwls(),
            fwhms,
            'k--',
            zorder=0
        )

        fwhm_ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        fwhm_ax.set_ylabel('FWHM (nm)', fontsize=cfg.LABEL_S)
        fwhm_ax.set_title(
                f'{self.name.title()} FWHM', fontsize=cfg.TITLE_S)

        # add minor grid lines at 50 nm intervals and major gridlines at 100 nm
        # or minor at 100 and major at 500, depending on spectral range
        spec_range = cfg.SAMPLE_RES['wvl_max'] - cfg.SAMPLE_RES['wvl_min']
        if spec_range <= 1000:
            fwhm_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(50))
            fwhm_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(100))
        elif spec_range <= 5000:
            fwhm_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(100))
            fwhm_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(500))
        else:
            fwhm_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(500))
            fwhm_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(1000))

        fwhm_ax.grid(True, which='major',axis='both', lw=0.6)
        fwhm_ax.grid(True, which='minor',axis='both', lw=0.3)

        # Set the font name for axis tick labels to be Comic Sans
        for tick in fwhm_ax.get_xticklabels():
            tick.set_fontname("Arial")
        for tick in fwhm_ax.get_yticklabels():
            tick.set_fontname("Arial")

        if subfig is None:
            plt.tight_layout()
            output_file = Path(self.object_dir, self.name+'_fwhms').with_suffix(cfg.PLT_FRMT)
            fig.savefig(output_file)
            print('Plots exported to '+str(Path(self.object_dir)))

        return fig, fwhm_ax
    
    def plot_snr_max(self, subfig: plt.figure=None):
        """Plot the max. signal-to-noise ratio of the instrument as a function
        of cwl
        """   

        print('Plotting Maximum Signal-to-Noise Ratio...')

        if subfig is not None:
            snr_ax = subfig.add_subplot()
            fig = subfig
        else:
            fltr_ax_size = (cfg.FIG_SIZE[0], cfg.FIG_SIZE[1])
            fig, snr_ax = plt.subplots(figsize=fltr_ax_size, dpi=cfg.DPI)

        # set up the plot
        # use futura font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Futura'
        sns.despine(right=True, top=True)

        try:
            snrs = self.main_df['snr']
            snr_max = np.nanmax(snrs)
        except KeyError:
            print('No SNR data available for this instrument.')
            return fig, snr_ax

        snr_ax.set(
            xbound=(cfg.SAMPLE_RES['wvl_min']-10, cfg.SAMPLE_RES['wvl_max']+10),
            ybound=(0, 1.1*snr_max),
            autoscale_on=False)

        # get colour palette from the filter colours DataFrame
        cwl_colours = self.filter_cols.to_numpy()        
        snr_ax.scatter(
            self.cwls(),
            snrs,
            marker='o',
            c=cwl_colours,
            zorder=1
        )

        snr_ax.plot(
            self.cwls(),
            snrs,
            'k--',
            zorder=0
        )
        
        snr_ax.set_xlabel('Wavelength (nm)', fontsize=cfg.LABEL_S)
        snr_ax.set_ylabel('Max. SNR', fontsize=cfg.LABEL_S)
        snr_ax.set_title(
                f'{self.name.title()} Signal-to-Noise Ratio', fontsize=cfg.TITLE_S)

        # add minor grid lines at 50 nm intervals and major gridlines at 100 nm
        # or minor at 100 and major at 500, depending on spectral range
        spec_range = cfg.SAMPLE_RES['wvl_max'] - cfg.SAMPLE_RES['wvl_min']
        if spec_range <= 1000:
            snr_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(50))
            snr_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(100))

        elif spec_range <= 5000:
            snr_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(100))
            snr_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(500))
        else:
            snr_ax.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(500))
            snr_ax.get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(1000))

        snr_ax.grid(True, which='major',axis='both', lw=0.6)
        snr_ax.grid(True, which='minor',axis='both', lw=0.3)

        # Set the font name for axis tick labels to be Comic Sans
        for tick in snr_ax.get_xticklabels():
           tick.set_fontname("Arial")
        for tick in snr_ax.get_yticklabels():
            tick.set_fontname("Arial")

        if subfig is None:
            plt.tight_layout()
            output_file = Path(self.object_dir, self.name+'_snr').with_suffix(cfg.PLT_FRMT)
            fig.savefig(output_file)
            print('Plots exported to '+str(Path(self.object_dir)))

        return fig, snr_ax    

    def plot_snr_r(self, subfig: plt.figure=None):
        """Plot the signal-to-noise ratio of the instrument as a function
        of reflectance
        """   

        print('Plotting Signal-to-Noise Ratio aas function of R...')

        if subfig is not None:
            snr_ax = subfig.add_subplot()
            fig = subfig
        else:
            fltr_ax_size = (cfg.FIG_SIZE[0], cfg.FIG_SIZE[1])
            fig, snr_ax = plt.subplots(figsize=fltr_ax_size, dpi=cfg.DPI)

        # set up the plot
        # use futura font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Futura'
        sns.despine(right=True, top=True)

        try:
            snrs = self.main_df['snr'].to_numpy()
            snr_max = np.nanmax(snrs)
        except KeyError:
            print('No SNR data available for this instrument.')
            return fig, snr_ax

        # set up according to optimal reflectance in each channel
        if 'refl_opt' in self.main_df.columns:
            refl_opt = self.main_df['refl_opt'].to_numpy()
        else:
            # assume optimised for reflectance of 1
            refl_opt = [1.0] * len(self.filter_ids)

        refl = np.linspace(0, 1, 100)
        snr_r = np.outer(np.sqrt(refl), snrs).T

        # snr_ax.set(
        #     xbound=(refl[0], refl[-1]),
        #     ybound=(0, 1.1*snr_max),
        #     autoscale_on=False)

        # get colour palette from the filter colours DataFrame
        cwl_colours = self.filter_cols.to_numpy()        

        for i, filter_id in enumerate(self.filter_ids):
            
            snr_ax.plot(
                refl * refl_opt[i],
                snr_r[i],
                label=filter_id,
                color=cwl_colours[i],
                linewidth=0.6,
                zorder=1
            )

        snr_ax.set_xlabel('I/F', fontsize=cfg.LABEL_S)
        snr_ax.set_ylabel('SNR(R)', fontsize=cfg.LABEL_S)
        snr_ax.set_title(
                f'{self.name.title()} Signal-to-Noise Ratio', fontsize=cfg.TITLE_S)

        # set legend font size
        snr_ax.legend(fontsize=cfg.LEGEND_S)

        snr_ax.grid(True, which='major',axis='both', lw=0.6)
        snr_ax.grid(True, which='minor',axis='both', lw=0.3)

        # Set the font name for axis tick labels to be Comic Sans
        for tick in snr_ax.get_xticklabels():
           tick.set_fontname("Arial")
        for tick in snr_ax.get_yticklabels():
            tick.set_fontname("Arial")

        if subfig is None:
            plt.tight_layout()
            output_file = Path(self.object_dir, self.name+'_snr').with_suffix(cfg.PLT_FRMT)
            fig.savefig(output_file)
            print('Plots exported to '+str(Path(self.object_dir)))

        return fig, snr_ax     

    def plot_instrument_characteristics(self):
        """Plot the instrument characteristics of the instrument as a function
        of cwl
        """
        # make a figure to hold the information. 2 x 2 grid
        fig = plt.figure(layout='constrained', figsize=(2*cfg.FIG_SIZE[0], 2*cfg.FIG_SIZE[1]), dpi=cfg.DPI)           
        subfig = fig.subfigures(2, 2)

        # aff filter profile plot
        subfig[0][0], filter_ax = self.plot_filter_profiles(subfig[0][0])
        # add 'A.' to the title
        filter_ax.set_title('A. Transmission Profiles', fontsize=cfg.TITLE_S)

        # add signal-to-noise ratio plot
        subfig[1][0], snr_ax = self.plot_snr_max(subfig[1][0])
        # add 'B.' to the title
        snr_ax.set_title('B. Max. Signal-to-Noise Ratio', fontsize=cfg.TITLE_S)
        
        # add fwhm plot
        subfig[0][1], fwhm_ax = self.plot_fwhm(subfig[0][1])
        # add 'C.' to the title
        fwhm_ax.set_title('C. FWHM', fontsize=cfg.TITLE_S)

        # if insturment is not spectrometer, add the SNR vs R plot
        if self.filter_ids[0][0] != 'S':
            subfig[1][1], d_ax = self.plot_snr_r(subfig[1][1])
            # add 'D.' to the title
            d_ax.set_title('D. Signal-to-Noise Ratio vs Reflectance', fontsize=cfg.TITLE_S)
        else:
            # add spectral resolution plot        
            subfig[1][1], d_ax = self.plot_spectral_resolution(subfig[1][1])
            # add 'D.' to the title
            d_ax.set_title('D. Spectral Resolution', fontsize=cfg.TITLE_S)
        
        axes = [filter_ax, snr_ax, fwhm_ax, d_ax]

        # activate the figure
        plt.figure(fig.number)
        
        # set plot title
        fig.suptitle(f'{self.name.title()} Characteristics', fontsize=cfg.TITLE_S)

        output_file = Path(self.object_dir, self.name+'_instrument_characteristics').with_suffix(cfg.PLT_FRMT)
        plt.savefig(output_file)

        print('Plots exported to '+str(Path(self.object_dir)))

        return fig, axes


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
            spectral_range: List,
            snr_range: List, # just specify the start and end SNR for now - simple
        ) -> None:

        self.name = instrument_name
        self.instrument_type = instrument_type
        self.sampling = sampling
        self.resolution = resolution
        self.spectral_range = spectral_range
        self.snr_range = snr_range

        self.main_df = self.build_instrument()

    def build_instrument(self) -> pd.DataFrame:
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
        return inst_df

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

        # start_cwl = wvl_lo / (1 - 1/self.resolution)
        # end_cwl = wvl_hi / (1 + 1/self.resolution)
        start_cwl = wvl_lo
        end_cwl = wvl_hi

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

        snrs = np.linspace(self.snr_range[0], self.snr_range[1], len(cwls))

        inst_df = pd.DataFrame(data={
                                    'filter_id':filter_ids,
                                    'cwl': cwls,
                                    'fwhm': fwhms,
                                    'snr': snrs})
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

        snrs = np.linspace(self.snr_range[0], self.snr_range[1], len(cwls))

        inst_df = pd.DataFrame(data={
                                    'filter_id':filter_ids,
                                    'cwl': cwls,
                                    'fwhm': fwhms,
                                    'snr': snrs})
    
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
        cwl: Union[float, np.array],
        resolution_model: Dict,
        ) -> Union[float, np.array]:
        """Compute the fwhm (nm) given a cwl (nm)

        :param cwl: central wavelength (nm) to compute fwhm at
        :type cwl: Union[n.float, np.array]
        :param resolution_model: linear model for converting wavenumber to
            spectral resolving power
        :type resolution_per_wavenumber: dict
        :return: full-width at half-maximum(maxima), for central wavelength(s)
        :rtype: Union[float, np.array]
        """
        cwn = 1E7 / cwl # cm^-1
        resolution_per_wavenumber = resolution_model['m'] # 1/cm^-1
        offset = resolution_model['c']
        power = cwn * resolution_per_wavenumber + offset
        fwhm = cwl / power
        return fwhm
