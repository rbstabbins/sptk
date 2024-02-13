"""Test data script
Generate synthetic test data for unit tests on the Spectral Parameters Toolkit.
Test data has two classes, test_target and test_background.
Data sets are output for each of these to the sptk/spectral_library/test_target_data and
sptk/spectral_library/test_background_data directories.
Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 26-04-2022
"""
import shutil
from datetime import date
from pathlib import Path
from typing import Union
import numpy.polynomial.polynomial as poly
import numpy as np
import pandas as pd
from sptk.config import PACKAGE_DIRECTORY, SAMPLE_RES

def generate_test_spectral_library(
        flat_target: float=None,
        flat_background: float=None,
        out_of_bounds: bool=False,
        n_samples: Union[int,dict]=5) -> dict:
    """Generate synthetic data for testing sptk.
    :param flat_target: generate a spectrally flat target spectrum, defaults to None
    :type flat_target: bool, optional
    :param flat_background: generate a spectrally flat background spectrum, defaults to None
    :type flat_background: bool, optional
    :param n_samples: number of samples, defaults to 5
    :type n_samples: int, optional
    :return: coefficients used for non-flat synthetic spectra.
    :rtype: dict
    """

    # hires_wavelengths = np.linspace(380, 1100, num=720)
    # set the desired datapoints, and add clamps at boundaries
    if out_of_bounds:
        lores_wavelengths = np.linspace(SAMPLE_RES['wvl_min'], 0.7*SAMPLE_RES['wvl_max'], num = 10)
        hires_wavelengths = np.arange(
                SAMPLE_RES['wvl_min'],
                0.7*SAMPLE_RES['wvl_max'],
                SAMPLE_RES['delta_wvl'],
                dtype=float)
    else:
        lores_wavelengths = np.linspace(SAMPLE_RES['wvl_min'], SAMPLE_RES['wvl_max'], num = 10)
        hires_wavelengths = np.arange(
                SAMPLE_RES['wvl_min'],
                SAMPLE_RES['wvl_max'],
                SAMPLE_RES['delta_wvl'],
                dtype=float)
    # lores_wavelengths =  np.array([380., 400., 450., 550., 650., 750., 850., 950., 1050, 1100])
    target_desired = np.array([0.80, 0.80, 0.50, 0.25, 0.50, 0.80, 0.50, 0.80, 0.50, 0.65])
    background_desired = np.array([0.80, 0.80, 0.50, 0.25, 0.50, 0.80, 0.50, 0.40, 0.30, 0.20])
    classes = {'test_target': target_desired, 'test_background':background_desired}

    # get date of data generation
    today = date.today()
    today_str = today.strftime("%d/%m/%Y")

    coef_dict = {}

    # update number of samples in each class according to flat target or background
    if flat_target and flat_background:
        n_samples = 1

    if isinstance(n_samples, int):
        n_samples = {'test_target': n_samples, 'test_background': n_samples}

    root_dir = Path(PACKAGE_DIRECTORY / '..' / 'data' / 'spectral_library' / 'test_library')
    for clss in classes:
        dir_clss = Path(root_dir / clss)
        if Path.exists(dir_clss):
            shutil.rmtree(dir_clss)
        dir_clss.mkdir(parents=True, exist_ok=True)

    # build data-base and add noise to the desired datapoints
    entry_list = []
    for clss in classes:
        for i in range(n_samples[clss]):
            # build template data entry
            hdr_lbls = ['Database of Origin',
                        'Sample Description',
                        'Date Added',
                        'Data ID',
                        'Sample ID',
                        'Mineral Name',
                        'Wavelength']
            data_id = clss + str(i+1).zfill(3)
            hdr_info = ['Test Library',
                        'Synthetic Test',
                        today_str,
                        data_id,
                        data_id,
                        clss,
                        '']
            if clss == 'test_target' and flat_target:
                hires_sample = np.zeros(len(hires_wavelengths)) + flat_target
            elif clss == 'test_background' and flat_background:
                hires_sample = np.zeros(len(hires_wavelengths)) + flat_background
            else:
                # add noise, unless it is the first entry - to give a determined example
                if i != 0:
                    noise = np.random.normal(0, 0.03, len(lores_wavelengths))
                    lores_sample = classes[clss] + noise
                else:
                    lores_sample = classes[clss]

                # interpolate the datapoints by fitting a polynomial to each.
                coefs = poly.polyfit(lores_wavelengths, lores_sample, 7)
                hires_sample = poly.polyval(hires_wavelengths, coefs)

                # log the polynomial coefficients in the dictionary for function verification
                coef_dict[data_id] = coefs

            # format the data correctly for export
            series = pd.Series(index=hdr_lbls + hires_wavelengths.tolist(), data=hdr_info + hires_sample.tolist())

            entry_list.append(series)

            # output the datasets to the test_data directories
            dir_out = Path(root_dir / clss )
            file_out = Path(dir_out / data_id).with_suffix('.csv')
            series.to_csv(file_out, header=False)
    return entry_list

if __name__ == '__main__':
    coef_dict_test = generate_test_spectral_library(flat_background=0.25, flat_target=0.75)