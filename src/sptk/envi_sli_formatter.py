"""ENVI SLI file Spectral Library Reader

Reads ENVI SLI spectral library data, and standardises the formatting for compatibility with the
VISOR spectral library interface tool.

Saves each spectral library entry in a new csv file with metadata included.

Author: Roger Stabbins, NHM
Date: 26-08-2022
"""
import os
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import pysptools.util as util

def envi_sli_formatter(file_name: Union[str, Path]):
    """Convert an ENVI SLI spectral library file to a standardised format for 
    the VISOR spectral library interface tool.

    :param file_name: Path to the HDR file of the ENVI SLI spectral library.
    :type file_name: Union[str, Path]
    """    

    # file_name = "/Users/rogs/Documents/data/spectral_libraries/mica_pete/PMG_MICA_CRISM_all_ratioed.hdr"

    if type(file_name) is str:
        file_name = Path(file_name)

    library_name = os.path.basename(file_name).split('.')[0]

    library_path = file_name.parent.parent # local spectral library path

    spectra, envi_hdr = util.load_ENVI_spec_lib(file_name) # open the sli file

    # get sptk spectral library path
    data_library = Path(library_path / library_name ) # local VISOR
    data_library.mkdir(parents=True, exist_ok=True)

    # parse into separate spectra
    sample_names = envi_hdr['spectra names']
    wvls = envi_hdr['wavelength'] # convert from microns to nanometers
    if (envi_hdr['wavelength units'] == 'Micrometers') or 
                                        (envi_hdr['wavelength units'] == 'um'):
        wvls = [float(x)*1000 for x in wvls] # convert to nm from um
        # TODO other conversions to nm
    for count, sample_name in enumerate(sample_names):
        spectrum = spectra[count, :].flatten() # get spectrum
        # set 0's to NaN
        spectrum[spectrum==0.0] = np.nan
        # set bad numbers to NaN
        spectrum[spectrum==65535] = np.nan
        data = pd.Series(data=spectrum, index=wvls).to_frame().reset_index()

        # TODO: need a new method for getting the Mineral Name out of the sample name

        # # get metadata
        # sample_name = sample_name.replace('/', '-')
        # mineral_name = sample_name.split(' ')[0].lower() # first string in sample name
        # # replace '/' with '-'
        # sample_id = sample_name.split(' ')[1] # second string in sample name

        # # get MICA CRISM ratioed metadata
        # sample_name = sample_name.replace('/', '-')
        # mineral_name = (' ').join(sample_name.split('_')[2:])
        # sample_id = mineral_name

        # build Pandas Series from this
            # put info in a new header DataFrame
        hdr = pd.Series({  # put lists in dataframe
            'Data ID': sample_name,
            'Sample ID': sample_id,
            'Mineral Name': mineral_name,
            'Sample Description': '',
            'Date Added': '',
            'Viewing Geometry': '',
            'Other Information': '',
            'Grain Size': '',
            'Database of Origin': '',
            'Wavelength': 'Response'}).to_frame().reset_index()
        sample_df = pd.concat([hdr, data], axis=0, ignore_index=True)
        # save each spectra in a new csv file
        mineral_name_dir = Path(data_library / mineral_name)
        mineral_name_dir.mkdir(parents=True, exist_ok=True)
        filepath = Path(mineral_name_dir / sample_name).with_suffix('.csv')
        sample_df.to_csv(filepath, index=False, header=False)
