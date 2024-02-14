"""PDS RELAB Spectral Library Reader

Reads RELAB spectral library data downloaded from the PDS Geosciences Node,
and standardises the formatting for compatibility with the VISOR spectral library
interface tool.

Saves each spectral library entry in a new csv file with metadata included.

Author: Roger Stabbins, NHM
Date: 20-7-2022
"""
from pathlib import Path
import csv
import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd
from matplotlib.pyplot import close

class PdsRelabFormatter():
    """Class for accessing a reading PDS RELAB Spectral Library data,
    and for formatting each entry into a csv file compatible with the VISOR
    database.
    """

    def __init__(
            self,
            sample_name: str,
            library_path: str) -> None:
        """PDS RELAB reader object.

        :param sample_name: sample ID string in the spectral library directory
        :type sample_name: str
        :param library_path: Path to the spectral library local directory
        :type library_path: str
        """
        self.sample_name = sample_name
        self.library_path = library_path
        self.sample_data = Path(library_path, sample_name).with_suffix('.tab')
        self.sample_xml = Path(library_path, sample_name).with_suffix('.xml')
        self.mineral_group = None
        self.mineral_name = None
        self.sample_id = None
        # check for existence of these files
        try:
            open(self.sample_data)
        except FileNotFoundError:
            print(f"No data file for {self.sample_name}.")
            raise
        try:
            open(self.sample_xml)
        except FileNotFoundError:
            print(f"No xml file for {self.sample_name}.")
            raise

    def load_sample(self):
        """Load the sample xml and data files
        """
        hdr, entries = self.read_xml()
        data = self.read_data(entries)
        hdr.columns = data.columns
        sample_df = pd.concat([hdr, data], axis=0, ignore_index=True)
        return sample_df

    def read_xml(self):
        """Read the xml file of the sample
        """
        xml_df = pd.read_xml(str(self.sample_xml))
        xml_data = ET.parse(str(self.sample_xml))  # Parse XML data
        root = xml_data.getroot()  # Root element

        ns = {'speclib': 'http://pds.nasa.gov/pds4/speclib/v1',
                'pds': 'http://pds.nasa.gov/pds4/pds/v1'}

        # add try and catches for all of these
        sample_id = root.find('.//speclib:specimen_id', ns).text
        try:
            mineral_name = root.find('.//speclib:mineral_subtype', ns).text.lower()
        except AttributeError:
            try:
                mineral_name = root.find('.//speclib:rock_subtype', ns).text.lower()
            except AttributeError:
                mineral_name = root.find('.//speclib:specimen_name', ns).text.lower()
        sample_description = root.find('.//speclib:specimen_name', ns).text
        date_added = root.find('.//pds:creation_date_time', ns).text
        incidence_angle = root.find('.//speclib:incidence_angle', ns).text
        if not incidence_angle:
            incidence_angle = 'NA'
        emission_unit = root.find('.//speclib:emission_angle', ns).attrib['unit']
        emission_angle = root.find('.//speclib:emission_angle', ns).text
        if not emission_angle:
            emission_angle = 'NA'
        incidence_unit = root.find('.//speclib:emission_angle', ns).attrib['unit']
        viewing_geometry = f'\
            i={incidence_angle} {incidence_unit} \
            e={emission_angle} {emission_unit}'
        min_grain_size = root.find('.//speclib:specimen_min_size', ns).text
        max_grain_size = root.find('.//speclib:specimen_max_size', ns).text
        grain_units = root.find('.//speclib:specimen_max_size', ns).attrib['unit']
        grain_size = f'{min_grain_size} - {max_grain_size} {grain_units}'
        instrument_name = root.find('.//speclib:instrument_name', ns).text
        range_min = root.find('.//speclib:spectral_range_min', ns).text
        range_max = root.find('.//speclib:spectral_range_max', ns).text
        range_unit = root.find('.//speclib:spectral_range_unit', ns).text
        try:
            mineral_group = root.find('.//speclib:mineral_type', ns).text.lower()
        except AttributeError:
            mineral_group = root.find('.//speclib:rock_type', ns).text.lower()

        other_information = f'Instrument name: {instrument_name}; \
                        Range: {range_min} - {range_max} {range_unit}; \
                        Mineral Group: {mineral_group}'

        self.mineral_group = mineral_group
        self.mineral_name = mineral_name
        self.sample_id = sample_id
        entries = int(root.find('.//pds:records', ns).text)

        # put info in a new header DataFrame
        hdr = pd.Series({  # put lists in dataframe
            'Data ID': self.sample_name,
            'Sample ID': sample_id,
            'Mineral Name': mineral_name,
            'Sample Description': sample_description,
            'Date Added': date_added,
            'Viewing Geometry': viewing_geometry,
            'Other Information': other_information,
            'Grain Size': grain_size,
            'Database of Origin': 'RELAB',
            'Wavelength': 'Response'}).to_frame().reset_index()
        return hdr, entries

    def read_data(self, entries: int):
        """Read the data file of the sample
        """
        data = pd.read_csv(
                self.sample_data,
                nrows=entries,
                delim_whitespace=True,
                header=None,
                skiprows=1)
        if len(data.columns) == 3:
            data = data.drop(columns=2) # drop the standard deviation data if present
        return data

    def write_sample(self, sample_df: pd.DataFrame):
        """write the sample to csv file and store in the sptk spectral library

        :param sample_df: the sample data
        :type sample_df: pd.DataFrame
        """
        # get sptk spectral library path
        home = os.path.expanduser("~")
        data_library = Path(home / Path(os.path.dirname(__file__)) / 'spectral_library' / 'RELAB') # local VISOR
        # look for path under the mineral type
        mineral_group_dir = Path(data_library / self.mineral_group)
        mineral_group_dir.mkdir(parents=True, exist_ok=True)
        # look for path under the mineral sub_type
        mineral_name_dir = Path(mineral_group_dir / self.mineral_name)
        mineral_name_dir.mkdir(parents=True, exist_ok=True)
        # # look for path under sample id
        # sample_id_dir = Path(mineral_name_dir / self.sample_id)
        # sample_id_dir.mkdir(parents=True, exist_ok=True)
        # export the DataFrame to csv file in this format
        filepath = Path(mineral_name_dir / self.sample_name).with_suffix('.csv')
        sample_df.to_csv(filepath, index=False, header=False)

def print_xml_tags(root: ET.Element) -> None:
    """Get and print the tag names of the xml file associated
    with the root of xml ElementTree.

    :param root: root of an xml ElementTree
    :type root: ET.Element
    """
    print('tags of current xml file:')
    for elem in root.iter():
        print(elem.tag)

if __name__ == '__main__':
    relab_dl_path = Path(os.path.expanduser("~") / 'Desktop' / 'cartorder')
    # iterate list of all unique samples
    for sample in relab_dl_path.glob('*.tab'):
        print(sample.stem)
        ldr = PdsRelabFormatter(sample.stem, relab_dl_path)
        sample_data = ldr.load_sample()
        ldr.write_sample(sample_data)