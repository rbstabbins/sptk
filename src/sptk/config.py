"""Configuration file for Spectral Parameters Toolkit

Sets variables for:
 - timing of procedures
 - sampling resolution

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 18-05-2022
"""

import os
from pathlib import Path
from typing import Tuple
import numpy as np

TIME_IT = False # report process durations

"""Spectral Resolution"""
SAMPLE_RES={"wvl_min": 400, # spectral range (nm)
            "wvl_max": 1100, # spectral range (nm)
            "delta_wvl": 1} # spectral intervals (nm)
WVLS = np.arange(
            SAMPLE_RES['wvl_min'],
            SAMPLE_RES['wvl_max'],
            SAMPLE_RES['delta_wvl'],
            dtype=float)

"""Plotting"""
PLOT_PROFILES = True
# INLINE = False # produce plots inline - i.e. for notebook
CM = 1/2.54  # set scale factor for specifying fig size in cms
FIG_SIZE = (8*CM, 7*CM)
DPI = 300
PLT_FRMT = '.png' # '.png' or '.pdf'
LABEL_S = 8
TITLE_S = 10
LEGEND_S = 6

"""Exports"""
EXPORT_DF = True
LOAD_EXISTING = True # if True load existing directories, else build new
DATA_DIRECTORY = Path('..', 'data') # to be updated during script running
OUTPUT_DIRECTORY = Path('..', 'projects')

def build_project_directory(
        project_name: str,
        class_name: str) -> Tuple[Path, str]:
    """Builds a directory for the project at the users Desktop.

    If the project already exists, give option for loading in the existing
    project, or writing a new one - specify this with a keyword - default is
    false. Duplicate project names have a number suffix applied to the project
    name.

    :return: project directory path, and the updated project name
    :rtype: Tuple[Path, str]
    """
    project_dir = OUTPUT_DIRECTORY / project_name
    dir_built = False
    pn_w_sffx = project_name
    sffx_n = 2  # original directory is '1', so start suffix at 2

    while not dir_built:
        try: # if the project_dir does not exist, move to exception
            Path(project_dir / class_name).mkdir(parents=True, exist_ok=False)
            project_name = pn_w_sffx
            dir_built = True
        except FileExistsError:
            if LOAD_EXISTING:
                dir_built = True # no new folder will be written
            else: # add suffix number to project name and try again
                pn_w_sffx = project_name + '_' + str(sffx_n)
                project_dir = Path(OUTPUT_DIRECTORY / pn_w_sffx)
                sffx_n += 1

    # if ~LOAD_EXISTING:
    #     Path(project_dir / class_name).mkdir(parents=True, exist_ok=True)

    return project_dir, project_name

def resolve_path(path: Path, root: str='data') -> Path:
    """Resolve a path relative to the data or output directory.

    :param path: Path to resolve
    :type path: Path
    :param root: Root is for data or output directory, defaults to 'data'
    :type root: str, optional
    :return: Complete resolved path
    :rtype: Path
    """
    if root == 'data':
        return DATA_DIRECTORY / path
    elif root == 'output':
        return OUTPUT_DIRECTORY / path
    else:
        raise ValueError('root must be "data" or "output"')