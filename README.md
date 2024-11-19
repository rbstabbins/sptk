<p align="center">
  <a href="" rel="noopener">
 <img max-width=960px src="https://github.com/rbstabbins/sptk/blob/main/title.gif?raw=true" alt="Project logo"></a>
</p>

<h3 align="center">sptk: The Spectral Parameters Toolkit</h3>

<div align='center'>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10692531.svg)](https://doi.org/10.5281/zenodo.10692531)

</div>

---

<p align="center">
<strong>sptk</strong> is a Python package for investigating the ability of a multispectral imaging system to identify distinct materials and material groups through differences in reflectance spectra.
    <br>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [About ](#about-)
- [Updates ](#updates-)
  - [V0 -\> V1 Changes to SPTK (currently hosted in cassis\_development branch).](#v0---v1-changes-to-sptk-currently-hosted-in-cassis_development-branch)
- [Installing ](#installing-)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
- [Running the Tests](#running-the-tests)
- [Running the Example Notebooks](#running-the-example-notebooks)
- [Authors](#authors)
- [Citing the Software](#citing-the-software)
- [Acknowledgements](#acknowledgements)

## About <a name = "about"></a>

**sptk** provides a simple interface for:
* simulating the spectral response of an instrument,
* sampling a spectral library with the instrument,
* measuring the reconstruction error of the instrument on the spectral library,
* evaluating the spectral parameters afforded by the instrument,
* evaluating and ranking the ability of the spectral parmameters, and spectral parameter combinations, to separate categories of materials.

## Updates <a name = "updates"></a>

### V0 -> V1 Changes to SPTK (currently hosted in cassis_development branch).

  **src/config.py**
  - Change of default plot output from ```".png"``` to ```".pdf"```.
  - New ```update_sample_res()``` function for changing the spectral sample resolution of the given instance of SPTK.

  **src/instrument.py** 
  - New ```shape``` option for generating ```"top-hat"``` profile transmission filters, as an alternative to ```"gauss"``` filters used previously.
  - New ability to resample input high-resolution spectral transmission profile files to match the spectral sample resolution of the given instance of SPTK, via linear interpolation.
  - New ability to estimate central-wavelength (CWL) and full-width-at-half-maximum (FWHM) from input high-resolution spectral transmission profiles.
  - Addition of optional Signal-to-Noise Ratio (SNR) property to the Instrument object. Searches for SNR information when loading instrument transmission profiles.
  - New method for estimating the standard-RGB colour of a given transmission filter, using the [```colour-science``` Python library](https://www.colour-science.org/).

  **src/linear_discriminant_analysis.py**
  - Minor change to handling of the ```singular_mask``` np.array to conform to pylint requirements, such that the inverse of the ```singular_mask``` matrix is calculated before passing to the ```bcsm``` (between-class scatter matrix), rather than during the passing.

  **src/material_collection.py**
  - ```plot_profiles``` function now returns a list of the matplotlib axes objects for the generated plots of the material collection.
  - A new ```render_colour``` function takes an argument of a standard illuminant type, and calls the new ```SpectralLibraryAnalyser.render_colour()``` function, producing the expected standard-RGB appearance of each entry of the given MaterialCollection object.

  **src/observation.py**
  - Now inherits project name from the associated Instrument object, such that new projects can be constructed for unique MaterialCollection x Instrument combinations.
  - New ```noisy`` attribute that tracks whether noise has been added to the given Observation object.
  - New function for computing the uncertainty on the Observation object reflectance data, given the Signal-to-noise Ratio of the associated Instrument object.
  - The ```add_noise``` function has been updated to replace the noise definition, that was previously defined as ```shot``` or ```thermal``, to now use the given SNR information, that can optionally be scaled by a given Spectral Power Distribution (SPD).
  - Noise function to be refined.
  - ```plot_profiles``` function now returns a list of the matplotlib axes objects for the generated plots of the material collection.
  - A new ```render_colour``` function takes an argument of a standard illuminant type, and calls the new ```SpectralLibraryAnalyser.render_colour()``` function, producing the expected standard-RGB appearance of each entry of the given Observation object.

  **src/spectral_library_analyser.py**
  - The ```plot_profiles``` function now returns a list of the matplotlib Axes objects produced during the plotting process.
  - Minor adjustments have been made to the ```render_plot_profile```, that now follows the Seaborn ```"paper"`` style.
  - A new ```render_colour``` function has been written that uses the ```colour-science``` Python library to compute the expected colour of each entry of the given spectral library object (MaterialCollection or Observation), under the given illuminant. The colour presentation is underdevelopment.

  **src/spectral_parameter_combination_classifier.py**
  - Minor updates have been made to plot figure formatting.

  **src/spectral_parameters.py**
  - Ratio spectral parameters are now plotted on a log-10 scale, for better symmetry for presenting ratios of ±1.

  **examples/material_collection_colour_example.ipynb**
  - This new notebook demonstrates the Material Collection colour rendering function.

  **examples/observation_colour_example.ipynb**
  - This new notebook demonstrates the Observation colour rendering function.


## Installing <a name = "installing"></a>

```sptk``` is available via PyPI. 

We recommend downloading a copy of the [https://github.com/rbstabbins/sptk](https://github.com/rbstabbins/sptk) repository, and running the unit tests and working through the example notebooks. 

To run the example notebooks you'll also need to download the accompanying Example Dataset, hosted in the following Zenodo repository: [doi:10.5281/zenodo.10683367](https://zenodo.org/doi/10.5281/zenodo.10683367).

### Prerequisites

First, prepare a new environment with Python=3.10.8, using your environment manager of choice. 

For example, with conda:
```
conda env create -n sptk python=3.10.8
```
and activate the environment:
```
conda activate sptk
```

Currently **sptk** is only available via pip, so make sure you have pip installed on your environment also, e.g.:

```
conda install pip
```


### Installing

Install the latest version of **sptk** with pip:

```
pip install sptk
```
you can also specify the version you'd like to install, e.g.:
```
pip install sptk=0.1
```

## Running the Tests<a name = "running-the-tests"></a>

The ```sptk/tests/``` directory hosts a set of unit tests for each module of the **sptk** package. These have been written for the ```unittest``` unit testing framework.

The unit tests can be executed by navigating to the ```sptk/tests``` directory and running:

```
python -m unittest -v
```

The unit tests provided are comprehensive but not exhaustive. We recommend also executing the example notebooks to test and understand the software.

## Running the Example Notebooks<a name = "running-the-example-notebooks"></a>

We recommend exploring the [example notebooks](./examples/) to become familiar with the software and the placement of directories in the repository.

Please follow the guidelines in the [README.md](./examples/README.md) to download the required [Example Dataset](https://zenodo.org/doi/10.5281/zenodo.10683367) for executing the example notebooks.

The ```sptk/tests/``` directory hosts a set of unit tests for each module of the **sptk** package. These have been written with

## Authors<a name = "authors"></a>

The Spectral Parameters Toolkit was designed and developed by [@rbstabbins](https://github.com/rbstabbins).

See also the list of [contributors](https://github.com/rbstabbins/sptk/contributors) who participated in this project.

## Citing the Software<a name = "citing-the-software"></a>

If you use **sptk** in your research, please provide acknowledgement to the authors with the following citation:

Roger Stabbins, & Grindrod, P. (2024). rbstabbins/sptk: Release v0.1 (v0.1). Zenodo. https://doi.org/10.5281/zenodo.10694286

## Acknowledgements

The development of this software has been funded by the following grants:
- UK Space Agency Aurora Science Programme: Geochemistry to Geology for ExoMars 2020 visible to near infrared spectral variability ST/T001747/1
- UK Space Agency Mars Exploration Science Standard Call 2023: Exploring the Limits of Material Discrimination with CaSSIS Multiband Imaging ST/Y005910/1