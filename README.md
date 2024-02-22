<p align="center">
  <a href="" rel="noopener">
 <img max-width=960px src="https://github.com/rbstabbins/sptk/blob/v0.1a1/title.gif?raw=true" alt="Project logo"></a>
</p>

<h3 align="center">sptk: The Spectral Parameters Toolkit</h3>

<div align='center'>

[![DOI](https://zenodo.org/badge/756902064.svg)](https://zenodo.org/badge/latestdoi/756902064)

</div>

---

<p align="center">
<strong>sptk</strong> is a Python package for investigating the ability of a multispectral imaging system to identify distinct materials and material groups through differences in reflectance spectra.
    <br>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [About ](#about-)
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

## Installing <a name = "installing"></a>

```sptk``` is available via PyPI and conda. 

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

### Installing

Install sptk v.0.1 with pip or conda, e.g.:

```
conda install sptk=0.1
```
or
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
Roger Stabbins, & Grindrod, P. (2024). rbstabbins/sptk: Release v0.1 (v0.1). Zenodo. https://doi.org/10.5281/zenodo.10692532

## Acknowledgements

The development of this software has been funded by the following grants:
- UK Space Agency Aurora Science Programme: Geochemistry to Geology for ExoMars 2020 visible to near infrared spectral variability ST/T001747/1
- UK Space Agency Mars Exploration Science Standard Call 2023: Exploring the Limits of Material Discrimination with CaSSIS Multiband Imaging ST/Y005910/1