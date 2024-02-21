# Example Notebooks

This directory hosts example notebooks describing and demonstrating the key processes that the Spectral Parameters Toolkit (sptk) performs.

[MaterialCollection Example Notebook:](./material_collection_example.ipynb) placing a spectral library in the data directory, and processing this into a material collection object.

[Instrument Example Notebook:](./instrument_example.ipynb) placing instrument information in the data directory, and processing this into an instrument object.

[Observation Example Notebook:](./observation_example.ipynb) sampling a material collection with an instrument, and printing key results, including plots and RMSE metrics.

[Spectral Parameters Example Notebook:](./spectral_parameters_example.ipynb) computing the spectral parameters afforded by an instrument.

[Spectral Parameters Combination Classifier Example Notebook:](./spectral_parameters_combination_classifier_example.ipynb) performing LDA and ranking the spectral parameter combinations afforded by the instrument on the dataset.

## Example Dataset

The notebooks involve the construction of an instrument file, and guides the user through the process of downloading and creating an example spectral library (via [ViSOR](https://westernreflectancelab.com/visor/)). 

For completeness and verification, this example dataset can be downloaded from Zenodo ([doi:10.5281/zenodo.10683367]((https://zenodo.org/doi/10.5281/zenodo.10683367))). Once downloaded, the example instrument file and example spectral library must be placed in the ```sptk/data/instruments/``` and ```sptk/data/spectral_library/``` directories, respectively, as instructed on the repository homepage.