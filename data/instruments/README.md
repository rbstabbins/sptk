Directory for hosting 'instrument' data.

Instrument data files are csv, saved as [instrument name].csv.

Instrument files can be of the format of [ID/name], [centre-wavelength (nm)], [full-width at half-maximum (nm)] e.g.:
|filter_id|cwl|fwhm|
|-------|-------|-------|
|F01|440|25|
|...|...|...|

or [wavelngth (nm)], [ID/Name Transmission], [...] e.g.:
|wvls|F01|F02|...|
|-------|-------|-------|-------|
|380|0.0|0.0|...|
|400|0.1|0.0|...|
|420|0.15|0.1|...|
|440|0.6|0.15|...|
|460|0.15|0.6|...|
|480|0.10|0.15|...|
|...|...|...|...|

An example instrument file can be found at: [doi:10.5281/zenodo.10683367](https://zenodo.org/doi/10.5281/zenodo.10683367).