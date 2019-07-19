SpectraML Project
=================

___Authors___  
Kevin T. Chu `<kevin@velexi.com>`
Bonita Song
Srikar Munukutla

------------------------------------------------------------------------------

Table of Contents
-----------------

1. [Overview][#1]

   1.1. [Software Dependencies][#1.1]

   1.2. [Directory Structure][#1.2]

   1.3. [Template Files][#1.3]

2. [Setting Up][#2]

   2.1. [Python Environment][#2.1]

   2.2: [Preparing Spectra Data][#2.2]

3. [References][#3]

------------------------------------------------------------------------------

## 1. Overview

The SpectraML project team researches applications of machine learning to
the analysis of spectroscopic data. We are currently focused on the following
core areas:

* feature engineering (e.g., preprocessing algorithms for spectra);

* machine learning algorithms (e.g., artificial neural networks, CNNs); and

* performance evaluation framework (e.g., bootstrap, k-fold cross-validation).

As a model problem, we are developing a machine learning system for
classifying reflectance spectra from the USGS Spectral Library Version 7
dataset.

### 1.1 Software Dependencies

#### Base Requirements

* Python

#### Required Python Packages ####

See `requirements.txt` for list of Python packages required for this project.

#### Recommended Python Packages ####

* `autoenv`
* `virtualenv`
* `virtualenvwrapper`

### 1.2 Directory Structure

    README.markdown
    requirements.txt
    bin/
    config/
    data/
    docs/
    lab-notebook/
    lib/
    reports/

* `README.markdown`: this file

* `requirements.txt`: `pip` requirements file containing Python packages for
  data science, testing, and assessing code quality

* `bin`: directory containing utility programs

* `config`: directory containing template configuration files (e.g., `autoenv`
  configuration file)

* `data`: directory where project datasets should be placed. __Note__: in
  general, datasets should not be committed to the git repository. Instead,
  datasets should be placed into this directory (either manually or using
  automation scripts) and referenced by Jupyter notebooks. See
  [Section 2][#2.2] for details.

* `docs`: directory containing project documentation and notes

* `lab-notebook`: directory containing Jupyter notebooks used for
  experimentation and development. Jupyter notebooks saved in this directory
  should (1) have a single author and (2) be dated.

* `lib`: directory containing source code developed to support project

* `reports`: directory containing Jupyter notebooks that present and record
  final results. Jupyter notebooks saved in this directory should be polished,
  contain final analysis results, and be the work product of the entire data
  science team.

### 1.3. Template Files

Template files and directories are indicated by the 'template' suffix. These
files and directories are intended to simplify the set up of the lab notebook.
When appropriate, they should be renamed (with the 'template' suffix removed).

------------------------------------------------------------------------------

## 2. Setting Up

### 2.1. Python Environment

* Create Python virtual environment for project.

    ```bash
    $ mkvirtualenv -p /PATH/TO/PYTHON PROJECT_NAME
    ```

* Install required Python packages.

    ```bash
    $ pip install -r requirements.txt
    ```

* Set up autoenv.

  - Copy `config/env.template` to `.env` in project root directory.

  - Set template variables in `.env` (indicated by `{{ }}` notation).

### 2.2. Preparing Spectra Data

A zip file containing the full USGS Spectra Library (Version 7) is included
in the `data` directory. To prepare the spectra data for use in Jupyter
notebooks, use following instructions.

* Extract the data files in `ASCIIdata_splib07a.zip`.

  ```bash
  $ cd data
  $ unzip ASCIIdata_splib07a.zip
  ```

* Generate standardized version of spectra by using the `standardize-spectra`
  script. `standardize-spectra` carries out the following operations:

  - fills in missing data points with interpolated values;

  - resamples spectra so that they all have the same abscissa values;

  - saves spectra to CSV files containing wavelength and reflectance values;

  - generate the `spectra-metadata.csv` database containing metadata for each
    spectrum; and

  - names each spectrum file using the unique ID (in `spectra-metadata.csv`)
    associated with the spectrum.

  ___Usage___

  The following provide several examples of how to use `standardize-spectra`.
  __Note__: if the `standardize-spectra` command cannot be found, check that
  `bin` is on your path.

  - Show help message.

    ```bash
    $ standardize-spectra --help
    ```

  - Basic usage uses default output directory and wavelength values.

    ```bash
    $ cd data
    $ standardize-spectra ASCIIdata_splib07a spectrometers
    ```

  - Set custom output directory by using the `-o OUTPUT_DIR` option.

    ```bash
    $ cd data
    $ standardize-spectra ASCIIdata_splib07a spectrometers -o custom-location
    ```

  - Set number of wavelengths in spectra directory by using the
    `--num-wavelengths NUM_WAVELENGTHS` option.

    ```bash
    $ cd data
    $ standardize-spectra ASCIIdata_splib07a spectrometers \
      --num-wavelengths 2000
    ```

* Use lists of spectra IDs to define collections of spectra. Within Jupyter
  notebook, use the following directory paths to facilitate access to spectra
  files.

  ```python
  # Data directories
  data_dir = os.environ['DATA_DIR']
  spectra_data_dir = os.path.join(data_dir, 'ASCIIdata_splib07a')

  # Path to data file for spectra with ID=12345
  spectrum_path = os.path.join(spectra_data_dir, '12345.csv')
  ```

------------------------------------------------------------------------------

## 3. References

* J. Whitmore.
  ["Jupyter Notebook Best Practices for Data Science"][#whitmore-2016]
  (2016/09).

------------------------------------------------------------------------------

[-----------------------------INTERNAL LINKS-----------------------------]: #

[#1]: #1-overview
[#1.1]: #11-software-dependencies
[#1.2]: #12-directory-structure
[#1.3]: #13-template-files

[#2]: #2-setting-up
[#2.1]: #21-python-environment
[#2.2]: #22-preparing-spectra-data

[#3]: #3-references

[-----------------------------EXTERNAL LINKS-----------------------------]: #

[#whitmore-2016]:
  https://www.svds.com/tbt-jupyter-notebook-best-practices-data-science/
