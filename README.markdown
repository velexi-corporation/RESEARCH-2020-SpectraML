SpectraML Project
=================

___Authors___  
Bonita Song  
Kevin T. Chu `<kevin@velexi.com>`

------------------------------------------------------------------------------

Table of Contents
-----------------

1. [Overview][#1]

   1.1. [Software Dependencies][#1.1]

   1.2. [Directory Structure][#1.2]

   1.3. [Template Files][#1.3]

2. [Setting Up][#2]

   2.1. [Python Environment][#2.1]

   2.2: [Spectra Data][#2.2]

3. [References][#3]

------------------------------------------------------------------------------

## 1. Overview

TODO

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
    config/
    data/
    docs/
    lab-notebook/
    lib/
    reports/

* `README.markdown`: this file

* `requirements.txt`: `pip` requirements file containing Python packages for
  data science, testing, and assessing code quality.

* `bin`: directory containing utility programs

* `config`: directory containing template configuration files (e.g., `autoenv`
  configuration file)

* `data`: directory where project data should be placed. TODO

* `docs`: directory containing project documentation and notes.

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

## 2.2. Spectra Data

* TODO: add instructions on how to generate standardized spectra data

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
[#2.2]: #22-spectra-data

[#3]: #3-references

[-----------------------------EXTERNAL LINKS-----------------------------]: #

[#whitmore-2016]:
  https://www.svds.com/tbt-jupyter-notebook-best-practices-data-science/
