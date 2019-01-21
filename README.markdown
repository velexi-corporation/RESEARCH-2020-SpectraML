Spectra Analysis Project
========================

___Authors___  
Bonita Song  
Kevin T. Chu `<kevin@velexi.com>`

------------------------------------------------------------------------------

Table of Contents
-----------------

1. [Overview][#1]

   1.1. [Software Dependencies][#1.1]

   1.2. [Directory Structure][#1.2]

2. [Usage][#2]

   2.1. [Setting Up][#2.1]

3. [References][#3]

------------------------------------------------------------------------------

1 Overview
----------

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
    reports/
    src/

* `README.markdown`: this file

* `requirements.txt`: `pip` requirements file containing Python packages for
  data science, testing, and assessing code quality.

* `config`: directory containing template configuration files (e.g., `autoenv`
  configuration file)

* `data`: directory where project data should be placed. TODO

* `docs`: directory containing project documentation and notes.

* `lab-notebook`: directory intended for Jupyter notebooks used for
  experimentation and development. Jupyter notebooks saved in this directory
  should (1) have a single author and (2) be dated.

* `reports`: directory intended for Jupyter notebooks that present and record
  final results. Jupyter notebooks saved in this directory should be polished,
  contain final analysis results, and be the work product of the entire data
  science team.

* `src`: directory intended for source code developed to support project

------------------------------------------------------------------------------

2 Usage
-------

### 2.1 Setting Up

* Create Python virtual environment for project.

    ```bash
    $ mkvirtualenv -p /PATH/TO/PYTHON PROJECT_NAME
    ```

* Install required Python packages.

    ```bash
    $ pip install -r requirements.txt
    ```

* Set up autoenv.

  - Copy `config/env` to `.env` in project root directory.

  - Set template variables in `.env` (indicated by `{{ }}` notation).

------------------------------------------------------------------------------

3 References
------------

* TODO

------------------------------------------------------------------------------

[-----------------------------INTERNAL LINKS-----------------------------]: #

[#1]: #1-overview
[#1.1]: #1-1-software-dependencies
[#1.2]: #1-2-directory-structure

[#2]: #2-usage
[#2.1]: #2-1-setting-up

[#3]: #3-references

[-----------------------------EXTERNAL LINKS-----------------------------]: #
