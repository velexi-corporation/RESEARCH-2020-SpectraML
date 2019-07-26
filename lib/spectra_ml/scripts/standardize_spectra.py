"""
Script functions to support standardizing all of the spectra contained in
a directory.

Notes
-----
* Spectra files are expected to be in USGS SPECPR format.
"""
# --- Imports

# Standard library
from collections import OrderedDict
import enum
import glob
import logging
import os
import re
import time

# External packages
import numpy as np
import pandas as pd
from progress.bar import Bar

# SpectraML
from spectra_ml import data
from spectra_ml import io


# --- Constants

# Default x-axis parameters
DEFAULT_X_LOWER = 0.37  # wavelength in microns
DEFAULT_X_UPPER = 2.5  # wavelength in microns
DEFAULT_NUM_GRID_POINTS = 1000  # number of grid points along x-axis


# --- Exit statuses

@enum.unique
class ExitStatus(enum.IntEnum):
    """
    Error codes.
    """
    SUCCESS = 0
    ERROR_CREATING_OUTPUT_DIRECTORY = 1


# --- Main program

def run(output_dir, raw_data_dir, spectrometers_dir,
        x_lower=DEFAULT_X_LOWER,
        x_upper=DEFAULT_X_UPPER,
        num_grid_points=DEFAULT_NUM_GRID_POINTS):
    """
    Standardize all spectra in specified directory.

    Outputs
    -------
    * 'spectra-metadata.csv': CSV-formatted database of spectra metadata. Each
      row is uniquely identified by a spectrum id (primary key) and contains
      fields such as: material description, spectrometer code, purity code,
      measurement type, and path (relative to raw_data_dir).

    * 'XXXXX.csv': CSV-formatted spectrum data. Each file is named by the id
      of the spectrum (identical to the id in 'spectra-metadata.csv' file).
      Each row contains a wavelength and a reflectance value.

    Parameters
    ----------
    output_dir: str
        path to directory where spectra metadata databse and standardized
        spectra are to be written

    raw_data_dir: str
        path to directory containing (1) raw spectra data in SPECPR-formatted
        files and (2) spectrometer metadata files.

    spectrometers_dir: str
        path to directory containing YAML-formatted spectometer metadata.
        Paths to abscissas files are expected to be relative to the
        'raw_data_dir' directory.

    x_lower: float
        lower end of x-axis

    x_upper: float
        upper end of x-axis

    num_grid_points: int
        number of grid points along x-axis

    Return value
    ------------
    (int) : exit status
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    # --- Check arguments

    if not os.path.isdir(output_dir):
        error = "'output_dir' (='{}') not found".format(output_dir)
        raise ValueError(error)

    if not os.path.isdir(raw_data_dir):
        error = "'raw_data_dir' (='{}') is not found".format(raw_data_dir)
        raise ValueError(error)

    if not os.path.isdir(spectrometers_dir):
        error = "'spectrometers_dir' (='{}') is not found" \
            .format(spectrometers_dir)
        raise ValueError(error)

    if not isinstance(x_lower, (int, float)):
        raise ValueError("'x_lower' should be an int or float")

    if not isinstance(x_upper, (int, float)):
        raise ValueError("'x_upper' should be an int or float")

    if not isinstance(num_grid_points, int):
        raise ValueError("'x_upper' should be an int")

    if x_lower > x_upper:
        raise ValueError("'x_upper' should be greater than or equal to "
                         "'x_lower'")

    if num_grid_points <= 0:
        raise ValueError("'num_grid_points' should be a positive integer")

    # --- Preparations

    # Initialize timing data
    timing_data = OrderedDict()
    timing_data['Load raw spectra'] = 0
    timing_data['Standardize spectra'] = 0
    timing_data['Save standardized spectra'] = 0
    timing_data['Miscellaneous'] = 0

    logging.info("Preparations: STARTED")
    t_start = time.time()

    # Load spectrometers
    spectrometers = io.load_spectrometers(spectrometers_dir, raw_data_dir)
    spectrometer_codes = spectrometers.keys()

    # Generate x-axis grid
    abscissas = pd.Series(np.linspace(x_lower, x_upper, num_grid_points))

    # Initialize spectra metadata database
    spectra_metadata_db = []
    spectra_metadata_db_columns = ['id', 'material',
                                   'spectrometer_purity_code',
                                   'measurement_type',
                                   'raw_data_path']
    spectra_metadata_db_path = os.path.join(output_dir,
                                            'spectra-metadata.csv')

    timing_data['Miscellaneous'] += time.time() - t_start
    logging.info("Preparations: FINISHED")

    # --- Load and standardize spectra

    # ------ Get list of raw spectra files

    # Get list of all directories in raw data directory that contain spectra
    spectra_dirs = [path for path in glob.glob(os.path.join(raw_data_dir, '*'))
                    if os.path.isdir(path)]

    # Get list of all spectra files in raw data directory
    spectra_files = []
    for spectra_dir in spectra_dirs:
        spectra_files.extend(glob.glob(os.path.join(spectra_dir, '*.txt'),
                                       recursive=True))

    # ------ Process spectra

    suffix = '%(index)d/%(max)d (ETA:%(eta)ds)'
    with Bar('Processing spectra', max=len(spectra_files), suffix=suffix) \
            as progress_bar:

        # Initialize spectra metadata
        for path in spectra_files:

            # --- Load raw spectra

            # Identify spectrometer
            t_start = time.time()

            spectrometer = None
            for code in spectrometer_codes:
                if code in path:
                    spectrometer = spectrometers[code]
                    break

            timing_data['Miscellaneous'] += time.time() - t_start

            # --- Load spectrum

            t_start = time.time()

            spectrum, metadata = io.load_spectrum(path, spectrometer)
            metadata['raw_data_path'] = os.path.relpath(path, raw_data_dir)
            spectra_metadata_db.append(metadata)

            timing_data['Load raw spectra'] += time.time() - t_start

            # --- Standardize spectra

            t_start = time.time()

            spectrum_standardized = data.resample_spectrum(spectrum, abscissas)

            timing_data['Standardize spectra'] += time.time() - t_start

            # --- Save standardized spectra

            t_start = time.time()

            # Construct filename for CSV file
            if not re.search('errorbars', path):
                filename = '{}.csv'.format(metadata['id'])
            else:
                filename = '{}-errorbars.csv'.format(metadata['id'])

            spectrum_standardized.to_csv(os.path.join(output_dir, filename))

            timing_data['Save standardized spectra'] += time.time() - t_start

            progress_bar.next()

        # Save spectra metadata database
        spectra_metadata_db = pd.DataFrame(spectra_metadata_db,
                                           columns=spectra_metadata_db_columns)
        spectra_metadata_db.set_index('id', inplace=True)
        spectra_metadata_db.to_csv(spectra_metadata_db_path, sep='|')

    # --- Emit timing data to log

    total_time = sum([time for time in timing_data.values()])
    msg = "Elapsed time: {:.1f}s".format(total_time)
    logging.info(msg)

    msg = "Timing Data"
    for stage in timing_data:
        msg += '\n    {}: {:.2f}s'.format(stage, timing_data[stage])
    msg += "\n"
    logging.info(msg)

    # --- Clean up

    return ExitStatus.SUCCESS
