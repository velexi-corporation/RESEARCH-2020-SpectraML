"""
spectra_ml.spectrometers module provides support for managing spectrometer
data.
"""

# --- Imports

# Standard library
import os

# External packages
import pandas as pd
import yaml

# --- Constants


# --- Public Functions

def load_spectra(spectra_path, spectrometer):
    """
    Load spectra data.

    Parameters
    ----------
    spectra_path: str
        path to spectra data file

    spectrometer: dict
        metadata and abscissa values for spectrometer

    Return value
    ------------
    spectra: pandas.DataFrame
        DataFrame containing spectra data indexed by wavelength
    """
    # --- Check arguments

    if not os.path.isfile(spectra_path):
        error = "'spectra_path' (='{}') is not a valid file" \
            .format(spectra_path)
        raise ValueError(error)

    # --- Load spectra data

    # TODO

    # --- Clean spectra data

    # TODO

    # --- Return spectra

    return spectra


def load_spectrometers(spectometers_path, splib07a_dir):
    """
    Load all spectrometer data contained in specified directory.

    Parameters
    ----------
    spectrometers_dir: str
        path to directory containing spectrometer metadata

    splib07a_dir: str
        path to top-level of splib07a directory

    Return value
    ------------
    spectrometers: dict
        metadata and abscissa values for spectrometers
    """
    # --- Check arguments

    if not os.path.isdir(spectometers_path):
        error = "'spectometers_path' (='{}') is not a valid directory" \
            .format(spectometers_path)
        raise ValueError(error)

    if not os.path.isdir(splib07a_dir):
        error = "'splib07a_dir' (='{}') is not a valid directory" \
            .format(splib07a_dir)
        raise ValueError(error)

    # --- Preparations

    # Initialize spectrometers
    spectrometers = {}

    # --- Load spectrometer data

    # Load metadata
    spectrometer_file_paths = \
        [os.path.join(spectometers_path, filename)
         for filename in os.listdir(spectometers_path)
         if not filename.startswith('.') and
         os.path.isfile(os.path.join(spectometers_path, filename))]

    for file_path in spectrometer_file_paths:
        name, _ = os.path.splitext(os.path.basename(file_path))

        with open(file_path, 'r') as file_:
            spectrometers[name] = yaml.safe_load(file_)

    # Load abscissas
    for spectrometer, metadata in spectrometers.items():
        # Check for required values
        if 'abscissas' not in metadata:
            error = "'abscissas' missing from '{}' spectrometer" \
                .format(spectrometer)
            raise RuntimeError(error)

        for abscissa in metadata['abscissas'].values():
            # Load values
            abscissa['values'] = pd.read_csv(
                os.path.join(splib07a_dir, abscissa['abscissas_file']))

            # Load bandpass values
            if 'bandpass_file' in abscissa:
                abscissa['bandpass_values'] = pd.read_csv(
                    os.path.join(splib07a_dir, abscissa['bandpass_file']))

    # --- Return results

    return spectrometers
