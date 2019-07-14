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

def load_spectrometers(metadata_dir, splib07a_dir):
    """
    Load all spectrometer data contained in specified directory.

    Parameters
    ----------
    metadata_dir: str
        path to directory containing spectrometer metadata

    splib07a_dir: str
        path to top-level of splib07a directory

    Return value
    ------------
    spectrometers: dict
        metadata and abscissa values for spectrometers
    """
    # --- Check arguments

    if not os.path.isdir(metadata_dir):
        error = "'metadata_dir' (='{}') is not a valid directory" \
            .format(metadata_dir)
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
        [os.path.join(metadata_dir, filename)
         for filename in os.listdir(metadata_dir)
         if not filename.startswith('.') and
         os.path.isfile(os.path.join(metadata_dir, filename))]

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
                os.path.join(splib07a_dir, abscissa['abscissas_file'])) \
                .values.flatten()

            # Load bandpass values
            if 'bandpass' in abscissa:
                abscissa['bandpass_values'] = pd.read_csv(
                    os.path.join(splib07a_dir, abscissa['bandpass_file'])) \
                    .values.flatten()

    # --- Return results

    return spectrometers
