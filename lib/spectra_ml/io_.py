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

def load_spectrum(spectrum_path, spectrometer):
    """
    Load spectrum data.

    Parameters
    ----------
    spectrum_path: str
        path to spectrum data file

    spectrometer: dict
        metadata and abscissa values for spectrometer

    Return values
    -------------
    spectrum : DataFrame
        DataFrame containing spectrum data indexed by wavelength

    spectrum_metadata: dict
        metadata for spectrum
    """
    # --- Check arguments

    if not os.path.isfile(spectrum_path):
        error = "'spectrum_path' (='{}') is not a valid file" \
            .format(spectrum_path)
        raise ValueError(error)

    # --- Load spectrum data

    # Load spectrum
    spectrum = pd.read_csv(spectrum_path)

    # Add wavelengths to DataFrame
    wavelengths = spectrometer['abscissas']['wavelengths']['values']
    if len(wavelengths) != len(spectrum):
        raise RuntimeError("Mismatch between spectrum length and "
                           "spectrometer wavelengths")

    spectrum['wavelength'] = spectrometer['abscissas']['wavelengths']['values']

    # Extract metadata string
    metadata_str = spectrum.columns[0]

    # --- Fill in missing values

    # Set index to 'wavelength' column
    spectrum.set_index('wavelength', inplace=True)

    # Rename columns
    column_names = {spectrum.columns[0]: 'reflectance'}
    spectrum.rename(columns=column_names, inplace=True)

    # Interpolate to fill in missing values
    spectrum.interpolate(method='values', inplace=True)

    # --- Parse spectrum metadata

    metadata_parts = metadata_str.split()
    record_id = metadata_parts[1].split('=')[-1].strip(':').strip()
    material = ' '.join(metadata_parts[2:-2])
    spectrometer_purity_code = metadata_parts[-2]
    measurement_type = metadata_parts[-1]

    # TODO: split spectrometer_code into spectrometer code and purity code
    spectrum_metadata = {
        'id': record_id,
        'material': material,
        'spectrometer_purity_code': spectrometer_purity_code,
        'measurement_type': measurement_type,
        }

    # --- Return spectrum and metadata

    return spectrum, spectrum_metadata


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
