"""
spectra_ml.data module provides support for managing and manipulating spectra
data.
"""

# --- Imports

# External packages
import numpy as np
import pandas as pd


# --- Constants

# Default parameter values
DEFAULT_NUM_ABSCISSAS = 1000


# --- Public Functions

# abscissa_type=DEFAULT_ABSCISSA_TYPE,
# num_abscissas=DEFAULT_NUM_ABSCISSAS):

# --- Spectra Functions

def resample_spectrum(spectrum, wavelengths):
    """
    Compute (approximate) values of spectrum at specified wavelengths.

    Parameters
    ----------
    spectrum: DataFrame
        spectrum to resample

    wavelengths: DataFrame or Series
        ordered list of wavelengths to sample spectrum at

    Return values
    -------------
    resampled_spectrum: DataFrame
        resampled spectrum
    """
    # --- Check arguments

    if not isinstance(spectrum, pd.DataFrame):
        raise ValueError("'spectrum' should be a DataFrame")

    if spectrum.index.name != 'wavelength':
        raise ValueError("'spectrum.index' should be named 'wavelength'")

    if 'reflectance' not in spectrum.columns:
        raise ValueError("'reflectance' should be a column in 'spectrum'")

    if not isinstance(wavelengths, (pd.DataFrame, pd.Series)):
        raise ValueError("'wavelengths' should be a DataFrame or Series")

    if isinstance(wavelengths, pd.DataFrame):
        if len(wavelengths.columns) != 1:
            raise ValueError("'wavelengths' should contain only one column")

    # --- Preparations

    resampled_spectrum = pd.DataFrame()
    resampled_spectrum['wavelength'] = wavelengths.copy()
    resampled_spectrum['reflectance'] = \
        np.interp(wavelengths, spectrum.index, spectrum.reflectance)
    resampled_spectrum.set_index('wavelength', inplace=True)

    # --- Return results

    return resampled_spectrum
