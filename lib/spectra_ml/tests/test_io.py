"""
Test spectra_ml.io module.
"""
# --- Imports

# Standard library
import os
import unittest

# External packages
import pandas as pd

# SpectraML
from spectra_ml import io


# --- spectrometers Module Tests

class io_tests(unittest.TestCase):  # pylint: disable=invalid-name
    """
    Unit tests for spectra_ml.io module.
    """
    # --- Test cases

    @staticmethod
    def test_load_spectrometers_1():
        """
        Test test_load_spectrometers(). Normal usage.
        """
        # --- Preparations

        data_dir = os.environ['DATA_DIR']
        spectrometers_dir = os.path.join(data_dir, 'spectrometers')
        splib07a_dir = os.path.join(data_dir, 'ASCIIdata_splib07a')

        # --- Exercise functionality and check results

        spectrometers = io.load_spectrometers(spectrometers_dir,
                                              splib07a_dir)

        for spectrometer in spectrometers.values():
            assert 'description' in spectrometer
            assert 'abscissas' in spectrometer

            abscissas = spectrometer['abscissas']
            assert 'wavelengths' in abscissas

            for abscissa in abscissas.values():
                assert 'unit' in abscissa
                assert 'min' in abscissa
                assert 'max' in abscissa
                assert 'values' in abscissa
                assert isinstance(abscissa['values'], pd.DataFrame)

                if 'bandpass_values' in abscissa:
                    assert isinstance(abscissa['bandpass_values'],
                                      pd.DataFrame)
