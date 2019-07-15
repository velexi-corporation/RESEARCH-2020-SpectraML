"""
Test spectra_ml.io module.
"""
# --- Imports

# Standard library
import os
import unittest

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

            wavelengths = abscissas['wavelengths']
            assert 'unit' in wavelengths
            assert 'min' in wavelengths
            assert 'max' in wavelengths
            assert 'values' in wavelengths
