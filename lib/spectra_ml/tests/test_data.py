"""
Test spectra_ml.data module.
"""
# --- Imports

# Standard library
import os
import unittest

# External packages
import numpy as np
import pandas as pd

# SpectraML
from spectra_ml import data
from spectra_ml import io


# --- spectrometers Module Tests

class data_tests(unittest.TestCase):  # pylint: disable=invalid-name
    """
    Unit tests for spectra_ml.data module.
    """
    # --- Test preparation and clean up

    def setUp(self):  # pylint: disable=invalid-name
        """
        Perform preparations required by most tests.

        - Get spectrum for tests
        """
        # Set data paths
        data_dir = os.environ['DATA_DIR']
        spectrometers_dir = os.path.join(data_dir, 'spectrometers')
        splib07a_dir = os.path.join(data_dir, 'ASCIIdata_splib07a')

        # Get spectrometer for tests
        spectrometers = io.load_spectrometers(spectrometers_dir, splib07a_dir)
        spectrometer = spectrometers['ASDFR']

        # Get spectrum for tests
        spectrum_path = os.path.join(
            splib07a_dir, 'ChapterM_Minerals',
            'splib07a_Stilbite_HS482.1B_Zeolite_ASDFRb_AREF.txt')

        self.spectrum, _ = io.load_spectrum(spectrum_path, spectrometer)

    # --- Test cases

    def test_resample_spectrum_1(self):
        """
        Test resample_spectrum(). Normal usage.
        """
        # --- Preparations

        # Set wavelengths to resample at
        num_wavelengths = 100
        wavelength_min = min(self.spectrum.index)
        wavelength_max = max(self.spectrum.index)
        wavelengths = pd.Series(np.linspace(wavelength_min, wavelength_max,
                                            num_wavelengths))

        # --- Exercise functionality and check results

        resampled_spectrum = \
            data.resample_spectrum(self.spectrum, wavelengths)

        # Check data structure
        assert isinstance(resampled_spectrum, pd.DataFrame)
        assert resampled_spectrum.index.name == 'wavelength'
        assert (resampled_spectrum.index == wavelengths).all()
        assert len(resampled_spectrum) == len(wavelengths)

        # Check data values
        assert (resampled_spectrum['reflectance'] >= 0).all()

    def test_resample_spectrum_2(self):
        """
        Test resample_spectrum(). Check resampled values.
        """
        # --- Preparations

        # Create test spectrum to resample
        # TODO: 5 wavelengths

        # Set wavelengths to resample at
        # TODO: 10 wavelengths

        # --- Exercise functionality and check results

        # TODO: fix as needed
        resampled_spectrum = \
            data.resample_spectrum(self.spectrum, wavelengths)

        # Check values of resampled spectrum
        # TODO
