"""
Test spectra_ml.io module.
"""
# --- Imports

# Standard library
import os
import unittest

# External packages
import pandas as pd
import pytest

# SpectraML
from spectra_ml import io


# --- spectrometers Module Tests

class io_tests(unittest.TestCase):  # pylint: disable=invalid-name
    """
    Unit tests for spectra_ml.io module.
    """
    # --- Test cases

    @staticmethod
    def test_load_spectrum_1():
        """
        Test load_spectrum(). Normal usage.
        """
        # --- Preparations

        # Data directories
        data_dir = os.environ['DATA_DIR']
        spectrometers_dir = os.path.join(data_dir, 'spectrometers')
        splib07a_dir = os.path.join(data_dir, 'ASCIIdata_splib07a')

        # Path to spectrum file to use for tests
        spectrum_path = os.path.join(
            splib07a_dir, 'ChapterM_Minerals',
            'splib07a_Stilbite_HS482.1B_Zeolite_ASDFRb_AREF.txt')

        # Get spectrometer for spectrum
        spectrometers = io.load_spectrometers(spectrometers_dir,
                                              splib07a_dir)
        spectrometer = spectrometers['ASDFR']

        # --- Exercise functionality and check results

        spectrum, spectrum_metadata = io.load_spectrum(spectrum_path,
                                                       spectrometer)

        # Check spectrum
        assert spectrum.index.name == 'wavelength'
        assert len(spectrum.columns) == 1
        assert 'reflectance' in spectrum

        # Check spectrum_metadata
        assert 'id' in spectrum_metadata
        assert 'material' in spectrum_metadata
        assert 'spectrometer_purity_code' in spectrum_metadata
        assert 'measurement_type' in spectrum_metadata

    @staticmethod
    def test_load_spectrum_2():
        """
        Test load_spectrum(). Mismatch between spectrum and spectrometer.
        """
        # --- Preparations

        # Data directories
        data_dir = os.environ['DATA_DIR']
        spectrometers_dir = os.path.join(data_dir, 'spectrometers')
        splib07a_dir = os.path.join(data_dir, 'ASCIIdata_splib07a')

        # Path to spectrum file to use for tests
        spectrum_path = os.path.join(
            splib07a_dir, 'ChapterM_Minerals',
            'splib07a_Stilbite_HS482.1B_Zeolite_ASDFRb_AREF.txt')

        # Get spectrometer for spectrum
        spectrometers = io.load_spectrometers(spectrometers_dir,
                                              splib07a_dir)
        spectrometer = spectrometers['BECK']

        # --- Exercise functionality and check results

        with pytest.raises(RuntimeError) as exception_info:
            io.load_spectrum(spectrum_path, spectrometer)

        error_message = \
            str(exception_info._excinfo[1])  # pylint: disable=protected-access
        assert 'Mismatch between spectrum length and ' \
               'spectrometer wavelengths' in error_message

    @staticmethod
    def test_load_spectrometers_1():
        """
        Test load_spectrometers(). Normal usage.
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
            assert 'x-axis' in spectrometer

            x_axis_options = spectrometer['x-axis']
            assert 'wavelength' in x_axis_options

            for axis_type, axis in x_axis_options.items():
                assert 'unit' in axis
                assert 'min' in axis
                assert 'max' in axis
                assert 'values' in axis
                assert isinstance(axis['values'], pd.DataFrame)
                assert axis['values'].columns == [axis_type]

                if 'bandpass_values' in axis:
                    assert isinstance(axis['bandpass_values'],
                                      pd.DataFrame)
                    assert axis['bandpass_values'].columns == ['bandpass']
