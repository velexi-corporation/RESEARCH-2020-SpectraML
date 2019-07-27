#!/usr/bin/env python
# coding: utf-8

# ## 2019-07-26: Testing standarize-specta
# 
# ### Authors
# * Kevin Chu (kevin@velexi.com)
# 
# ### Notes
# * In this Jupyter notebook, we check that standardize spectra is correctly filling in missing values for spectrum id=2714
# 
#   - ChapterM_Minerals/splib07a_Chlorite_SMR-13.c_45-60um_BECKa_AREF.txt
# 

# ## Preparations

# In[1]:


# --- Imports

# Standard libraries
import os

# External packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SpectraML
from spectra_ml import data
from spectra_ml import io


# In[2]:


# --- Configuration Parameters

# Data directories
data_dir = os.environ['DATA_DIR']
splib07a_dir = os.path.join(data_dir, 'ASCIIdata_splib07a')
splib07a_standardized_dir = os.path.join(data_dir, 'ASCIIdata_splib07a.standardized')
spectrometers_dir = os.path.join(data_dir, 'spectrometers')

# Spectrometers
spectrometers = io.load_spectrometers(spectrometers_dir, splib07a_dir)

# Test Spectrum
raw_spectrum_path = os.path.join(splib07a_dir, 'ChapterM_Minerals', 'splib07a_Chlorite_SMR-13.c_45-60um_BECKa_AREF.txt')
standardized_spectrum_path = os.path.join(splib07a_standardized_dir, '2714.csv')


# ## Read and Plot Spectra Data

# In[3]:


# --- Read spectra data

raw_spectrum = pd.read_csv(raw_spectrum_path)
spectrometer_wavelengths = spectrometers['BECK']['x-axis']['wavelength']['values']

standardized_spectrum = pd.read_csv(standardized_spectrum_path)
standardized_wavelengths = standardized_spectrum.wavelength

# --- Plot raw and filled-in spectra

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(spectrometer_wavelengths, raw_spectrum)
plt.show()
    
plt.subplot(2, 1, 2)
plt.plot(standardized_wavelengths, standardized_spectrum.reflectance)
    
plt.show()

