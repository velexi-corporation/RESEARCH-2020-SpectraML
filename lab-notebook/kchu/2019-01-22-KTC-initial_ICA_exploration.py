#!/usr/bin/env python
# coding: utf-8

# ## 2019-01-22: Initial ICA Exploration
# 
# ### Authors
# * Kevin Chu (kevin@velexi.com)
# 
# ### Notes
# 
# * In theory, ICA should "learn" spectra features that are common across multiple materials. Projection of a new spectra onto these "component spectra" could be used to construct the input vector to a supervised learning system.
# 
# * The independent components can be computed from the mixing matrix (FastICA.mixing_). Note that FastICA automatically "whitens" the training dataset, so the mean spectra (FastICA.mean_) needs to be added to each column of the mixing matrix.
# 
# * To compute the representation (i.e., coefficients) of a spectra with respect to the independent components, multiply the spectra by the unmixing matrix (FastICA.components_).

# ## Preparations

# In[1]:


# --- Imports

# Standard libraries
import os
import re

# External packages
import matplotlib.pyplot as plt
import numpy
import pandas

from sklearn.decomposition import FastICA                  


# In[2]:


# --- Configuration Parameters

# Data directory
data_dir = os.environ['DATA_DIR']

# Materials
materials = {
    'actinolite': 0,
    'alunite': 1,
    'calcite': 2,
}


# ### Data Preparation

# In[3]:


# --- Load data from files

# Get file list
data_files = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)
              if not file_name.startswith('.') and
              os.path.isfile(os.path.join(data_dir, file_name))]

# Initialize spectra dataset
spectra_data = pandas.DataFrame()

# Initialize material labels
class_labels = []

# Load data files
for file_name in data_files:
    # Read data into DataFrame
    raw_data = pandas.read_csv(file_name)
    
    # Clean up header
    spectra_id = raw_data.columns[0].strip()
    raw_data.columns = [spectra_id]

    # Replace missing values (set to -1.23e+34 in raw data) with 0
    raw_data[spectra_id][raw_data[spectra_id] < 0] = 0.0

    # Append spectra
    spectra_data[spectra_id] = raw_data[spectra_id]

    # Assign class label
    for material, label in materials.items():
        if re.search(material, spectra_id, re.IGNORECASE):
            class_labels.append(label)
            break

# Calculate dataset parameters
spectrum_length, num_spectra = spectra_data.shape

# Convert labels to numpy array
class_labels = numpy.array(class_labels)
class_labels.resize([num_spectra, 1])


# ## Data Exploration

# In[4]:


# --- Plot spectra by material

for material_name, material_id in materials.items():
    # Get indices for spectra for material
    spectra_indices = numpy.argwhere(class_labels==material_id)[:, 0]

    # Plot spectra in their own figure
    plt.figure()
    plt.title(material_name)
    plt.plot(spectra_data.iloc[:, spectra_indices])


# ## ICA Exploration

# In[5]:


# --- Generate ICA model

# ICA Parameters
num_components = 5

# Create FastICA object
ica = FastICA(n_components=num_components)

# Fit ICA model
X = spectra_data.values.T
S = ica.fit_transform(X)

# Compute independent spectra components
# Note: mixing (not unmixing) matrix holds independent components
mean_spectra = ica.mean_.reshape([spectrum_length, 1])
spectra_components = ica.mixing_ + numpy.tile(mean_spectra, [1, num_components])

# Display results
print("Number of components generated:", num_components)
print("Number of fitting iterations:", ica.n_iter_)

# Display independent spectra components
for i in range(spectra_components.shape[1]):
    plt.title('Component {}'.format(i))
    plt.plot(spectra_components[:, i])
    plt.figure()


# In[6]:


# --- Compute representation for spectra

# Get unmixing matrix
unmixing_matrix = ica.components_

# spectra_data[0]
print("Coefficients from fit_transform():", S[0, :])
coefficients = numpy.dot(unmixing_matrix,
                         X[0, :].reshape([spectrum_length, 1]) - mean_spectra).T
print("Coefficients from multiplying by unmixing matrix:", coefficients)
      

