#!/usr/bin/env python
# coding: utf-8

# ## 2019-08-09: Testing tensorflow.keras
# 
# ### Authors
# * Kevin Chu (kevin@velexi.com)
# 
# ### Notes
# * In this Jupyter notebook, we test that tensorflow.keras is compatible with the keras functionality that we have been using for our ANN experiments.

# In[1]:


# --- Imports

# Standard library
import os
import random

# External packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Tensorflow & Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# In[2]:


# --- Configuration Parameters

# Data directories
data_dir = os.environ['DATA_DIR']
splib07a_dir = os.path.join(data_dir, 'ASCIIdata_splib07a')
splib07a_standardized_dir = os.path.join(data_dir, 'ASCIIdata_splib07a.standardized')
spectrometers_dir = os.path.join(data_dir, 'spectrometers')

# Spectra parameters
spectrum_len = 1000


# In[3]:


# --- Initialization

# Random seed
random.seed(0)


# In[4]:


# --- Prepare spectra data

# Load spectra metadata
metadata_path = os.path.join(splib07a_standardized_dir,"spectra-metadata.csv")
metadata = pd.read_csv(metadata_path, sep="|")
metadata.head()

# Remove NIC4 spectra
metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]


# In[5]:


# --- Prepare samples

record_nums = []
y = []
spectrum_names = []

num_act = 0
num_aln = 0
num_chl = 0

for i in range(metadata.shape[0]):
    record = metadata.iloc[i, :]
    if record[2].find("Actinolite") != -1: # if material name contains actinolite
        record_nums.append(record[0])
        y.append(int(0))
        spectrum_names.append("Actinolite")
        num_act += 1
    elif record[2].find("Alun") != -1:
        record_nums.append(record[0])
        y.append(int(1))
        spectrum_names.append("Alunite")
        num_aln += 1
    elif (record[2].find("Chlorit") != -1 or record[2].find("Chlor.") != -1 or record[2].find("Chlor+") != -1 or record[2].find("Chl.") != -1):
        record_nums.append(record[0])
        y.append(int(2))
        spectrum_names.append("Chlorite")
        num_chl += 1

y = np.reshape(y, (len(y), 1))
num_samples = len(record_nums)

print("Number of samples: ", num_samples)
print("Number of Actinolite samples: ", num_act)
print("Number of Alunite samples: ", num_aln)
print("Number of Chlorite samples: ", num_chl)


# In[6]:


# --- Load spectra

# Initialize variables
spectra = np.zeros((num_samples, spectrum_len))
wavelengths = np.zeros((1,spectrum_len))

for i in range(num_samples):
    spectra_path = os.path.join(splib07a_standardized_dir,"{}.csv".format(record_nums[i]))
    data = pd.read_csv(spectra_path)
    if i == 0:
        wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()

# Verify that all spectra values are non-negative
assert (spectra >= 0).all()


# In[7]:


# --- Plot spectra by material

# plot each class in a separate plot
# plot spectra names in legend
# plot minerals and mixtures w diff line widths

mineral_names = ["Actinolite", "Alunite", "Chlorite"]

# Number of samples in each class
num0 = 0 
num1 = 0
num2 = 0

# Plot parameters
linewidth = 2

# count the number of each class to make spectra0, spectra1, spectra2 databases
for i in range(num_samples):
    if y[i,0]== 0:
        num0 += 1
    elif y[i,0]== 1:
        num1 += 1
    elif y[i,0]== 2:
        num2 += 1

# make class-specific databases spectra0, ...1, ...2
spectra0 = np.zeros((num0, spectrum_len)) 
spectra1 = np.zeros((num1, spectrum_len)) 
spectra2 = np.zeros((num2, spectrum_len)) 

labels0 = ["" for x in range(num0)]
labels1 = ["" for x in range(num1)]
labels2 = ["" for x in range(num2)]

linewidth0 = np.zeros(num0)
linewidth1 = np.zeros(num1)
linewidth2 = np.zeros(num2)

# Initialize counter variables
i0 = 0
i1 = 0
i2 = 0

# Populate class-specific databases spectra0, ...1, ...2
for i in range(num_samples):
    
    # populate matrices for making each class plot
    if y[i,0]== 0:
        spectra0[i0,:] = spectra[i,:]
        labels0[i0] = spectrum_names[i]
        linewidth0[i0] = linewidth
        i0 +=1
    elif y[i,0]== 1:
        spectra1[i1,:] = spectra[i,:]
        labels1[i1] = spectrum_names[i]
        linewidth1[i1] = linewidth
        i1 +=1
    else:
        spectra2[i2,:] = spectra[i,:]
        labels2[i2] = spectrum_names[i]
        linewidth2[i2] = linewidth
        i2 +=1

# plot each class-specific database separately
for i in range(num0):
    plt.plot(wavelengths[0,:], spectra0[i,:]) # remove linewidth for all mixtures/minerals to be standard
plt.show()

for i in range(num1):
    plt.plot(wavelengths[0,:], spectra1[i,:])
plt.show()

for i in range(num2):
    plt.plot(wavelengths[0,:], spectra2[i,:])
plt.show()


# In[8]:


# --- Prepare ML datasets

# Construct train, dev, and test datasets
sample_indices = list(range(0, num_samples))
print(num_samples)
random.shuffle(sample_indices)
train_set_size = 3*(num_samples//5)
dev_set_size = (num_samples//5)
test_set_size = num_samples-dev_set_size - train_set_size

train_set_indices = sample_indices[:train_set_size]
dev_set_indices = sample_indices[train_set_size: train_set_size+dev_set_size]
test_set_indices= sample_indices[train_set_size+dev_set_size: num_samples]

# Training dataset
train_set = spectra[train_set_indices, :]

train_labels = y[train_set_indices, :].flatten()
train_labels = to_categorical(train_labels)

# Dev dataset
dev_set = spectra[dev_set_indices, :]

dev_labels = y[dev_set_indices, :].flatten()
dev_labels = to_categorical(dev_labels)

# Test dataset
test_set = spectra[test_set_indices, :]

test_labels = y[test_set_indices, :].flatten()
test_labels = to_categorical(test_labels)


# In[9]:


print(train_set.shape)
model = Sequential()
model.add(Dense(32, input_shape=(train_set.shape[1],)))
model.add(Dense(3, activation='softmax'))
print(model.summary())


# In[10]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 12
EPOCHS = 50

model.fit(train_set, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(dev_set, dev_labels)) 


# In[11]:


y_pred = model.predict(test_set)
y_pred


# In[12]:


model.evaluate(test_set, test_labels)

