#!/usr/bin/env python
# coding: utf-8

# In[1]:


# environment set up
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf # only use tensorflow keras
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd

# working folder
# directory = "/Users/Srikar/Desktop/Velexi/spectra-ml/data"
directory = os.environ['DATA_DIR']
os.chdir(directory)

# print(os.getcwd())


# In[4]:


stddata_path = os.path.join(directory,"Srikar-Standardized")
metadata = pd.read_csv(os.path.join(stddata_path,"spectra-metadata.csv"), sep="|")
metadata.head()


# In[5]:


metadata.iloc[327, 1]


# In[6]:


# metadata.columns
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
# ~metadata['spectrometer_purity_code'].str.contains("NIC4")
metadata.shape


# In[7]:


x = 0
for i in range(metadata.shape[0]):
    if metadata.iloc[i, 1] == 'reflectance':
        spectrum = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(metadata.iloc[i, 0])))
        for j in range(spectrum.shape[0]):
            if np.isnan(spectrum.iloc[j, 1]):
                print(str(metadata.iloc[i, 3]) + " " + str(metadata.iloc[i, 0]) + " " + str(metadata.iloc[i, 2]))
                x += 1
                break

print(x)


# In[8]:


num_nic = 0
for i in range(metadata.shape[0]):
    if metadata.iloc[i, 3].find("NIC") != -1:
        num_nic += 1

print(num_nic)


# In[ ]:




