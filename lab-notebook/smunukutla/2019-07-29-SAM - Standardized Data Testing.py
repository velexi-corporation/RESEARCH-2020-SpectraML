#!/usr/bin/env python
# coding: utf-8

# In[141]:


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
from spectra_ml.io_ import load_spectra_metadata


# In[142]:


spectrum_len = 500 # automate this
parent_dir = os.environ['PWD']
stddata_path = os.path.join(os.environ['DATA_DIR'], "StdData-" + str(spectrum_len))
os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[143]:


metadata = load_spectra_metadata(os.path.join(stddata_path,"spectra-metadata.csv"))

metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")]

metadata.head()


# In[144]:


nan_records = set()
x = 0
for i in range(metadata.shape[0]):
    spectrum = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(str(metadata.iloc[i, 0]))))
    for j in range(spectrum.shape[0]):
        if np.isnan(spectrum.iloc[j, 1]):
            print(str(metadata.iloc[i, 3]) + " " + str(metadata.iloc[i, 0]) + " " + str(metadata.iloc[i, 2]))
            nan_records.add(metadata.iloc[i, 0])
            x += 1
            break

print(x)


# In[146]:


nan_records


# In[149]:


# for row in range(metadata.iloc[:, 0].shape[0]):
#     print(metadata.index[row])
#     print(metadata.iat[row, 0])


# In[150]:


def removenans(metadata, nan_set):
#     cols = ['index', 'notnan']
    indices = []
    ret = []
    for row in range(metadata.iloc[:, 0].shape[0]):
        indices.append(metadata.index[row])
        if metadata.iat[row, 0] in nan_set:
            ret.append(False)
        else:
            ret.append(True)
    ret = pd.Series(ret, index=indices)
    return ret


# In[153]:


# for y in metadata:
#     print(y)


# In[154]:


removenans(metadata, nan_records)


# In[155]:


metadata = metadata[removenans(metadata, nan_records)]


# In[156]:


metadata


# In[157]:


num_nic = 0
for i in range(metadata.shape[0]):
    if metadata.iloc[i, 3].find("NIC") != -1:
        num_nic += 1

print(num_nic)


# In[ ]:




