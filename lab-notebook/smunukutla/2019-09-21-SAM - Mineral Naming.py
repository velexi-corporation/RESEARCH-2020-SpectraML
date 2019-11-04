#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import ast
from spectra_ml.io_ import load_spectra_metadata


# In[2]:


spectrum_len = 500 # automate this
parent_dir = os.environ['PWD']
stddata_path = os.path.join(os.environ['DATA_DIR'], "StdData-" + str(spectrum_len))
os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[3]:


metadata = load_spectra_metadata(os.path.join(stddata_path,"spectra-metadata.csv"))

metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")] # add in ChapterS Soils and Mixtures later


# In[4]:


metadata.sort_values('material',inplace=True)


# In[5]:


print(metadata.to_string())


# In[6]:


names = []
num = []

frame = pd.DataFrame(columns=['material', 'count'])

series = metadata['material']
series = series.apply(lambda x: x.split(" ")[0])

series = series.value_counts()
# series = series.to_frame()

frame['count'] = series.values
frame['material'] = series.index
# frame = frame[frame['count'] >= 12]
# series.columns = ['count']
# series['material'] = series.index
# series.reset_index([], inplace=True)
# # series.columns = ['material', 'count']

# # for i in range(counts.size):
# #     print(counts.index[i] + " " + str(counts[i]))
# series
frame.iloc[:, 0].tolist()

print(frame.to_string())


# In[7]:


dictionary = {frame.iloc[:, 0].tolist()[i] : i for i in range(len(frame.iloc[:, 0].tolist()))}
dictionary


# In[ ]:





# In[ ]:




