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

data_dir = os.environ['DATA_DIR']
stddata_path = os.path.join(data_dir,"Srikar-Standardized")
metadata = pd.read_csv(os.path.join(stddata_path,"spectra-metadata.csv"), sep="|", dtype={"spectrum_id":str})

metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")] # add in ChapterS Soils and Mixtures later


# In[2]:


metadata.sort_values('material',inplace=True)


# In[3]:


print(metadata.to_string())


# In[9]:


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

frame


# In[7]:


dictionary = {frame.iloc[:, 0].tolist()[i] : i for i in range(len(frame.iloc[:, 0].tolist()))}
dictionary


# In[ ]:




