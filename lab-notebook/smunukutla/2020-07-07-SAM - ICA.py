#!/usr/bin/env python
# coding: utf-8

# In[1]:


# environment set up
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score
import os
import random
import pandas as pd
import ast
from scipy import stats as st
import time


# In[2]:


spectrum_len = 500 # automate this
parent_dir = os.environ['PWD']
stddata_path = os.path.join(os.environ['DATA_DIR'], "StdData-" + str(spectrum_len))
os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[3]:


data = pd.read_csv("data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[4]:


spectra = np.zeros((num_samples,spectrum_len))


# In[5]:


for i in range(len(record_nums)):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
#     if i == 0:
#         wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()


# In[6]:


spectra.shape


# In[7]:


y_cat = to_categorical(y)


# In[8]:


from sklearn.decomposition import FastICA


# In[ ]:


model = FastI(n_components=10, alpha=1, verbose=True)

