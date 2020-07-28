#!/usr/bin/env python
# coding: utf-8

# In[19]:


# environment set up
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import time
import ast
from scipy import stats as st

# working folder
directory = os.environ['DATA_DIR']


# In[20]:


spectrum_len = 500 # automate this
parent_dir = os.environ['PWD']
stddata_path = os.path.join(os.environ['DATA_DIR'], "StdData-" + str(spectrum_len))
os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[21]:


data = pd.read_csv("data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[22]:


data


# In[23]:


spectra = np.zeros((num_samples,spectrum_len))


# In[24]:


for i in range(len(record_nums)):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
#     if i == 0:
#         wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()


# In[26]:


data


# In[27]:


spectra.shape


# In[30]:


y_cat = to_categorical(y)


# In[31]:


from sklearn.decomposition import DictionaryLearning


# In[32]:


model = DictionaryLearning(n_components=10, alpha=1, verbose=True)


# In[44]:


results = model.fit_transform(spectra)


# In[45]:


for row in results:
    print(row)


# In[46]:


model2 = DictionaryLearning(n_components=10, alpha=1, transform_algorithm='threshold', verbose=True)


# In[47]:


results2 = model2.fit_transform(spectra)


# In[48]:


for row in results2:
    print(row)


# In[49]:


results.shape


# In[ ]:




