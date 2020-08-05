#!/usr/bin/env python
# coding: utf-8

# In[42]:


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


# In[43]:


spectrum_len = 500 # automate this
parent_dir = os.environ['PWD']
stddata_path = os.path.join(os.environ['DATA_DIR'], "StdData-" + str(spectrum_len))
os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[44]:


data = pd.read_csv("data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[45]:


data


# In[46]:


spectra = np.zeros((num_samples,spectrum_len))


# In[47]:


for i in range(len(record_nums)):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
#     if i == 0:
#         wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()


# In[48]:


y_cat = to_categorical(y)


# In[8]:


data.head(5)


# In[9]:


spectra.shape


# In[10]:


spectra


# In[11]:


y_cat = to_categorical(y)


# In[12]:


from sklearn.decomposition import DictionaryLearning


# In[13]:


model = DictionaryLearning(n_components=10, alpha=1, verbose=True)


# In[14]:


results = model.fit_transform(spectra)


# In[15]:


results.shape


# In[16]:


print(results)


# In[17]:


model2 = DictionaryLearning(n_components=10, alpha=1, transform_algorithm='threshold', verbose=True)


# In[18]:


results2 = model2.fit_transform(spectra)


# In[19]:


results2.shape


# In[20]:


print(results2)


# In[21]:


model.get_params()


# In[22]:


print(model.components_)


# In[23]:


model.components_.shape


# In[24]:


print(model2.components_)


# In[25]:


model2.components_.shape


# In[ ]:


approximate with the training data
run transform on the training data to find the reconstructed spectra

166 x 10 X 10 x 500

approximation of an integral

thresholding is bad to get coefficients


# In[26]:


results.dot(model.components_)


# In[27]:


dist = np.linalg.norm(results.dot(model.components_) - spectra)


# In[28]:


# max(results.dot(model.components_) - spectra, key=max)


# In[29]:


dist


# In[30]:


dist2 = np.linalg.norm(results2.dot(model2.components_) - spectra)


# In[31]:


dist2


# In[39]:


model.transform(spectra)


# In[40]:


results


# In[49]:


np.linalg.norm(model.transform(spectra) - results)


# In[ ]:




