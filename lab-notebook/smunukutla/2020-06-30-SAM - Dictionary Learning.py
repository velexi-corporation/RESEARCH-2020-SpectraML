#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


spectrum_len = 500 # automate this

parent_dir = os.environ['PWD']
data_dir = os.environ['DATA_DIR']
stddata_path = os.path.join(data_dir, "StdData-" + str(spectrum_len))

os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[3]:


data = pd.read_csv("data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[4]:


num_samples


# In[5]:


data


# In[6]:


spectra = np.zeros((num_samples,spectrum_len))
wavelengths = np.zeros((1,spectrum_len))


# In[7]:


for i in range(len(record_nums)):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
    if i == 0:
        wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()


# In[8]:


type(spectra)


# In[9]:


y_cat = to_categorical(y)


# In[10]:


data.head(5)


# In[11]:


spectra.shape


# In[12]:


spectra


# In[13]:


y_cat = to_categorical(y)


# In[14]:


from sklearn.decomposition import DictionaryLearning


# In[15]:


model = DictionaryLearning(n_components=10, alpha=1, verbose=True)


# In[16]:


dictionary = model.fit_transform(spectra)


# In[17]:


dictionary.shape


# In[18]:


print(dictionary)


# In[19]:


model2 = DictionaryLearning(n_components=10, alpha=1, transform_algorithm='threshold', verbose=True)


# In[20]:


dictionary2 = model2.fit_transform(spectra)


# In[21]:


dictionary2.shape


# In[22]:


print(dictionary2)


# In[23]:


model.get_params()


# In[24]:


atoms = model.components_
print(atoms)


# In[25]:


atoms.shape


# In[26]:


atoms2 = model2.components_
print(atoms2)


# In[27]:


atoms2.shape


# approximate with the training data
# run transform on the training data to find the reconstructed spectra
# 
# 166 x 10 X 10 x 500
# 
# approximation of an integral
# 
# thresholding is bad to get coefficients
# 
# use max distance
# 
# take each row at a time and then take the norm of each
# 
# L2 norm / number of points
# 
# model.transform(spectra) is the same thing as dictionary

# In[28]:


reconstructed_spectra = dictionary.dot(atoms)


# In[29]:


reconstructed_spectra


# In[30]:


reconstructed_spectra.shape


# In[31]:


distances = []
for i in range(len(spectra)):
    distances.append(np.linalg.norm(spectra[i] - reconstructed_spectra[i]))


# In[32]:


distances


# In[33]:


reconstructed_spectra2 = dictionary2.dot(atoms2)


# In[34]:


reconstructed_spectra2


# In[35]:


distances2 = []
for i in range(len(spectra)):
    distances2.append(np.linalg.norm(spectra[i] - reconstructed_spectra2[i]))


# In[36]:


distances2


# In[37]:


reconstructed_spectra


# In[38]:


spectra


# In[39]:


height = 3
width = 1.5*height
linewidth = 4
# for i in range(num_samples):
examples = 20
lst = [35, 49, 137, 127, 108, 40, 72, 33, 29, 64, 127, 11, 98, 86, 8, 74, 85, 55, 17, 61]
for index in lst:
    fig = plt.figure(figsize=(width, height), dpi=60)
    plt.plot(wavelengths[0,:], reconstructed_spectra[index,:], linewidth = linewidth, color='k')
    plt.xticks([])
    plt.yticks([])
    ax = fig.axes
    ax[0].axis('off')
    print("Reconstructed Spectra:", index)
    plt.show()
# path = os.path.join(data_dir, "plots-" + str(spectrum_len), record_nums[i] + "-" + spectrum_names[i] + ".png")
# fig.savefig(path, format = "PNG")
# plt.close(fig)


# In[40]:


height = 3
width = 1.5*height
linewidth = 4
lst = [35, 49, 137, 127, 108, 40, 72, 33, 29, 64, 127, 11, 98, 86, 8, 74, 85, 55, 17, 61]
for index in lst:
    fig = plt.figure(figsize=(width, height), dpi=60)
    plt.plot(wavelengths[0,:], reconstructed_spectra2[index,:], linewidth = linewidth, color='k')
    plt.xticks([])
    plt.yticks([])
    ax = fig.axes
    ax[0].axis('off')
    print("Reconstructed Spectra2:", index)
    plt.show()


# In[45]:


num_zero = 0
for row in atoms:
    done = False
    for col in row:
        if col == 0 and not done:
            num_zero += 1
            done = True
            print(row)
num_zero


# In[46]:


num_zero = 0
for row in atoms2:
    done = False
    for col in row:
        if col == 0 and not done:
            num_zero += 1
            done = True
            print(row)
num_zero


# In[47]:


num_zero = 0
for row in dictionary:
    done = False
    for col in row:
        if col == 0 and not done:
            num_zero += 1
            done = True
            print(row)
num_zero


# In[48]:


num_zero = 0
for row in dictionary2:
    done = False
    for col in row:
        if col == 0 and not done:
            num_zero += 1
            done = True
            print(row)
num_zero


# In[ ]:




