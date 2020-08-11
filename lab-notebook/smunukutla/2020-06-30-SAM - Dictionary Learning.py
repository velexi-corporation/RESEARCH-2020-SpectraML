#!/usr/bin/env python
# coding: utf-8

# In[86]:


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


# In[87]:


spectrum_len = 500 # automate this

parent_dir = os.environ['PWD']
data_dir = os.environ['DATA_DIR']
stddata_path = os.path.join(data_dir, "StdData-" + str(spectrum_len))

os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[88]:


data = pd.read_csv("data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[89]:


num_samples


# In[90]:


data


# In[91]:


spectra = np.zeros((num_samples,spectrum_len))
wavelengths = np.zeros((1,spectrum_len))


# In[92]:


for i in range(len(record_nums)):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
    if i == 0:
        wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()


# In[93]:


type(spectra)


# In[94]:


y_cat = to_categorical(y)


# In[95]:


data.head(5)


# In[96]:


spectra.shape


# In[97]:


spectra


# In[98]:


y_cat = to_categorical(y)


# In[99]:


from sklearn.decomposition import DictionaryLearning


# In[100]:


model = DictionaryLearning(n_components=10, alpha=1, verbose=True)


# In[101]:


atoms = model.fit_transform(spectra)


# In[102]:


atoms.shape


# In[103]:


print(atoms)


# In[104]:


model2 = DictionaryLearning(n_components=10, alpha=1, transform_algorithm='threshold', verbose=True)


# In[105]:


atoms2 = model2.fit_transform(spectra)


# In[106]:


atoms2.shape


# In[107]:


print(atoms2)


# In[108]:


model.get_params()


# In[109]:


print(model.components_)


# In[110]:


model.components_.shape


# In[111]:


print(model2.components_)


# In[112]:


model2.components_.shape


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
# model.transform(spectra) is the same thing as atoms

# In[113]:


reconstructed_spectra = atoms.dot(model.components_)


# In[114]:


reconstructed_spectra


# In[115]:


reconstructed_spectra.shape


# In[116]:


distances = []
for i in range(len(spectra)):
    distances.append(np.linalg.norm(spectra[i] - reconstructed_spectra[i]))


# In[132]:


distances


# In[118]:


reconstructed_spectra2 = atoms2.dot(model2.components_)


# In[119]:


distances2 = []
for i in range(len(spectra)):
    distances2.append(np.linalg.norm(spectra[i] - reconstructed_spectra2[i]))


# In[120]:


distances2


# In[121]:


reconstructed_spectra


# In[122]:


spectra


# In[140]:


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


# In[ ]:




