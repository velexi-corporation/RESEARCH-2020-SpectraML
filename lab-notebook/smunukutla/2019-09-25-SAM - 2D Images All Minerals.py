#!/usr/bin/env python
# coding: utf-8

# In[20]:


# environment set up
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from spectra_ml.io_ import load_spectra_metadata

# working folder
# data_dir = "/Users/Srikar/Desktop/Velexi/spectra-ml/data/"
data_dir = os.environ['DATA_DIR']
os.chdir(data_dir)


# In[21]:


stddata_path = os.path.join(data_dir,"Srikar-Standardized")
metadata = load_spectra_metadata(os.path.join(stddata_path,"spectra-metadata.csv"))

metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")]


# In[22]:


data = pd.read_csv("/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla/data.csv", sep=",", dtype=str)
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[23]:


spectrum_len = 500
spectra = np.zeros((num_samples,spectrum_len))
wavelengths = np.zeros((1,spectrum_len))


# In[24]:


for i in range(num_samples):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
    if i == 0:
        wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()


# In[25]:


num_samples


# In[29]:


height = 1.5
width = 1.5*height
linewidth = 3
# figsize=(width, height), dpi=100
for i in range(num_samples):
    fig = plt.figure()
    plt.plot(wavelengths[0,:], spectra[i,:], linewidth = linewidth, color='k')
    plt.xticks([])
    plt.yticks([])
    ax = fig.axes
    ax[0].axis('off')
    path = os.path.join(data_dir, "plots", record_nums[i] + "-" + spectrum_names[i] + ".png")
    fig.savefig(path, format = "PNG")
    plt.close(fig)


# In[ ]:




