#!/usr/bin/env python
# coding: utf-8

# In[9]:


# environment set up
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from spectra_ml.io_ import load_spectra_metadata


# In[10]:


# working folder
spectrum_len = 500

parent_dir = os.environ['PWD']
data_dir = os.environ['DATA_DIR']
stddata_path = os.path.join(data_dir, "StdData-" + str(spectrum_len))


# In[11]:


metadata = load_spectra_metadata(os.path.join(stddata_path,"spectra-metadata.csv"))

metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")]


# In[12]:


os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))
data = pd.read_csv("data.csv", sep=",", dtype=str)
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[13]:


spectra = np.zeros((num_samples,spectrum_len))
wavelengths = np.zeros((1,spectrum_len))


# In[14]:


for i in range(num_samples):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
    if i == 0:
        wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()


# In[15]:


num_samples


# In[16]:


spectra


# In[29]:


examples = 20
lst = []
for i in range(examples):
    lst.append(random.randint(0, num_samples-1))
print(lst)


# In[32]:


height = 3
width = 1.5*height
linewidth = 4
# figsize=(width, height), dpi=96 default
# for i in range(num_samples):
for index in lst:
    fig = plt.figure(figsize=(width, height), dpi=60)
    plt.plot(wavelengths[0,:], spectra[index,:], linewidth = linewidth, color='k')
    plt.xticks([])
    plt.yticks([])
    ax = fig.axes
    ax[0].axis('off')
    print("Original Spectra:", index)
    plt.show()
# path = os.path.join(data_dir, "plots-" + str(spectrum_len), record_nums[i] + "-" + spectrum_names[i] + ".png")
# fig.savefig(path, format = "PNG")
# plt.close(fig)


# In[ ]:




