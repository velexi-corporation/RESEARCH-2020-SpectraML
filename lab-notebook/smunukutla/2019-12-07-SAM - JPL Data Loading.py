#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import os
import matplotlib.pyplot as plt


# In[77]:


data_dir = os.environ['DATA_DIR']
data_path = os.path.join(data_dir, "JPL_data")
os.chdir(data_path)


# In[78]:


metadata_path = os.path.join(data_path, "Manifest.txt")


# In[79]:


# fi = open(os.listdir()[1], "r")

# data = pd.read_csv(metadata_path, sep = "\s+", header=None, skipinitialspace=True, skiprows=21)
metadata = pd.read_csv(metadata_path, header=None, index_col=False, sep = "\s+")
# data.columns = {"Wavelength", "Reflectance"}
metadata.columns = {"Spectrum"}
metadata = metadata[metadata.iloc[:, 0].str.contains("spectrum")]
metadata.reset_index(inplace=True, drop=True)
# print(type(metadata))
for i in range(metadata.shape[0]):
    print(metadata.iloc[i])


# In[80]:


i = 0
for row in metadata.itertuples():
    print(i)
    i = i + 1
#     print(row.Spectrum)
#     print(row.Spectrum) # or iterrows() (returns index and row) and then row["Spectrum"]
    data = pd.read_csv(os.path.join(data_path, row.Spectrum), sep = "\s+", header=None, skipinitialspace=True, skiprows=21)
    fi = open(row.Spectrum, "r")
#     print(fi.readline().strip().split()[1])
    print(fi.readline())


# In[48]:


# plt.plot(data["Reflectance"])
plt.plot(data.iloc[:, 1])
plt.title('Reflectance vs Wavelength')
plt.ylabel('Reflectance')
plt.xlabel('Wavelength')
# plt.legend(['Reflectance'], loc='upper right')


# In[ ]:




