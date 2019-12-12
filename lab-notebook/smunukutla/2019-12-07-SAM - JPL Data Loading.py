#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import os
import matplotlib.pyplot as plt


# In[44]:


data_dir = os.environ['DATA_DIR']
data_path = os.path.join(data_dir, "JPL_data")
os.chdir(data_path)


# In[45]:


os.listdir()[1]


# In[46]:


fi = open(os.listdir()[1], "r")

data = pd.read_csv(os.path.join(data_path, "mineral.silicate.nesosilicate.coarse.vswir.ns-6a.jpl.beckman.spectrum.txt"), sep = "\s+", header=None, skipinitialspace=True, skiprows=21)
data.columns = {"Wavelength", "Reflectance"}
print(type(data))
data


# In[48]:


# plt.plot(data["Reflectance"])
plt.plot(data.iloc[:, 1])
plt.title('Reflectance vs Wavelength')
plt.ylabel('Reflectance')
plt.xlabel('Wavelength')
# plt.legend(['Reflectance'], loc='upper right')


# In[ ]:




