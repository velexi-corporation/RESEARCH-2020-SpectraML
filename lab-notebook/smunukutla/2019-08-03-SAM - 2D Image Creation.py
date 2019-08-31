#!/usr/bin/env python
# coding: utf-8

# In[90]:


# environment set up
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd

# working folder
# data_dir = "/Users/Srikar/Desktop/Velexi/spectra-ml/data/"
data_dir = os.environ['DATA_DIR']
os.chdir(data_dir)


# In[99]:


stddata_path = os.path.join(data_dir,"Srikar-Standardized")
metadata = pd.read_csv(os.path.join(stddata_path,"spectra-metadata.csv"), sep="|")
metadata.head()


# In[100]:


metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")] # add in ChapterS Soils and Mixtures later
metadata.shape


# In[101]:


record_nums = []
y = []
spectrum_names = []

act = 0
aln = 0
chl = 0

for i in range(metadata.shape[0]):
    data = metadata.iloc[i, :]
    if data[2].find("Actinolite") != -1: # if material name contains actinolite
        record_nums.append(data[0])
        y.append(int(0))
        spectrum_names.append("Actinolite")
        act += 1
    elif data[2].find("Alun") != -1:
        record_nums.append(data[0])
        y.append(int(1))
        spectrum_names.append("Alunite")
        aln += 1
    elif (data[2].find("Chlorit") != -1 or data[2].find("Chlor.") != -1 or data[2].find("Chlor+") != -1 or data[2].find("Chl.") != -1):
        record_nums.append(data[0])
        y.append(int(2))
        spectrum_names.append("Chlorite")
        chl += 1

y = np.reshape(y, (len(y), 1))
num_samples = len(record_nums)
print(num_samples)
print(len(y))
print(type(y))
print(act)
print(aln)
print(chl)


# In[102]:


spectrum_len = 500
spectra = np.zeros((num_samples,spectrum_len))
wavelengths = np.zeros((1,spectrum_len))


# In[103]:


num_neg = 0
for i in range(num_samples):
    hasnegative = False
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
    if i == 0:
        wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()
    for j in range(spectrum_len):
        if spectra[i,j] < 0:
            hasnegative = True
            spectra[i,j] = 0
    if hasnegative:
        print(record_nums[i])
        num_neg += 1
print(num_neg)


# In[111]:


# --- plot the classes

# plot each class in a separate plot
# plot spectra names in legend
# plot minerals and mixtures w diff line widths

mineral_names = ["Actinolite", "Alunite", "Chlorite"]

# variables
num0 = 0 #number of samples of class 0
num1 = 0
num2 = 0

mineral_linewidth = 1         # linewidth = 1 is default
mixture_linewidth = 3         

# count the number of each class to make spectra0, spectra1, spectra2 databases
for i in range(num_samples):
    if y[i,0]== 0:
        num0 += 1
    elif y[i,0]== 1:
        num1 += 1
    elif y[i,0]== 2:
        num2 += 1

# make class-specific databases spectra0, ...1, ...2
spectra0 = np.zeros((num0,spectrum_len)) 
spectra1 = np.zeros((num1,spectrum_len)) 
spectra2 = np.zeros((num2,spectrum_len)) 

labels0 = ["" for x in range(num0)]
labels1 = ["" for x in range(num1)]
labels2 = ["" for x in range(num2)]

linewidth0 = np.zeros(num0)
linewidth1 = np.zeros(num1)
linewidth2 = np.zeros(num2)


# make counters for each database to place spectra
i0 = 0
i1 = 0
i2 = 0

# set linewidth for the spectrum 
# populate class-specific databases spectra0, ...1, ...2
for i in range(num_samples):
    
    # set linewidth
    #testcode
    #print(spectrum_categories)
    #print(spectrum_categories[i])
    
#     if spectrum_categories[i] == is_a_mineral:
#         linewidth = mineral_linewidth
        
#         #testcode
#         #print('min')
#     else: 
#         linewidth = mixture_linewidth
    linewidth = 2
        
        #testcode
        #print('mix')
    
    # populate matrices for making each class plot
    if y[i,0]== 0:
        spectra0[i0,:] = spectra[i,:]
        labels0[i0] = spectrum_names[i]
        linewidth0[i0] = linewidth
        i0 +=1
    elif y[i,0]== 1:
        spectra1[i1,:] = spectra[i,:]
        labels1[i1] = spectrum_names[i]
        linewidth1[i1] = linewidth
        i1 +=1
    else:
        spectra2[i2,:] = spectra[i,:]
        labels2[i2] = spectrum_names[i]
        linewidth2[i2] = linewidth
        i2 +=1

# plot each class-specific database separately
# remove linewidth for all mixtures/minerals to be standard
for i in range(i0):
#     plt.plot(range(1, spectrum_len+1), spectra0[i,:], label = labels0[i], linewidth = linewidth0[i])
    fig = plt.figure()
    plt.plot(wavelengths[0,:], spectra0[i,:], label = labels0[i], linewidth = linewidth0[i], color='k')
    plt.xticks([])
    plt.yticks([])
#     fig.patch.set_visible(False)
#     plt.show()
    path = os.path.join(data_dir, "plots", mineral_names[0] + str(i+1) + ".png")
    ax = fig.axes
    ax[0].axis('off')
    fig.savefig(path, format = "PNG")
    plt.close(fig)
# plt.legend(bbox_to_anchor=(1.1, 1.05))
# plt.show()

for i in range(i1):
#     plt.plot(range(1, spectrum_len+1), spectra1[i,:], label = labels1[i], linewidth = linewidth1[i])
    fig = plt.figure()
    plt.plot(wavelengths[0,:], spectra1[i,:], label = labels1[i], linewidth = linewidth1[i], color='k')
    plt.xticks([])
    plt.yticks([])
#     fig.patch.set_visible(False)
#     plt.show()
    path = os.path.join(data_dir, "plots", mineral_names[1] + str(i+1) + ".png")
    ax = fig.axes
    ax[0].axis('off')
    fig.savefig(path, format = "PNG")
    plt.close(fig)
# plt.legend(bbox_to_anchor=(1.1, 1.05))
# plt.show()

for i in range(i2):
#     plt.plot(range(1, spectrum_len+1), spectra2[i,:], label = labels2[i], linewidth = linewidth2[i])
    fig = plt.figure()
    plt.plot(wavelengths[0,:], spectra2[i,:], label = labels2[i], linewidth = linewidth2[i], color='k')
    plt.xticks([])
    plt.yticks([])
#     fig.patch.set_visible(False)
#     plt.show()
    path = os.path.join(data_dir, "plots", mineral_names[2] + str(i+1) + ".png")
    ax = fig.axes
    ax[0].axis('off')
    fig.savefig(path, format = "PNG")
    plt.close(fig)
# plt.legend(bbox_to_anchor=(1.1, 1.05))
# plt.show()


# In[106]:


num_neg = 0
for file in os.listdir(stddata_path):
    data = pd.read_csv(os.path.join(stddata_path,file))
    if data.shape[1] == 2:
        arr = data.iloc[:, 1].to_numpy()
    if np.isnan(arr[0]) or np.isnan(arr[len(arr)-1]):
        print(file)
        num_neg += 1
        continue
#     for j in range(len(arr)):
#         if np.isnan(arr[j]):
#             print(file)
#             num_neg += 1
#             break
print(num_neg)


# In[107]:


metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")]
metadata.shape

record_nums = []
mineral_names = []
for i in range(metadata.shape[0]):
    data = metadata.iloc[i, :]
    record_nums.append(data[0])
    mineral_names.append(data[2])


# In[108]:


num_neg = 0
print(len(record_nums))
for i in range(len(record_nums)):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
    if data.shape[1] == 2:
        arr = data.iloc[:, 1].to_numpy()
    if np.isnan(arr[0]) or np.isnan(arr[len(arr)-1]):
        print(record_nums[i])
        print(mineral_names[i])
        num_neg += 1
        continue
#     for j in range(len(arr)):
#         if np.isnan(arr[j]):
#             print(file)
#             num_neg += 1
#             break
print(num_neg)


# In[109]:


# os.listdir(stddata_path)


# In[110]:


# data = pd.read_csv(os.path.join(stddata_path,"211.csv"))
# arr = data.iloc[:, 1].to_numpy()
# for j in range(spectrum_len):
#     print(type(arr[j]))


# In[ ]:




