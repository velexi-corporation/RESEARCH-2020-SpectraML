#!/usr/bin/env python
# coding: utf-8

# In[1]:


# environment set up
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd

# working folder
# directory = "/Users/Srikar/Desktop/Velexi/spectra-ml/data"
directory = os.environ['DATA_DIR']
os.chdir(directory)

# print(os.getcwd())


# In[2]:


stddata_path = os.path.join(directory,"Srikar-Standardized")
metadata = pd.read_csv(os.path.join(stddata_path,"spectra-metadata.csv"), sep="|")
metadata.head()


# In[3]:


metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")]
metadata.shape


# In[4]:


# def find_record(string):
#     ind = string.find("|")
#     return string[:ind]


# In[5]:


# data = metadata.iloc[0]
# type(data)

# metadata[metadata["material"].str.match("Chlor")].shape[0]


# In[6]:


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
# print(num_samples)
# print(len(y))
# print(type(y))
# print(act)
# print(aln)
# print(chl)


# In[7]:


spectrum_len = 500
spectra = np.zeros((num_samples,spectrum_len))
wavelengths = np.zeros((1,spectrum_len))

# y = np.zeros((num_samples, 1))


# In[8]:


# import shutil

# for num in actinolite:
#     shutil.copy2(stddata_path+"/{}.csv".format(num), directory+"/Std_Actinolite")
# for num in alunite:
#     shutil.copy2(stddata_path+"/{}.csv".format(num), directory+"/Std_Alunite")
# for num in chlorite:
#     shutil.copy2(stddata_path+"/{}.csv".format(num), directory+"/Std_Chlorite")


# In[9]:


# num_neg = 0
# for i in range(num_samples):
#     hasnegative = False
#     data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
#     if i == 0:
#         wavelengths[i,:] = data.iloc[:, 0].to_numpy()
#     spectra[i,:] = data.iloc[:, 1].to_numpy()
#     for j in range(spectrum_len):
#         if spectra[i,j] < 0:
#             hasnegative = True
#             spectra[i,j] = 0
#     if hasnegative:
#         print(record_nums[i])
#         num_neg += 1
# print(num_neg)
# wavelengths
# print(record_nums[43])
# print(spectra[43])


# In[10]:


# spectrum_len = 480
# spectra = np.zeros((num_samples,spectrum_len))

# spectrum_categories = np.zeros(num_samples)
# first_record_of_mixtures_chapter = 11602
# is_a_mineral = 1                                   # these numbers match the chapter numbers given by usgs
# is_a_mixture = 2
# spectrum_names = ["" for x in range(num_samples)]

# y = np.zeros((num_samples, 1))

# os.chdir(dataset)

# i = 0

# for filename in os.listdir(dataset):
#     file_object  = open(filename, 'r').readlines()
# #     strip off header, add to matrix 'spectra'
#     spectra[i,:] = file_object[1:]

# #     label spectrum class, based on header
# #     actinolite: 0, alunite: 1, chlorite: 2
#     material_name = file_object[0]
    
#     spectrum_names[i] = material_name
    
#     start = 'Record='
#     end = ':'
#     record_number = int((material_name.split(start))[1].split(end)[0])
#     # print(record_number)
#     if record_number < first_record_of_mixtures_chapter:
#         spectrum_categories[i] = is_a_mineral
#     else:
#         spectrum_categories[i] = is_a_mixture

# #     print(material_name)

#     if material_name.find('Actinolite',) != -1: # if material name contains actinolite
#         y[i,0] = 0
#     elif material_name.find('Alun',)!= -1:
#         y[i,0] = 1
#     else: # chlorite
#         y[i,0] = 2

# #     turn missing points into 0
#     for j in range(spectrum_len):
#         if spectra[i,j] < 0:
#             spectra[i,j] = 0
#     i+=1


# In[11]:


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
# for i in range(i0):
#     fig = plt.figure()
#     plt.plot(range(1, spectrum_len+1), spectra0[i,:], label = labels0[i], linewidth = linewidth0[i]) # remove linewidth for all mixtures/minerals to be standard
#     plt.plot(wavelengths[0,:], spectra0[i,:]) # remove linewidth for all mixtures/minerals to be standard
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#     path = "/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla/plots/" + mineral_names[0] + str(i) + ".png"
#     fig.savefig(path, format = "PNG")
# plt.legend(bbox_to_anchor=(1.1, 1.05))
# plt.show()

# for i in range(i1):
#     plt.plot(range(1, spectrum_len+1), spectra1[i,:], label = labels1[i], linewidth = linewidth1[i])
#     plt.plot(wavelengths[0,:], spectra1[i,:])
# plt.legend(bbox_to_anchor=(1.1, 1.05))
# plt.show()

# for i in range(i2):
#     plt.plot(range(1, spectrum_len+1), spectra2[i,:], label = labels2[i], linewidth = linewidth2[i])
#     plt.plot(wavelengths[0,:], spectra2[i,:])
# plt.legend(bbox_to_anchor=(1.1, 1.05))
# plt.show()


# In[12]:


# random.seed(660)


# In[ ]:


os.chdir("/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla")
fi = open("indices.txt", "r")

for i in range(10):
    train_set_indices = ast.literal_eval(fi.readline())
    test_set_indices = ast.literal_eval(fi.readline())
    dev_set_indices = ast.literal_eval(fi.readline())
    
    for j in train_set_indices:
        j = int(j)
    for j in test_set_indices:
        j = int(j)
    for j in dev_set_indices:
        j = int(j)
    
    train_set = spectra[train_set_indices, :]
    train_labels = y[train_set_indices, :]
    dev_set = spectra[dev_set_indices, :]
    dev_labels = y[dev_set_indices, :]
    test_set = spectra[test_set_indices, :]
    test_labels = y[test_set_indices, :]

    train_labels = train_labels.flatten()
    dev_labels = dev_labels.flatten()
    test_labels = test_labels.flatten()
    
    train_set = np.reshape(train_set, (train_set.shape[0], spectrum_len, 1))
    dev_set = np.reshape(dev_set, (dev_set.shape[0], spectrum_len, 1))
    test_set = np.reshape(test_set, (test_set.shape[0], spectrum_len, 1))

    train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
    dev_labels = np.reshape(dev_labels, (dev_labels.shape[0], 1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

    train_labels = to_categorical(train_labels)
    dev_labels = to_categorical(dev_labels)
    test_labels = to_categorical(test_labels)


# In[13]:


# sample_indices = list(range(0, num_samples))
# print(num_samples)
# random.shuffle(sample_indices) # stratified sklearn split
# train_set_size = 3*(num_samples//5)
# dev_set_size = (num_samples//5)
# test_set_size= num_samples-dev_set_size - train_set_size
# print(train_set_size)
# print(test_set_size)
# print(dev_set_size)
# train_set_indices = sample_indices[:train_set_size]
# dev_set_indices = sample_indices[train_set_size: train_set_size+dev_set_size]
# test_set_indices= sample_indices[train_set_size+dev_set_size: num_samples]
# print(train_set_indices)
# print(test_set_indices)
# print(dev_set_indices)

# train_set = spectra[train_set_indices, :]
# train_labels = y[train_set_indices, :]
# dev_set = spectra[dev_set_indices, :]
# dev_labels = y[dev_set_indices, :]
# test_set = spectra[test_set_indices, :]
# test_labels = y[test_set_indices, :]


# In[14]:


# train_labels = train_labels.flatten()
# dev_labels = dev_labels.flatten()
# test_labels = test_labels.flatten()
# type(train_labels)


# In[15]:


# len(train_set)
# len(train_set[17])
# print(test_set.shape)


# In[16]:


# train_set = np.reshape(train_set, (train_set.shape[0], spectrum_len, 1))
# dev_set = np.reshape(dev_set, (dev_set.shape[0], spectrum_len, 1))
# test_set = np.reshape(test_set, (test_set.shape[0], spectrum_len, 1))

# train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
# dev_labels = np.reshape(dev_labels, (dev_labels.shape[0], 1))
# test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

# train_labels = to_categorical(train_labels)
# dev_labels = to_categorical(dev_labels)
# test_labels = to_categorical(test_labels)


# In[17]:


# print(dev_set)


# In[217]:


# random.seed()
model = Sequential()
# model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model.add(Conv1D(64, 25, activation='relu', input_shape=(train_set.shape[1], 1)))
model.add(Conv1D(64, 25, activation='relu'))
model.add(MaxPooling1D(4)) # 108 by 64 so far
model.add(Conv1D(100, 25, activation='relu'))
model.add(Conv1D(100, 25, activation='relu'))
model.add(MaxPooling1D(4))
# model.add(Dropout(0.5))
# model.add(GlobalAveragePooling1D())
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
print(model.summary())


# In[218]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 12
EPOCHS = 50

print(train_labels.shape)
model.fit(train_set, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(dev_set, dev_labels)) 


# In[188]:


y_pred = model.predict(test_set)
y_pred


# In[189]:


test_labels


# In[190]:


model.evaluate(test_set, test_labels)


# In[ ]:




