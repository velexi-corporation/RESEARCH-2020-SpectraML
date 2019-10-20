#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import random
import ast
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from scipy import stats as st

# directory = "/Users/Srikar/Desktop/Velexi/spectra-ml/data/plots"
data_dir = os.environ['DATA_DIR']
data_dir = os.path.join(data_dir, "plots")
os.chdir(data_dir)


# In[33]:


num_samples = len(os.listdir(os.getcwd()))
img = mpimg.imread(os.path.join(data_dir,os.listdir(os.getcwd())[0]))
spectrum_height = img.shape[0]
spectrum_width = img.shape[1]


# In[34]:


def convertimg(img):
    newimg = np.empty([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            row = img[i][j]
            newimg[i][j] = (row[0] + row[1] + row[2])/3
    return newimg


# In[35]:


data = pd.read_csv("/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla/data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[36]:


spectra = np.zeros((num_samples, spectrum_height, spectrum_width))
i = 0
for num in record_nums:
    img = plt.imread(os.path.join(data_dir,num+"-"+spectrum_names[i]+".png")) # os.path.join here, look into timeit, pickle file
    spectra[i] = convertimg(img)
    i += 1


# In[37]:


spectra = spectra.reshape(spectra.shape[0], spectra.shape[1], spectra.shape[2], 1)
spectra.shape


# In[38]:


os.chdir("/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla")
fi = open("indices.txt", "r")

stats = []

for i in range(10):
    train_set_indices = ast.literal_eval(fi.readline())
    test_set_indices = ast.literal_eval(fi.readline())
    dev_set_indices = ast.literal_eval(fi.readline())
    
    for j in train_set_indices:
        j = int(j)
    for k in test_set_indices:
        k = int(k)
    for m in dev_set_indices:
        m = int(m)
        
#     print(train_set_indices)
#     print(test_set_indices)
#     print(dev_set_indices)
    
    train_set = spectra[train_set_indices, :]
    train_labels = y[train_set_indices, :]
    dev_set = spectra[dev_set_indices, :]
    dev_labels = y[dev_set_indices, :]
    test_set = spectra[test_set_indices, :]
    test_labels = y[test_set_indices, :]

    train_labels = to_categorical(train_labels)
    dev_labels = to_categorical(dev_labels)
    test_labels = to_categorical(test_labels)
    
    model = Sequential()
    #add model layers
    model.add(Conv2D(32, kernel_size=10, strides=(6,6), activation='relu', input_shape=(spectra.shape[1],spectra.shape[2], 1))) # finer features at the first layer
    model.add(Conv2D(32, kernel_size=3, activation='relu')) # larger features at later layer
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_set, train_labels, validation_data=(dev_set, dev_labels), epochs=10, verbose=1)
    
    my_list = model.evaluate(test_set, test_labels, verbose=0)
    
    stats.append(my_list[1])

print("2D CNN:", stats)


# In[40]:


print("2D CNN Results:", st.describe(stats))


# In[ ]:




