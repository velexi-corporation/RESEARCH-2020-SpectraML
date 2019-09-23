#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# directory = "/Users/Srikar/Desktop/Velexi/spectra-ml/data/plots"
data_dir = os.environ['DATA_DIR']
data_dir = os.path.join(data_dir, "plots")
os.chdir(data_dir)


# In[5]:


num_samples = len(os.listdir(os.getcwd()))
img = mpimg.imread(os.path.join(data_dir,os.listdir(os.getcwd())[0]))
spectrum_height = img.shape[0]
spectrum_width = img.shape[1]


# In[6]:


def convertimg(img):
    newimg = np.empty([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            row = img[i][j]
            newimg[i][j] = (row[0] + row[1] + row[2])/3
    return newimg


# In[7]:


data = pd.read_csv("/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla/data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[8]:


spectra = np.zeros((num_samples, spectrum_height, spectrum_width))
i = 0
for num in record_nums:
    img = plt.imread(os.path.join(data_dir,num+"-"+spectrum_names[i]+".png")) # os.path.join here, look into timeit, pickle file
    spectra[i] = convertimg(img)
    i += 1


# In[9]:


spectra.shape


# In[ ]:


os.chdir("/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla")
fi = open("indices.txt", "r")

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

    train_labels = train_labels.flatten()
    dev_labels = dev_labels.flatten()
    test_labels = test_labels.flatten()

    
    train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
    dev_labels = np.reshape(dev_labels, (dev_labels.shape[0], 1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

    train_labels = to_categorical(train_labels)
    dev_labels = to_categorical(dev_labels)
    test_labels = to_categorical(test_labels)

    from sklearn.model_selection import train_test_split

    y_new = np.copy(y)
    y_new = np.reshape(y_new, (len(y_new), ))
    X_train, X_test, y_train, y_test = train_test_split(spectra, y_new, test_size=0.2, stratify=y_new)
    # clf.fit(train_set, train_labels)
    clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    # preds = clf.predict(test_set)
    # print("Accuracy:", accuracy_score(test_labels, preds))
    preds = clf.predict(X_test)
#     print("Accuracy:", accuracy_score(y_test, preds))
    print(accuracy_score(y_test, preds))

