#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as st


# In[2]:


# directory = "/Users/Srikar/Desktop/Velexi/spectra-ml/data/plots"
spectrum_len = 500
data_dir = os.environ['DATA_DIR']
parent_dir = os.environ['PWD']
plots_dir = os.path.join(data_dir, "plots")
stddata_path = os.path.join(data_dir, "StdData-" + str(spectrum_len))
os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[3]:


img = mpimg.imread(os.path.join(plots_dir, os.listdir(plots_dir)[0]))
spectrum_height = img.shape[0]
spectrum_width = img.shape[1]


# In[4]:


def convertimg(img):
    newimg = np.empty([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            row = img[i][j]
            newimg[i][j] = (row[0] + row[1] + row[2])/3
    return newimg


# In[5]:


data = pd.read_csv("data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[6]:


spectra = np.zeros((num_samples, spectrum_height, spectrum_width))
i = 0
for num in record_nums:
    img = plt.imread(os.path.join(plots_dir, num + "-" + spectrum_names[i] + ".png")) # look into timeit, pickle file
    spectra[i] = convertimg(img)
    i += 1


# In[7]:


spectra = spectra.reshape(spectra.shape[0], spectra.shape[1], spectra.shape[2], 1)
spectra.shape


# In[ ]:


fi = open("indices.txt", "r")
num_runs = int(fi.readline())
num_minerals = int(fi.readline())

stats = []

for i in range(num_runs):
    train_set_indices = ast.literal_eval(fi.readline())
    test_set_indices = ast.literal_eval(fi.readline())
    dev_set_indices = ast.literal_eval(fi.readline())

    for j in train_set_indices:
        j = int(j)
    for k in test_set_indices:
        k = int(k)
    for m in dev_set_indices:
        m = int(m)
    
    train_set = spectra[train_set_indices, :]
    dev_set = spectra[dev_set_indices, :]
    test_set = spectra[test_set_indices, :]
    
    train_labels = y_cat[train_set_indices, :]
    dev_labels = y_cat[dev_set_indices, :]
    test_labels = y_cat[test_set_indices, :]
    
    model = Sequential()
    #add model layers
    model.add(Conv2D(32, kernel_size=10, strides=(6,6), activation='relu', input_shape=(spectra.shape[1],spectra.shape[2], 1))) # finer features at the first layer
    model.add(Conv2D(32, kernel_size=3, activation='relu')) # larger features at later layer
    model.add(Flatten())
    
    from tensorflow.keras import backend as K

    # with a Sequential model
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[3].output])
    layer_output = get_3rd_layer_output(dev_set)
    
    clf = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='entropy')
    
    # clf.fit(train_set, train_labels)
    clf.fit(train_set, train_labels)
    
    # preds = clf.predict(test_set)
    # print("Accuracy:", accuracy_score(test_labels, preds))
    preds = clf.predict(test_set)
#     print("Accuracy:", accuracy_score(y_test, preds))
    stats.append(accuracy_score(test_labels, preds))

print("2D CNN + Random Forest Results:", st.describe(stats))

