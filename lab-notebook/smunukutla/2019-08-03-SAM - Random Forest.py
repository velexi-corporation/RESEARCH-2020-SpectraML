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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as st

# directory = "/Users/Srikar/Desktop/Velexi/spectra-ml/data/plots"
data_dir = os.environ['DATA_DIR']
data_dir = os.path.join(data_dir, "plots")
os.chdir(data_dir)


# In[2]:


num_samples = len(os.listdir(os.getcwd()))
img = mpimg.imread(os.path.join(data_dir,os.listdir(os.getcwd())[0]))
spectrum_height = img.shape[0]
spectrum_width = img.shape[1]


# In[3]:


def convertimg(img):
    newimg = np.empty([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            row = img[i][j]
            newimg[i][j] = (row[0] + row[1] + row[2])/3
    return newimg


# In[4]:


data = pd.read_csv("/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla/data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[5]:


spectra = np.zeros((num_samples, spectrum_height, spectrum_width))
i = 0
for num in record_nums:
    img = plt.imread(os.path.join(data_dir,num+"-"+spectrum_names[i]+".png")) # os.path.join here, look into timeit, pickle file
    spectra[i] = convertimg(img)
    i += 1


# In[6]:


spectra = spectra.reshape(spectra.shape[0], spectra.shape[1]*spectra.shape[2])


# In[7]:


os.chdir("/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla")
fi = open("indices.txt", "r")

stats = []

for i in range(20):
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
    
    clf = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='entropy')
    
    # clf.fit(train_set, train_labels)
    clf.fit(train_set, train_labels)
    
    # preds = clf.predict(test_set)
    # print("Accuracy:", accuracy_score(test_labels, preds))
    preds = clf.predict(test_set)
#     print("Accuracy:", accuracy_score(y_test, preds))
    stats.append(accuracy_score(test_labels, preds))

print("Random Forest:", stats) # add averages of the accuracy


# In[8]:


stats


# In[10]:


print("Random Forest Results:", st.describe(stats))


# In[ ]:




