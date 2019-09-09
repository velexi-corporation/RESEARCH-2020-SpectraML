#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import random
import ast

# directory = "/Users/Srikar/Desktop/Velexi/spectra-ml/data/plots"
directory = os.environ['DATA_DIR']
directory = os.path.join(directory, "plots")
os.chdir(directory)


# In[8]:


# for i in range(10):
#     print(random.randint(0, 1000))


# In[9]:


num_samples = len(os.listdir(os.getcwd()))
img = mpimg.imread(os.path.join(directory,os.listdir(os.getcwd())[0]))
spectrum_height = img.shape[0]
spectrum_width = img.shape[1]


# In[10]:


def convertimg(img):
    newimg = np.empty([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            row = img[i][j]
            newimg[i][j] = (row[0] + row[1] + row[2])/3
    return newimg


# In[11]:


spectra = np.zeros((num_samples, spectrum_height, spectrum_width))
y = []
i = 0
for name in os.listdir():
#     print(name)
    if name.find("Actinolite") != -1:
        y.append(int(0))
    elif name.find("Alunite") != -1:
        y.append(int(1))
    else:
        y.append(int(2))
    img = plt.imread(os.path.join(directory,name)) # os.path.join here, look into timeit, pickle file
    spectra[i] = convertimg(img)
    i += 1


# In[12]:


spectra = spectra.reshape(spectra.shape[0], spectra.shape[1]*spectra.shape[2])


# In[32]:


spectra.shape


# In[33]:


y = np.reshape(y, (len(y), 1))
y.shape


# In[34]:


# random.seed(0)


# In[35]:


# sample_indices = list(range(0, num_samples))
# print(num_samples)
# random.shuffle(sample_indices)
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

# fi = open("indices.txt", "r")

# train_set = spectra[train_set_indices, :]
# train_labels = y[train_set_indices, :]
# dev_set = spectra[dev_set_indices, :]
# dev_labels = y[dev_set_indices, :]
# test_set = spectra[test_set_indices, :]
# test_labels = y[test_set_indices, :]


# In[7]:


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

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(bootstrap=True, criterion='entropy')

    from sklearn.model_selection import train_test_split

    y = np.reshape(y, (len(y), ))
    X_train, X_test, y_train, y_test = train_test_split(spectra, y, test_size=0.2, random_state=42)
    # clf.fit(train_set, train_labels)
    clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    # preds = clf.predict(test_set)
    # print("Accuracy:", accuracy_score(test_labels, preds))
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))


# In[36]:


# train_labels = train_labels.flatten()
# dev_labels = dev_labels.flatten()
# test_labels = test_labels.flatten()


# In[37]:


# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(bootstrap=True, criterion='entropy')


# In[38]:


# from sklearn.model_selection import train_test_split

# y = np.reshape(y, (len(y), ))
# X_train, X_test, y_train, y_test = train_test_split(spectra, y, test_size=0.2, random_state=42)
# # clf.fit(train_set, train_labels)
# clf.fit(X_train, y_train)


# In[39]:


# from sklearn.metrics import accuracy_score
# # preds = clf.predict(test_set)
# # print("Accuracy:", accuracy_score(test_labels, preds))
# preds = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, preds))


# In[ ]:




