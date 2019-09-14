#!/usr/bin/env python
# coding: utf-8

# In[48]:


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


# In[49]:


num_samples = len(os.listdir(os.getcwd()))
img = mpimg.imread(os.path.join(directory,os.listdir(os.getcwd())[0]))
spectrum_height = img.shape[0]
spectrum_width = img.shape[1]


# In[50]:


def convertimg(img):
    newimg = np.empty([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            row = img[i][j]
            newimg[i][j] = (row[0] + row[1] + row[2])/3
    return newimg


# In[51]:


data = pd.read_csv("/Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla/data.csv", sep=",")
record_nums = data.iloc[0, :].astype(int).tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[54]:


os.listdir()


# In[53]:


spectra = np.zeros((num_samples, spectrum_height, spectrum_width))
y = []
i = 0
for name in os.listdir():
    print(name)
    if name.find("Actinolite") != -1:
        y.append(int(0))
    elif name.find("Alunite") != -1:
        y.append(int(1))
    else:
        y.append(int(2))
    img = plt.imread(os.path.join(directory,name)) # os.path.join here, look into timeit, pickle file
    spectra[i] = convertimg(img)
    i += 1


# In[32]:


spectra = spectra.reshape(spectra.shape[0], spectra.shape[1]*spectra.shape[2])


# In[33]:


spectra.shape


# In[34]:


y = np.reshape(y, (len(y), 1))
y.shape


# In[42]:


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
        
    print(train_set_indices)
    print(test_set_indices)
    print(dev_set_indices)
    
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
    clf = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='entropy')

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
    print("Accuracy:", accuracy_score(y_test, preds))


# In[24]:


# train_labels = train_labels.flatten()
# dev_labels = dev_labels.flatten()
# test_labels = test_labels.flatten()


# In[25]:


# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(bootstrap=True, criterion='entropy')


# In[26]:


# from sklearn.model_selection import train_test_split

# y = np.reshape(y, (len(y), ))
# X_train, X_test, y_train, y_test = train_test_split(spectra, y, test_size=0.2, random_state=42)
# # clf.fit(train_set, train_labels)
# clf.fit(X_train, y_train)


# In[27]:


# from sklearn.metrics import accuracy_score
# # preds = clf.predict(test_set)
# # print("Accuracy:", accuracy_score(test_labels, preds))
# preds = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, preds))


# In[ ]:




