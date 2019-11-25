#!/usr/bin/env python
# coding: utf-8

# In[79]:


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
import time


# In[80]:


# directory = "/Users/Srikar/Desktop/Velexi/spectra-ml/data/plots"
spectrum_len = 500
data_dir = os.environ['DATA_DIR']
parent_dir = os.environ['PWD']
plots_dir = os.path.join(data_dir, "plots-" + str(spectrum_len))
stddata_path = os.path.join(data_dir, "StdData-" + str(spectrum_len))
os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[81]:


img = mpimg.imread(os.path.join(plots_dir, os.listdir(plots_dir)[0]))
spectrum_height = img.shape[0]
spectrum_width = img.shape[1]


# In[82]:


def convertimg(img):
    newimg = np.empty([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            row = img[i][j]
            newimg[i][j] = (row[0] + row[1] + row[2])/3
    return newimg


# In[83]:


data = pd.read_csv("data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[84]:


start_time = time.time()
spectra = np.zeros((num_samples, spectrum_height, spectrum_width))
i = 0
for num in record_nums:
    img = plt.imread(os.path.join(plots_dir, num + "-" + spectrum_names[i] + ".png")) # os.path.join here, look into timeit, pickle file
    spectra[i] = convertimg(img)
    i += 1

end_time = time.time()
print(end_time - start_time)


# In[85]:


spectra = spectra.reshape(spectra.shape[0], spectra.shape[1]*spectra.shape[2])


# In[104]:


fi = open("indices.txt", "r")
num_runs = int(fi.readline())
num_minerals = int(fi.readline())

stats = []

init_time = time.time()

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
        
#     print(train_set_indices)
#     print(test_set_indices)
#     print(dev_set_indices)
    
    train_set = spectra[train_set_indices, :]
    train_labels = y[train_set_indices, :]
    dev_set = spectra[dev_set_indices, :]
    dev_labels = y[dev_set_indices, :]
    test_set = spectra[test_set_indices, :]
    test_labels = y[test_set_indices, :]
    
    train_plus_dev_set = spectra[train_set_indices+dev_set_indices, :]

    train_labels = train_labels.flatten()
    dev_labels = dev_labels.flatten()
    test_labels = test_labels.flatten()
    
    train_plus_dev_labels = y[train_set_indices+dev_set_indices, :]
    
    train_plus_dev_labels = train_plus_dev_labels.reshape(train_plus_dev_labels.shape[0],)
    print(train_plus_dev_labels.shape)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, bootstrap=True, criterion='entropy')
    
    # clf.fit(train_set, train_labels)
    clf.fit(train_plus_dev_set, train_plus_dev_labels)
    
    # preds = clf.predict(test_set)
    # print("Accuracy:", accuracy_score(test_labels, preds))
    preds = clf.predict(test_set)
#     print("Accuracy:", accuracy_score(y_test, preds))
    stats.append(accuracy_score(test_labels, preds))

print("Random Forest Results:", st.describe(stats))
total_seconds = time.time() - init_time
print(total_seconds)


# In[ ]:




