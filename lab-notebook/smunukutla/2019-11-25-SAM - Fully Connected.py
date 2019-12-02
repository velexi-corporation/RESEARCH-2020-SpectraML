#!/usr/bin/env python
# coding: utf-8

# In[34]:


# environment set up
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import time
import ast
from scipy import stats as st

# working folder
directory = os.environ['DATA_DIR']


# In[17]:


spectrum_len = 500 # automate this
parent_dir = os.environ['PWD']
stddata_path = os.path.join(directory, "StdData-" + str(spectrum_len))
os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[18]:


data = pd.read_csv("data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[19]:


spectra = np.zeros((num_samples,spectrum_len))


# In[20]:


for i in range(len(record_nums)):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
#     if i == 0:
#         wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()


# In[26]:


spectra.shape


# In[21]:


y_cat = to_categorical(y)


# In[35]:


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
    
    train_set = spectra[train_set_indices, :]
    train_labels = y_cat[train_set_indices, :]
    dev_set = spectra[dev_set_indices, :]
    dev_labels = y_cat[dev_set_indices, :]
    test_set = spectra[test_set_indices, :]
    test_labels = y_cat[test_set_indices, :]

#     train_set = np.reshape(train_set, (train_set.shape[0], spectrum_len, 1))
#     dev_set = np.reshape(dev_set, (dev_set.shape[0], spectrum_len, 1))
#     test_set = np.reshape(test_set, (test_set.shape[0], spectrum_len, 1))
    
#     print(train_set.shape)
#     print(train_labels.shape)
    
    model = Sequential() # tf upgrading to 2.0, after that we need to specify the dtype/construct all layers at once
    model.add(Dense(num_minerals*10, input_dim=train_set.shape[1], activation='relu'))
    model.add(Dense(num_minerals*3, activation='relu'))
    model.add(Dense(num_minerals, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    BATCH_SIZE = 32
    EPOCHS = 80
    
#     checkpointer = ModelCheckpoint(filepath="model.h5",
#                                verbose=0,
#                                save_best_only=True)
#     tensorboard = TensorBoard(log_dir='./logs',
#                           histogram_freq=0,
#                           write_graph=True,
#                           write_images=True)

#     history = model.fit(train_set, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, validation_data=(dev_set, dev_labels), callbacks=[checkpointer, tensorboard]).history
    model.fit(train_set, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, validation_data=(dev_set, dev_labels))
    
    predictions = model.evaluate(test_set, test_labels, verbose=0)
    
    stats.append(predictions[1])

print("Fully Connected ANN Results:", st.describe(stats))
total_seconds = time.time() - init_time
print(total_seconds)


# In[ ]:




