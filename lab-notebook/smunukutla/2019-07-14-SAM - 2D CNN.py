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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from scipy import stats as st
import time


# In[2]:


spectrum_len = 500 # automate this
data_dir = os.environ['DATA_DIR']
parent_dir = os.environ['PWD']
stddata_path = os.path.join(data_dir, "StdData-" + str(spectrum_len))
plots_dir = os.path.join(data_dir, "plots-" + str(spectrum_len))
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


start_time = time.time()
spectra = np.zeros((num_samples, spectrum_height, spectrum_width))
i = 0
for num in record_nums:
    img = plt.imread(os.path.join(plots_dir, num + "-" + spectrum_names[i] + ".png")) # look into timeit, pickle file
    spectra[i] = convertimg(img)
    i += 1

end_time = time.time()
print(end_time - start_time)


# In[7]:


spectra = spectra.reshape(spectra.shape[0], spectra.shape[1], spectra.shape[2], 1)
spectra.shape


# In[8]:


y_cat = to_categorical(y)


# In[10]:


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
#     train_labels = y[train_set_indices, :]
    dev_set = spectra[dev_set_indices, :]
#     dev_labels = y[dev_set_indices, :]
    test_set = spectra[test_set_indices, :]
#     test_labels = y[test_set_indices, :]

#     train_labels = to_categorical(train_labels)
#     dev_labels = to_categorical(dev_labels)
#     test_labels = to_categorical(test_labels) # take apart the input and the output
    
    train_labels = y_cat[train_set_indices, :]
    dev_labels = y_cat[dev_set_indices, :]
    test_labels = y_cat[test_set_indices, :]
    
    model = Sequential()
    #add model layers
    model.add(Conv2D(32, kernel_size=10, strides=(6,6), activation='relu', input_shape=(spectra.shape[1],spectra.shape[2], 1))) # finer features at the first layer
    model.add(Conv2D(32, kernel_size=3, activation='relu')) # larger features at later layer
    model.add(Flatten())
    model.add(Dense(num_minerals, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    BATCH_SIZE = 32
    EPOCHS = 25
    
#     checkpointer = ModelCheckpoint(filepath="model.h5",
#                                verbose=0,
#                                save_best_only=True)
#     tensorboard = TensorBoard(log_dir='./logs',
#                           histogram_freq=0,
#                           write_graph=True,
#                           write_images=True)
    
#     history = model.fit(train_set, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, validation_data=(dev_set, dev_labels), callbacks=[checkpointer, tensorboard]).history
    model.fit(train_set, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, validation_data=(dev_set, dev_labels))
    
    preds = model.evaluate(test_set, test_labels, verbose=0)
    
    stats.append(preds[1])

print("2D CNN Results:", st.describe(stats))
total_seconds = time.time() - init_time
print(total_seconds)


# In[ ]:


# model.layers[3].output


# In[ ]:


# loaded_model = load_model('model.h5')


# In[ ]:


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('2D CNN loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')


# In[ ]:


plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('2D CNN accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')


# In[ ]:


# model.save('2dcnn.h5')


# In[ ]:




