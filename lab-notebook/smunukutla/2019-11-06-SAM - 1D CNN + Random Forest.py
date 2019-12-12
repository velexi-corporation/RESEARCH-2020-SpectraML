#!/usr/bin/env python
# coding: utf-8

# In[11]:


# environment set up
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score
import os
import random
import pandas as pd
import ast
from scipy import stats as st
import time


# In[12]:


# working folder = "/Users/Srikar/Desktop/Velexi/spectra-ml/data"

spectrum_len = 500 # automate this
parent_dir = os.environ['PWD']
stddata_path = os.path.join(os.environ['DATA_DIR'], "StdData-" + str(spectrum_len))
os.chdir(os.path.join(parent_dir, "lab-notebook", "smunukutla"))


# In[13]:


# data extraction

data = pd.read_csv("data.csv", sep=",")
record_nums = data.iloc[0, :].tolist()
spectrum_names = data.iloc[1, :].tolist()
y = data.iloc[2, :].astype(int).tolist()
y = np.reshape(y, (len(y), 1))
num_samples = len(y)


# In[14]:


spectra = np.zeros((num_samples,spectrum_len))

# wavelengths = np.zeros((1,spectrum_len))
# y = np.zeros((num_samples, 1))


# In[15]:


for i in range(len(record_nums)):
    data = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(record_nums[i])))
#     if i == 0:
#         wavelengths[i,:] = data.iloc[:, 0].to_numpy()
    spectra[i,:] = data.iloc[:, 1].to_numpy()


# In[16]:


y_cat = to_categorical(y)


# In[18]:


fi = open("indices.txt", "r")
num_runs = int(fi.readline())
num_minerals = int(fi.readline())

combo_stats = []
cnn_stats = []

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
#     train_labels = y[train_set_indices, :]
    dev_set = spectra[dev_set_indices, :]
#     dev_labels = y[dev_set_indices, :]
    test_set = spectra[test_set_indices, :]
#     test_labels = y[test_set_indices, :]

#     train_labels = train_labels.flatten()
#     dev_labels = dev_labels.flatten()
#     test_labels = test_labels.flatten()

    train_set = np.reshape(train_set, (train_set.shape[0], spectrum_len, 1))
    dev_set = np.reshape(dev_set, (dev_set.shape[0], spectrum_len, 1))
    test_set = np.reshape(test_set, (test_set.shape[0], spectrum_len, 1))

#     train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
#     dev_labels = np.reshape(dev_labels, (dev_labels.shape[0], 1))
#     test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

#     train_labels = to_categorical(train_labels)
#     dev_labels = to_categorical(dev_labels)
#     test_labels = to_categorical(test_labels)

    train_labels = y_cat[train_set_indices, :]
    dev_labels = y_cat[dev_set_indices, :]
    test_labels = y_cat[test_set_indices, :]
    
#     print(train_labels)
    
    model = Sequential() # tf upgrading to 2.0, after that we need to specify the dtype/construct all layers at once
    # model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
    model.add(Conv1D(32, 25, activation='relu', input_shape=(train_set.shape[1], 1))) # optional: , dtype=tf.dtypes.float64
    model.add(Conv1D(32, 25, activation='relu'))
    model.add(MaxPooling1D(4)) # 108 by 64 so far
#     model.add(Conv1D(100, 25, activation='relu'))
#     model.add(Conv1D(100, 25, activation='relu'))
#     model.add(MaxPooling1D(4))
    # model.add(Dropout(0.5))
    # model.add(GlobalAveragePooling1D())
    model.add(Flatten())
    model.add(Dense(num_minerals, activation='softmax'))
    
    flatten_ind = 3
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    BATCH_SIZE = 32
    EPOCHS = 25

    model.fit(train_set, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, validation_data=(dev_set, dev_labels))
    
    my_list = model.evaluate(test_set, test_labels, verbose=0)

#     print(tf.keras.losses.categorical_crossentropy())
    
#     print(model.predict(test_set))
    
#     print(model.layers[0].get_weights())

#     from tensorflow.keras import backend as K

#     # with a Sequential model
#     get_3rd_layer_output = K.function([model.layers[0].input],
#                                       [model.layers[7].output])
#     layer_output = get_3rd_layer_output(dev_set)
#     print(layer_output)
    
#     print(dev_labels[13])
#     print(dev_set[13])

    FC_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=flatten_ind).output)
#     FC_layer_model.summary()
#     print(model.input.shape)
#     print(model.get_layer(index=flatten_ind).output.shape)
    
    features = np.zeros(shape = (train_set.shape[0], model.layers[flatten_ind].output.shape[1]))
#     print(features.shape)
    for p in range(train_set.shape[0]):
        spectra_in = train_set[p]
        spectra_in = np.expand_dims(spectra_in, axis=0)
        FC_output = FC_layer_model.predict(spectra_in)
#         print(FC_output.shape)
        features[p] = FC_output
    
#     np.save('features', features)
    
    feature_col = []
    for p in range(model.layers[flatten_ind].output.shape[1]):
        feature_col.append("f_" + str(p))
    
    train_features = pd.DataFrame(data = features, columns = feature_col)
    feature_col = np.array(feature_col)
    
    train_label_ids = y[train_set_indices, :]
    train_class = list(np.unique(train_label_ids))
#     print('Training Features Shape:', train_features.shape)
#     print('Training Labels Shape:', train_label_ids.shape)
    
#     my_list = model.evaluate(test_set, test_labels, verbose=0)
    
#     stats.append(my_list[1])
    
#     print(model.layers[2].input)
#     print(model.layers[2].output.shape[1])

    clf = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='entropy')
    
#     # clf.fit(train_set, train_labels)
    clf.fit(train_features, train_label_ids)
    
    features_test = np.zeros(shape = (test_set.shape[0], model.layers[flatten_ind].output.shape[1]))
    for p in range(test_set.shape[0]):
        spectra_in = test_set[p]
        spectra_in = np.expand_dims(spectra_in, axis=0)
        FC_output = FC_layer_model.predict(spectra_in)
        features_test[p] = FC_output
    
    test_features = pd.DataFrame(data = features_test, columns = feature_col)
    feature_col = np.array(feature_col)
    
    test_label_ids = y[test_set_indices, :]
    test_class = list(np.unique(test_label_ids))
#     print('Test Features Shape:', test_features.shape)
#     print('Test Labels Shape:', test_label_ids.shape)
    
    cnn_preds = model.evaluate(test_set, test_labels, verbose=0)
    combo_predictions = clf.predict(test_features)
    
#     accuracy=accuracy_score(predictions , test_label_ids)
#     print('Accuracy:', accuracy*100, '%.')
#     # preds = clf.predict(test_set)
#     # print("Accuracy:", accuracy_score(test_labels, preds))
#     preds = clf.predict(test_set)
# #     print("Accuracy:", accuracy_score(y_test, preds))
    cnn_stats.append(cnn_preds[1])
    combo_stats.append(accuracy_score(combo_predictions, test_label_ids))

print("1D CNN Results:", st.describe(cnn_stats))
print("1D CNN + RF Results:", st.describe(combo_stats))
total_seconds = time.time() - init_time
print(total_seconds)


# In[ ]:




