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
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from scipy import stats as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time


# In[2]:


spectrum_len = 500 # automate this
data_dir = os.environ['DATA_DIR']
parent_dir = os.environ['PWD']
stddata_path = os.path.join(data_dir, "StdData-" + str(spectrum_len))
plots_dir = os.path.join(data_dir, "plots")
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


# In[9]:


spectra[[0, 1], :].shape


# In[27]:


# fi = open("indices.txt", "r")
# num_runs = int(fi.readline())
# num_minerals = int(fi.readline())

# stats = []

# for i in range(num_runs):
#     train_set_indices = ast.literal_eval(fi.readline())
#     test_set_indices = ast.literal_eval(fi.readline())
#     dev_set_indices = ast.literal_eval(fi.readline())

#     for j in train_set_indices:
#         j = int(j)
#     for k in test_set_indices:
#         k = int(k)
#     for m in dev_set_indices:
#         m = int(m)
    
#     train_set = spectra[train_set_indices, :]
#     dev_set = spectra[dev_set_indices, :]
#     test_set = spectra[test_set_indices, :]
    
#     train_labels = y_cat[train_set_indices, :]
#     dev_labels = y_cat[dev_set_indices, :]
#     test_labels = y_cat[test_set_indices, :]
    
#     model = Sequential()
#     #add model layers
#     model.add(Conv2D(32, kernel_size=10, strides=(6,6), activation='relu', input_shape=(spectra.shape[1],spectra.shape[2], 1))) # finer features at the first layer
#     model.add(Conv2D(32, kernel_size=3, activation='relu')) # larger features at later layer
#     model.add(Flatten())
    
#     from tensorflow.keras import backend as K

#     # with a Sequential model
#     get_3rd_layer_output = K.function([model.layers[0].input],
#                                       [model.layers[3].output])
#     layer_output = get_3rd_layer_output(dev_set)
    
#     clf = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='entropy')
    
#     # clf.fit(train_set, train_labels)
#     clf.fit(train_set, train_labels)
    
#     # preds = clf.predict(test_set)
#     # print("Accuracy:", accuracy_score(test_labels, preds))
#     preds = clf.predict(test_set)
# #     print("Accuracy:", accuracy_score(y_test, preds))
#     stats.append(accuracy_score(test_labels, preds))

# print("2D CNN + Random Forest Results:", st.describe(stats))


# In[37]:


fi = open("indices.txt", "r")
num_runs = int(fi.readline())
num_minerals = int(fi.readline())

stats = []

init_time = time.time()

for i in range(1):
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
    model.add(Conv2D(64, kernel_size=5, strides=(6,6), activation='relu', input_shape=(spectra.shape[1],spectra.shape[2], 1))) # finer features at the first layer
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu')) # larger features at later layer
#     model.add(Conv2D(32, kernel_size=3, activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(num_minerals*5, activation='softmax'))
    model.add(Flatten())
    model.add(Dense(num_minerals*5, activation='softmax'))
#     model.add(Dropout(0.5))
    model.add(Dense(num_minerals, activation='softmax'))
    model.summary()
    
    flatten_ind = 2
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    BATCH_SIZE = 32
    EPOCHS = 25
    
    checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
    
    history = model.fit(train_set, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_data=(dev_set, dev_labels), callbacks=[checkpointer, tensorboard]).history
    
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
    
    np.save('features', features)
    
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
    
    predictions = clf.predict(test_features)
    
#     accuracy=accuracy_score(predictions , test_label_ids)
#     print('Accuracy:', accuracy*100, '%.')
#     # preds = clf.predict(test_set)
#     # print("Accuracy:", accuracy_score(test_labels, preds))
#     preds = clf.predict(test_set)
# #     print("Accuracy:", accuracy_score(y_test, preds))
    stats.append(accuracy_score(predictions, test_label_ids))

print("2D CNN + RF Results:", st.describe(stats))
total_seconds = time.time() - init_time
print(total_seconds)


# In[18]:


model.evaluate(test_set, test_labels, verbose=0)


# In[ ]:




