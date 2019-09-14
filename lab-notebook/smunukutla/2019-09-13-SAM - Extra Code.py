#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# splitting the dataset

# num_samples = len(os.listdir(os.getcwd()))

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

# train_set = spectra[train_set_indices, :]
# train_labels = y[train_set_indices, :]
# dev_set = spectra[dev_set_indices, :]
# dev_labels = y[dev_set_indices, :]
# test_set = spectra[test_set_indices, :]
# test_labels = y[test_set_indices, :]

