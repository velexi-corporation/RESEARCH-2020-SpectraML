# module defines functions to
#- split data randomly into train, dev, test sets by index
#- split train set into bootstrap sets
#- run a model several

# external libs
import numpy as np
import random

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import keras

def addchannel(X):
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    #train_set = np.reshape(train_set, (train_set.shape[0], train_set.shape[1], 1))
    #dev_set = np.reshape(dev_set, (dev_set.shape[0], train_set.shape[1], 1))
    return X

def split(num_samples):
    sample_indices = list(range(0, num_samples))
    random.seed(None)
    random.shuffle(sample_indices)
    train_population_size = 3*(num_samples//5) + (num_samples - 5*(num_samples//5))
    dev_population_size = (num_samples//5)
    test_population_size= num_samples-dev_population_size - train_population_size
    train_population_indices = sample_indices[0:train_population_size]
    dev_population_indices = sample_indices[train_population_size: train_population_size+dev_population_size]
    test_population_indices= sample_indices[train_population_size+dev_population_size: num_samples]

    return (train_population_indices, dev_population_indices, test_population_indices)

#TODO
def test(model,X,y,num_epochs, batch_size):
    spectra = X                 #name support
    num_samples = X.shape[0]

    X_old = X
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    #y_old = y
    #y = keras.utils.to_categorical(y)

    #split data into populations
    itrain,idev,itest = split(num_samples)
    train_population_indices = itrain   #name support
    dev_set = idev                      #name support
    BATCH_SIZE=batch_size               #name support
    EPOCHS = num_epochs                 #name support

    History = model.fit(train_set, train_set_labels, batch_size=BATCH_SIZE,\
    epochs=EPOCHS, verbose=1, validation_data=(dev_set, dev_set_labels))
    acc = History.history['acc']
    bresults[itrainresults, run] = acc[num_epochs-1]
    #record final epoch, plain nn train result, acc[...]), for each run

    # test on dev set
    dev_loss, dev_acc = model.evaluate(dev_set, dev_set_labels)
    test_loss, test_acc = model.evaluate(test_set, test_set_labels)
    return test

def bootstrap(model,X,y,num_epochs, batch_size, num_bootstrap_runs):
# --- bootstrap loop

# -select bootstrap sets
# -run algorithm
# -repeat num_bootstrap_runs times
# -collect mean and variance of train dev accuracies
# and store in bresults for run in range(num_bootstrap_runs)

    #read data
    spectra = X                 #name support
    num_samples = X.shape[0]

    y_old = y
    y = keras.utils.to_categorical(y)

    #split data into populations
    itrain,idev,itest = split(num_samples)
    train_population_indices = itrain   #name support
    dev_set_indices = idev              #name support
    BATCH_SIZE=batch_size               #name support
    EPOCHS = num_epochs                 #name support

    #array for results
    num_tests = 2                       #train, dev
    itrainresults = 0                          #row index of train results
    idevresults = 1                            #row index of dev results
    bresults = np.zeros((num_tests, num_bootstrap_runs))

    ## TODO:
    #handle model

    for run in range(num_bootstrap_runs):
        # - bootstrap sample the populations to make sets
        # make train, dev, and test sets from their respective populations

        # draw with replacement from the train population
        # make the validation and test sets the same as their populations
        train_set_indices = random.choices(train_population_indices, k=len(train_population_indices))

        # make train and test sets
        train_set = spectra[train_set_indices, :]
        train_set_labels = y[train_set_indices, :]
        dev_set = spectra[dev_set_indices, :]
        dev_set_labels = y[dev_set_indices,:]

        train_set = np.reshape(train_set, (train_set.shape[0], train_set.shape[1], 1))
        dev_set = np.reshape(dev_set, (dev_set.shape[0], train_set.shape[1], 1))


        # train
        History = model.fit(train_set, train_set_labels, batch_size=BATCH_SIZE,\
         epochs=EPOCHS, verbose=1, validation_data=(dev_set, dev_set_labels))
        acc = History.history['acc']
        bresults[itrainresults, run] = acc[num_epochs-1]
        #record final epoch, plain nn train result, acc[...]), for each run

        # test on dev set
        dev_loss, dev_acc = model.evaluate(dev_set, dev_set_labels)
        bresults[idevresults,run] = dev_acc

    return bresults

def bootstrap2(model,X,y,num_epochs, batch_size, num_bootstrap_runs):
# --- bootstrap loop

# -select bootstrap sets
# -run algorithm
# -repeat num_bootstrap_runs times
# -collect mean and variance of train dev accuracies
# and store in bresults for run in range(num_bootstrap_runs)

    #read data
    spectra = X                 #name support
    num_samples = X.shape[0]

    #y_old = y
    #y = keras.utils.to_categorical(y)

    #split data into populations
    itrain,idev,itest = split(num_samples)
    train_population_indices = itrain   #name support
    dev_population_indices = idev        #name support
    dev_set_indices = dev_population_indices
    BATCH_SIZE=batch_size               #name support
    EPOCHS = num_epochs                 #name support

    #array for bootstrap results
    num_tests = 2                       #train, dev
    itrainresults = 0                          #row index of train results
    idevresults = 1                            #row index of dev results
    bresults = np.zeros((num_tests, num_bootstrap_runs))

    ## TODO:
    #handle model

    for run in range(num_bootstrap_runs):
        # - bootstrap sample the populations to make sets
        # make train, dev, and test sets from their respective populations

        # draw with replacement from the train population
        # make the validation and test sets the same as their populations
        train_set_indices = random.choices(train_population_indices, k=len(train_population_indices))
        dev_set_indices = dev_population_indices

        # make train and test sets
        train_set = spectra[train_set_indices, :]
        train_set_labels = y[train_set_indices, :]
        dev_set = spectra[dev_set_indices, :]
        dev_set_labels = y[dev_set_indices,:]

        #train_set = np.reshape(train_set, (train_set.shape[0], train_set.shape[1], 1))
        #dev_set = np.reshape(dev_set, (dev_set.shape[0], train_set.shape[1], 1))


        # train
        History = model.fit(train_set, train_set_labels, batch_size=BATCH_SIZE,\
         epochs=EPOCHS, verbose=1, validation_data=(dev_set, dev_set_labels))
        acc = History.history['acc']
        bresults[itrainresults, run] = acc[num_epochs-1]
        #record final epoch, plain nn train result, acc[...]), for each run

        # test on dev set
        dev_loss, dev_acc = model.evaluate(dev_set, dev_set_labels)
        bresults[idevresults,run] = dev_acc

    return bresults

#TODO
def bstats(bresults_array):

    return 5
