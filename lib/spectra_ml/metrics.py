# module defines functions to
#- split data randomly into train, dev, test sets by index
#- split train set into bootstrap sets
#- run a model several

# external libs
import numpy as np
import random

def split(num_samples):
    sample_indices = list(range(0, num_samples))
    random.seed(0)
    random.shuffle(sample_indices)
    train_population_size = 3*(num_samples//5) + (num_samples - 5*(num_samples//5))
    dev_population_size = (num_samples//5)
    test_population_size= num_samples-dev_population_size - train_population_size
    train_population_indices = sample_indices[0:train_population_size]
    dev_population_indices = sample_indices[train_population_size: train_population_size+dev_population_size]
    test_population_indices= sample_indices[train_population_size+dev_population_size: num_samples]

    return (train_population_indices, dev_population_indices, test_population_indices)

#TODO
def aggregate()

#TODO
def bootstrap(model,X,y)
# --- bootstrap loop

# -select bootstrap sets
# -run algorithm
# -repeat num_bootstrap_runs times
# -collect mean and variance of train dev accuracies
# and store in bresults for run in range(num_bootstrap_runs)

    #read data
    spectra = X                 #name support
    num_samples = X.shape[0]

    #split data into populations
    itrain,idev,itest = split(num_samples)
    train_population_indices = itrain   #name support
    dev_set_indices = idev              #name support

    #array for results
    num_tests = 2                       #train, dev
    itrain = 0                          #row index of train results
    idev = 1                            #row index of dev results
    bresults = np.zeros((num_tests, num_bootstrap_runs))

    ## TODO:
    #handle model

    for run in range(num_bootstrap_runs):
        # - bootstrap sample the populations to make sets
        # make train, dev, and test sets from their respective populations

        # draw with replacement from the train population
        # make the validation and test sets the same as their populations
        train_set_indices = random.choices(train_population_indices, k=train_set_size)
        dev_set_indices = dev_population_indices
        test_set_indices = test_population_indices

        # make train and test sets
        train_set = spectra[train_set_indices, :]
        train_set_labels = y[train_set_indices, :]
        dev_set = spectra[dev_set_indices, :]
        dev_set_labels = y[dev_set_indices,:]
        test_set = spectra[test_set_indices, :]
        test_set_labels = y[test_set_indices, :]

        # use model from test cell

        # train
        History = model.fit(train_set, train_set_labels, epochs=num_epochs)

        acc = History.history['acc']
        bresults[itrain, run] = acc[num_epochs-1]   #record final epoch, plain nn train result, acc[...]), for each run

        #testcode
        #print(bruns)    # --- test plain nn on dev set

        # test on dev set
        dev_loss, dev_acc = model.evaluate(dev_set, dev_set_labels)
        bresults[idev,run] = dev_acc

        return bresults
