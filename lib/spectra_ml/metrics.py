# module defines functions to
#- split data randomly into train, dev, test sets by index
#- split train set into bootstrap sets
#- run a model several

# external libs
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

def bootstrap():
    foo = 3
    return foo
