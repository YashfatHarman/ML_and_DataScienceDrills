#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:37:35 2019

@author: rahman
"""

#import everything
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools

import time
import pickle
import datetime
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, LocallyConnected2D
from sklearn.model_selection import train_test_split

from keras import backend as K

import sys
from contextlib import contextmanager

print("All imported successfully!")

def set_globals():
    global num_classes
    #global batch_size
    global epochs
    global input_shape
    
    num_classes = 10
    #batch_size = 64
    epochs = 20
    
#read data
def prepare_data():
    '''
    Fixed train and test files. KISS.
    '''
    #Read data

    train_file = "Data/zip_train.txt"    
    train_data = pd.read_csv(train_file, sep=' ', header = None)
    
    #drop last column
    train_data.drop([257], axis = 1, inplace = True)
    
    
    test_file = "Data/zip_test.txt"    
    test_data = pd.read_csv(test_file, sep=' ', header = None)    
    #drop second column
    test_data.drop([1], axis = 1, inplace = True)
    
    x_train = (train_data.iloc[:, 1:].values).astype("float32")
    y_train = (train_data.iloc[:, 0].values).astype("int32")
    x_test = (test_data.iloc[:, 1:].values).astype("float32")
    y_test = (test_data.iloc[:, 0].values).astype("int32")
    
    #convert class vectors to binary clas metrices with one-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    #print(y_train_cat.shape)
    #print(y_test_cat.shape)

    x_train_train, x_train_val, y_train_train, y_train_val = train_test_split(x_train, y_train_cat, test_size = 0.1, random_state = 42)
    print("x_train_train shape:", x_train_train.shape)
    print("y_train_train shape:",y_train_train.shape)
    print("x_train_val shape:",x_train_val.shape)
    print("y_train_val shape:",y_train_val.shape)    
    print("x_test shape",x_test.shape)
    print("y_test shape",y_test.shape)
    print("data read.")
    
    return x_train_train, y_train_train, x_train_val, y_train_val, x_test, y_test



def create_model1(input_shape, kernel_initializer = "he_normal", lr = 0.1, decay = 0.0, momentum = 0.0):
    # first deep network. Need to have at least four layers, 
    # relu must be used in at least one layer, sigmoid/tanh must be used in at least one.
    
    model_name = "model1_deep"
    
    model1 = Sequential()
    
    model1.add(Dense(20, activation = "tanh", input_shape = (input_shape,), kernel_initializer=kernel_initializer))
    model1.add(Dense(20, activation = "relu", kernel_initializer=kernel_initializer))
    model1.add(Dense(20, activation = "relu", kernel_initializer=kernel_initializer))
    
    model1.add(Dense(num_classes, activation = "softmax"))
    
    sgd = keras.optimizers.SGD(lr = lr, decay = decay, momentum = momentum)
    #default: keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    
    model1.compile(loss = keras.losses.categorical_crossentropy, optimizer = sgd, 
                  metrics = ["accuracy"])
    
    print("model1 created.")
    
    #print(model1.summary())
    
    return model1, model_name

def create_model2(input_shape, kernel_initializer = "he_normal", lr = 0.1, decay = 0.0, momentum = 0.0):
    # Locally connected network without shared weights. Need to have at least four layers, 
    # relu must be used in at least one layer, sigmoid/tanh must be used in at least one.

    model_name = "model2_local_no_weight_share"
    
    model2 = Sequential()
    
    # loss: 0.0046 - acc: 0.9988 - val_loss: 0.0770 - val_acc: 0.9849
    #model2.add(LocallyConnected2D(8, (3, 3), input_shape = input_shape, activation = "tanh", kernel_initializer = "he_normal"))
    
    model2.add(LocallyConnected2D(8, (3, 3), input_shape = input_shape, activation = "tanh", kernel_initializer = kernel_initializer))
    model2.add(LocallyConnected2D(8, (3, 3), activation = "relu", kernel_initializer = kernel_initializer))
    model2.add(LocallyConnected2D(8, (3, 3), activation = "relu", kernel_initializer = kernel_initializer))
    
    model2.add(Flatten())
    model2.add(Dense(num_classes, activation = "softmax"))

    sgd = keras.optimizers.SGD(lr = lr, decay = decay,  momentum = momentum)
    #default: keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    
    model2.compile(loss = keras.losses.categorical_crossentropy, optimizer = sgd, 
              metrics = ["accuracy"])

    print("model2 created.")
    
    #print(model2.summary())
    
    return model2, model_name
    
def create_model3(input_shape, kernel_initializer = "he_normal", lr = 0.1, decay = 0.0 , momentum = 0.0):
    # Locally connected network with shared weights. So this is a CNN.
    # First let's just use Conv2D layers, then we'll add max pooling.
    
    # Need to have at least four layers, 
    # relu must be used in at least one layer, sigmoid/tanh must be used in at least one.
    
    
    model_name = "model3_local_weight_share_basically_CNN"
    
    model3 = Sequential()
    
    model3.add(Conv2D(4, (3, 3), input_shape = input_shape, activation = "tanh", kernel_initializer = kernel_initializer))
    model3.add(Conv2D(4, (3, 3), activation = "relu", kernel_initializer = kernel_initializer))
    model3.add(Conv2D(4, (3, 3), activation = "relu", kernel_initializer = kernel_initializer))
    
    model3.add(Flatten())
    model3.add(Dense(num_classes, activation = "softmax"))
    
    sgd = keras.optimizers.SGD(lr = lr, decay = decay, momentum = momentum)
    #default: keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    
    model3.compile(loss = keras.losses.categorical_crossentropy, optimizer = sgd, metrics = ["accuracy"])
    
    print("model3 created.")
    
    return model3, model_name
    



def get_datetime():
    #get the date and time and return as a string
    
    dt = datetime.datetime.now()
    
    dt_string = "_" + str(dt.year) + "_" + str(dt.month) + "_" + str(dt.day) + "_" + str(dt.hour) + "_" + str(dt.minute) + "_" + str(dt.second) + "_"
    
    return dt_string


@contextmanager
def stdout_redirected(new_stdout):
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout

def print_list(lst):
    for x in lst:
        print("{:.3f}".format(x), end = "  ")
    print()

def print_results(results):
    
    #open a file and write results in the the file as well.
    for element in results:
        print("model: ", element["name"])
        print("params: ", element["params"])
        print("accuracy: ", end = "")
        print_list(element["accuracy"])
        print("val_accuracy: ", end = "") 
        print_list(element["val_accuracy"])
        print("last epoch accuracy (test, val) : {:.4f}, {:.4f}".format(element["accuracy"][-1], element["val_accuracy"][-1]))  
        print()
    
    #print the results in a file named with date and time.
        
    pass
    
    

#run model
def run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val, batch_size = 64):
    print("running model ", model_name)
    start_time = time.time()
    
    if model_name == "model2_local_no_weight_share" or model_name == "model3_local_weight_share_basically_CNN":
        x_train_train = x_train_train.reshape(x_train_train.shape[0], 16, 16, 1)
        x_train_val = x_train_val.reshape(x_train_val.shape[0], 16, 16, 1)
    
    h = model.fit(x_train_train, y_train_train, batch_size = batch_size, epochs = epochs, validation_data = (x_train_val, y_train_val), verbose = 2)
    
    model.save_weights(model_name + "_weights.h5")
    pickle.dump(h.history, open("Pickles/" + model_name + get_datetime() + "history.pk","wb"))
    
    end_time = time.time()
    
    print("total training time: ", end_time - start_time)
    
    accuracy = h.history["acc"]
    val_accuracy = h.history["val_acc"]
    loss = h.history["loss"]
    val_loss = h.history["val_loss"]
    
    print("accuracy: ", end = "")
    print_list(accuracy)
    print("val_accuracy: ", end = "") 
    print_list(val_accuracy)
    print("last epoch accuracy (test, val) : {:.4f}, {:.4f}".format(accuracy[-1], val_accuracy[-1]))
    
    return accuracy, val_accuracy

#set parameters, run the models, collect results, show
def design_test_for_kernel_initializer(x_train_train, y_train_train, x_train_val, y_train_val):
    
    kernel_initializer_modes = {
            "ru_point_zero_five" : keras.initializers.RandomUniform(minval = -0.05, maxval = 0.05, seed = 42),
            "ru_one" : keras.initializers.RandomUniform(minval = -1.0, maxval = 1.0, seed = 42),
            "ru_five" : keras.initializers.RandomUniform(minval = -5.0, maxval = 5.0, seed = 42),
            "zeros" : keras.initializers.Zeros(),
            "he_normal" : keras.initializers.he_normal(seed = 42),
            "glorot_normal" : keras.initializers.glorot_normal(seed = 42),
    }

    
    
    results = []
    
    for kernel_initializer_name, kernel_initializer in kernel_initializer_modes.items():
        model, model_name = create_model1(input_shape = x_train_train.shape[1], kernel_initializer = kernel_initializer)
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val)
            
        results.append({"name": model_name, 
                       "params": "kernel_initializer_mode__" + kernel_initializer_name,
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy})
        
    for kernel_initializer_name, kernel_initializer in kernel_initializer_modes.items():
        model, model_name = create_model2(input_shape = (16,16,1), kernel_initializer = kernel_initializer)
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val)
            
        results.append({"name": model_name, 
                       "params": "kernel_initializer_mode__" + kernel_initializer_name,
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy})
        
    for kernel_initializer_name, kernel_initializer in kernel_initializer_modes.items():
        model, model_name = create_model3(input_shape = (16,16,1), kernel_initializer = kernel_initializer)
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val)
            
        results.append({"name": model_name, 
                       "params": "kernel_initializer_mode__" + kernel_initializer_name,
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy})
        
    with(open("Results/" + "parameter_initialization" + get_datetime() + ".txt","w")) as f:
        with stdout_redirected(f):
            print("final results for perameter initialization test. \n")
            print_results(results)

        
    pass
    

def design_test_for_learning_rate(x_train_train, y_train_train, x_train_val, y_train_val):
    
    learning_rates = [0.001, 0.01, 0.1, 1, 10]
    
 
    results = []
    
    for learning_rate in learning_rates:
        print("running model1 with learning rate ", learning_rate)
        model, model_name = create_model1(input_shape = x_train_train.shape[1], lr = learning_rate)
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val)
            
        results.append({"name": model_name, 
                       "params": "learning_rate__" + str(learning_rate),
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy})
#    Results for model 1: 
#            0.001 is too low (accuracy increasing and will reach to optimum but will need much more epochs), 
#            0.01 is okay,
#            0.1 is perfect for us,
#            1 is too high, accuracy starts fluctuating,
#            10 is way too high.
#   
        
        
    for learning_rate in learning_rates:
        print("running model2 with learning rate ", learning_rate)
        model, model_name = create_model2(input_shape = (16,16,1), lr = learning_rate)
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val)
            
        results.append({"name": model_name, 
                       "params": "learning_rate__" + str(learning_rate),
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy})
        
#            Results for model 2:
#            0.001 is too low, and no learning advancement at all.
#            0.01 is exact same (why?)
#            0.1 is okay
#            1 is perfect
#            10 is way too high   
        
        
    for learning_rate in learning_rates:
        print("running model3 with learning rate ", learning_rate)
        model, model_name = create_model3(input_shape = (16,16,1), lr = learning_rate)
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val)
            
        results.append({"name": model_name, 
                       "params": "learning_rate__" + str(learning_rate),
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy})

#            Results for model 3:
#            0.001 is too low, but learning is progressing and will eventually get there.
#            0.01 is okay
#            0.1 is perfect
#            1 is too high, almost no learning is one
#            10 is same.

        
    with(open("Results/" + "learning_rate" + get_datetime() + ".txt","w")) as f:
        with stdout_redirected(f):
            print("final results for perameter learning rate. \n")
            print_results(results)

    pickle.dump(results, open("Results/" + "learning_rate" + get_datetime() + "results.pk","wb"))
    
    pass


def design_test_for_learning_rate_dacay(x_train_train, y_train_train, x_train_val, y_train_val):
    '''
    After learning rate, lets play a little with diminishing learning rate, that is 
    learning rate decay. As we already saw, 0.1 works best for model 1 and 3, 
    1 works best for model 2. Let's start with somethign bigger that that and start
    decaying over time. So may be 0.5 and 5 might be good starting points. 
    '''
    
    learning_rates = [ 0.1, 1, 10]
    
    results = []
    
    outfilename = "Results/" + "learning_rate" + "_decay_" + get_datetime()
    
    for learning_rate in learning_rates:
        for decay in [0.0, learning_rate/epochs]:
            print("running model1 with learning rate {} decay {}".format( learning_rate, decay) )
            model, model_name = create_model1(input_shape = x_train_train.shape[1], lr = learning_rate, decay = decay)
            accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val)
                
            results.append({"name": model_name, 
                           "params": "learning_rate__" + str(learning_rate) + "__decay_" + str(decay),
                            "accuracy" : accuracy,
                            "val_accuracy" : val_accuracy})
        
    for learning_rate in learning_rates:
        for decay in [0.0, learning_rate/epochs]:
            print("running model1 with learning rate {} decay {}".format( learning_rate, decay) )
            model, model_name = create_model2(input_shape = (16,16,1), lr = learning_rate, decay = decay)
            accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val)
                
            results.append({"name": model_name, 
                           "params": "learning_rate__" + str(learning_rate) + "__decay_" + str(decay),
                            "accuracy" : accuracy,
                            "val_accuracy" : val_accuracy})
        
    for learning_rate in learning_rates:
        for decay in [0.0, learning_rate/epochs]:
            print("running model1 with learning rate {} decay {}".format( learning_rate, decay) )
            model, model_name = create_model3(input_shape = (16,16,1), lr = learning_rate, decay = decay)
            accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val)
                
            results.append({"name": model_name, 
                           "params": "learning_rate__" + str(learning_rate) + "__decay_" + str(decay),
                            "accuracy" : accuracy,
                            "val_accuracy" : val_accuracy})
        
        
    with(open( outfilename + ".txt","w")) as f:
        with stdout_redirected(f):
            print("final results for perameter learning rate. \n")
            print_results(results)

    pickle.dump(results, open(outfilename + "_results.pk","wb"))
    
    pass


'''
Now play with batch size
'''
def design_test_for_batch_size(x_train_train, y_train_train, x_train_val, y_train_val):
    '''
    Let's try with batch size of 1, 64, 256, 1024, and x_train_train.shape[0] (basically all data at a time.)
    '''
    
   # batch_sizes = [1, 64, 256, 1024] 
        # removing x_train_train.shape[0]
        # the test was stopping abruptly last time. May be the big batch size to blame?
    
    '''
    Finally got it working with batch size upto 512. Not sure what went wrong with sies bigger than that.
    Not worth spending more time on it I guess.
    
    For model1, batch size 1 was terrible. Batch size 64, 256, 512 all worked okay.
    64 quickly reached to equilibrium, 256 and 512 was slightly slower.
    512 did generalize better though, like no fluctuations with test data.
    
    For model2, batch size 1 fluctuated a lot. Specially for test data.
    64 was best, and worked quickly, generalized well.
    256 took time to rise, but eventually was heading to the right direction. Slow though.
    512 didn't work at all (not sure why.)
    Would it have worked better for larger epochs. Not likely, the accuracy was flat around 10% 
    for both test and train.
    
    For model3, batch size 1 didn't work.
    64 was best, and worked quickly.
    256 and 512 showed the same trend, thoguh slightly slower.
    
    512 was the slowest among three, but it generalized best. No fluctuations on test data.
    '''
    
    batch_sizes = [1, 64, 256, 512]

    results = []
    
    outfilename = "Results/" + "batch_size_" + get_datetime()
    
    pickle_file = open(outfilename + ".pk","wb")
    
    for batch_size in batch_sizes:
        print("running model1 with batch size {}".format( batch_size) )
        model, model_name = create_model1(input_shape = x_train_train.shape[1])
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val, batch_size = batch_size)
        
        result = {"name": model_name, 
                       "params": "batch_size__"  + str(batch_size),
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy}
        results.append(result)
        pickle.dump(result, pickle_file)
    
        print("running model2 with batch size {}".format( batch_size) )
        model, model_name = create_model2(input_shape = (16,16,1) )
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val, batch_size = batch_size)
                
        result = {"name": model_name, 
                       "params": "batch_size__"  + str(batch_size),
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy}
        results.append(result)
        pickle.dump(result, pickle_file)
    
        print("running model3 with batch size {}".format( batch_size) )
        model, model_name = create_model3(input_shape = (16,16,1))
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val, batch_size = batch_size)
                
        result = {"name": model_name, 
                       "params": "batch_size__"  + str(batch_size),
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy}
        results.append(result)
        pickle.dump(result, pickle_file)
    
    pickle_file.close()
        
        
        
#    with(open( outfilename + ".txt","w")) as f:
#        with stdout_redirected(f):
#            print("final results for batch size. \n")
#            print_results(results)
#
#    pickle.dump(results, open(outfilename + "_results.pk","wb"))
#   
        
        
    pass

def design_test_for_momentum(x_train_train, y_train_train, x_train_val, y_train_val):
    '''
    Commonly used momentum coefficient values are 0.5, 0.9, and 0.99. Using the best
    parameter initialization strategy, the best learning rate, and the best batch size you have found so far,
    experiment with the three different momentum values on the three networks you have and document
    the results.
    
    model 1:
        param init strategy: he_normal
        learning rate: 0.1
        batch size: 64
            
    
    model 2:
        param init strategy: he_normal
        learning rate: 1
        batch size: 64
            
    
    model 2:
        param init strategy: he_normal
        learning rate: 0.1
        batch size: 64
            
    '''
    
    '''
    Results:
        Model 1: momentum 0.5 worked best, 0.9 was almost as good. 0.99 didnt work at all.
        
        Model 2: momentum 0.5 worked great. 0.9 and 0.99 didnt work at all.
        
        Model 3: momentum 0.5 worked best, 0.9 was almost as good. 0.99 didnt work at all.
    '''
    results = []
    
    outfilename = "Results/" + "momentum_" + get_datetime()
    
    pickle_file = open(outfilename + ".pk","wb")
    
    momentums = [0.5, 0.9, 0.99]
    
    kernel_initializer = "he_normal"
    batch_size = 64
    for momentum in momentums:
        print("running model1 with momentum {}".format( momentum) )
        learning_rate = 0.1
        model, model_name = create_model1(input_shape = x_train_train.shape[1], kernel_initializer = kernel_initializer, lr = learning_rate, momentum = momentum)
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val, batch_size = batch_size)
        
        result = {"name": model_name, 
                       "params": "momentum__"  + str(momentum),
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy}
        results.append(result)
        pickle.dump(result, pickle_file)
    
        print("running model2 with momentum {}".format( momentum) )
        learning_rate = 1
        model, model_name = create_model2(input_shape = (16,16,1), kernel_initializer = kernel_initializer, lr = learning_rate, momentum = momentum)
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val, batch_size = batch_size)
                
        result = {"name": model_name, 
                       "params": "momentum__"  + str(momentum),
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy}
        results.append(result)
        pickle.dump(result, pickle_file)
    
        print("running model3 with momentum {}".format( momentum) )
        learning_rate = 0.1
        model, model_name = create_model3(input_shape = (16,16,1),  kernel_initializer = kernel_initializer, lr = learning_rate, momentum = momentum)
        accuracy, val_accuracy = run_model(model, model_name, x_train_train, y_train_train, x_train_val, y_train_val, batch_size = batch_size)
                
        result = {"name": model_name, 
                       "params": "momentum__"  + str(momentum),
                        "accuracy" : accuracy,
                        "val_accuracy" : val_accuracy}
        results.append(result)
        pickle.dump(result, pickle_file)
    
    pickle_file.close()
    
    
    
    pass

    
if __name__ == "__main__":
    print("Hello World!")
    os.makedirs("Results", exist_ok = True)
    os.makedirs("Pickles", exist_ok = True)
    
    set_globals()
    x_train_train, y_train_train, x_train_val, y_train_val, x_test, y_test = prepare_data()
    
    design_test_for_momentum(x_train_train, y_train_train, x_train_val, y_train_val)
    print("Done normally!")    
