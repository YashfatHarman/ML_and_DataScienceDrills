
# coding: utf-8

# ## Locally conncted with no weights shared 
# in the first three layers, where each input/neuron is connected to the neurons in a local neighbor in the next layer

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
#import keras

import time
import pickle

#get_ipython().run_line_magic('matplotlib', 'inline')

print("All imported successfully!")


# In[2]:


#Read data

train_file = "Data/zip_train.txt"

train_data = pd.read_csv(train_file, sep=' ', header = None)

#drop last column
train_data.drop([257], axis = 1, inplace = True)

print(train_data.shape)


# In[3]:


#read test file

test_file = "Data/zip_test.txt"

test_data = pd.read_csv(test_file, sep=' ', header = None)

test_data.drop([1], axis = 1, inplace = True)

print(test_data.shape)


# In[5]:


#get x and y datasets

x_train = (train_data.iloc[:, 1:].values).astype("float32")
y_train = (train_data.iloc[:, 0].values).astype("int32")
x_test = (test_data.iloc[:, 1:].values).astype("float32")
y_test = (test_data.iloc[:, 0].values).astype("int32")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[6]:


#for Dense, input reshaping is not needed. For CNN, reshape the model here.
X_train = x_train.reshape(x_train.shape[0], 16, 16, 1)
X_test = x_test.reshape(x_test.shape[0], 16, 16, 1)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# In[7]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, LocallyConnected2D
from sklearn.model_selection import train_test_split

print("all imported.")


# In[10]:


num_classes = 10

input_shape = (16,16,1)
print("input_shape:", input_shape)

batch_size = 64
epochs = 20


# In[12]:


#convert class vectors to binary clas metrices with one-hot encoding
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)
print(y_train_cat.shape)
print(y_test_cat.shape)


X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train_cat, test_size = 0.1, random_state = 42)
print("X_train_train shape:", X_train_train.shape)
print("y_train_train shape:",y_train_train.shape)
print("X_train_val shape:",X_train_val.shape)
print("y_train_val shape:",y_train_val.shape)


# In[20]:




model3 = Sequential()

model3.add(Conv2D(4, (11, 11), input_shape = input_shape, activation = "tanh", kernel_initializer = "he_normal"))

model3.add(Flatten())
model3.add(Dense(num_classes, activation = "softmax"))

model3.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.RMSprop(), 
              metrics = ["accuracy"])

print("model3 created.")

model3.summary()


# ## trainings are done in Tesla. So comment the block here.

# In[14]:



#now train it. Save the weights and history.

start_time = time.time()

h = model3.fit(X_train_train, y_train_train, batch_size = batch_size, epochs = epochs, validation_data = (X_train_val, y_train_val), verbose = 1)

model3.save_weights("model3_weights.h5")
pickle.dump(h.history, open("model3_history.pk","wb"))

end_time = time.time()

print("total training time: ", end_time - start_time)


# In[15]:


#traing done. So load the weights and test.

