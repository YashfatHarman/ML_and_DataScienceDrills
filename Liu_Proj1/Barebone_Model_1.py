
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
#import keras

import time
import pickle

#get_ipython().magic('matplotlib inline')

print("All imported successfully!")


# In[27]:


#Read data

train_file = "Data/zip_train.txt"

train_data = pd.read_csv(train_file, sep=' ', header = None)
#print("train shape:", train_data.shape)
#print(type(train_data))
#print(train_data.head(3))


# In[28]:


#for some reason there is an extra column at the end where all the values seemed like nonzero. 
# We can safely drop that one, but just checking if it indeed has all nonzeroes, or if there are 
# any other missing value in any other column.

#print(train_data.isnull().sum(axis = 0).nonzero())


# In[29]:


#drop last column
train_data.drop([257], axis = 1, inplace = True)
#print("train shape:", train_data.shape)
#print(train_data.head(3))


# In[30]:


#read test file

test_file = "Data/zip_test.txt"

test_data = pd.read_csv(test_file, sep=' ', header = None)
#print("test shape:", test_data.shape)
#print(type(test_data))
#print(test_data.head(3))


# In[31]:


#check for null values
#print(test_data.isnull().sum(axis = 0).nonzero()) 
#test_data.isnull().sum(axis = 0)



# In[32]:


#this time the column 1 is null, so drop it.
test_data.drop([1], axis = 1, inplace = True)
#print("test shape:", test_data.shape)
#print(test_data.head(3))


# In[33]:


#count frequency of each label

#z_train = Counter(train_data[0])
#print(sorted(z_train.items()))


## In[34]:


#sns.countplot(train_data[0])


## In[35]:


#z_test = Counter(test_data[0])
#print(sorted(z_test.items()))


## In[36]:


#sns.countplot(test_data[0])


# In[37]:


#get x and y datasets

x_train = (train_data.iloc[:, 1:].values).astype("float32")
y_train = (train_data.iloc[:, 0].values).astype("int32")
x_test = (test_data.iloc[:, 1:].values).astype("float32")
y_test = (test_data.iloc[:, 0].values).astype("int32")

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)


# In[38]:


#get_ipython().magic('matplotlib inline')
##visualize some of the digits

#plt.figure(figsize = (12,10))
#x,y = 10,4
#for i in range(40):
#    plt.subplot(y,x,i+1)
#    plt.imshow(x_train[i].reshape((16,16)), interpolation = "nearest")
#plt.show()


# In[39]:


#reshape to match Keras's expectations
#not needed for the fully conencted layer

#X_train = x_train.reshape(x_train.shape[0], 16, 16, 1)
#X_test = x_test.reshape(x_test.shape[0], 16, 16, 1)
X_train = x_train
X_test = x_test
    
#print(X_train.shape)
#print(X_test.shape)


# In[40]:


#ok, lets start the fun

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, LocallyConnected2D
from sklearn.model_selection import train_test_split

print("all imported.")


# In[42]:


num_classes = 10
#input_shape = X_train.shape[1]
#print("input_shape:", input_shape)

batch_size = 64
epochs = 20


# In[43]:


#convert class vectors to binary clas metrices with one-hot encoding
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)
#print(y_train_cat.shape)
#print(y_test_cat.shape)


# In[45]:


X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train_cat, test_size = 0.1, random_state = 42)
print("X_train_train shape:", X_train_train.shape)
print("y_train_train shape:",y_train_train.shape)
print("X_train_val shape:",X_train_val.shape)
print("y_train_val shape:",y_train_val.shape)


# In[46]:


# first deep network. Need to have at least four layers, 
# relu must be used in at least one layer, sigmoid/tanh must be used in at least one.
# This one needs to be fully connected.

model1 = Sequential()

#lets try and if increasing the number of neurons help or not.
model1.add(Dense(20, activation = "tanh", input_shape = (X_train.shape[1],)))
model1.add(Dense(20, activation = "relu"))
model1.add(Dense(20, activation = "relu"))
#this offered 98.8% accuracy in training, 95.5% in validation, and 92% in testing.

model1.add(Dense(num_classes, activation = "softmax"))

model1.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.RMSprop(), 
              metrics = ["accuracy"])

print("model1 created.")

model1.summary()


# In[ ]:


#parameters: 256 * 10 + 10 = 2570, 10 * 10 + 10 = 110 etc.

#now train it. Save the weights and history.

start_time = time.time()

h = model1.fit(X_train_train, y_train_train, batch_size = batch_size, epochs = epochs, validation_data = (X_train_val, y_train_val), verbose = 1)

model1.save_weights("model1_weights.h5")
pickle.dump(h.history, open("model1_history.pk","wb"))

end_time = time.time()

print("total training time: ", end_time - start_time)

