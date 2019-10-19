
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
from subprocess import check_output
print(check_output(["ls", "Data"]).decode("utf8"))
#get_ipython().magic('matplotlib inline')

import time
import pickle

# In[5]:


#load dataset

train = pd.read_csv("Data/train.csv")
#print(train.shape)
#print(train.head(5))


# In[10]:


z_train = Counter(train["label"])
#print(sorted(z_train.items()))


# In[15]:


#sns.countplot(train["label"])


# In[16]:


#load the dataset for test

test = pd.read_csv("Data/test.csv")
#print(test.shape)
#print(test.head(5))


# In[34]:


x_train = (train.iloc[:, 1:].values).astype("float32")
y_train = (train.iloc[:, 0].values).astype("int32")
x_test = test.values.astype("float32")

print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)
print("x_test shape: ",x_test.shape)


# In[35]:


#get_ipython().magic('matplotlib inline')
#visualize some of the digits

#plt.figure(figsize = (12,10))
#x,y = 10,4
#for i in range(40):
#    plt.subplot(y,x,i+1)
#    plt.imshow(x_train[i].reshape((28,28)), interpolation = "nearest")
#plt.show()


# In[36]:


#normalizing the data
x_train = x_train/255.0
x_test = x_test / 255.0


# In[37]:


#y_train


# In[38]:


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[39]:


#reshape to match Keras's expectations

X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[44]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
print("all keras packages imported.")


# In[45]:


batch_size = 64
num_classes = 10
epochs = 10
input_shape = (28,28,1)


# In[46]:


#cnvert class vectros to binary clas metrices with one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)


# In[47]:


print("X_train shape: ",X_train.shape)
print("X_val shape: ",X_val.shape)
print("Y_train shape: ",Y_train.shape)
print("Y_val shape: ",Y_val.shape)


# In[61]:


#Linear Model

model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", input_shape = input_shape))
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal"))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(64, (3,3), activation = "relu", padding = "same", kernel_initializer = "he_normal"))
model.add(Conv2D(64, (3,3), activation = "relu", padding = "same", kernel_initializer = "he_normal"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation = "relu", padding = "same", kernel_initializer = "he_normal"))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(num_classes, activation = "softmax"))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.RMSprop(), metrics = ["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor = "val_acc", patience = 3, verbose = 1, factor = 0.5, min_lr = 0.0001)

datagen = ImageDataGenerator(
        featurewise_center = False, #set input mean to 0 over dataset
        samplewise_center = False, #set each sample mean to 0
        featurewise_std_normalization = False, #divide inputs by std of the dataset
        samplewise_std_normalization = False, #divide each intut by its std
        zca_whitening = False, #apply ZCA whitening
        rotation_range = 15,    #randomly rotate images in the range (egrees, 0 to 180)
        zoom_range = 0.1,       #randmly zoom image
        width_shift_range = 0.1, #randomly shift images horizontally (fraction to total width)
        height_shift_range = 0.1, #randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  #randomly flip images
        vertical_flip = False    #randomly flip images
    )

print("model created.")


# In[62]:


model.summary()


# In[64]:

start_time = time.time()

datagen.fit(X_train)
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size), epochs = epochs, validation_data = (X_val, Y_val), verbose = 1, steps_per_epoch = X_train.shape[0], callbacks = [learning_rate_reduction],)

model.save("model_epoch10.h5")
pickle.dump(h.history, open("history_epoch10.pk","wb"))

end_time = time.time()

print("total training time: ", end_time - start_time)


