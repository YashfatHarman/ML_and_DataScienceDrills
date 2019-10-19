
# coding: utf-8

# In[31]:


'''
Just copying code from Keras Tutorial

#Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import pickle 
import numpy as np

print("All imported!")


# In[6]:


#set parameters

batch_size = 32
num_classes = 10
epochs = 100 #change to 100 in gpu
data_augmentation = False #change to True later
num_predictions = 20
save_dir = os.path.join(os.getcwd(), "saved_models")
model_name = "keras_cifar10_trained_model.h5"


# In[37]:


#get data
#(x_train, y_train),(x_test, y_test) = cifar10.load_data()
# x_train shape (50000, 32, 32, 3)
# y_train shape (50000, 1)
# x_test shape (10000, 32, 32, 3)
# y_test shape (10000, 1)

#lets load data from local drive
#for this, cifar10 needs to be downloaded from "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

data_dir = "cifar-10-batches-py"
data_files = [file for file in os.listdir(data_dir) if file.startswith("data")]
print(data_files)

test_files = [file for file in os.listdir(data_dir) if file.startswith("test")]
print(test_files)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

X_train =  []
y_train = []

for data_file in data_files:
    dic = unpickle(os.path.join(data_dir,data_file))
    data = dic[b"data"]
    labels = dic[b"labels"]
    X_train.append(data)
    y_train.append(labels)
    
X_train = np.concatenate(X_train, axis = 0)
print(X_train.shape)
y_train = np.concatenate(y_train, axis = 0)
print(y_train.shape)

X_test =  []
y_test = []

for test_file in test_files:
    dic = unpickle(os.path.join(data_dir,test_file))
    data = dic[b"data"]
    labels = dic[b"labels"]
    X_test.append(data)
    y_test.append(labels)
    
X_test = np.concatenate(X_test, axis = 0)
print(X_test.shape)
y_test = np.concatenate(y_test, axis = 0)
print(y_test.shape)



# In[38]:


#convert clas lables to binary class metrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y after converting to binary class metrics:")
print(y_train.shape)
print(y_test.shape)



# In[39]:


#reshape the X values to *,32,32,3
X_train = X_train.reshape(X_train.shape[0],32,32,3)
X_test = X_test.reshape(X_test.shape[0],32,32,3)
print("X after reshaping:")
print(X_train.shape)
print(X_test.shape)


# In[45]:


#create model

model = Sequential()

model.add(Conv2D(32, (3,3), padding = "same", input_shape = X_train.shape[1:], activation = "relu"))
model.add(Conv2D(32, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), padding = "same",  activation = "relu"))
model.add(Conv2D(32, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = "softmax"))

#initialize rmsprop optimizer
opt = keras.optimizers.rmsprop(lr = 0.0001, decay = 1e-6)

#train the model using rpsprop
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

model.summary()



# In[46]:


X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255



# In[47]:


if not data_augmentation:
    print("Not using data augmentation.")
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test, y_test), shuffle = True, verbose = 1)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        workers=4)

#save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print("saved trained model at {}".format(model_path) )

#score trained model
scores = model.evaluate(X_test, y_test, verbose = 1)
print("test loss:", scores[0])
print("test accuracy:", scores[1])

