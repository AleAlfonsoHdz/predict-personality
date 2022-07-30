from __future__ import print_function
import pickle
import csv
import numpy as np
from zipfile import ZipFile
import os
import shutil
import cv2
import math
import sys
import glob
import time
import random
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # specify which GPU(s) to be used

import keras
import tensorflow as tf
#from keras.applications import VGG16
from keras.engine import  Model
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation
from keras.layers.core import Dense,Flatten
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping

#--------------------------------------------------------------------------------------------------------------

def extract_picklefile(picklefile):
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
    return data

train_OCEAN_features = extract_picklefile('train_OCEAN_features.pkl')
validation_OCEAN_features = extract_picklefile('validation_OCEAN_features.pkl')
test_OCEAN_features = extract_picklefile('test_OCEAN_features.pkl')


train_size = len(train_OCEAN_features.values())
validation_size = len(validation_OCEAN_features.values())


train_path = 'dataset/all_classes_train/train'
validation_path = 'dataset/all_classes_validation/validation'




vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))

# we are gonna freeze the parameters
for layer in vgg_model.layers:
    layer.trainable = False

last_layer = vgg_model.get_layer('pool5').output


conv6 = Conv2D(512, kernel_size=1, activation='relu', name='conv6')(last_layer)
pool6 = MaxPooling2D(pool_size=(3, 3), name='pool6')(conv6)

flatten = Flatten(name='flatten1')(pool6)

dense1024 = Dense(1024, activation='relu', name='fc6')(flatten)
dense512 = Dense(512, activation='relu', name='fc7')(dense1024)
dense128 = Dense(128, activation='relu', name='fc8')(dense512)
dense32 = Dense(32, activation='relu', name='fc9')(dense128)


output = Dense(5, activation='sigmoid', name='output')(dense32)

vgg_faces = Model([vgg_model.input], output)

vgg_faces.summary()

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # default parameters
adam = Adam(lr=1e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

print("[INFO] compiling model...")
vgg_faces.compile(adam, loss='mean_squared_error', metrics=['mse', 'mae'])



# mean substraction by channel
# data_format : 'channels_last' (we are using Keras backend)
mean_face = load_img('mean_face.png')
mean_face_arr = img_to_array(mean_face)

mean_R = np.mean(mean_face_arr[..., 0])
mean_G = np.mean(mean_face_arr[..., 1])
mean_B = np.mean(mean_face_arr[..., 2])

def mean_substraction(img):

    x = img_to_array(img)
    x = x[..., ::-1]
    x[..., 0] -= mean_R # mean_R # 94.06396 
    x[..., 1] -= mean_G # mean_G # 101.61101
    x[..., 2] -= mean_B # mean_B # 129.55186

    return x

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, labels, dataset_path, batch_size=32, dim=(224, 224), output_size=5, n_channels=3, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: diccionary of image labels ()
        :param dataset_path: path to dataset location (train set or validation set)
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dim = dim
        self.output_size = output_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)"""
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.output_size), dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # INPUT IMAGE (we load and normalize image from image_path)
            img = cv2.imread( os.path.join(self.dataset_path, ID) + '.png' )
            img_n = mean_substraction(img)
            X[i,] = img_n

            # OCEAN LABEL
            # in this case, OCEAN dicc has a tuple of values: (OCEAN values, features values)
            # so we only select the OCEAN values
            y[i] = self.labels[ID][0]

        return X, y

#--------------------------------------------------------------------------------------------------------------
train_IDs = [f.rsplit('.', 1)[0] for f in os.listdir(train_path)  if not f.startswith('.')]
validation_IDs = [f.rsplit('.', 1)[0] for f in os.listdir(validation_path)  if not f.startswith('.')]

train_generator = DataGenerator(train_IDs, train_OCEAN_features, train_path)
validation_generator = DataGenerator(validation_IDs, validation_OCEAN_features, validation_path)

#train_generator = generator(train_path, train_OCEAN_features, batch_size)
#validation_generator = generator(validation_path, validation_OCEAN_features, batch_size)

#--------------------------------------------------------------------------------------------------------------

# callback function to print output layer weights during training
print_output_weights = LambdaCallback( on_epoch_end=lambda epoch, logs: print(vgg_faces.layers[-1].get_weights()[0]) )

# callback function to keep best model weights
checkpoint = ModelCheckpoint(filepath='vgg_faces_weights.hdf5', monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')

# callback function to stop when the model do not improve whatever value we are monitoring
early_stopping_monitor = EarlyStopping(monitor='val_mean_squared_error', patience=5)

callbacks_list = [checkpoint]

# Training only top layers (all the other layers are frozen)
print("[INFO] training only top layers...")
history = vgg_faces.fit_generator( generator = train_generator, validation_data = validation_generator, epochs = 50, verbose = 1, callbacks = callbacks_list)

# summarize history for mse
figure()
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model VGG_Faces mse [ONLY TOP LAYERS]')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_vggFace_mse[TOP_LAYERS].png')

# summarize history for mse
figure()
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model VGG_Faces mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_vggFace_mae[TOP_LAYERS].png')

# summarize history for loss
figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model VGG_Faces loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_vggFace_loss[TOP_LAYERS].png')

with open('vgg_faces_history[TOP_LAYERS].pkl', 'wb') as f:
    pickle.dump(history.history, f, protocol=2)

# serialize the model to disk
print("[INFO] serializing model to disk...")
vgg_faces.save("vgg_faces[TOP_LAYERS].h5")

# we are gonna unfreeze the rest of the network
for layer in vgg_faces.layers:
    layer.trainable = True

# for the changes to the model to take affect we need to recompile
print("[INFO] re-compiling model...")
vgg_faces.compile(Adam(lr=1e-05), loss='mean_squared_error', metrics=['mse', 'mae'])


# Training only top layers (all the other layers are frozen)
print("[INFO] training all the model...")
history = vgg_faces.fit_generator( generator = train_generator, validation_data = validation_generator, epochs = 20, verbose = 1, callbacks = callbacks_list)



# summarize history for mse
figure()
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model VGG_Faces mse [ALL LAYERS]')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_vggFace_mse[ALL_LAYERS].png')

# summarize history for mse
figure()
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model VGG_Faces mae [ALL LAYERS]')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_vggFace_mae[ALL_LAYERS].png')

# summarize history for loss
figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model VGG_Faces loss [ALL LAYERS]')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_vggFace_loss[ALL_LAYERS].png')

with open('vgg_faces_history[ALL_LAYERS].pkl', 'wb') as f:
    pickle.dump(history.history, f, protocol=2)


# serialize the model to disk
print("[INFO] serializing model to disk...")
vgg_faces.save("vgg_faces[ALL_LAYERS].h5")
