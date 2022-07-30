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
os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU(s) to be used

import keras
import tensorflow as tf
#from keras.applications import VGG16
from keras.engine import  Model
from keras.utils import Sequence
from keras.models import Sequential, load_model
from keras.layers import Activation, concatenate
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


images_train_path = 'dataset/all_classes_train/train'
images_validation_path = 'dataset/all_classes_validation/validation'

audios_train_path = 'audio_dataset/audio_train'
audios_validation_path = 'audio_dataset/audio_validation'

#--------------------------------------------------------------------------------------------------------------

# we load trained nn vgg with faces
print("[INFO] loading trained vgg model...")
vgg_model = load_model('vgg_faces[ALL_LAYERS].h5')

# we load trained nn with audio
print("[INFO] loading trained nn with audio...")
audio_model = load_model('audio_model.h5')

vgg_dense32 = vgg_model.get_layer('fc9').output
audio_dense32 = audio_model.get_layer('fc3').output

models_fusion = concatenate(([vgg_dense32, audio_dense32]), axis=1, name='vgg_audio_fusion')

dense512 = Dense(512, activation='relu', name='d1')(models_fusion)
dense512_1 = Dense(512, activation='relu', name='d2')(dense512)
dense32 = Dense(32, activation='relu', name='d3')(dense512_1)

output = Dense(5, activation='sigmoid', name='output')(dense32)

# two inputs to this nn, face image and audio
vgg_audio_model = Model([vgg_model.input, audio_model.input], output)

vgg_audio_model.summary()

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # default parameters
adam = Adam(lr=1e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

print("[INFO] compiling model...")
vgg_audio_model.compile(adam, loss='mean_squared_error', metrics=['mse', 'mae'])

#--------------------------------------------------------------------------------------------------------------


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
    #Generates data for Keras
    #Sequence based data generator. Suitable for building data generator for training and prediction.
 
    def __init__(self, list_IDs, labels, images_path, audios_path, batch_size=32, dim=(224, 224), audio_shape=224938, output_size=5, n_channels=3, shuffle=True):
        #Initialization
        #:param list_IDs: list of all 'label' ids to use in the generator
        #:param labels: diccionary of image labels ()
        #:param images_path: path to images dataset location (train set or validation set)
        #:param audios_path: path to audios dataset location (train set or validation set)
        #:param batch_size: batch size at each iteration
        #:param dim: tuple indicating image dimension
        #:param n_channels: number of image channels
        #:param n_classes: number of output masks
        #:param shuffle: True to shuffle label indexes after every epoch
        
        self.list_IDs = list_IDs
        self.labels = labels
        self.images_path = images_path
        self.audios_path = audios_path
        self.batch_size = batch_size
        self.dim = dim
        self.audio_shape = audio_shape
        self.output_size = output_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        #:return: number of batches per epoch
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #Generate one batch of data
        #:param index: index of the batch
        #:return: X and y
       
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X_1, X_2, y = self.__data_generation(list_IDs_temp)
        return [X_1, X_2], y

    def on_epoch_end(self):
        #Updates indexes after each epoch
       
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        #Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_1 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        X_2 = np.empty((self.batch_size, self.audio_shape))

        y = np.empty((self.batch_size, self.output_size), dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # INPUT IMAGE (we load and normalize image from image_path)
            img = cv2.imread( os.path.join(self.images_path, ID) + '.png' )
            img_n = mean_substraction(img)
            X_1[i,] = img_n

            # INPUT AUDIO ARRAY
            X_2[i,] = np.load(os.path.join(self.audios_path, ID.rsplit('-', 1)[0]) + '.npy')

            # OCEAN LABEL
            # in this case, OCEAN dicc has a tuple of values: (OCEAN values, features values)
            # so we only select the OCEAN values
            y[i] = self.labels[ID][0]

        return X_1, X_2, y

#--------------------------------------------------------------------------------------------------------------

train_IDs = [f.rsplit('.', 1)[0] for f in os.listdir(images_train_path)  if not f.startswith('.')]
validation_IDs = [f.rsplit('.', 1)[0] for f in os.listdir(images_validation_path)  if not f.startswith('.')]

train_generator = DataGenerator(train_IDs, train_OCEAN_features, images_train_path, audios_train_path)
validation_generator = DataGenerator(validation_IDs, validation_OCEAN_features, images_validation_path, audios_validation_path)

#--------------------------------------------------------------------------------------------------------------


# callback function to print output layer weights during training
print_output_weights = LambdaCallback( on_epoch_end=lambda epoch, logs: print(vgg_audio_model.layers[-1].get_weights()[0]) )

# callback function to keep best model weights
checkpoint = ModelCheckpoint(filepath='vgg_audio_model_weights.hdf5', monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')

# callback function to stop when the model do not improve whatever value we are monitoring
early_stopping_monitor = EarlyStopping(monitor='val_mean_squared_error', patience=6)

callbacks_list = [checkpoint]

# Train model on dataset
print("[INFO] training model...")
history = vgg_audio_model.fit_generator( generator = train_generator, validation_data = validation_generator, epochs = 70, verbose = 1, callbacks = callbacks_list )


# summarize history for mse
figure()
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('vgg_audio_model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_vgg_audio_model_MSE.png')

# summarize history for mae
figure()
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('vgg_audio_model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_vgg_audio_model_MAE.png')

# summarize history for loss
figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('vgg_audio_model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_vgg_audio_model_loss.png')

with open('vgg_audio_model_history.pkl', 'wb') as f:
    pickle.dump(history.history, f, protocol=2)

# serialize the model to disk
print("[INFO] serializing model to disk...")
vgg_audio_model.save("vgg_audio_model.h5")

