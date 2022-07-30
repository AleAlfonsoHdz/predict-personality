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
os.environ["CUDA_VISIBLE_DEVICES"]="3" # specify which GPU(s) to be used

import keras
import tensorflow as tf
#from keras.applications import VGG16
from keras.engine import  Model
from keras.utils import Sequence
from keras.models import Sequential, load_model
from keras.layers import Activation, concatenate
from keras.layers.core import Dense,Flatten
from keras.layers import Input
#from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping

#--------------------------------------------------------------------------------------------------------------

images_test_path = 'dataset/all_classes_test/test'
audios_test_path = 'audio_dataset/audio_test'

#--------------------------------------------------------------------------------------------------------------

# we load trained nn vgg_audio model
print("[INFO] loading trained vgg_audio model...")
vgg_audio_model = load_model('vgg_audio_model.h5')
vgg_audio_model.load_weights('vgg_audio_model_weights.hdf5') # loading best weights


vgg_audio_model.summary()


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
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, images_path, audios_path, batch_size=32, dim=(224, 224), audio_shape=224938, output_size=5, n_channels=3, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param images_path: path to images dataset location (train set or validation set)
        :param audios_path: path to audios dataset location (train set or validation set)
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
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
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return len(self.list_IDs) # only for predictions

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
        X_1, X_2 = self.__data_generation(list_IDs_temp)
        return [X_1, X_2]

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        #Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)"""
        # Initialization
        X_1 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        X_2 = np.empty((self.batch_size, self.audio_shape), dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # INPUT IMAGE (we load and normalize image from image_path)
            img = cv2.imread( os.path.join(self.images_path, ID) + '.png' )
            img_n = mean_substraction(img)
            X_1[i,] = img_n

            # INPUT AUDIO ARRAY
            X_2[i,] = np.load(os.path.join(self.audios_path, ID.rsplit('-', 1)[0]) + '.npy')


        return X_1, X_2

#--------------------------------------------------------------------------------------------------------------

test_IDs = [f.rsplit('.', 1)[0] for f in os.listdir(images_test_path)  if not f.startswith('.')]
np.save('test_IDs_vgg+audio', test_IDs)
test_generator = DataGenerator(test_IDs, images_test_path, audios_test_path)

#--------------------------------------------------------------------------------------------------------------

# Train model on dataset
print("[INFO] predicting...")
predictions = vgg_audio_model.predict_generator(generator = test_generator, verbose = 1)

print("[INFO] printing predictions...")
print(predictions)

print("[INFO] saving predictions...")
np.save('predictions_vgg+audio', predictions) # save

