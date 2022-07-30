from __future__ import print_function
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3" # specify which GPU(s) to be used

import pickle
import numpy as np
import random
import tensorflow as tf
import keras
from keras.engine import Model
from keras.utils import Sequence
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers import Input, Conv1D, MaxPool1D, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping


#--------------------------------------------------------------------------------------------------------------

test_path = 'audio_dataset/audio_test'

#--------------------------------------------------------------------------------------------------------------

# we load trained nn with audio
print("[INFO] loading trained nn with audio...")
audio_model = load_model('audio_model.h5')
audio_model.load_weights('audio_weights.hdf5') # loading best weights


audio_model.summary()

#--------------------------------------------------------------------------------------------------------------

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, dataset_path, batch_size=32, audio_shape=224938, output_size=5, shuffle=False):
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
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.audio_shape = audio_shape
        self.output_size = output_size
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
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)"""
        # Initialization
        X = np.empty((self.batch_size, self.audio_shape), dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # INPUT ARRAY
            X[i,] = np.load(os.path.join(self.dataset_path, ID) + '.npy')

        return X

#--------------------------------------------------------------------------------------------------------------

test_IDs = [f.rsplit('.', 1)[0] for f in os.listdir(test_path)  if not f.startswith('.')]
np.save('test_IDs_audio', test_IDs)
test_generator = DataGenerator(test_IDs, test_path)

#--------------------------------------------------------------------------------------------------------------

print("[INFO] predicting...")
predictions = audio_model.predict_generator(generator = test_generator, verbose = 1)

print("[INFO] printing predictions...")
print(predictions)

print("[INFO] saving predictions...")
np.save('predictions_audio', predictions) # save