from __future__ import print_function
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3" # specify which GPU(s) to be used

import pickle
import numpy as np
import random
import tensorflow as tf
import keras
from keras.engine import Model
from keras.utils import Sequence
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers import Input, Conv1D, MaxPool1D, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping


model = Input(shape = (224938, ), name='audio_vector')


dense1 = Dense(512, activation='relu', name='fc1')(model)
dense2 = Dense(128, activation='relu', name='fc2')(dense1)
dense3 = Dense(32, activation='relu', name='fc3')(dense2)

output = Dense(5, activation='sigmoid', name='output')(dense3)

audio_model = Model([model], output)

audio_model.summary()

print("[INFO] compiling model...")
audio_model.compile(Adam(1e-05),loss='mean_squared_error',metrics=['mse', 'mae'])

batch_size = 64


def extract_picklefile(picklefile):
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
    return data


train_path = 'audio_dataset/audio_train'
validation_path = 'audio_dataset/audio_validation'

train_OCEAN = extract_picklefile('train_audio_OCEAN.pkl')
validation_OCEAN= extract_picklefile('validation_audio_OCEAN.pkl')

train_size = len(train_OCEAN.values())
validation_size = len(validation_OCEAN.values())



class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, labels, dataset_path, batch_size=32, audio_shape=224938, output_size=5, shuffle=True):
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
        self.audio_shape = audio_shape
        self.output_size = output_size
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
        X = np.empty((self.batch_size, self.audio_shape))
        y = np.empty((self.batch_size, self.output_size), dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # INPUT ARRAY
            X[i,] = np.load(os.path.join(self.dataset_path, ID) + '.npy')

            # OCEAN LABEL
            # in this case, OCEAN dicc has a tuple of values: (OCEAN values, features values)
            # so we only select the OCEAN values
            y[i] = self.labels[ID][0]

        return X, y

#--------------------------------------------------------------------------------------------------------------

train_IDs = [f.rsplit('.', 1)[0] for f in os.listdir(train_path)  if not f.startswith('.')]
validation_IDs = [f.rsplit('.', 1)[0] for f in os.listdir(validation_path)  if not f.startswith('.')]

train_generator = DataGenerator(train_IDs, train_OCEAN, train_path)
validation_generator = DataGenerator(validation_IDs, validation_OCEAN, validation_path)


#--------------------------------------------------------------------------------------------------------------


# callback function to print output layer weights during training
print_output_weights = LambdaCallback( on_epoch_end=lambda epoch, logs: print(audio_model.layers[-1].get_weights()[0]) )

# callback function to keep best model weights
checkpoint = ModelCheckpoint(filepath='vgg_faces_weights.hdf5', monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')

# callback function to stop when the model do not improve whatever value we are monitoring
early_stopping_monitor = EarlyStopping(monitor='val_mean_squared_error', patience=6)

callbacks_list = [checkpoint]

# Train model on dataset
print("[INFO] training model...")
history = audio_model.fit_generator( generator = train_generator, validation_data = validation_generator, epochs = 150, verbose = 1, callbacks = callbacks_list )


# summarize history for mse
figure()
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model_audio mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_audio_mse.png')

# summarize history for mae
figure()
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model_audio mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_audio_mae.png')

# summarize history for loss
figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_audio loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plot_audio_loss.png')

with open('audio_nn_history.pkl', 'wb') as f:
    pickle.dump(history.history, f, protocol=2)

# serialize the model to disk
print("[INFO] serializing model to disk...")
audio_model.save("audio_model.h5")


