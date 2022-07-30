#-----------------------------------------------------------------------------------------------------------------------------------------------------
#imports

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

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# global variables

path = sys.argv[1] # full path where all folders with faces are

train_path = 'dataset/all_classes_train/train/'
validation_path = 'dataset/all_classes_validation/validation/'
test_path = 'dataset/all_classes_test/test/'

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# functions

def extract_picklefile(picklefile):
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
    return data

# return full path of each folder name in a given list
def get_folders_path(path, lista):
    
    for i, f in enumerate(lista):
        lista[i] = os.path.join(path, f)
        
    return lista

# if the output directory does not exist, create it
def create_folder(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

# construct a dictionary where the key is the video name and the value is a list of OCEAN values [e, n, a, c, o]
def labels_list(list_keys, list_of_dics):
    return dict([(k.rsplit('.', 1)[0], [dicc[k] for dicc in list_of_dics]) for k in list_keys])

def get_name_dataset(dicc):
    lista = list(dicc.keys())
    for i, f in enumerate(lista):
        lista[i] = f.rsplit('.', 1)[0]
    return lista

# all files from each folder in the list intersections are copied to its corresponding dataset folder
def build_dataset(dataset_path, intersections):
    
    create_folder(dataset_path)

    for folder in intersections:     
        for file in os.listdir(folder):
            if file.endswith(".png"):
                filename = os.path.join(folder, file)
                shutil.copy(filename, dataset_path)


# it creates a diccionary with the following structure : {'image_name' : (OCEAN_ndarray, histogram+gender+age vector), ...}
def construct_dicc_OCEAN_features(dicc, dataset_path):
    all_faces = [f.rsplit('.', 1)[0] for f in os.listdir(dataset_path)  if not f.startswith('.')]

    dicc_OCEAN_histogram = {}
    for face in all_faces:
        dicc_OCEAN_histogram[face] = ( np.asarray(dicc[face.rsplit('-', 1)[0]]),
                            np.load('feature_vectors/' + face.rsplit('-', 1)[0] + '.npy') )
    return dicc_OCEAN_histogram

#---------------------------------------------------------------------------------------------------------------------------------------------------------

train_data = extract_picklefile('annotation_training.pkl')
validation_data = extract_picklefile('annotation_validation-1.pkl')
test_data = extract_picklefile('annotation_test.pkl')

#---------------------------------------------------------------------------------------------------------------------------------------------------------

# train
list_of_dics_train = [value for value in train_data.values()] # extraversion, neuroticism, agreeableness, conscientiousness, interview, openness
del list_of_dics_train[4] # we dont need 'interview' values, so we delete them from the list  now ---> e, n, a, c, o

list_train_keys = list((list(train_data.items())[0][1]).keys()) # video names for train data


# validation
list_of_dics_validation = [value for value in validation_data.values()]
del list_of_dics_validation[4]

list_validation_keys = list((list(validation_data.items())[0][1]).keys())


# test
list_of_dics_test = [value for value in test_data.values()]
del list_of_dics_test[4]

list_test_keys = list((list(test_data.items())[0][1]).keys()) # video names for test data


train_labels = labels_list(list_train_keys, list_of_dics_train)
validation_labels = labels_list(list_validation_keys, list_of_dics_validation)
test_labels = labels_list(list_test_keys, list_of_dics_test)

#---------------------------------------------------------------------------------------------------------------------------------------------------------

train_openness = train_data.get('openness')
validation_openness = validation_data.get('openness')
test_openness = test_data.get('openness')


train_names = get_name_dataset(train_openness)
validation_names = get_name_dataset(validation_openness)
test_names = get_name_dataset(test_openness)

#---------------------------------------------------------------------------------------------------------------------------------------------------------

list_initial = next(os.walk(path))[1]


train_intersections = get_folders_path(path, list(np.intersect1d(list_initial, train_names)))

validation_intersections = get_folders_path(path, list(np.intersect1d(list_initial, validation_names)))

test_intersections = get_folders_path(path, list(np.intersect1d(list_initial, test_names)))


start_time = time.time()
print("[INFO] executing build_dataset ...")
#--------------------------------------------------------------------------------------------------------
# Now, all files from each folder in the list intersections are copied to its corresponding dataset folder
build_dataset(train_path, train_intersections)
build_dataset(validation_path, validation_intersections)
build_dataset(test_path, test_intersections)
#--------------------------------------------------------------------------------------------------------
print("[INFO] build_dataset has finished in %s minutes ---" % ((time.time() - start_time)/60))


start_time = time.time()
print("[INFO] executing construct_dicc_OCEAN_features ...")
#--------------------------------------------------------------------------------------------------------
# it creates a diccionary with the following structure : {'image_name' : (OCEAN_ndarray, histogram+gender+age vector), ...}
train_OCEAN_features = construct_dicc_OCEAN_features(train_labels, train_path)
validation_OCEAN_features = construct_dicc_OCEAN_features(validation_labels, validation_path)
test_OCEAN_features = construct_dicc_OCEAN_features(test_labels, test_path)
#--------------------------------------------------------------------------------------------------------
print("[INFO] construct_dicc_OCEAN_features has finished in %s minutes" % ((time.time() - start_time)/60))



#--------------------------------------------------------------------------------------------------------

print("[INFO] saving train_OCEAN_histogram as a pickle file")
with open('train_OCEAN_features.pkl', 'wb') as f:
    pickle.dump(train_OCEAN_features, f, protocol=2)

print("[INFO] saving validation_OCEAN_histogram as a pickle file")
with open('validation_OCEAN_features.pkl', 'wb') as f:
    pickle.dump(validation_OCEAN_features, f, protocol=2)

print("[INFO] saving test_OCEAN_histogram as a pickle file")
with open('test_OCEAN_features.pkl', 'wb') as f:
    pickle.dump(test_OCEAN_features, f, protocol=2)

#--------------------------------------------------------------------------------------------------------
