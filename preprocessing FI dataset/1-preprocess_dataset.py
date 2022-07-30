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

train_realpath = os.path.realpath(train_path)
validation_realpath = os.path.realpath(validation_path)
test_realpath = os.path.realpath(test_path)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# functions

def extract_picklefile(picklefile):
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
    return data

# if the output directory does not exist, create it
def create_folder(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

# construct a dictionary where the key is the video name and the value is a list of OCEAN values [e, n, a, c, o]
def labels_list(list_keys, list_of_dics):
    return dict([(k.rsplit('.', 1)[0], [dicc[k] for dicc in list_of_dics]) for k in list_keys])

# only keeps n random images from each folder (the rest is deleted)
def keep_n(n, images_path):
    
    all_faces = os.listdir(images_path)
    
    if len(all_faces) >= n:
        random_faces_list = random.sample(all_faces, n)
    else:
        random_faces_list = random.sample(all_faces, len(all_faces))
    
    faces_to_delete = np.setdiff1d(all_faces, random_faces_list)
    
    for f in faces_to_delete:
        os.remove(os.path.join(images_path, f))


# rename each image in each folder like: 'folder name' + '-' + 'face number'
def rename_file_as_directory(folder_path):
    files = os.listdir(folder_path)
    
    for filename in files:
        new_filename = folder_path.rsplit('/', 1)[1] + "-" + filename.rsplit('.', 1)[0] + ".png"
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    

# return full path of each folder name in a given list
def get_folders_path(path, lista):
    
    for i, f in enumerate(lista):
        lista[i] = os.path.join(path, f)
        
    return lista

# This saves txt files per each video. That will be the necessary inputs to the nn in inference.py
# to predict the emotions vectors of each video.
create_folder('values_txt/')
def prepare_inputs_to_emotions_nn(folder, output_name):
    output_to_file = open('values_txt/' + output_name, "w")
    
    files = os.listdir(folder)
   
    filename =''
    for file in files:
        if file.endswith(".png"):
            file_splitted = file.rsplit('-', 1)[0]
            if file_splitted in train_names:
                filename = os.path.join(path+folder.rsplit('/', 1)[1], file)
            if file_splitted in validation_names:
                filename = os.path.join(path+folder.rsplit('/', 1)[1], file)
            if file_splitted in test_names:
                filename = os.path.join(path+folder.rsplit('/', 1)[1], file)
            
            output_to_file.write(filename + " \n")

    output_to_file.close()

def get_name_dataset(dicc):
    lista = list(dicc.keys())
    for i, f in enumerate(lista):
        lista[i] = f.rsplit('.', 1)[0]
    return lista

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




#---------------------------------------------------------------------------------------------------------------------------------------------------------

faces_folders = next(os.walk('faces/'))[1]
faces_folders = get_folders_path(path, faces_folders)

#---------------------------------------------------------------------------------------------------------------------------------------------------------

start_time = time.time()
print("[INFO] Keeping 10 random faces from each folder...")
print("[INFO] Renaming each image in each folder as 'folder name' + '-' + 'face number'...")
print("[INFO] Saving a txt file for each video in 'values_txt/'...")
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
for folder in faces_folders:
    print(folder)
    # First, we only keep 10 random faces from each folder (the rest is deleted)
    keep_n(10, folder)
    
    # Afterwards, we rename each image in each folder as: 'folder name' + '-' + 'face number'
    rename_file_as_directory(folder)
    
    # Then, we save txt files per each video. That will be the necessary inputs to the nn in inference.py
    # to predict the emotions vectors of each video.
    output_name = folder.rsplit('/', 1)[1] + '.txt'
    prepare_inputs_to_emotions_nn(folder, output_name)


#--------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
print("[INFO] 1-preprocess_dataset.py has finished in %s minutes ---" % ((time.time() - start_time)/60))
