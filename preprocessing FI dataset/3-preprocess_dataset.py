import numpy as np
from ast import literal_eval
import sys
import pickle
import time
import os
import csv

#------------------------------------------------------------------------------------------------------

def extract_gender(csv_path):
    gender = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in list(csv_reader)[1:]:
            gender[row[0].rsplit('.', 1)[0]] = float(row[3])
    return gender

def extract_age(csv_path):
    age = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in list(csv_reader):
            age[row[0]] = float(row[1])
    return age

# if the output directory does not exist, create it
def create_folder(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

def normalize_data(data):
    return data / np.sum(data, dtype=np.float32)


#------------------------------------------------------------------------------------------------------

vectors_txt = os.listdir('emotions_vectors/')

# folder where all the histogram+gender+age vectors per each video are gonna be saved
create_folder('feature_vectors/')



gender = {}
gender = extract_gender('./gender_age/eth_gender/eth_gender_annotations_test.csv')
gender.update(extract_gender('./gender_age/eth_gender/eth_gender_annotations_dev.csv'))

age = {}
age = extract_age('./gender_age/age/luis_edad_train.csv')
age.update(extract_age('./gender_age/age/luis_edad_validation.csv'))
age.update(extract_age('./gender_age/age/luis_test_edad.csv'))


start_time = time.time()
#--------------------------------------------------------------------------------------------------------

# we are gonna save each histogram vectors in a pickle file
for output_npy in vectors_txt:

	emotions_vector = np.load('emotions_vectors/' + output_npy)

	# emotions arrays
	neutral = emotions_vector[:,0]
	anger = emotions_vector[:,1]
	disgust = emotions_vector[:,2]
	fear = emotions_vector[:,3]
	happy = emotions_vector[:,4]
	sadness = emotions_vector[:,5]
	surprise = emotions_vector[:,6]


	b = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # bins

	hist_neutral, bins_neutral = np.histogram(neutral, bins = b)
	hist_anger, bins_anger = np.histogram(anger, bins = b)
	hist_disgust, bins_disgust = np.histogram(disgust, bins = b)
	hist_fear, bins_fear = np.histogram(fear, bins = b)
	hist_happy, bins_happy = np.histogram(happy, bins = b)
	hist_sadness, bins_sadness = np.histogram(sadness, bins = b)
	hist_surprise, bins_surprise = np.histogram(surprise, bins = b)



	# normalizing histograms
	hist_neutral = normalize_data(hist_neutral)
	hist_anger = normalize_data(hist_anger)
	hist_disgust = normalize_data(hist_disgust)
	hist_fear = normalize_data(hist_fear)
	hist_happy = normalize_data(hist_happy)
	hist_sadness = normalize_data(hist_sadness)
	hist_surprise = normalize_data(hist_surprise)

	histogram_vector = np.concatenate([hist_neutral, hist_anger, hist_disgust,hist_fear, hist_happy, hist_sadness, hist_surprise])

	gender_age = [ gender[output_npy.rsplit('.', 1)[0]], age[output_npy.rsplit('.', 1)[0]] ]

	feature_vector = np.concatenate([histogram_vector, gender_age])

	print(feature_vector)

	# saving feature_vector as a npy file
	np.save('feature_vectors/' + output_npy.rsplit('.', 1)[0], feature_vector) # save

#--------------------------------------------------------------------------------------------------------
print("[INFO] 3-preprocess dataset.py has finished in %s minutes" % ((time.time() - start_time)/60))

