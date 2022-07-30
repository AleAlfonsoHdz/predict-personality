import os, numpy, PIL
from PIL import Image
import os
import numpy as np

train_path = 'dataset/all_classes_train/train'
validation_path = 'dataset/all_classes_validation/validation'

arr=numpy.zeros((224,224,3),numpy.float)

train_array = np.array([np.array(Image.open(train_path + "/" + f)) for f in os.listdir(train_path) if not f.startswith('.')])
validation_array = np.array([np.array(Image.open(validation_path + "/" + f)) for f in os.listdir(validation_path) if not f.startswith('.')])

images_array = np.concatenate([train_array, validation_array])

arr = np.array(np.mean(images_array, axis=(0)), dtype=np.uint8)

print("[INFO] saving mean_face.png...")
# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
out.save("mean_face.png")
#out.show()