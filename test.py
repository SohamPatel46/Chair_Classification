from tensorflow import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow import keras
import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import sys 

f=__file__
f=f.replace('test.py', '')
root=f+'Chair'
model = keras.models.load_model(root+'/my.h5')
img_vector_features=np.load(root+'/n.npy')
img_paths=np.load(root+'/n1.npy')

query_path=sys.argv[1]
#query_path = input("Enter test path : ")
img = image.load_img(query_path, target_size=(150,150))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_feature = model.predict(img_data)
vgg16_feature = np.array(vgg16_feature)
query_feature = vgg16_feature.flatten()

# Numbers of similar images that we want to show
N_QUERY_RESULT = 3
nbrs = NearestNeighbors(n_neighbors=N_QUERY_RESULT, metric="cosine").fit(img_vector_features)

distances, indices = nbrs.kneighbors([query_feature])
similar_image_indices = indices.reshape(-1)

print(similar_image_indices)

fig = plt.figure()
a = fig.add_subplot(2, 2, 1)
img = cv.imread(query_path) 
imgplot = cv.imshow(img)
a = fig.add_subplot(2, 2, 2)
img = cv.imread(img_paths[similar_image_indices[0]]) 
imgplot = cv.imshow(img)
a = fig.add_subplot(2, 2, 3)
img = cv.imread(img_paths[similar_image_indices[1]]) 
imgplot = cv.imshow(img)
a = fig.add_subplot(2, 2, 4)
img = cv.imread(img_paths[similar_image_indices[2]]) 
imgplot = cv.imshow(img)