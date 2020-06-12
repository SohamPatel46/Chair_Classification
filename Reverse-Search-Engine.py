# -*- coding: utf-8 -*-
"""Task1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a7E_qBhnYRsMGrtGqAuOxm9Ym0OD1D8p
"""

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors





img=os.listdir('/tmp')
img_paths=[]
for i in img:
  s='/tmp/'+i
  img_paths.append(s)
print(len(img_paths))

model = VGG16(weights='imagenet', include_top=False)

img_vector_features = []
i=0
for img_path in img_paths:
  img = image.load_img(img_path, target_size=(224, 224))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)

  vgg16_feature = model.predict(img_data)
  vgg16_feature = np.array(vgg16_feature)
  vgg16_feature = vgg16_feature.flatten()
  img_vector_features.append(vgg16_feature)

"""# New Section"""



query_path = '/usr/test/3.jpg'
img = image.load_img(query_path, target_size=(224, 224))
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

#img = cv.imread(img_paths[similar_image_indices[2]]) 
#plt.imshow(img)

fig = plt.figure()
a = fig.add_subplot(2, 2, 1)
img = cv.imread(query_path) 
imgplot = plt.imshow(img)
a = fig.add_subplot(2, 2, 2)
img = cv.imread(img_paths[similar_image_indices[0]]) 
imgplot = plt.imshow(img)
a = fig.add_subplot(2, 2, 3)
img = cv.imread(img_paths[similar_image_indices[1]]) 
imgplot = plt.imshow(img)
a = fig.add_subplot(2, 2, 4)
img = cv.imread(img_paths[similar_image_indices[2]]) 
imgplot = plt.imshow(img)

