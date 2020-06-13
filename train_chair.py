# -*- coding: utf-8 -*-
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
print("Trainning started".center(30,"*"))
f=__file__
f=f.replace('train_chair.py', '')
root=f+'Chair'
root_dir = os.path.join(root)

img_dir = os.path.join(root_dir+'/All Chair')
img=os.listdir(img_dir)
img_paths=[]
for i in img:
  s=img_dir+'/'+i
  img_paths.append(s)

model = VGG16(weights='imagenet', include_top=False)

img_vector_features = []
i=0
for img_path in img_paths:
  img = image.load_img(img_path, target_size=(150,150))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)

  vgg16_feature = model.predict(img_data)
  vgg16_feature = np.array(vgg16_feature)
  vgg16_feature = vgg16_feature.flatten()
  img_vector_features.append(vgg16_feature)

model.save(root_dir+"/my.h5")
img_vector_features=np.array(img_vector_features)
img_paths=np.array(img_paths)
np.save(root_dir+'/n.npy',img_vector_features)
np.save(root_dir+'/n1.npy',img_paths)

