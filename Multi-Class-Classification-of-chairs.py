import os
from tensorflow import keras
import numpy as np
from google.colab import files
from keras.preprocessing import image
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing.image import ImageDataGenerator

from google.colab import drive
drive.mount('/content/drive')

#/content/drive/My Drive/Multiclass chair/Chair
root_dir = input("Enter extracted zip file path (Chair) : ")
Accent_dir = os.path.join(root_dir+'/Accent Chairs')
Armpit_dir = os.path.join(root_dir+'/Armpit')
Beambag_dir = os.path.join(root_dir+'/BeamBags')
DiningChairs_dir = os.path.join(root_dir+'/Dining Chairs')
FootStall_dir = os.path.join(root_dir+'/Footstools')
BarStool_dir = os.path.join(root_dir+'/BarStools')
Benches_dir = os.path.join(root_dir+'/benches')
GardenChairs_dir = os.path.join(root_dir+'/Garden Chairs')
OfficeChairs_dir = os.path.join(root_dir+'/Office Chairs')
labels=["Accent Chairs","Armpit","BarStools","BeamBags","Dining Chairs","Footstools","Garden Chairs","Office Chairs","Benches"]

TRAINING_DIR = root_dir 
training_datagen = ImageDataGenerator(
      rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=20
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])

#model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=20, steps_per_epoch=20, verbose = 1)
#model.save("rps.h5")

# predicting images 
query_path = input("Enter test path : ")
img = image.load_img(query_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(labels)
print(classes)
