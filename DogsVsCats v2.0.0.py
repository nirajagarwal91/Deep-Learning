# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 21:20:05 2018

@author: Niraj
"""

# Building Convolutional Neural Network

from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dropout
from keras.layers import Dense
from tensorflow.contrib.keras import backend

classify = Sequential()

# Adding Convolution Layer using Feature Maps
# Filter : Feature Detectors 
# Can use more bits if running on GPU for input shape
# Activation function can be used to remove non linearity in image
classify.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Pooling Step Max Pooling (Get reduced feature Map using Pooling)

classify.add(MaxPooling2D(pool_size=(2,2)))

# Adding another CNN layer

classify.add(Conv2D(32, (3, 3), activation = 'relu'))
classify.add(MaxPooling2D(pool_size=(2,2)))

classify.add(Conv2D(64, (3, 3), activation = 'relu'))
classify.add(MaxPooling2D(pool_size=(2,2)))

classify.add(Conv2D(64, (3, 3), activation = 'relu'))
classify.add(MaxPooling2D(pool_size=(2,2)))

# Flattenning of the pooled features to get feature vector

classify.add(Flatten())

# Creating Hidden Layer

classify.add(Dense(units= 64, activation='relu'))
classify.add(Dropout(0.6))
classify.add(Dense(units= 64, activation='relu'))
classify.add(Dense(units= 64, activation='relu'))
classify.add(Dropout(0.3))
# Output Layer
classify.add(Dense(units=1, activation='sigmoid'))

# Since our outcome is for only cats and dogs there fore binary cross entropy
classify.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## Fitting CNN to images

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_gen = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classify.fit_generator(
        train_gen,
        steps_per_epoch=8000,
        epochs=100,
        validation_data=test_gen,
        validation_steps=2000)

import os
script_dir = os.path.dirname(_file_)
model_backup_path = os.path.join('../dataset/cat_or_dog_model')
classify.save(model_backup_path)
print("Model saved to", model_backup_path)
backend.clear_session()
print("The model class indices are:", train_gen.class_indices)







