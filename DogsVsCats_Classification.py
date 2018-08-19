# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 21:20:05 2018

@author: Niraj
"""

# Building Convolutional Neural Network

from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classify = Sequential()

# Adding Convolution Layer using Feature Maps
# Filter : Feature Detectors 
# Can use more bits if running on GPU for input shape
# Activation function can be used to remove non linearity in image
classify.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling Step Max Pooling (Get reduced feature Map using Pooling)

classify.add(MaxPooling2D(pool_size=(2,2)))

# Flattenning of the pooled features to get feature vector

classify.add(Flatten())

# Creating Hidden Layer

classify.add(Dense(units= 128, activation='relu'))
# Output Layer
classify.add(Dense(units=1, activation='sigmoid'))

# Since our outcome is for only cats and dogs there fore binary cross entropy
classify.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])












