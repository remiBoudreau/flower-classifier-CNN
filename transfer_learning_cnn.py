# Filename: transfer_learning_cnn.py
# Dependencies: transfer_learning_cnn_util.py, collections, keras, math, os, pathlib, random, shutil, sys, tensorflow, warnings
# Author: Jean-Michel Boudreau
# Date: October 21, 2019

'''
Queries user about whether to download creative-commons licensed flower photos 
from Google or to specify the path of one existing locally. Furthermore,
queries to split data into training and validation subsets or train on the 
whole dataset. Finally, trains a convolutional neural network constructed using
the VGG16 CNN as the base model (with  imagenet's pre-trained weights but 
without its fully connected top layer) with these additional layers at the top:
    1. Flatten
    2. Fully connected hidden layer with 256 neurons
    3. Softmax layer
to predict whether a given image (resized to 150 px by 150 px) is of a daisy,
dandelion, rose, sunflower, or tulip.
'''

# Import libraries
from transfer_learning_cnn_util import clean_data_exists
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import warnings
warnings.simplefilter("ignore")

# Parameters for resizing images
img_height = 150
img_width = 150

# Parameters for training
batch_size = 32
num_epochs = 10
frozen_layer_lower = 0
frozen_layer_upper = 18
'''
frozen_layer_lower and frozen_layer_upper correspond to the lower and upper 
bound for the range layers that will not be trained on. 
The vgg16 model.sumamry() is shown below. 
layer   name
0       input_3 
1       block1_conv1
2       block1_conv2
3       block1_pool
4       block2_conv1
5       block2_conv2 
6       block2_pool
7       block3_conv1
8       block3_conv2
9       block3_conv3
10      block3_pool
11      block4_conv1
12      block4_conv2
13      block4_conv3
14      block4_pool
15      block5_conv1
16      block5_conv2
17      block5_conv3
18      block5_pool
'''

# Create training generator and (if applicable), validation generator for keras
# CNN training
train_generator, validation_generator = clean_data_exists(img_height, img_width, batch_size)

# VGG16 pre-trained model without fully connected layers and with different 
# input dimensions
vgg16 = VGG16(
        # Use pretrained weights from imagenet
        weights = "imagenet", 
        # Remove top/fully-connected layers
        include_top=False, 
        # Change input dimensions
        input_shape = (img_width, img_height, 3))

# Set trainable layers
for layer in vgg16.layers[frozen_layer_lower:frozen_layer_upper]:
    layer.trainable=False

# Add custom layers to create new model
base_model = vgg16.output
GAP = GlobalAveragePooling2D()(base_model)
fc_layer = Dense(256, activation="relu")(GAP)
predictions = Dense(5, activation="softmax")(fc_layer)
model = Model(input = vgg16.input, output = predictions)

# Construct CNN
model.compile(loss="categorical_crossentropy", 
              optimizer="nadam", 
              metrics=["accuracy"])

# Train CNN
model.fit_generator(generator = train_generator,
                    validation_data = validation_generator,
                    epochs=num_epochs)