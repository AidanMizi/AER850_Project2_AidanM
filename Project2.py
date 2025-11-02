
# imports
import sys
import numpy as np
import pandas as pd
import keras as kp
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU

# VERSIONS
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"numpy Version: {np.__version__}")
print(f"pandas Version: {pd.__version__}")
print(f"Python Version: {sys.version}")


# Step 1: data processing
# define image shape and channel and batch size
image_width, image_height, image_channel, batch_size = 500, 500, 3, 32;

# define image data directories
training_data = r"Data\train"
validation_data = r"Data\valid"

# create training data image augmentation object
train_data_aug = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    rotation_range = 40,
    horizontal_flip = True
    )
    
# create validation data image augmentation object
valid_data_aug = ImageDataGenerator(
    rescale = 1./255
    )

# create train and validation generators
train_gen = train_data_aug.flow_from_directory(
    training_data,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'  
    )

valid_gen = valid_data_aug.flow_from_directory(
    validation_data,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
    )























