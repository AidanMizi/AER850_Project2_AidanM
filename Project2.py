
# imports
import numpy as np
import pandas as pd
import keras as kp
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU



# Step 1: data processing

image_width, image_height, image_channel = 500, 500, 3;

training_data = r"Data\train"
validation_data = r"Data\valid"











