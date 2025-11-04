
# imports
import sys
import numpy as np
import pandas as pd
# import keras as kp
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
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
    #rotation_range = 40,
    #horizontal_flip = True
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


# Step 2 and 3: Neural Network Architecture Design & Hyperparameter Analysis
DCNN_model = Sequential()
DCNN_model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu',input_shape=(image_width, image_height, image_channel)))
DCNN_model.add(MaxPooling2D(pool_size=(2, 2)))

# second stack
DCNN_model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
DCNN_model.add(MaxPooling2D(pool_size=(2, 2)))

# third stack
DCNN_model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
DCNN_model.add(MaxPooling2D(pool_size=(2, 2)))

#fourth stack 
#DCNN_model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
#DCNN_model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten 
DCNN_model.add(Flatten()),
DCNN_model.add(Dense(128, activation = 'relu')), 
#DCNN_model.add(Dense(32, activation = 'relu'))
DCNN_model.add(Dropout(0.25))
DCNN_model.add(Dense(3, activation = 'softmax')) 

print(DCNN_model.summary())


learning_rate = 1e-3
DCNN_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])




# Step 4: Model Evaluation

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=6,
    mode='min',
    restore_best_weights=True
    )

history1 = DCNN_model.fit(x=train_gen, validation_data=valid_gen, epochs=15, batch_size=128, validation_split=0.001, callbacks=[early_stopping])




# Plot the validation and train


# Plotting accuracy over epochs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'], label='Training Accuracy')
plt.plot(history1.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy for Model 1')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting Loss over epochs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history1.history['loss'], label='Training Loss')  # Changed 'Loss' to 'loss'
plt.plot(history1.history['val_loss'], label='Validation Loss')  # Changed 'val_Loss' to 'val_loss'
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Model 1')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



# Save the model
DCNN_model.save('DCNN_model.h5')

















