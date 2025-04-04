import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import os

if __name__ == '__main__':

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Define image size and batch size
    IMG_SIZE = 224
    BATCH_SIZE = 32     #amount of data clustered tongether in every training set


    train_dir = 'C:\\Users\\danie\\OneDrive\\Desktop\\AI projects\\CatAndDogCNN\\PetImages\\test'   #dir to test data
    test_dir = 'C:\\Users\\danie\\OneDrive\\Desktop\\AI projects\\CatAndDogCNN\\PetImages\\train'   #dir to training data

    #Creating data Generators for preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalize pixel values to [0, 1]
        rotation_range=20,           # Randomly rotate images by up to 20 degrees
        width_shift_range=0.2,       # Randomly shift images horizontally by 20% of the width
        height_shift_range=0.2,      # Randomly shift images vertically by 20% of the height
        shear_range=0.2,             # Apply shearing transformations
        zoom_range=0.2,              # Randomly zoom into images
        horizontal_flip=True,        # Randomly flip images horizontally
        fill_mode='nearest'          # Fill in new pixels that may appear after a transformation
    )

    test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for test data

    # Load the training and testing data using the ImageDataGenerator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),  # Resize images
        batch_size=BATCH_SIZE,
        class_mode='binary'  # Binary classification (cats vs dogs)
    )
    # same thing but for test data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),  # Resize images
        batch_size=BATCH_SIZE,
        class_mode='binary'  # Binary classification (cats vs dogs)
    )

    # Build a simple CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)), #applies fetures (using a kernal to change the image to extract data)
        layers.MaxPooling2D((2, 2)),    #reduce image after change to reduce computation load
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),   #changes imiage into a 1D vector
        layers.Dense(128, activation='relu'),   #real input network layer
        layers.Dense(1, activation='sigmoid')  # Binary output (cat or dog)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=10,
        validation_data=test_generator,
        validation_steps=test_generator.samples // BATCH_SIZE
    )


# Save the model's weights to a file
model.save_weights('cat_dog_model.h5')