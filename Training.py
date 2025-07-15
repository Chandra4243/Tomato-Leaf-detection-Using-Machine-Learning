from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Configure GPU options
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Define the CNN architecture
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data
training_set = train_datagen.flow_from_directory('C:/Users/Chandrabhushan/.vscode/Project from scratch/Plant-Leaf-Disease-Prediction-main/Dataset/train', 
                                                 target_size=(128, 128),
                                                 batch_size=6, 
                                                 class_mode='categorical')

valid_set = test_datagen.flow_from_directory('C:/Users/Chandrabhushan/.vscode/Project from scratch/Plant-Leaf-Disease-Prediction-main/Dataset/val', 
                                             target_size=(128, 128), 
                                             batch_size=3, 
                                             class_mode='categorical')

# Train the classifier
history = classifier.fit(training_set,
                         epochs=50,
                         validation_data=valid_set)

# Save the model
classifier.save("model.h5")
print("Model saved to disk.")
