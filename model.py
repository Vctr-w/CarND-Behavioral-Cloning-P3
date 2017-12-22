import csv
import cv2
import numpy as np
from random import *

def get_path(pref, source_path):
    filename = source_path.split('/')[-1]
    return pref + filename

def get_image(pref, source_path):
    img = cv2.imread(get_path(pref, source_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

correction = 0.2            # Correction angle added to left and right camera images to augment our centre camera data
thresh = 0.00               # Steering angles <= thresh are considered "steering straight"
probability_reject = 0.9    # To remove bias in steering straight, we reject 90% of data containing steering input <= thresh

lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read input images and steering measurements

images = []
measurements = []
for line in lines:	
    steering_centre = float(line[3])
    steering_left = steering_centre + correction
    steering_right = steering_centre - correction
    
    img_centre = get_image('IMG/', line[0])
    img_left = get_image('IMG/', line[1])
    img_right = get_image('IMG/', line[2])

    images.extend([img_centre, img_left, img_right])
    measurements.extend([steering_centre, steering_left, steering_right])

# Augment data by flipping across vertical axis and steering angle

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):

    # To remove bias, include a higher proportion of corners
    # So if steering straight, then reject with x%

    if abs(measurement) <= thresh and random() < probability_reject:
        pass

    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, -1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

# Include a dropout probability of 0.5 to prevent overfitting

dropout_rate = 0.5

# Use the NVIDIA architecture with dropout layers added
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(dropout_rate))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(dropout_rate))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
exit()

