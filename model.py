import argparse
import base64
import json
import csv
import cv2
import numpy as np
import os
from urllib.request import urlretrieve
from os.path import isfile
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, ELU, Conv2D, ConvLSTM2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from PIL import Image
from PIL import ImageOps
from io import BytesIO
from keras.models import load_model

def normalize_data(x):
    x = (x/255.)*2 - 1
    return x

# Define model
def myNvidiaModel():
    model = Sequential()
    model.add(Convolution2D(36, 3, 3, subsample=(2,2), input_shape=(50, 100, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, subsample=(2,2))) #, input_shape=(24, 49, 36)))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Convolution2D(64, 3, 3, subsample=(2,2))) #, input_shape=(11, 24, 48)))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1))) #, input_shape=(5, 11, 64)))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1))) #, input_shape=(3, 9, 64)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

def mySimpleModel():
    model = Sequential()
    model.add(Convolution2D(36, 3, 3, subsample=(2,2), input_shape=(50, 100, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, subsample=(2,2), input_shape=(12, 25, 36)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), input_shape=(5, 12, 48)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(25))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

def myCommaModel():
    model = Sequential()
    model.add(Convolution2D(16, 4, 4, subsample=(4, 4), border_mode="same", input_shape=(50, 100, 3)))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(256))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model

def mySimpleModel2():
    model = Sequential([
        Conv2D(32, 3, 3, input_shape=(50, 100, 3), border_mode='same', activation='relu'),
        Conv2D(64, 3, 3, border_mode='same', activation='relu'),
        Dropout(0.5),
        Conv2D(128, 3, 3, border_mode='same', activation='relu'),
        Conv2D(256, 3, 3, border_mode='same', activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, name='output', activation='tanh'),
    ])
    return model

cwd = os.getcwd()
# Read in Udacity sample data
driving_log = cwd + '/data/driving_log.csv'
print('Reading csv file', driving_log)
data = csv.reader(open(driving_log), delimiter=",",quotechar='|')
img_center = []
img_left = []
img_right = []
steering = []
print('Looping CSV')
for row in data:
    imgString = cwd + '/data/' + row[0]
    #imgString = cwd + '/myDrivingData/' + row[0]
    #image = Image.open(imgString)
    #print('Resizing image', imgString)
    #image = image.resize((100,50))
    image = cv2.imread(imgString)
    image = cv2.resize(image, (100,50))
    img_center.append(image)
    tmp = row[1].strip()
    imgString = cwd + '/data/' + tmp
    image = cv2.imread(imgString)
    #print('Resizing image', imgString)
    image = cv2.resize(image, (100,50))
    img_left.append(image)
    tmp = row[2].strip()
    imgString = cwd + '/data/' + tmp
    image = cv2.imread(imgString)
    image = cv2.resize(image, (100,50))
    img_right.append(image)
    steering.append(row[3])
img_center = np.asarray(img_center)
img_left = np.asarray(img_left)
img_right = np.asarray(img_right)
steering_center = np.asarray(steering, dtype=np.float32)

# Merge left/right image data
#X_train = img_center
#y_train = steering
tmp_image = np.concatenate([img_center, img_left])
steering_left = steering_center
steering_left[steering_left>0] = steering_left[steering_left>0]*1.8
steering_left[steering_left<0] = steering_left[steering_left<0]*0.5
tmp_steering = np.concatenate([steering_center, steering_left])
X_train = np.concatenate([tmp_image, img_right])
steering_right = steering_center
steering_right[steering_right>0] = steering_right[steering_right>0]*0.5
steering_right[steering_right<0] = steering_right[steering_right<0]*1.8
y_train = np.concatenate([tmp_steering, steering_right])
y_train[y_train > 1] = 1.0
y_train[y_train < -1] = -1.0

# Read in my Sample data
driving_log = cwd + '/myDrivingData/driving_log.csv'
print('Reading csv file', driving_log)
data = csv.reader(open(driving_log), delimiter=",",quotechar='|')
img_center = []
steering = []
print('Looping CSV')
for row in data:
    imgString = row[0]
    image = cv2.imread(imgString)
    image = cv2.resize(image, (100,50))
    img_center.append(image)
    steering.append(row[3])
img_center = np.asarray(img_center)
steering_center = np.asarray(steering, dtype=np.float32)
X_train = np.concatenate([X_train, img_center])
y_train = np.concatenate([y_train, steering_center])

print('Final Img Size: ', X_train.shape)
print('Final Steering Size: ', y_train.shape)

# Shuffle data
X_train, y_train = shuffle(X_train, y_train)

# Normalize data
X_normalized = normalize_data(X_train)

model = mySimpleModel2()
# TODO: Compile and train the model
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_normalized, y_train, nb_epoch=5, validation_split=0.2)

################################################################

# Save the model and weights

################################################################

model_json = model.to_json()

with open("./model.json", "w") as json_file:

    json.dump(model_json, json_file)

model.save_weights("./model.h5")

print("Saved model to disk")
