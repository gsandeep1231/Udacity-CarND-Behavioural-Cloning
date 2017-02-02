import argparse
import base64
import json
from urllib.request import urlretrieve
from os.path import isfile
from tqdm import tqdm
import csv
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from PIL import Image
from PIL import ImageOps
from io import BytesIO
import cv2
from keras.models import load_model


data = csv.reader(open('myDrivingData/driving_log.csv'), delimiter=",",quotechar='|')
img_center = []
img_left = []
img_right = []
steering = []
for row in data:
    imgString = row[0]
    image = cv2.imread(imgString)
    image = cv2.resize(image, (100,50))
    img_center.append(image)
    img_left.append(row[1])
    img_right.append(row[2])
    steering.append(row[3])
img_center = np.asarray(img_center)
img_left = np.asarray(img_left)
img_right = np.asarray(img_right)
steering = np.asarray(steering, dtype=np.float32)

#print(img_center[0:15])
#print(steering[0:15])

# Shuffle data
X_train = img_center
y_train = steering
X_train, y_train = shuffle(X_train, y_train)

# Normalize data
def normalize_data(x):
    x = (x/255.)*2 - 1
    return x

X_normalized = normalize_data(X_train)


# Define model
model = Sequential()
model.add(Convolution2D(36, 3, 3, subsample=(2,2), input_shape=(50, 100, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3, subsample=(2,2), input_shape=(24, 49, 36)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2,2), input_shape=(11, 24, 48)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1,1), input_shape=(5, 11, 64)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1,1), input_shape=(3, 9, 64)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))
    
# TODO: Compile and train the model
model.compile(optimizer=Adam(), loss="mse")
history = model.fit(X_normalized, y_train, nb_epoch=10, validation_split=0.2)
model.save('my_model.h5')