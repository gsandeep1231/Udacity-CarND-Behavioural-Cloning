import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import csv
import os
import json
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, ELU, Conv2D, ConvLSTM2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from img_utils import *
#from other_utils import *
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
NUM_EPOCHS = 5

######################################
# READ CSV 
######################################

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
    imgString = cwd + '/data/' + row[0].strip()
    img_center.append(imgString)
    imgString = cwd + '/data/' + row[1].strip()
    img_left.append(imgString)
    imgString = cwd + '/data/' + row[2].strip()
    img_right.append(imgString)
    steering.append(row[3])
img_center = np.asarray(img_center)
img_left = np.asarray(img_left)
img_right = np.asarray(img_right)
steering_center = np.asarray(steering, dtype=np.float32)
X_train=[]
y_train=[]

# Merge left/right image data
correction = 0.15
tmp_image = np.concatenate([img_center, img_left])
steering_left = steering_center
steering_left[steering_left>0] = steering_left[steering_left>0] + correction
steering_left[steering_left<0] = steering_left[steering_left<0] - correction
tmp_steering = np.concatenate([steering_center, steering_left])
X_train = np.concatenate([tmp_image, img_right])
steering_right = steering_center
steering_right[steering_right>0] = steering_right[steering_right>0] - correction
steering_right[steering_right<0] = steering_right[steering_right<0] + correction
y_train = np.concatenate([tmp_steering, steering_right])
y_train[y_train > 1] = 1.0
y_train[y_train < -1] = -1.0
'''
# Read in my Sample data
driving_log = cwd + '/myDrivingData/driving_log.csv'
print('Reading csv file', driving_log)
data = csv.reader(open(driving_log), delimiter=",",quotechar='|')
img_center = []
steering = []
print('Looping CSV')
for row in data:
    img_center.append(row[0])
    steering.append(row[3])
img_center = np.asarray(img_center)
steering_center = np.asarray(steering, dtype=np.float32)
#steering_center[steering_center<-0.33] = steering_center[steering_center<-0.33]*0.6
'''
#X_train = np.concatenate([X_train, img_center])
#y_train = np.concatenate([y_train, steering_center])

print('Final Img Size: ', X_train.shape)
print('Final Steering Size: ', y_train.shape)

# Shuffle and split
X_train, y_train = shuffle(X_train, y_train)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

def mySimpleModel1():
    model = Sequential([
        Conv2D(32, 3, 3, input_shape=(25, 100, 1), border_mode='same', activation='relu'),
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

def mySimpleModel2():
    model = Sequential()
    model.add(Convolution2D(36, 3, 3, subsample=(2,2), input_shape=(25, 100, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(25))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

def mySimpleModel3():
    input_shape = (25, 100, 1)
    #input_shape = (64, 64, 3)
    filter_size = 3
    pool_size = (2,2)
    model = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5,input_shape=input_shape))
    model.add(Convolution2D(3,1,1, border_mode='valid', name='conv0', init='he_normal'))
    model.add(Convolution2D(32,filter_size,filter_size, border_mode='valid', name='conv1', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(32,filter_size,filter_size, border_mode='valid', name='conv2', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,filter_size,filter_size, border_mode='valid', name='conv3', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64,filter_size,filter_size, border_mode='valid', name='conv4', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128,filter_size,filter_size, border_mode='valid', name='conv5', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(128,filter_size,filter_size, border_mode='valid', name='conv6', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512,name='hidden1', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(64,name='hidden2', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(16,name='hidden3',init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1, name='output', init='he_normal'))
    return model
    
    
    

model = mySimpleModel2()
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit_generator(process_batch(X_train, y_train, BATCH_SIZE),
                                  len(X_train),
                                  NUM_EPOCHS,
                                  validation_data=process_batch(X_validation, y_validation, BATCH_SIZE),
                                  nb_val_samples=len(X_validation))

################################################################
# Save the model and weights
################################################################

model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights("./model.h5")
print("Saved model to disk")

################################################################
# visualize model history for loss
################################################################

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('model_loss_plot.png')

################################################################
# visualize model
################################################################

#from keras.utils.visualize_util import plot
#plot(model, to_file='model_graph.png')
