import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


EPOCHS = 10
BATCH_SIZE = 128

data = csv.reader(open('myDrivingData/driving_log.csv'), delimiter=",",quotechar='|')
img_center = []
img_left = []
img_right = []
steering = []
for row in data:
      img_center.append(row[0])
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

# Split data
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

def myNvidiaNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # O = (H - F + 1) / S
    # Layer 1: Convolutional. Input = 66x200x3. Output = 31x98x24.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 24), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(24))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 2, 2, 1], padding='VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)
    
    # Layer 2: Convolutional. Input = 31x98x24. Output = 14x47x36.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 24, 36), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(36))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 2, 2, 1], padding='VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Layer 3: Convolutional. Input = 14x47x36. Output = 5x22x48.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 36, 48), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(48))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 2, 2, 1], padding='VALID') + conv3_b
    # Activation.
    conv3 = tf.nn.relu(conv3)
    
    # Layer 4: Convolutional. Input = 5x22x48. Output = 3x20x64.
    conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 48, 64), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(64))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_b
    # Activation.
    conv4 = tf.nn.relu(conv4)
    
    # Layer 5: Convolutional. Input = 3x20x64. Output = 1x18x64.
    conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(64))
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='VALID') + conv5_b
    # Activation.
    conv5 = tf.nn.relu(conv5)
    
    # Flatten. Input = 1x18x64. Output = 1152.
    fc0   = flatten(conv5)
    
    # Layer 6: Fully Connected. Input = 1152. Output = 100.
    fc6_W = tf.Variable(tf.truncated_normal(shape=(1152, 100), mean = mu, stddev = sigma))
    fc6_b = tf.Variable(tf.zeros(100))
    fc6   = tf.matmul(fc0, fc6_W) + fc6_b
    # Activation.
    fc6    = tf.nn.relu(fc6)
    
    # Layer 7: Fully Connected. Input = 100. Output = 50.
    fc7_W = tf.Variable(tf.truncated_normal(shape=(100, 50), mean = mu, stddev = sigma))
    fc7_b = tf.Variable(tf.zeros(50))
    fc7   = tf.matmul(fc6, fc7_W) + fc7_b
    # Activation.
    fc7    = tf.nn.relu(fc7)
    
    # Layer 8: Fully Connected. Input = 50. Output = 10.
    fc8_W = tf.Variable(tf.truncated_normal(shape=(50, 10), mean = mu, stddev = sigma))
    fc8_b = tf.Variable(tf.zeros(10))
    fc8   = tf.matmul(fc7, fc8_W) + fc8_b
    # Activation.
    fc8    = tf.nn.relu(fc8)
    
    # Layer 9: Fully Connected. Input = 10. Output = 1.
    fc9_W = tf.Variable(tf.truncated_normal(shape=(10, 1), mean = mu, stddev = sigma))
    fc9_b = tf.Variable(tf.zeros(1))
    logits = tf.matmul(fc8, fc9_W) + fc9_b
    
    return logits
    
# Training the model

### FEATURES & LABELS ###
x = tf.placeholder(tf.float32, (None, 66, 200, 3))
y = tf.placeholder(tf.int32, (None))

### TRAINING PIPELINE ###
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

### MODEL EVALUATION ###
correct_prediction = tf.equal(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### TRAIN THE MODEL ###
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, 'lenet')
    print("Model saved")