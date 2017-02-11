import numpy as np
import matplotlib.image as mpimg
import cv2

######################################
###### PREPROCESSING FUNCTIONS #######
######################################

def read_images(img_paths):
    imgs = np.empty([len(img_paths), 160, 320, 3], dtype=np.uint8)
    for i, path in enumerate(img_paths):
        #print('Reading file', path)
        imgs[i] = mpimg.imread(path, 1)
    return imgs

def trim_images(imgs):
    #img = cv2.imread(file)
    imgs = imgs[:,60:135,:,:]
    return imgs

def threshold_images(imgs):
    imgs_thresh = np.empty([len(imgs), 75, 320])
    for i, img in enumerate(imgs):
        #print(img.shape)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_gray = np.array([0,0,50])
        upper_gray = np.array([255,80,255])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res = cv2.bitwise_and(gray, gray, mask=mask_gray)
        imgs_thresh[i] = res
    return imgs_thresh

def normalize_images(imgs):
    imgs = (imgs/255.)*2 - 1
    return imgs

def resize_images(imgs):
    imgs_resized = np.empty([len(imgs), 25, 100])
    for i, img in enumerate(imgs):
        imgs_resized[i] = cv2.resize(img, (100, 25))
    return imgs_resized

def preprocess_batch(img_paths):
    imgs = read_images(img_paths)
    imgs = trim_images(imgs)
    imgs = threshold_images(imgs)
    imgs = normalize_images(imgs)
    imgs = resize_images(imgs)
    imgs = imgs.reshape(imgs.shape[0], 25, 100, 1)
    return imgs

def preprocess_test_img(imgs):
    imgs = trim_images(imgs)
    imgs = threshold_images(imgs)
    imgs = normalize_images(imgs)
    imgs = resize_images(imgs)
    imgs = imgs.reshape(imgs.shape[0], 25, 100, 1)
    return imgs

def process_batch(imgs, steerings, batch_size):
    size_images = len(imgs)
    while True:
        batch = np.random.choice(size_images, batch_size)
        batch_imgs, batch_steerings = preprocess_batch(imgs[batch]), steerings[batch].astype(float)
        yield batch_imgs, batch_steerings