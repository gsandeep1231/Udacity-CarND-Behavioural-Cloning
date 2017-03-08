import numpy as np
import matplotlib.image as mpimg
import cv2
import math

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
    #imgs = (imgs/255.)*2 - 1
    imgs = (imgs/255.) - 0.5
    return imgs

def resize_images(imgs):
    new_imgs = []
    for i, img in enumerate(imgs):
        img1 = cv2.resize(img, (64, 64))
        new_imgs.append(img1)
    new_imgs = np.asarray(new_imgs)
    return new_imgs

def augment_mirror_img(imgs, steerings):
    aug_imgs = []
    aug_steerings = []
    #print("current img size:", len(imgs))
    #print("current steering size:", len(steerings))
    for img, steering in zip(imgs, steerings):
        aug_imgs.append(img)
        aug_steerings.append(steering)
        flipped_img = cv2.flip(img, 1)
        flipped_steering = steering * -1.0
        aug_imgs.append(flipped_img)
        aug_steerings.append(flipped_steering)
    #print("new img size:", len(aug_imgs))
    #print("new steering size:", len(aug_steerings))
    aug_imgs = np.asarray(aug_imgs)
    aug_steerings = np.asarray(aug_steerings)
    return aug_imgs, aug_steerings
        
def shift_image(imgs,steerings,shift_range):
    shift_imgs = []
    shift_steerings = []
    for img, steering in zip(imgs, steerings):
        shift_imgs.append(img)
        shift_steerings.append(steering)
        rows,cols,channels = img.shape
        tr_x = shift_range*np.random.uniform()-shift_range/2
        new_steer = steering + tr_x/shift_range*2*.2
        tr_y = 10*np.random.uniform()-10/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        new_img = cv2.warpAffine(img,Trans_M,(cols,rows))
        shift_imgs.append(new_img)
        shift_steerings.append(new_steer)
    shift_imgs = np.asarray(shift_imgs)
    shift_steerings = np.asarray(shift_steerings)
    return shift_imgs, shift_steerings

def add_brightness(imgs):
    new_imgs = []
    for i, img in enumerate(imgs):
        img1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        img1[:,:,2] = img1[:,:,2]*random_bright
        img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2RGB)
        new_imgs.append(img1)
    new_imgs = np.asarray(new_imgs)
    return new_imgs

def trim_image_new(imgs):
    new_imgs = []
    for i, img in enumerate(imgs):
        shape = img.shape
        img = img[math.floor(shape[0]/4+10):shape[0]-25, 0:shape[1]]
        new_imgs.append(img)
    new_imgs = np.asarray(new_imgs)
    return new_imgs

def preprocess_batch(img_paths, steerings):
    imgs = read_images(img_paths)
    imgs, steerings = shift_image(imgs, steerings, 150)
    imgs = add_brightness(imgs)
    imgs = trim_image_new(imgs)
    imgs = resize_images(imgs)
    imgs, steerings = augment_mirror_img(imgs, steerings)
    return imgs, steerings

def preprocess_test_img(imgs):
    imgs = trim_image_new(imgs)
    imgs = resize_images(imgs)
    return imgs
    
def preprocess_batch_old(img_paths, steerings):
    imgs = read_images(img_paths)
    imgs = trim_images(imgs)
    imgs = threshold_images(imgs)
    imgs = normalize_images(imgs)
    imgs = resize_images(imgs)
    imgs, steerings = augment_mirror_img(imgs, steerings)
    imgs = imgs.reshape(imgs.shape[0], 25, 100, 1)
    return imgs, steerings

def preprocess_test_img_old(imgs):
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
        batch_imgs, batch_steerings = preprocess_batch(imgs[batch], steerings[batch].astype(float))
        yield batch_imgs, batch_steerings