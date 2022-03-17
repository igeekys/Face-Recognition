import os
import random
import shutil

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.utils import to_categorical
from PIL import Image
from imageio import imread


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def LoadDataset(self) :

    num_classes = self.Nclass
    print('Loading ' + str(num_classes) + ' classes')
    
    if self.split_dataset:
        train_dir, valid_dir, test_dir, n_train, n_valid, n_test = data_split(self.dataset_path, self.dataset_path, TRAIN_RATIO=0.8, VALID_RATIO = 0.1)
    else:
        [train_dir, valid_dir, test_dir], [n_train, n_valid, n_test] = scan_dataset(self.dataset_path)

    print('loading training images...')
    i = 0
    class_idx = 0
    X_train = np.zeros([n_train, self.img_size, self.img_size, self.c_dim], dtype=np.float32)
    y_train = np.zeros([n_train], dtype=np.float32)
    annotations = {}
    for img_class in os.listdir(train_dir):
        class_path = os.path.join(train_dir, img_class)
        if os.path.isdir(class_path):
            annotations[img_class] = class_idx
            for ImgFile in os.listdir(class_path):
                X = imread(os.path.join(class_path, ImgFile))
                X = np.array(Image.fromarray(X).resize((self.img_size, self.img_size)))
                if len(np.shape(X)) == 2:
                    X_train[i] = np.transpose(np.array([X, X, X]),(1,2,0))
                else:
                    X_train[i] = X
                y_train[i] = class_idx
                i += 1
            class_idx += 1
    assert i==n_train
    print('finished loading {} training images'.format(i))

    print('loading validate images...')
    X_valid = np.zeros([n_valid, self.img_size, self.img_size, self.c_dim], dtype=np.float32)
    y_valid = np.zeros([n_valid], dtype=np.float32)
    i = 0
    for img_class in os.listdir(valid_dir):
        class_path = os.path.join(valid_dir, img_class)
        if os.path.isdir(class_path):
            for ImgFile in os.listdir(class_path):
                X = imread(os.path.join(class_path, ImgFile))
                X = np.array(Image.fromarray(X).resize((self.img_size, self.img_size)))
                if len(np.shape(X)) == 2:
                    X_valid[i] = np.transpose(np.array([X, X, X]),(1,2,0))
                else:
                    X_valid[i] = X
                y_valid[i] = annotations[img_class]
                i += 1
    assert i == n_valid
    print('finished loading {} validation images'.format(i))
    

    print('loading test images...')
    X_test = np.zeros([n_test, self.img_size, self.img_size, self.c_dim], dtype=np.float32)
    y_test = np.zeros([n_test], dtype=np.float32)
    i = 0
    for img_class in os.listdir(test_dir):
        class_path = os.path.join(test_dir, img_class)
        if os.path.isdir(class_path):
            for ImgFile in os.listdir(class_path):
                X = imread(os.path.join(class_path, ImgFile))
                X = np.array(Image.fromarray(X).resize((self.img_size, self.img_size)))
                if len(np.shape(X)) == 2:
                    X_test[i] = np.transpose(np.array([X, X, X]),(1,2,0))
                else:
                    X_test[i] = X
                y_test[i] = annotations[img_class]
                i += 1
    assert i == n_test
    print('finished loading {} test images'.format(i))

    print('data preprocessing...')
    X_train /= 255.0
    X_valid /= 255.0
    X_test /= 255.0

    ## image normalization
    X_train = normalize(X_train)
    X_valid= normalize(X_valid)
    X_test = normalize(X_test)

    # convert class vectors to binary class matrices(one-hot representation)
    y_train = to_categorical(y_train, num_classes)
    y_valid = to_categorical(y_valid, num_classes)
    y_test = to_categorical(y_test, num_classes)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def normalize(img_data,ImageNet=True):
    if ImageNet:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
    else:
        mean = np.mean(img_data, axis=(0, 1, 2))
        std = np.std(img_data, axis=(0, 1, 2))
    img_data = (img_data - mean) / std


    return img_data

# def get_annotations_map():
#     valAnnotationsPath = './tiny-imagenet-200/val/val_annotations.txt'
#     valAnnotationsFile = open(valAnnotationsPath, 'r')
#     valAnnotationsContents = valAnnotationsFile.read()
#     valAnnotations = {}

#     for line in valAnnotationsContents.splitlines():
#         pieces = line.strip().split()
#         valAnnotations[pieces[0]] = pieces[1]

#     return valAnnotations

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch, img_size):
    # random_rotation 
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [img_size, img_size], 10)

    return batch

# split dateset into train/valid/test set 
def data_split(data_dir, images_dir, TRAIN_RATIO=0.8, VALID_RATIO = 0.1):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    valid_dir = os.path.join(data_dir, 'valid')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir) 
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)
    classes = os.listdir(images_dir)
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(valid_dir)

    N_train, N_valid, N_test = 0, 0, 0 
    for c in classes:
        class_dir = os.path.join(images_dir, c)
        if not os.path.isdir(class_dir):
            continue
        images = os.listdir(class_dir)
        n_train = int(len(images) * TRAIN_RATIO)
        n_valid = int(n_train * VALID_RATIO)
        n_test = len(images) - n_train - n_valid
        N_train += n_train
        N_valid += n_valid
        N_test += n_test
        
        train_images = images[:n_train+n_valid]
        seed = 777
        np.random.seed(seed)
        random.shuffle(train_images)
        valid_images = train_images[n_train:n_train+n_valid]
        train_images = train_images[:n_train]
        test_images = images[n_train+n_valid:]
        
        os.makedirs(os.path.join(train_dir, c), exist_ok = True)
        os.makedirs(os.path.join(valid_dir, c), exist_ok = True)
        os.makedirs(os.path.join(test_dir, c), exist_ok = True)
        
        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
        for image in valid_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(valid_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
    print(f'Number of training examples: {N_train}')
    print(f'Number of validation examples: {N_valid}')
    print(f'Number of testing examples: {N_test}')
    return train_dir, valid_dir, test_dir, N_train, N_valid, N_test

def scan_dataset(data_dir):
    dirs = ['', '', '']   # train_dir, valid_dir, test_dir
    nums = [0, 0, 0]      # n_train, n_valid, n_test
    for idx, phase in enumerate(['train','valid','test']):
        dirs[idx] = os.path.join(data_dir, phase)
        class_dirs = os.listdir(dirs[idx])
        for c in class_dirs:
            img_dir = os.path.join(dirs[idx], c)
            if os.path.isdir(img_dir):
                nums[idx] += len(os.listdir(img_dir))
    return dirs, nums
