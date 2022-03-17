import copy
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import plot_images


# split dateset into train/test set 
def data_split(data_dir, images_dir, TRAIN_RATIO=0.8):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir) 
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    classes = os.listdir(images_dir)
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    for c in classes:
        class_dir = os.path.join(images_dir, c)
        if not os.path.isdir(class_dir):
            continue
        images = os.listdir(class_dir)
        n_train = int(len(images) * TRAIN_RATIO)
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        os.makedirs(os.path.join(train_dir, c), exist_ok = True)
        os.makedirs(os.path.join(test_dir, c), exist_ok = True)
        
        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
            
        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
    return train_dir, test_dir

# calculate the mean/std of images 
def cal_data(train_dir):
    train_data = datasets.ImageFolder(root = train_dir, transform = transforms.ToTensor())
    means = torch.zeros(3)
    stds = torch.zeros(3)
    for img, label in train_data:
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))
    means /= len(train_data)
    stds /= len(train_data)
    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')
    return means, stds

# data augmentation
def data_aug(pretrained_size, pretrained_means, pretrained_stds, train_dir, test_dir):
    train_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(pretrained_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                        ])

    test_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.CenterCrop(pretrained_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                        ])
    train_data = datasets.ImageFolder(root = train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(root = test_dir, transform = test_transforms)
    return train_data, test_data, test_transforms

# create validation set
def create_val_set(train_data, test_data, test_transforms,VALID_RATIO=0.9):
    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples
    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    return valid_data

# plot image examples
def plot_img_examples(train_data, test_data, N_IMAGES=25):
    images, labels = zip(*[(image, label) for image, label in [train_data[i] for i in range(N_IMAGES)]])
    classes = test_data.classes
    plot_images(images, labels, classes)
