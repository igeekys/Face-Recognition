import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

from DataProcessing import (cal_data, create_val_set, data_aug, data_split,
                            plot_img_examples)
from model_op import (count_parameters, epoch_time, evaluate, get_predictions,
                      train)
from resnet import build_model
from utils import (LRFinder, get_pca, get_representations, get_tsne,
                   plot_confusion_matrix, plot_lr_finder, plot_most_incorrect,
                   plot_representations)

# dataset processing #################################
data_dir = './datasets'
images_dir = data_dir

# split dateset into train/test set 
# train_dir, test_dir = data_split(data_dir, images_dir, TRAIN_RATIO=0.8)
train_dir, test_dir = './datasets/train/', './datasets/test/'

# calculate the mean/std of images 
# means, stds = cal_data(train_dir)
# output:
# Calculated means: tensor([0.6420, 0.5958, 0.5712])
# Calculated stds: tensor([0.2472, 0.2372, 0.2327])

# data augmentation
pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]
train_data, test_data,test_transforms = data_aug(pretrained_size, pretrained_means, pretrained_stds, train_dir, test_dir)

# create validation set
valid_data = create_val_set(train_data, test_data, test_transforms,VALID_RATIO = 0.9)

# create dataloarder with the given batch_size 
BATCH_SIZE = 64
train_iterator = data.DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
valid_iterator = data.DataLoader(valid_data, batch_size = BATCH_SIZE)
test_iterator = data.DataLoader(test_data, batch_size = BATCH_SIZE)

# # plot image examples
# plot_img_examples(train_data, test_data, N_IMAGES=25)
###################################################

# build model and load pre-trained parameters #####
OUTPUT_DIM = len(test_data.classes)
BASE_MODEL = 'resnet50'
PRETRAINED_MODEL = './model/resnet50-19c8e357.pth'
SE = False
if SE:
    SAVED_MODEL = BASE_MODEL+'-SE-finetune.pth'
else:
    SAVED_MODEL = BASE_MODEL+'-finetune.pth'
model = build_model(BASE_MODEL, OUTPUT_DIM, pretrained=False, model_path=PRETRAINED_MODEL,is_se=SE)

# training ########################################
TRAIN = True
if TRAIN:
    START_LR = 1e-7
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # # find best learning rate
    # END_LR = 10
    # NUM_ITER = 100
    # lr_finder = LRFinder(model, optimizer, criterion, device)
    # lrs, losses = lr_finder.range_test(train_iterator, END_LR, NUM_ITER)
    # plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)

    FOUND_LR = 1e-3
    params = [
            {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
            {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
            {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
            {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
            {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
            {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
            {'params': model.fc.parameters()}
            ]
    optimizer = optim.Adam(params, lr = FOUND_LR)

    EPOCHS = 10
    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
    MAX_LRS = [p['lr'] for p in optimizer.param_groups]
    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr = MAX_LRS,
                                        total_steps = TOTAL_STEPS)

    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):
        start_time = time.monotonic()
        
        train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, scheduler, device)
        valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), SAVED_MODEL)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
            f'Train Acc @5: {train_acc_5*100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
            f'Valid Acc @5: {valid_acc_5*100:6.2f}%')

# model testing ####################################
model.load_state_dict(torch.load(SAVED_MODEL))
test_loss, test_acc_1, test_acc_5 = evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
    f'Test Acc @5: {test_acc_5*100:6.2f}%')

# # result analysis ###################################
# images, labels, probs = get_predictions(model, test_iterator, device)
# pred_labels = torch.argmax(probs, 1)

# classes = train_data.classes
# plot_confusion_matrix(labels, pred_labels, classes)

# corrects = torch.eq(labels, pred_labels)
# incorrect_examples = []
# for image, label, prob, correct in zip(images, labels, probs, corrects):
#     if not correct:
#         incorrect_examples.append((image, label, prob))
# incorrect_examples.sort(reverse = True, key = lambda x: torch.max(x[2], dim = 0).values)
# N_IMAGES = 36
# plot_most_incorrect(incorrect_examples, classes, N_IMAGES)

# # representation visualization ######################
# outputs, labels = get_representations(model, train_iterator, device)
# output_pca_data = get_pca(outputs)
# plot_representations(output_pca_data, labels, classes)
# output_tsne_data = get_tsne(outputs)
# plot_representations(output_tsne_data, labels, classes)
