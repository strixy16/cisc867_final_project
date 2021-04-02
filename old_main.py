# Main file for deep_icc
# Description: Main script for running image loading and model training in development
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: April 1, 2021

from importlib import reload
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from skimage.transform import resize
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, MaxPool2D
from tensorflow.keras.optimizers import Adam
import tensorflow.compat.v2.summary as summary
from tensorflow.python.ops import summary_ops_v2
from tqdm import tqdm
from typing import Any, Dict, Iterable, Sequence, Tuple, Optional, Union

# Imports from my own code
from patient_data_split import pat_train_test_split

# Constants for development
FILESTOLOAD = 10  # 2888 is all of them
imdim_from_preprocessing = 256  # must match opt.ImageSize in image preprocessing configuration files
imdim_for_network = 256
random_seed = 16

# Path to CSVs that connect patient id to slices and rfs label
zero_info_path = "/Users/katyscott/Documents/ICC/Data/Labels/" + str(imdim_from_preprocessing) +"/RFS_all_tumors_zero.csv"
nan_info_path = "/Users/katyscott/Documents/ICC/Data/Labels/" + str(imdim_from_preprocessing) +"/RFS_all_tumors_NaN.csv"

zero_image_path = '/Users/katyscott/Documents/ICC/Data/Images/Tumors/' + str(imdim_from_preprocessing) + '/Zero/'
nan_image_path = '/Users/katyscott/Documents/ICC/Data/Images/Tumors/' + str(imdim_from_preprocessing) + '/NaN/'

# Reading in info for zero background images
info = pd.read_csv(zero_info_path)
image_fnames = np.asarray(info.iloc[:, 0])
pat_num = np.asarray(info.iloc[:, 1])
slice_num = np.asarray(info.iloc[:, 2])
rfs_event = np.asarray(info.iloc[:, 3])
rfs_time = np.asarray(info.iloc[:, 4])

print(rfs_event.shape)
print(rfs_time[1])


images = np.empty((1, imdim_for_network, imdim_for_network))
file_count = 0
for image_file in tqdm(image_fnames):
    if file_count >= FILESTOLOAD:
        break
    else:
        file_count += 1
    #     print("Loading: ", image_file)
        # Load in file as an numpy array
        img = np.fromfile(zero_image_path + image_file)
        # Reshape image from 1D to 2D array - need to nothardcode this, square root?
        img_2D = np.reshape(img, (imdim_from_preprocessing,imdim_from_preprocessing))
        # Scale image to this dimension, smooth image with Gaussian filter, pads with the reflection of the vector
        # mirrored on the first and last values of the vector along each axis.
        img_final = resize(img_2D, (imdim_for_network, imdim_for_network), anti_aliasing=True, mode='reflect')
        # Not sure this next line is working, want an array with all the images as their own array in it
        img_final_3D = np.reshape(img_final, (1,) + img_final.shape)
        images = np.append(images, img_final_3D, axis=0)

images = np.delete(images, 0, axis=0)

# plt.imshow(images[8], cmap='Greys')

# Training and testing split
split = 0.9
train_slice_indices, test_slice_indices = pat_train_test_split(pat_num[:FILESTOLOAD], rfs_event[:FILESTOLOAD], 0.9, random_seed)

print("Train: ", np.array(train_slice_indices).shape)
print("Test: ", np.array(test_slice_indices).shape)

train_slices = images[train_slice_indices,:,:]#[:][:]
train_slices = train_slices.squeeze() # Remove first dim of size 1

train_time = rfs_time[train_slice_indices]
train_event = rfs_event[train_slice_indices]
print("Training set: ", train_slices.shape)
print("Training time labels: ", train_time.shape)
print("Training event labels: ", train_event.shape)

test_slices = images[test_slice_indices,:,:]
test_slices = test_slices.squeeze() # Remove first dim of size 1

test_time = rfs_time[test_slice_indices]
test_event = rfs_event[test_slice_indices]
print("Testing set: ", test_slices.shape)
print("Testing time labels: ", test_time.shape)
print("Testing event labels: ", test_event.shape)

#### MODEL CODE ####
# Model Setup
model = Sequential([
        Conv2D(6, kernel_size=(5,5), activation='relu', name='conv_1'),
        MaxPool2D(pool_size=(2,2)),
        Conv2D(16, (5,5), activation='relu', name='conv_2'),
        MaxPool2D(pool_size=(2,2)),
        Flatten(),
        Dense(120, activation='relu', name='dense_1'),
        Dense(84, activation='relu', name='dense_2'),
        Dense(1, activation='linear', name='dense_3')
    ])

# Start with just 2 images to get everything working
howmany = 2
small_train = train_slices[:howmany]
small_time = train_time[:howmany]
small_event = train_event[:howmany]

small_test = test_slices[:howmany]
small_test_time = test_time[:howmany]
small_test_event = test_event[:howmany]

from input_function import InputFunction, _make_riskset
from train_and_evaluate import TrainAndEvaluateModel
import cph_loss
import cindex_metric

train_fn = InputFunction(small_train, small_time, small_event, drop_last = True, shuffle=True)

eval_fn = InputFunction(small_test, small_test_time, small_test_event)

trainer = TrainAndEvaluateModel(
    model=model,
    model_dir="/Users/katyscott/Documents/ICC/small-cnn/",
    train_dataset=train_fn(),
    eval_dataset=eval_fn(),
    learning_rate=0.0001,
    num_epochs=15,
)

trainer.train_and_evaluate()
