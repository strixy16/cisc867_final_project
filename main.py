# Main file for deep_icc
# Description: Main script for running image loading and model training in development
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: February 25, 2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, MaxPool2D

from patient_data_split import pat_train_test_split

'''
Loader needs
CSV path  - path to CSV file that contains
Data path - path to the bin files. I want to have two of these, one for NaN and one for zeros
rfs_path -
'''

# Path to csvs that connect patient id to slices and rfs label
zero_info_path = '/Users/katyscott/Documents/ICC/Data/Labels/RFS_all_tumors_zero.csv'
nan_info_path = '/Users/katyscott/Documents/ICC/Data/Labels/RFS_all_tumors_NaN.csv'

zero_image_path = '/Users/katyscott/Documents/ICC/Data/Images/Tumors/Zero/'
nan_image_path = '/Users/katyscott/Documents/ICC/Data/Images/Tumors/NaN/'

# Need to split data up into train and test and then perform augmentation
# set up all data first
FILESTOLOAD = 100
imdim = 224
random_seed = 16
info = pd.read_csv(zero_info_path)
image_fnames = np.asarray(info.iloc[:, 0])
pat_num = np.asarray(info.iloc[:, 1])
slice_num = np.asarray(info.iloc[:, 2])
label = np.asarray(info.iloc[:, 3])
# Only loading in 100 number of files for development
images = np.empty((1,imdim,imdim))
file_count = 0
for image_file in image_fnames:
    if file_count >= FILESTOLOAD:
        break
    else:
        print("Loading: ", image_file)
        # Load in file as an numpy array
        img = np.fromfile(zero_image_path + image_file)
        # Reshape image from 1D to 2D array - need to not hardcode this, square root?
        img_2D = np.reshape(img, (299, 299))
        # Scale image to this dimension, smooth image with Gaussian filter, pads with the reflection of the vector
        # mirrored on the first and last values of the vector along each axis.
        img_final = resize(img_2D, (imdim, imdim), anti_aliasing=True, mode='reflect')
        # Not sure this next line is working, want an array with all the images as their own array in it
        img_final_3D = np.reshape(img_final, (1,) + img_final.shape)
        images = np.append(images, img_final_3D, axis=0)
        file_count += 1

images = np.delete(images, 0, axis=0)

plt.imshow(images[1], cmap='Greys')
print("Did it work?")

# Training and testing split
train_slice, test_slice = pat_train_test_split(pat_num[:FILESTOLOAD], label[:FILESTOLOAD], 70, random_seed)

# Model
# img_in = Input(shape=(imdim, imdim,)) # [None, imdim, imdim]






