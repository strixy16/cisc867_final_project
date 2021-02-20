import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import resize


'''
Loader needs
CSV path  - path to CSV file that contains
Data path - path to the bin files. I want to have two of these, one for NaN and one for zeros
rfs_path -
'''


# class Image_Loader(Dataset):
#
#     def __init__(self, csv_path, image_path, idx, mean, std, img_dim):
#         self.info = pd.read_csv(csv_path)
#         self.image_path = image_path
#         self.image_fname = np.asarray(self.info.iloc[idx, 0])
#         self.pat_num = np.asarray(self.info.iloc[idx, 1])
#         self.slice_num = np.asarray(self.info.iloc[idx, 2])
#         self.label = np.asarray(self.info.iloc[idx, 3])




# Path to csvs that connect patient id to slices and rfs label
zero_info_path = '/Users/katyscott/Documents/ICC/Data/Labels/RFS_all_tumors_zero.csv'
nan_info_path = '/Users/katyscott/Documents/ICC/Data/Labels/RFS_all_tumors_NaN.csv'

zero_image_path = '/Users/katyscott/Documents/ICC/Data/Images/Tumors/Zero/'
nan_image_path = '/Users/katyscott/Documents/ICC/Data/Images/Tumors/NaN/'

# Need to split data up into train and test and then perform augmentation
# set up all data first

info = pd.read_csv(zero_info_path)

images = np.array([])
image_fnames = np.asarray(info.iloc[:, 0])
pat_num = np.asarray(info.iloc[:, 1])
slice_num = np.asarray(info.iloc[:, 2])
label = np.asarray(info.iloc[:, 3])
for image_file in image_fnames:
    print("Processing: ", image_file)
    # Load in file as an numpy array
    img = np.fromfile(zero_image_path + image_file)
    # Reshape image from 1D to 2D array
    img_2D = np.reshape(img, (299, 299))
    # Scale image to this dimension, smooth image with Gaussian filter, pads with the reflection of the vector
    # mirrored on the first and last values of the vector along each axis.
    img_final = resize(img_2D, (224, 224), anti_aliasing=True, mode='reflect')
    # Not sure this next line is working, want an array with all the images as their own array in it
    images = np.append(images, [img_final])

print("Did it work?")



