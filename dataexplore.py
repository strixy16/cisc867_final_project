# Name: dataexplore.py
# Environment: Python 3.8
# Author: Katy Scott
# Created: April 1, 2021
# Script to do preliminary data analysis and generate distribution plots and Kaplan Meieir analysis

from importlib import reload
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sksurv.nonparametric import kaplan_meier_estimator

# Imports from my own code
from patient_data_split import pat_train_test_split
from input_function import InputFunction, _make_riskset
import cph_loss
import cindex_metric
from train_and_evaluate import TrainAndEvaluateModel

# Constants for development
FILESTOLOAD = 2888  # 2888 is all of them
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
print(rfs_time[0])

# Training and testing split
split = 0.9
train_slice_indices, test_slice_indices = pat_train_test_split(pat_num[:FILESTOLOAD], rfs_event[:FILESTOLOAD], 0.9, random_seed)
# print("Train: ", np.array(train_slice_indices).shape)
# print("Test: ", np.array(test_slice_indices).shape)

train_time = rfs_time[train_slice_indices]
train_event = rfs_event[train_slice_indices]
print("Training time labels: ", train_time.shape)
print("Training event labels: ", train_event.shape)

test_time = rfs_time[test_slice_indices]
test_event = rfs_event[test_slice_indices]
print("Testing time labels: ", test_time.shape)
print("Testing event labels: ", test_event.shape)

info.rename(columns={"Pat ID": "Pat_ID", "Slice Num": "Slice_Num", "RFS Code": "RFS_Code", "RFS Time":"RFS_Time"}, inplace=True)

# Distribution plots for slice level labels, total dataset
slice_rfs_code_plt = sns.displot(info, x="RFS_Code")
plt.xticks([0,1])
plt.xlabel("RFS Code")
plt.title("Slice RFS Code")

slice_rfs_time_plt = sns.displot(info, x="RFS_Time")
plt.xlabel("RFS Time (months)")
plt.title("Slice RFS Time")

# Distribution plots for patient level labels, total dataset
pat_info = info.drop_duplicates("Pat_ID")
pat_rfs_code_plt = sns.displot(pat_info, x="RFS_Code")
plt.xticks([0,1])
plt.xlabel("RFS Code")
plt.title("Patient RFS Code")

pat_rfs_time_plt = sns.displot(pat_info, x="RFS_Time", )
plt.xlabel("RFS Time (months)")
plt.title("Patient RFS Time")

# Getting label info for train and testing sets
training_info = info.iloc[train_slice_indices]
testing_info = info.iloc[test_slice_indices]

# Distribution plots for slice level labels, training set
trainslice_rfs_code_plt = sns.displot(training_info, x="RFS_Code")
plt.xticks([0,1])
plt.xlabel("RFS Code")
plt.title("Training Set - Slice RFS Code")

trainslice_rfs_time_plt = sns.displot(training_info, x="RFS_Time")
plt.xlabel("RFS Time (months)")
plt.title("Training Set - Slice RFS Time")

# Distribution plots for patient level labels, training set
train_pat_info = training_info.drop_duplicates("Pat_ID")
trainpat_rfs_code_plt = sns.displot(train_pat_info, x="RFS_Code")
plt.xticks([0,1])
plt.xlabel("RFS Code")
plt.title("Training Set - Patient RFS Code")

trainpat_rfs_time_plt = sns.displot(train_pat_info, x="RFS_Time", )
plt.xlabel("RFS Time (months)")
plt.title("Training Set - Patient RFS Time")

# Distribution plots for slice level labels, testing set
testslice_rfs_code_plt = sns.displot(testing_info, x="RFS_Code")
plt.xticks([0,1])
plt.xlabel("RFS Code")
plt.title("Testing Set - Slice RFS Code")

testslice_rfs_time_plt = sns.displot(testing_info, x="RFS_Time")
plt.xlabel("RFS Time (months)")
plt.title("Testing Set - Slice RFS Time")

# Distribution plots for patient level labels, testing set
test_pat_info = testing_info.drop_duplicates("Pat_ID")
testpat_rfs_code_plt = sns.displot(test_pat_info, x="RFS_Code")
plt.xticks([0,1])
plt.xlabel("RFS Code")
plt.title("Testing Set - Patient RFS Code")

testpat_rfs_time_plt = sns.displot(test_pat_info, x="RFS_Time", )
plt.xlabel("RFS Time (months)")
plt.title("Testing Set - Patient RFS Time")

# KAPLAN MEIER ANALYSIS
# Convert rfs_event from numeric to boolean for use in sksurv
b_rfs_event = np.array(rfs_event, dtype=bool)
# Creating structured array for kaplan_meier_estimator
rfs_type = np.dtype([('Status','bool'), ('Time', 'f')])
rfs = np.empty(len(rfs_event),dtype=rfs_type)
rfs['Status'] = b_rfs_event
rfs['Time'] = rfs_time

time, survival_prob = kaplan_meier_estimator(rfs['Status'], rfs['Time'])
plt.step(time, survival_prob, where="post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time in months $t$")
plt.title("Kaplan-Meier Survival Curve")

plt.show()

