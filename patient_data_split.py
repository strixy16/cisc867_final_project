# Name: patient_train_test_split.py
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: February 28, 2021

import numpy as np


def pat_train_test_split(pat_num, label, split_perc, seed = 16):
    """
    Function to split data into training and testing, keeping slices from one patient in one class
    Args:
        pat_num - numpy array of patient numbers or ids to be split
        label - numpy array of binary labels for the data, indicating recurrence or non-recurrence (censoring)
        split_perc - float value < 1, percentage of data to put in training set, 1 - split_perc will be the testing size
        seed - seed for patient index shuffling
    Returns:
        sets - tuple of training and testing slice indices in a list
    """
    # Checking that split percentage is between 0 and 1 to print better error message
    if split_perc > 1.0 or split_perc < 0.0:
        print("Invalid split percentage. Must be between 0 and 1.")
        return -1
    
    # Separate out positive and negative labels to evenly distribute them between classes
    # Get index of slices with 0 and 1 label
    # z => zero, o => one
    z_idx = np.asarray(np.where(label == 0)).squeeze()
    o_idx = np.asarray(np.where(label == 1)).squeeze()

    # Get patient ids of 0 and 1 labels
    z_pats = pat_num[z_idx]
    o_pats = pat_num[o_idx]

    # Remove repeat patient ids (repeats are there because pat_nums has number for every slice)
    # u => unique
    uz_pats = np.unique(z_pats)
    uo_pats = np.unique(o_pats)

    np.random.seed(seed)
    # Shuffle patient index for splitting
    np.random.shuffle(uz_pats)
    np.random.shuffle(uo_pats)

    # Find index to split data at from input
    split_z = int(split_perc * len(uz_pats))
    split_o = int(split_perc * len(uo_pats))

    # Training patient set
    train_z_pat = uz_pats[:split_z]
    train_o_pat = uo_pats[:split_o]

    # Testing patient set
    test_z_pat = uz_pats[split_z:]
    test_o_pat = uo_pats[split_o:]
    
    # Training slice set for censored patients (rfs_code = 0)
    train_z_slice = []
    for pat in train_z_pat:
        tr_slice_z = np.asarray(np.where(pat_num == pat)).squeeze()
        if len(tr_slice_z.shape) == 0:
            tr_slice_z = np.expand_dims(tr_slice_z, axis=0)
        train_z_slice = np.concatenate((train_z_slice, tr_slice_z))

    # Training slice set for non-censored patients 
    train_o_slice = []
    for pat in train_o_pat:
        tr_slice_o = np.asarray(np.where(pat_num == pat)).squeeze()
        if len(tr_slice_o.shape) == 0:
            tr_slice_o = np.expand_dims(tr_slice_o, axis=0)
        train_o_slice = np.concatenate((train_o_slice, tr_slice_o))

    # Testing slice set for censored patients (rfs_code = 0)
    test_z_slice = []
    for pat in test_z_pat:
        ts_slice_z = np.asarray(np.where(pat_num == pat)).squeeze()
        if len(ts_slice_z.shape) == 0:
            ts_slice_z = np.expand_dims(ts_slice_z, axis=0)
        test_z_slice = np.concatenate((test_z_slice, ts_slice_z))

    # Testing slice set for non-censored patients
    test_o_slice = []
    for pat in test_o_pat:
        ts_slice_o = np.asarray(np.where(pat_num == pat)).squeeze()
        if len(ts_slice_o.shape) == 0:
            ts_slice_o = np.expand_dims(ts_slice_o, axis=0)
        test_o_slice = np.concatenate((test_o_slice, ts_slice_o))

    # Combine censored and non-censored slice sets
    train_slice = np.concatenate((train_z_slice, train_o_slice)).astype(int)
    test_slice = np.concatenate((test_z_slice, test_o_slice)).astype(int)

    # Tuple of indices for training and testing slices
    sets = (train_slice, test_slice)
    return sets

