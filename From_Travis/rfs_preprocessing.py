import torch
import numpy as np
from scipy import ndimage
import operator
import pandas as pd
from skimage.transform import resize
import cv2 as cv
import random
from scipy.stats import iqr
from imblearn.over_sampling import SMOTE


def binaryImage(img):
    img = (~np.isnan(img)).astype(int)
    img = ndimage.binary_fill_holes(img).astype(int)
    return img


def brightnessContrast(img):
    alpha = np.random.uniform(0, 5)
    beta = np.random.randint(-50, 25)
    tmp = (~np.isnan(img)).astype(float)
    tmp = ndimage.binary_fill_holes(tmp).astype(float)
    img = np.uint8(img)
    img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    img = img.astype('float')
    img = img * tmp
    tmp[tmp == 0] = np.nan
    img = img * tmp

    return img


def create_SMOTE(csv_path, data_path, idx):
    imgs, lbs, ids, sls = [], [], [], []
    data = pd.read_csv(csv_path)
    for i in idx:
        name = np.asarray(data.iloc[i, 0])
        image = np.nan_to_num(np.fromfile(data_path + str(name)))
        label = str(np.asarray(data.iloc[i, 3])).zfill(3)
        id = str(np.asarray(data.iloc[i, 1])).zfill(3)
        slice = str(np.asarray(data.iloc[i, 2])).zfill(3)

        labels = np.asarray(data.iloc[i, 3])
        ids = np.expand_dims(np.asarray(data.iloc[i, 1]), axis=1)
        slices = np.expand_dims(np.asarray(data.iloc[i, 2]), axis=1)
        imgs.append(np.concatenate((image, ids, slices), axis=0))

        lbs.append(labels)
        # lbs.append(label + id + slice)
        # lbs = np.hstack((lbs, label))
        # ids = np.hstack((ids, id))
        # sls = np.hstack((sls, slice))
    imgs = np.asarray(imgs)
    lbs = np.asarray(lbs)
    sm = SMOTE()
    imgs_smote, y_smote = sm.fit_resample(imgs, lbs)

    return imgs_smote, y_smote


def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def get_imbalanced(csv_path, idx):
    data = pd.read_csv(csv_path)
    labels = np.asarray(data.iloc[idx, 3])
    u_labels = np.unique(labels)
    n_classes = np.zeros((1, u_labels.size))
    for i in u_labels:
        n_classes[0, i] = np.count_nonzero(labels == u_labels[i])
    min_idx = np.asarray(np.where(n_classes == np.amin(n_classes))[0]).squeeze()
    imb_idx = idx[np.asarray(np.where(labels == u_labels[min_idx])).squeeze()]
    return imb_idx


def get_mean_and_std(dataset, channels, max_load=10000):
    '''Compute the mean and std value of dataset.'''
    # mean = torch.zeros(channels)
    # std = torch.zeros(channels)
    median = np.zeros(channels)
    iq_r = np.zeros(channels)
    print('==> Computing mean and std..')
    N = min(max_load, len(dataset))
    for i, data in enumerate(dataset, 0):
        im,_,_,_ = data
        for j in range(channels):
            img = im[j,:,:].numpy()
            median[j] += np.median(img)
            iq_r[j] += iqr(img)
            # tmp = (im[j,:,:] != 0).astype(int)
            # tmp = ndimage.binary_fill_holes(tmp).astype(int)
            # tidx = np.asarray(np.where(tmp.reshape(-1) == 1)).T
            # mean[j] += tidx.float().mean()
            # std[j] += tidx.float().std()
            # mean[j] += im[j, :, :].float().mean()
            # std[j] += im[j, :, :].float().std()
    mean = torch.from_numpy(np.divide(median, N))
    std = torch.from_numpy(np.divide(iq_r, N)) # ACTUALLY INTERQUARTILE RANGE
    # mean.div_(N).tolist()
    # std.div_(N).tolist()
    return mean, std


def randomCrop(img, size):
    w, h = img.shape
    th, tw = size
    i = random.randint(0, w - tw)
    j = random.randint(0, h - th)
    img = img[i:i + th, j:j + tw]

    return img

# made area outside tumour NaNs to see how much of image was tumor, if it was under threshold
# take out hard coded values
def removeSmallScans(csv_path, file_path, thresh):
    data = pd.read_csv(csv_path)
    image = np.asarray(data.iloc[:,0])
    nz = np.zeros(len(image))
    o_idx = list(range(len(image)))
    for i in o_idx:
        name = image[i]
        img = np.fromfile(file_path + str(name))
        img = np.reshape(img, (299, 299), order='F') # shouldn't hard code this
        # img = cropND(img, (260, 260))
        img[np.isnan(img)] = 0
        img = resize(img, (224, 224), anti_aliasing=True, mode='reflect')
        tmp = (img != 0).astype(float)
        tmp = ndimage.binary_fill_holes(tmp).astype(float)
        tmp[tmp == 0] = np.nan
        img = img * tmp
        img = binaryImage(img)
        # img = cv.convertScaleAbs(img)
        # contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # contour_area = cv.contourArea(np.float32(contours))
        # area = contour_area
        area = np.count_nonzero(img)
        nz[i] = area
        # print(area)
    r_idx = np.asarray(np.where((nz<thresh))).squeeze()
    n_idx = np.delete(o_idx, r_idx)
    print(len(r_idx))

    return n_idx

# data splitting train test validate
def separateClasses(csv_path, idx, split, val, seed):
    data = pd.read_csv(csv_path)
    label = np.asarray(data.iloc[idx, 3])
    id_data = np.asarray(data.iloc[idx, 1])
    z_idx = np.asarray(np.where(label == 0)).squeeze() # zero index
    o_idx = np.asarray(np.where(label == 1)).squeeze() # one index
    id_z = id_data[z_idx]
    id_o = id_data[o_idx]
    uid_z = np.unique(id_z) # how keeping patients slices together in train and test
    uid_o = np.unique(id_o)
    np.random.seed(seed)
    np.random.shuffle(uid_z)
    np.random.shuffle(uid_o)
    split_z = int(split * len(uid_z))
    split_o = int(split * len(uid_o))
    u_train_z = uid_z[:split_z]
    u_train_o = uid_o[:split_o]
    u_val_z = []
    u_val_o = []

    u_test_z = uid_z[split_z:]
    u_test_o = uid_o[split_o:]
    if val == 1:
        u_val_z = u_test_z[:int(len(u_test_z) * 0.5)]
        u_val_o = u_test_o[:int(len(u_test_o) * 0.5)]
        u_test_z = u_test_z[int(len(u_test_z) * 0.5):]
        u_test_o = u_test_o[int(len(u_test_o) * 0.5):]
    tr_z, train_z = [], []
    for i in u_train_z:
        tr_z = np.asarray(np.where(id_data == i)).squeeze()
        if len(tr_z.shape) == 0:
            tr_z = np.expand_dims(tr_z, axis=0)
        train_z = np.concatenate((train_z, tr_z))

    tr_o, train_o = [], []
    for i in u_train_o:
        tr_o = np.asarray(np.where(id_data == i)).squeeze()
        if len(tr_o.shape) == 0:
            tr_o = np.expand_dims(tr_o, axis=0)
        train_o = np.concatenate((train_o, tr_o))

    tst_z, test_z = [], []
    for i in u_test_z:
        tst_z = np.asarray(np.where(id_data == i)).squeeze()
        if len(tst_z.shape) == 0:
            tst_z = np.expand_dims(tst_z, axis=0)
        test_z = np.concatenate((test_z, tst_z))

    tst_o, test_o = [], []
    for i in u_test_o:
        tst_o = np.asarray(np.where(id_data == i)).squeeze()
        if len(tst_o.shape) == 0:
            tst_o = np.expand_dims(tst_o, axis=0)
        test_o = np.concatenate((test_o, tst_o))

    val_z, val_o = [], []
    if val == 1:
        v_z, val_z = [], []
        for i in u_val_z:
            v_z = np.asarray(np.where(id_data == i)).squeeze()
            if len(v_z.shape) == 0:
                v_z = np.expand_dims(v_z, axis=0)
            val_z = np.concatenate((val_z, v_z))

        v_o, val_o = [], []
        for i in u_val_o:
            v_o = np.asarray(np.where(id_data == i)).squeeze()
            if len(v_o.shape) == 0:
                v_o = np.expand_dims(v_o, axis=0)
            val_o = np.concatenate((val_o, v_o))
    print(len(train_z), len(train_o))
    print(len(val_z), len(val_o))
    print(len(test_z), len(test_o))

    train = idx[np.concatenate((train_z, train_o)).astype(int)]
    test = idx[np.concatenate((test_z, test_o)).astype(int)]
    if val == 1:
        valid = idx[np.concatenate((val_z,val_o)).astype(int)]
        sets = (train, (valid, test))
    else:
        sets = (train, test)
    return sets


# def separateClasses(csv_path, idx, split, val, seed):
#     data = pd.read_csv(csv_path)
#     label = np.asarray(data.iloc[idx, 3])
#     id_data = np.asarray(data.iloc[idx, 1])
#     z_idx = np.asarray(np.where(label == 0)).squeeze()
#     o_idx = np.asarray(np.where(label == 1)).squeeze()
#     np.random.seed(seed)
#     np.random.shuffle(z_idx)
#     np.random.shuffle(o_idx)
#     split_z = int(split * len(z_idx))
#     split_o = int(split * len(o_idx))
#     train_z = z_idx[:split_z]
#     train_o = o_idx[:split_o]
#     test_z = z_idx[split_z:]
#     test_o = o_idx[split_o:]
#     if val == 1:
#         val_z = z_idx[split_z:]
#         val_o = o_idx[split_o:]
#
#     unique_id = np.unique(id_data)
#     tick = 0
#     for i in unique_id:
#         tmp_tz = np.asarray(np.where(id_data[train_z] == i)).squeeze()
#         if len(tmp_tz.shape) == 0:
#             tmp_tz = np.expand_dims(tmp_tz, axis=0)
#         tmp_tsz = np.asarray(np.where(id_data[test_z] == i)).squeeze()
#         if len(tmp_tsz.shape) == 0:
#             tmp_tsz = np.expand_dims(tmp_tsz, axis=0)
#         tmp_to = np.asarray(np.where(id_data[train_o] == i)).squeeze()
#         if len(tmp_to.shape) == 0:
#             tmp_to = np.expand_dims(tmp_to, axis=0)
#         tmp_tso = np.asarray(np.where(id_data[test_o] == i)).squeeze()
#         if len(tmp_tso.shape) == 0:
#             tmp_tso = np.expand_dims(tmp_tso, axis=0)
#
#         if 0.6*tmp_tz.size > tmp_tsz.size:
#             train_z = np.concatenate((train_z, test_z[tmp_tsz]))
#             test_z = np.delete(test_z, tmp_tsz)
#             if val == 1:
#                 val_z = np.delete(val_z, tmp_tsz)
#         if 0.6*tmp_tz.size <= tmp_tsz.size:
#             if val == 1:
#                 if (tick % 2) == 0:
#                     val_z = np.concatenate((val_z, train_z[tmp_tz]))
#                     test_z = np.delete(test_z, tmp_tz)
#                 else:
#                     val_z = np.delete(val_z, tmp_tz)
#                     test_z = np.concatenate((test_z, train_z[tmp_tz]))
#             else:
#                 test_z = np.concatenate((test_z, train_z[tmp_tz]))
#             train_z = np.delete(train_z, tmp_tz)
#         if 0.6*tmp_to.size > tmp_tso.size:
#             train_o = np.concatenate((train_o, test_o[tmp_tso]))
#             test_o = np.delete(test_o, tmp_tso)
#             if val == 1:
#                 val_o = np.delete(val_o, tmp_tso)
#         if 0.6*tmp_to.size <= tmp_tso.size:
#             if val == 1:
#                 if (tick % 2) == 0:
#                     val_o = np.concatenate((val_o, train_o[tmp_to]))
#                     test_o = np.delete(test_o, tmp_to)
#                 else:
#                     val_o = np.delete(val_o, tmp_to)
#                     test_o = np.concatenate((test_o, train_o[tmp_to]))
#             else:
#                 test_o = np.concatenate((test_o, train_o[tmp_to]))
#             train_o = np.delete(train_o, tmp_to)
#         tick = tick + 1
#
#     print(len(train_z), len(train_o))
#     if val == 1:
#         print(len(val_z), len(val_o))
#         valid = idx[np.concatenate((val_z, val_o))]
#     print(len(test_z), len(test_o))
#
#     train = idx[np.concatenate((train_z, train_o))]
#     test = idx[np.concatenate((test_z, test_o))]
#     print(valid)
#     print('Now look at test for comparison')
#     print(test)
#     if val == 1:
#         sets = (train, (valid, test))
#     else:
#         sets = (train, test)
#     return sets


def translation(img, tx, ty):
    rows, cols = img.shape

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img[np.isnan(img)] = 0
    img = cv.warpAffine(img, M, (cols, rows))
    tmp = (img != 0).astype(float)
    tmp = ndimage.binary_fill_holes(tmp).astype(float)
    tmp[tmp == 0] = np.nan
    img = img * tmp

    return img
