import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset
import pandas as pd
from rfs_preprocessing import brightnessContrast, cropND, randomCrop, translation
from skimage.transform import resize
from scipy import ndimage
from scipy.ndimage import rotate
import random
from sklearn.preprocessing import MinMaxScaler
import torchvision.transforms as transforms
import pywt


# Class to build the dataset for the Cholangio data to detect RFS
class RFS_Loader(Dataset):

    def __init__(self, csv_path, data_path, idx, mean, std, img_dim):
        self.data = pd.read_csv(csv_path)
        self.path = data_path
        self.image = np.asarray(self.data.iloc[idx, 0])
        self.label = np.asarray(self.data.iloc[idx, 3])
        self.id = np.asarray(self.data.iloc[idx, 1])
        self.slice = np.asarray(self.data.iloc[idx, 2])
        self.dim = img_dim
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)] # mean and std come from preprocessing
        )

    def __getitem__(self, index):
        name = self.image[index]
        hgp = self.label[index] # replace hpg with rfs
        id_orig = self.id[index]
        slice_orig = self.slice[index]
        dim = self.dim
        data_path = self.path

        img = np.fromfile(data_path + str(name))

        # fixing dimensions problem
        hgp = hgp.squeeze()
        id_orig = id_orig.squeeze()
        slice_orig = slice_orig.squeeze()

        # process img
        # fix the hardcoding here, make it read the size in and check
        # this has to be the same number of pixels as are in bin file
        # taking from 1D to 2D
        img = np.reshape(img, (299, 299), order='F')
        # img = cropND(img, (260, 260))

        img[np.isnan(img)] = 0
        # compressing image to dimension
        img = resize(img, (dim, dim), anti_aliasing=True, mode='reflect')
        # can probably get rid of this next part
        tmp = (img != 0).astype(float)
        tmp = ndimage.binary_fill_holes(tmp).astype(float)
        tmp[tmp == 0] = np.nan
        img = img * tmp

        # make augmentation option?
        rnum = random.randint(0, 3)
        if rnum == 2:
            img = np.flip(img, 0)
        elif rnum == 1:
            img = np.flip(img, 1)

        # Scale image [0,1]
        img = np.nan_to_num(img) # can ignore this thing,
        img = np.stack((img, img, img), axis=2) # make slices 3D (RGB)
        img = self.transform(img)
        return img, hgp, id_orig, slice_orig

    def __len__(self):

        return len(self.label)


class SMOTE_Loader(Dataset):

    def __init__(self, imgs_smote, y_smote, mean, std, img_dim):
        self.image = np.asarray(imgs_smote)
        self.label = np.asarray(y_smote)
        self.dim = img_dim
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)]
        )

    def __getitem__(self, index):
        img = self.image[index][:-2]
        hgp = int(self.label[index])
        id_orig = int(self.image[index][-2])
        slice_orig = int(self.image[index][-1])
        dim = self.dim

        # hgp = hgp.squeeze()
        # id_orig = id_orig.squeeze()
        # slice_orig = slice_orig.squeeze()

        # process img
        img = np.reshape(img, (299, 299), order='F')
        # img = cropND(img, (260, 260))

        img = resize(img, (dim, dim), anti_aliasing=True, mode='reflect')

        rnum = random.randint(0, 3)
        if rnum == 2:
            img = np.flip(img, 0)
        elif rnum == 1:
            img = np.flip(img, 1)

        # Scale image [0,1]
        img = np.stack((img, img, img), axis=2)
        img = self.transform(img)
        return img, hgp, id_orig, slice_orig

    def __len__(self):

        return len(self.label)

# Data augmentation loader - called by JoinDatasets
class SyntheticLoader(Dataset):

    def __init__(self, csv_path, data_path, idx, synth, mean, std, img_dim):
        self.data = pd.read_csv(csv_path)
        self.path = data_path
        self.image = np.asarray(self.data.iloc[idx, 0])
        self.label = np.asarray(self.data.iloc[idx, 3])
        self.id = np.asarray(self.data.iloc[idx, 1])
        self.slice = np.asarray(self.data.iloc[idx, 2])
        self.newdata = synth
        self.dim = img_dim
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)]
        )

    def __getitem__(self, index):
        name = self.image[index]
        hgp = self.label[index]
        id_synth = self.id[index]
        slice_synth = self.slice[index]
        dim = self.dim

        data_path = self.path

        img = np.fromfile(data_path + str(name))
        hgp = hgp.squeeze()
        id_synth = id_synth.squeeze()
        slice_synth = slice_synth.squeeze()

        # process img
        img = np.reshape(img, (299, 299), order='F')
        # img = cropND(img, (260, 260))
        img[np.isnan(img)] = 0
        img = resize(img, (dim, dim), anti_aliasing=True, mode='reflect')
        tmp = (img != 0).astype(float)
        tmp = ndimage.binary_fill_holes(tmp).astype(float)
        tmp[tmp == 0] = np.nan
        img = img * tmp

        rnum = random.randint(0, 3)
        if self.newdata == 'hflip':
            img = np.flip(img, 0)
        elif self.newdata == 'vflip':
            img = np.flip(img, 1)
        elif self.newdata == 'crop':
            sig = 0
            while sig == 0:
                img = randomCrop(img, (210, 210))
                img[np.isnan(img)] = 0
                if img.mean() > 0:
                    break
            img = resize(img, (dim, dim), anti_aliasing=True, mode='reflect')
            tmp = (img != 0).astype(float)
            tmp = ndimage.binary_fill_holes(tmp).astype(float)
            tmp[tmp == 0] = np.nan
            img = img * tmp
        elif self.newdata == 'trans':
            tx = np.random.randint(-20, 50)
            ty = np.random.randint(-40, 50)
            img = translation(img, tx, ty)
        elif self.newdata == 'bright':
            img = brightnessContrast(img)
        elif self.newdata == 'noise':
            noise = np.random.randn(dim, dim) * 0.2
            tmp = (~np.isnan(img)).astype(float)
            noise = noise * tmp
            img = img + noise
        elif self.newdata == 'rotate':
            deg = random.randint(0, 360)
            img[np.isnan(img)] = 0
            tmp = (img != 0).astype(float)
            tmp = rotate(tmp, deg, reshape=False)
            img = rotate(img, deg, reshape=False)
            tmp = (tmp >= 0.65).astype(float)
            tmp = ndimage.binary_fill_holes(tmp).astype(float)
            tmp[tmp == 0] = np.nan
            img = img * tmp
        else:
            img = img
        if rnum == 2:
            img = np.flip(img, 0)

        # Scale image [0,1]
        img = np.nan_to_num(img)
        img = np.stack((img, img, img), axis=2)
        img = self.transform(img)
        return img, hgp, id_synth, slice_synth

    def __len__(self):

        return len(self.label)

# ignore this
class WaveletLoader(Dataset):

    def __init__(self, csv_path, data_path, idx, synth, mean, std):
        self.data = pd.read_csv(csv_path)
        self.path = data_path
        self.image = np.asarray(self.data.iloc[idx, 0])
        self.label = np.asarray(self.data.iloc[idx, 3])
        self.id = np.asarray(self.data.iloc[idx, 1])
        self.slice = np.asarray(self.data.iloc[idx, 2])
        self.newdata = synth
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)]
        )

    def __getitem__(self, index):
        name = self.image[index]
        hgp = self.label[index]
        id_synth = self.id[index]
        slice_synth = self.slice[index]

        data_path = self.path

        img = np.fromfile(data_path + str(name))
        hgp = hgp.squeeze()
        id_synth = id_synth.squeeze()
        slice_synth = slice_synth.squeeze()

        # process img
        img = np.reshape(img, (299, 299), order='F')
        img = cropND(img, (256, 256))

        rnum = random.randint(0, 3)
        if self.newdata == 'hflip':
            img = np.flip(img, 0)
        elif self.newdata == 'vflip':
            img = np.flip(img, 1)
        elif self.newdata == 'crop':
            sig = 0
            while sig == 0:
                img = randomCrop(img, (200, 200))
                img[np.isnan(img)] = 0
                if img.mean() > 0:
                    break
            img = resize(img, (256, 256), anti_aliasing=True, mode='reflect')
            tmp = (img != 0).astype(float)
            tmp = ndimage.binary_fill_holes(tmp).astype(float)
            tmp[tmp == 0] = np.nan
            img = img * tmp
        elif self.newdata == 'trans':
            tx = np.random.randint(-20, 75)
            ty = np.random.randint(-40, 75)
            img = translation(img, tx, ty)
        elif self.newdata == 'bright':
            img = brightnessContrast(img)
        elif self.newdata == 'noise':
            noise = np.random.randn(256, 256) * 0.2
            tmp = (~np.isnan(img)).astype(float)
            noise = noise * tmp
            img = img + noise
        elif self.newdata == 'rotate':
            deg = random.randint(0, 360)
            img[np.isnan(img)] = 0
            tmp = (img != 0).astype(float)
            tmp = rotate(tmp, deg, reshape=False)
            img = rotate(img, deg, reshape=False)
            tmp = (tmp >= 0.65).astype(float)
            tmp = ndimage.binary_fill_holes(tmp).astype(float)
            tmp[tmp == 0] = np.nan
            img = img * tmp
        else:
            img = img
        if rnum == 2:
            img = np.flip(img, 0)

        coeffs = pywt.dwt2(img, 'haar')
        (cA,(cH,cV,cD)) = coeffs
        cAt, cHt, cVt, cDt = cA.reshape(-1,1), cH.reshape(-1,1), cV.reshape(-1,1), cD.reshape(-1,1)

        tmpA, tmpH, tmpV, tmpD = (~np.isnan(cA)).astype(int), (~np.isnan(cH)).astype(int), (~np.isnan(cV)).astype(int), \
                                 (~np.isnan(cD)).astype(int)
        tidxA, tidxH, tidxV, tidxD = np.asarray(np.where(tmpA.reshape(-1) == 1)).T, np.asarray(np.where(tmpH.reshape(-1) == 1)).T, \
                                     np.asarray(np.where(tmpV.reshape(-1) == 1)).T, np.asarray(np.where(tmpD.reshape(-1) == 1)).T
        cAg, cHg, cVg, cDg = self.scaler.fit_transform(cAt[tidxA.squeeze()]), self.scaler.fit_transform(cHt[tidxH.squeeze()]), \
                             self.scaler.fit_transform(cVt[tidxV.squeeze()]), self.scaler.fit_transform(cDt[tidxD.squeeze()])

        cAt[tidxA.squeeze()], cHt[tidxH.squeeze()], cVt[tidxV.squeeze()], cDt[tidxD.squeeze()] = cAg, cHg, cVg, cDg
        cAt, cHt, cVt, cDt = cAt.reshape(cA.shape), cHt.reshape(cH.shape), cVt.reshape(cV.shape), cDt.reshape(cD.shape)

        img = np.stack((np.nan_to_num(cAt), np.nan_to_num(cHt), np.nan_to_num(cVt), np.nan_to_num(cDt)), axis=2)

        img = self.transform(img)
        return img, hgp, id_synth, slice_synth

    def __len__(self):

        return len(self.label)


def JoinDatasets(csv_path, data_path, idx, types, mean, std, img_dim, loader):
    init = types[0]
    dataset = types[1:]
    if loader=='wave':
        concatData = WaveletLoader(csv_path, data_path, idx, init, mean, std)
    else:
        # load in data without augmentation
        concatData = SyntheticLoader(csv_path, data_path, idx, init, mean, std, img_dim)
    for i, aug in enumerate(dataset):
        # create augmentations
        synth = SyntheticLoader(csv_path, data_path, idx, aug, mean, std, img_dim)
        # join original and augmented data
        concatData = ConcatDataset((synth, concatData))
    return concatData
