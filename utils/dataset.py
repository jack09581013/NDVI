from torch.utils.data import Dataset, Subset
import torch
import tools
import os
import cv2
import xml.etree.ElementTree as ET
import utils
import numpy as np


class 瑞評大帥哥幫我train資料集(Dataset):
    ROOT = 'D:\Dataset\瑞評大帥哥幫我train'

    def __init__(self, crop_size=(896, 1216), category='all', crop_seed=None):
        self.size = 0
        self.crop_size = crop_size
        self.crop_seed = crop_seed

        self.image_folders = []
        if category == 'all':
            category_folders = os.listdir(self.ROOT)
            for folder in category_folders:
                codes = os.listdir(os.path.join(self.ROOT, folder))
                self.size += len(codes)
                self.image_folders += [os.path.join(folder, code) for code in codes]

        else:
            codes = os.listdir(os.path.join(self.ROOT, category))
            self.size += len(codes)
            self.image_folders += [os.path.join(category, code) for code in codes]

    def __getitem__(self, index):
        X = cv2.imdecode(np.fromfile(os.path.join(self.ROOT, self.image_folders[index], f'GSD_RGB.tiff'),
                                     dtype=np.uint8), -1)
        X = utils.rgb2bgr(X)

        NIR = cv2.imdecode(np.fromfile(os.path.join(self.ROOT, self.image_folders[index], f'GSD_NIR.tiff'),
                                     dtype=np.uint8), -1)
        cropper = RandomCropper(X.shape[0:2], self.crop_size, seed=self.crop_seed)

        X = X.astype('float32')
        NIR = NIR.astype('float32')

        NDVI = np.zeros(NIR.shape, dtype=np.float32)
        mask = (NIR + X[..., 2]) != 0
        NDVI[mask] = (NIR[mask] - X[..., 2][mask]) / (NIR[mask] + X[..., 2][mask])

        X, Y = torch.from_numpy(X), torch.from_numpy(NDVI).unsqueeze(2)
        X, Y = X.permute(2, 0, 1), Y.permute(2, 0, 1)
        X, Y = cropper.crop(X) / 255, cropper.crop(Y)
        return X, Y

    def __len__(self):
        return self.size

    def __str__(self):
        return type(self).__name__


def random_subset(dataset, size, seed=None):
    assert size <= len(dataset), 'subset size cannot larger than dataset'
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    indexes = indexes[:size]
    return Subset(dataset, indexes)


def random_split(dataset, train_ratio=0.8, seed=None):
    assert 0 <= train_ratio <= 1
    train_size = int(train_ratio * len(dataset))
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    train_indexes = indexes[:train_size]
    test_indexes = indexes[train_size:]
    return Subset(dataset, train_indexes), Subset(dataset, test_indexes)


def sub_sampling(X, Y, ratio):
    X = X[:, ::ratio, ::ratio]
    Y = Y[::ratio, ::ratio] / ratio
    return X, Y


class RandomCropper:
    def __init__(self, image_size, crop_size, seed=None):
        H, W = crop_size
        assert image_size[0] >= H, 'image height must larger than crop height'
        assert image_size[1] >= W, 'image width must larger than crop width'

        H_range = image_size[0] - H
        W_range = image_size[1] - W

        np.random.seed(seed)
        if H_range > 0:
            self.min_row = np.random.randint(0, H_range + 1)
        else:
            self.min_row = 0

        if W_range > 0:
            self.min_col = np.random.randint(0, W_range + 1)
        else:
            self.min_col = 0

        self.max_row = self.min_row + H
        self.max_col = self.min_col + W

    def crop(self, I):
        return I[..., self.min_row:self.max_row, self.min_col:self.max_col]
