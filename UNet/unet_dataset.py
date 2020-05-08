import os
import numpy as np

import torch
import torch.utils.data

import random
import lmdb
from torch.utils.data import WeightedRandomSampler

from isg_ai_pb2 import ImageMaskPair
import augment


class UnetDataset(torch.utils.data.Dataset):
    """
    data set for UNet, image-mask pairs
    """

    def __init__(self, lmdb_filepath, augment=False):
        self.lmdb_filepath = lmdb_filepath

        self.augment = augment
        self.nb_classes = 2
        self.__init_database()
        self.lmdb_txn = self.lmdb_env.begin(write=False)  # hopefully a shared instance across all threads

    def __init_database(self):
        random.seed()

        # get a list of keys from the lmdb
        self.keys_flat = list()
        self.keys = list()
        self.keys.append(list())  # there will always be at least one class

        self.lmdb_env = lmdb.open(self.lmdb_filepath, map_size=int(2e10), readonly=True)  # 1e10 is 10 GB

        #present_classes_str_flat = list()

        datum = ImageMaskPair()
        print('Initializing image database')

        with self.lmdb_env.begin(write=False) as lmdb_txn:
            cursor = lmdb_txn.cursor()

            # move cursor to the first element
            cursor.first()
            # get the first serialized value from the database and convert from serialized representation
            datum.ParseFromString(cursor.value())
            # record the image size
            self.image_size = [datum.img_height, datum.img_width, datum.channels]

            # iterate over the database getting the keys
            for key, val in cursor:
                self.keys_flat.append(key)

        print('Dataset has {} examples'.format(len(self.keys_flat)))

    def get_image_count(self):
        # tie epoch size to the number of images
        return int(len(self.keys_flat))

    def get_image_size(self):
        return self.image_size

    def get_image_tensor_shape(self):
        # HWC to CHW
        #return [self.image_size[2], self.image_size[0], self.image_size[1]]
        return [self.image_size[0], self.image_size[1], self.image_size[2]]

    def get_label_tensor_shape(self):
        return [self.image_size[0], self.image_size[1]]

    def get_number_classes(self):
        return self.number_classes

    def get_image_shape(self):
        return self.image_size

    def __len__(self):
        return len(self.keys_flat)

    @staticmethod
    def format_image(x):
        # reshape into tensor (CHW)
        x = np.transpose(x, [2, 0, 1])
        return x

    @staticmethod
    def zscore_normalize(x):
        x = x.astype(np.float32)

        std = np.std(x)
        mv = np.mean(x)
        if std <= 1.0:
            # normalize (but dont divide by zero)
            x = (x - mv)
        else:
            # z-score normalize
            x = (x - mv) / std
        return x

    def __getitem__(self, index):
        datum = ImageMaskPair()  # create a datum for decoding serialized caffe_pb2 objects
        fn = self.keys_flat[index]

        # extract the serialized image from the database
        value = self.lmdb_txn.get(fn)
        # convert from serialized representation
        datum.ParseFromString(value)

        # convert from string to numpy array
        img = np.fromstring(datum.image, dtype=datum.img_type)
        # reshape the numpy array using the dimensions recorded in the datum
        img = img.reshape((datum.img_height, datum.img_width, datum.channels))
        if np.any(img.shape != np.asarray(self.image_size)):
            raise RuntimeError("Encountered unexpected image shape from database. Expected {}. Found {}.".format(self.image_size, img.shape))
        # convert from string to numpy array
        M = np.fromstring(datum.mask, dtype=datum.mask_type)
        # reshape the numpy array using the dimensions recorded in the datum
        M = M.reshape(datum.img_height, datum.img_width, datum.channels)

        # format the image into a tensor
        img = self.format_image(img)
        img = self.zscore_normalize(img)

        M = M.astype(np.int32)
        # convert to a one-hot (HWC) representation
        h, w, d = M.shape
        M = M.reshape(-1)
        fM = np.zeros((len(M), self.nb_classes), dtype=np.int32)
        fM[np.arange(len(M)), M] = 1
        fM = fM.reshape((h, w, d, self.nb_classes))

        img = torch.from_numpy(img)
        M = torch.from_numpy(M)

        return img, M
