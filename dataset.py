from typing import Optional, Callable

import os
import numpy as np
import skimage.io
import copy
import torch
import torchvision
import albumentations
import albumentations.pytorch

class SemanticSegmentationDataset(torch.utils.data.Dataset):
    default_tensor_transform = albumentations.Compose(
        [
            albumentations.pytorch.ToTensorV2(),
        ]
    )

    def __init__(self, image_filepath: str, mask_filepath: str, extension: str, transform: Optional[Callable] = None, preload_into_memory: bool = False, cast_grayscale_to_color=False, disable_zscore_norm=False):
        self.image_filepath = image_filepath
        self.mask_filepath = mask_filepath
        self.extension = extension
        self.transform = transform
        self.cast_grayscale_to_color = cast_grayscale_to_color
        self.disable_zscore_norm = disable_zscore_norm

        self.preload_into_memory = preload_into_memory

        self.image_filepath_list = list()
        for root, dirs, files in os.walk(self.image_filepath):
            for fn in files:
                if fn.endswith(self.extension):
                    self.image_filepath_list.append(os.path.join(root, fn))

        self.mask_filepath_list = list()
        for fn in self.image_filepath_list:
            self.mask_filepath_list.append(fn.replace(self.image_filepath, self.mask_filepath))

        self.image_list = None
        self.mask_list = None
        # preload the image data if requested
        self.preload_data()

    def __getitem__(self, index: int):
        if self.preload_into_memory:
            image = self.image_list[index]
            mask = self.mask_list[index]
        else:
            image = SemanticSegmentationDataset.read_image(self.image_filepath_list[index])
            mask = SemanticSegmentationDataset.read_image(self.mask_filepath_list[index])

        image = image.astype(np.float32)
        if mask.dtype == np.floating:
            mask = mask.astype(np.float32)
        else:
            mask = mask.astype(np.int32)

        if not self.disable_zscore_norm:
            image = self.zscore_normalize(image)

        is_1_channel = len(image.shape) == 2 or image.shape[2] == 1
        if is_1_channel and self.cast_grayscale_to_color:
            image = np.dstack((image, image, image))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
        else:
            # get a default transform to generate tensors
            transformed = SemanticSegmentationDataset.default_tensor_transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        return image, mask

    def __len__(self):
        return len(self.image_filepath_list)

    def get_number_channels(self):
        if len(self.image_filepath_list) == 0:
            return None
        # TODO this assumes HWC dimension ordering
        img = SemanticSegmentationDataset.read_image(self.image_filepath_list[0])
        if len(img.shape) == 2:
            return 1
        else:
            return img.shape[2]

    def preload_data(self):
        print("Pre-Loading image data into memory as part of caching")
        self.image_list = list()
        self.mask_list = list()
        if self.preload_into_memory:
            for fn in self.image_filepath_list:
                image = SemanticSegmentationDataset.read_image(fn)
                self.image_list.append(image)
            for fn in self.mask_filepath_list:
                mask = SemanticSegmentationDataset.read_image(fn)
                self.mask_list.append(mask)

    @staticmethod
    def read_image(fp):
        return skimage.io.imread(fp)
        # return PIL.Image.open(fp)

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

