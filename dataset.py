# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

from typing import Optional, Callable
import os
import numpy as np
import skimage.io
import copy
import torch
import random
import albumentations.pytorch
import logging

# local imports
import augment


class SemanticSegmentationDataset(torch.utils.data.Dataset):
    TRANSFORM_TRAIN = albumentations.Compose(
        [
            albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            augment.ZScoreNorm(),
            albumentations.pytorch.ToTensorV2()

        ]
    )
    TRANSFORM_TEST = albumentations.Compose([
        augment.ZScoreNorm(),
        albumentations.pytorch.ToTensorV2()
    ])

    def __init__(self, image_dirpath: str, mask_dirpath: str, img_ext: str, mask_ext: str, transform: Optional[Callable] = None, tile_size:int = None):
        self.image_dirpath = image_dirpath
        self.mask_dirpath = mask_dirpath
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.tile_size = tile_size  # None means no tiling
        self.test_every_n_steps = None

        self.image_filepath_list = list()
        for root, dirs, files in os.walk(self.image_dirpath):
            for fn in files:
                if fn.endswith(self.img_ext):
                    self.image_filepath_list.append(os.path.join(root, fn))
        self.image_filepath_list.sort()

        self.mask_filepath_list = list()
        for root, dirs, files in os.walk(self.mask_dirpath):
            for fn in files:
                if fn.endswith(self.mask_ext):
                    self.mask_filepath_list.append(os.path.join(root, fn))
        self.mask_filepath_list.sort()

        assert len(self.image_filepath_list) == len(self.mask_filepath_list)

        self.image_list = list()
        self.mask_list = list()
        self.preload_data()

    def set_test_every_n_steps(self, N: int, batch_size: int):
        if N is not None and N > 0:
            new_len = N * batch_size
            logging.info("Setting len(dataset) = {} from {}, due to test_every_n_steps={} and batch_size={}.".format(new_len, len(self.image_list), N, batch_size))
            self.test_every_n_steps = new_len

    def __len__(self):
        if self.test_every_n_steps is not None:
            return int(self.test_every_n_steps)
        else:
            return len(self.image_list)

    def __getitem__(self, index: int):

        # ensure the index is within the valid range of the list
        # this index math enables setting the dataset "len()" longer than the real data
        index = index % len(self.image_list)

        image = self.image_list[index]
        mask = self.mask_list[index]

        # apply augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # loss needs a long type tensor
        mask = mask.type(torch.LongTensor)
        image = image.type(torch.FloatTensor)

        return image, mask

    def get_number_channels(self):
        # this assumes HWC dimension ordering
        img = self.image_list[0]
        if len(img.shape) == 2:
            return 1
        else:
            return img.shape[2]

    def preload_data(self):
        logging.info("Loading the image data into memory")
        use_tiling = False
        if self.tile_size is not None and self.tile_size > 0:
            use_tiling = True
            logging.info("Tiling the images to {}x{}".format(self.tile_size, self.tile_size))

        for i in range(len(self.image_filepath_list)):
            if len(self.image_filepath_list) > 20 and i % int(len(self.image_filepath_list) / 10) == 0:
                logging.info("  loaded {}/{}".format(i, len(self.image_filepath_list)))
            img_fn = self.image_filepath_list[i]
            image = SemanticSegmentationDataset.read_image(img_fn)
            msk_fn = self.mask_filepath_list[i]
            mask = SemanticSegmentationDataset.read_image(msk_fn)

            if use_tiling:
                # break the image into tiles
                i_tile_list, m_tile_list, _ = self.tile_image_mask_pair(image, mask, self.tile_size)
                # extend the pixel data (np.ndarray) lists with the tiles
                self.image_list.extend(i_tile_list)
                self.mask_list.extend(m_tile_list)
            else:
                self.image_list.append(image)
                self.mask_list.append(mask)

    @staticmethod
    def read_image(fp):
        return skimage.io.imread(fp)
        # return PIL.Image.open(fp)

    @staticmethod
    def tile_image_mask_pair(img: np.ndarray, msk: np.ndarray, tile_size, tile_overlap: float = 0.1) -> (list[np.ndarray], list[np.ndarray]):
        tile_overlap = np.clip(tile_overlap, 0.0, 0.99)

        # get the height of the image
        height = img.shape[0]
        width = img.shape[1]

        img_list = list()
        msk_list = list()

        location_list = list()

        delta = int(tile_size - int(tile_overlap * tile_size))
        for x_st in range(0, width, delta):
            for y_st in range(0, height, delta):
                x_end = x_st + tile_size
                y_end = y_st + tile_size
                if x_st < 0 or y_st < 0:
                    # should never happen, but stranger things and all...
                    continue
                if x_end > width:
                    # slide box to fit within image
                    dx = width - x_end
                    x_st = x_st + dx
                    x_end = x_end + dx
                if y_end > height:
                    # slide box to fit within image
                    dy = height - y_end
                    y_st = y_st + dy
                    y_end = y_end + dy

                # handle if the image is smaller than the tile size
                x_st = max(0, x_st)
                y_st = max(0, y_st)
                location_list.append((x_st, y_st))

                # crop out the tile
                img_pixels = img[y_st:y_end, x_st:x_end]
                img_list.append(img_pixels)
                if msk is not None:
                    msk_pixels = msk[y_st:y_end, x_st:x_end]
                    msk_list.append(msk_pixels)

        return img_list, msk_list, location_list

    def set_transforms(self, transforms):
        self.transform = transforms

    def train_val_split(self, val_fraction: float = 0.1):
        if val_fraction is None:
            raise RuntimeError("Requesting train/val split with a val_fraction = None")
        if val_fraction < 0.0 or val_fraction > 1.0:
            raise RuntimeError("Impossible validation fraction {}.".format(val_fraction))

        val_size = int(val_fraction * len(self.image_list))

        idx = list(range(len(self.image_list)))
        random.shuffle(idx)
        v_idx = idx[0:val_size]
        t_idx = idx[val_size:]
        t_idx.sort()
        v_idx.sort()

        # duplicate this instance, but don't deep copy the loaded image data
        img_list = self.image_list
        msk_list = self.mask_list
        self.image_list = None
        self.mask_list = None
        train_dataset = copy.deepcopy(self)
        val_dataset = copy.deepcopy(self)
        self.image_list = img_list
        self.mask_list = msk_list

        train_dataset.image_list = list()
        train_dataset.mask_list = list()
        for i in t_idx:
            train_dataset.image_list.append(self.image_list[i])
            train_dataset.mask_list.append(self.mask_list[i])

        val_dataset.image_list = list()
        val_dataset.mask_list = list()
        for i in v_idx:
            val_dataset.image_list.append(self.image_list[i])
            val_dataset.mask_list.append(self.mask_list[i])

        assert len(train_dataset.image_list) > 0
        assert len(train_dataset.mask_list) > 0
        assert len(val_dataset.image_list) > 0
        assert len(val_dataset.mask_list) > 0

        return train_dataset, val_dataset

