import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import skimage.io
import numpy as np
import os
import skimage
import skimage.transform
from isg_ai_pb2 import ImageMaskPair
import shutil
import lmdb
import random
import argparse


def read_image(fp):
    img = skimage.io.imread(fp, as_grey=True)
    return img


def write_img_to_db(txn, img, msk, key_str):
    if type(img) is not np.ndarray:
        raise Exception("Img must be numpy array to store into db")
    if type(msk) is not np.ndarray:
        raise Exception("Img must be numpy array to store into db")

    # get the list of labels in the image
    labels = np.unique(msk)

    datum = ImageMaskPair()
    datum.channels = 1
    datum.img_height = img.shape[0]
    datum.img_width = img.shape[1]

    datum.img_type = img.dtype.str
    datum.mask_type = msk.dtype.str

    datum.image = img.tobytes()
    datum.mask = msk.tobytes()

    datum.labels = labels.tobytes()

    txn.put(key_str.encode('ascii'), datum.SerializeToString())
    return


def enforce_size_multiple(img):
    h = img.shape[0]
    w = img.shape[1]

    # this function crops the input image down slightly to be a size multiple of 16
    factor = 32
    tgt_h = int(np.floor(h / factor) * factor)
    tgt_w = int(np.floor(w / factor) * factor)

    dh = h - tgt_h
    dw = w - tgt_w

    img = img[int(dh/2):, int(dw/2):]
    img = img[0:tgt_h, 0:tgt_w]

    return img


def generate_database(img_list, database_name, image_filepath, mask_filepath, output_folder):
    output_image_lmdb_file = os.path.join(output_folder, database_name)

    if os.path.exists(output_image_lmdb_file):
        print('Deleting existing database')
        shutil.rmtree(output_image_lmdb_file)

    image_env = lmdb.open(output_image_lmdb_file, map_size=int(5e12))
    image_txn = image_env.begin(write=True)

    for i in range(len(img_list)):
        print('  {}/{}'.format(i, len(img_list)))
        txn_nb = 0
        img_file_name = img_list[i]
        block_key = img_file_name.replace('.tif','')

        img = read_image(os.path.join(image_filepath, img_file_name))
        msk = read_image(os.path.join(mask_filepath, img_file_name))

        img = enforce_size_multiple(img)
        msk = enforce_size_multiple(msk)

        key_str = '{}_{:08d}'.format(block_key, txn_nb)
        txn_nb += 1
        write_img_to_db(image_txn, img, msk, key_str)

        if txn_nb % 1000 == 0:
            image_txn.commit()
            image_txn = image_env.begin(write=True)

    image_txn.commit()
    image_env.close()


if __name__ == "__main__":
    # Define the inputs
    # ****************************************************

    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_lmdb', description='Script which converts two folders of images and masks into a pair of lmdb databases for training.')

    parser.add_argument('--image_folder', dest='image_folder', type=str, help='filepath to the folder containing the images', required=True)
    parser.add_argument('--mask_folder', dest='mask_folder', type=str, help='filepath to the folder containing the masks', required=True)
    parser.add_argument('--output_folder', dest='output_folder', type=str, help='filepath to the folder where the outputs will be placed', required=True)
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, help='name of the dataset to be used in creating the lmdb files', required=True)
    parser.add_argument('--train_fraction', dest='train_fraction', type=float, help='what fraction of the dataset to use for training (0.0, 1.0)', default=0.8)
    parser.add_argument('--image_format', dest='image_format', type=str, help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')
    parser.add_argument('--annotation_count', dest='annotation_count', type=str, help='Number of annotations to sample randomly without replacement from images on disk to add to lmdb database (use inf to select all)', default='inf')

    args = parser.parse_args()
    image_folder = args.image_folder
    mask_folder = args.mask_folder
    output_folder = args.output_folder
    dataset_name = args.dataset_name
    train_fraction = args.train_fraction
    image_format = args.image_format
    annotation_count = args.annotation_count
    annotation_count = np.asarray(annotation_count, dtype=np.float32)

    if image_format.startswith('.'):
        # remove leading period
        image_format = image_format[1:]

    # ****************************************************

    image_folder = os.path.abspath(image_folder)
    output_folder = os.path.abspath(output_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # find the image files for which annotations exist
    img_files = [f for f in os.listdir(mask_folder) if f.endswith('.{}'.format(image_format))]

    # in place shuffle
    random.shuffle(img_files)

    if np.isfinite(annotation_count):
        annotation_count = min(int(annotation_count), len(img_files))
        img_files = img_files[0:annotation_count]

    idx = int(train_fraction * len(img_files))
    train_img_files = img_files[0:idx]
    test_img_files = img_files[idx:]

    print('building train database')
    database_name = 'train-{}.lmdb'.format(dataset_name)
    generate_database(train_img_files, database_name, image_folder, mask_folder, output_folder)

    print('building test database')
    database_name = 'test-{}.lmdb'.format(dataset_name)
    generate_database(test_img_files, database_name, image_folder, mask_folder, output_folder)


