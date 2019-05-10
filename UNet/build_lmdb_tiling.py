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
    img = skimage.io.imread(fp, as_gray=True)
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


def process_slide_tiling(img, msk, tile_size):
    # get the height of the image
    height = img.shape[0]
    width = img.shape[1]
    delta = int(tile_size - int(0.1*tile_size))

    assert img.shape == msk.shape, 'Image and Mask must be the same dimensions/shape'

    img_list = []
    msk_list = []

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

            # crop out the tile
            img_pixels = img[y_st:y_end, x_st:x_end]
            msk_pixels = msk[y_st:y_end, x_st:x_end]

            img_list.append(img_pixels)
            msk_list.append(msk_pixels)

    return (img_list, msk_list)


def generate_database(img_list, database_name, image_filepath, mask_filepath, output_folder, tile_size):
    output_image_lmdb_file = os.path.join(output_folder, database_name)

    if os.path.exists(output_image_lmdb_file):
        print('Deleting existing database')
        shutil.rmtree(output_image_lmdb_file)

    image_env = lmdb.open(output_image_lmdb_file, map_size=int(5e12))
    image_txn = image_env.begin(write=True)

    with open(os.path.join(output_image_lmdb_file, 'img_filenames.csv'), 'w') as csvfile:
        for fn in img_list:
            csvfile.write(fn + '\n')

    for i in range(len(img_list)):
        print('  {}/{}'.format(i, len(img_list)))
        txn_nb = 0
        img_file_name = img_list[i]
        block_key = img_file_name.replace('.tif','')

        img = read_image(os.path.join(image_filepath, img_file_name))
        msk = read_image(os.path.join(mask_filepath, img_file_name))
        msk = msk.astype(np.uint8)

        # convert the image mask pair into tiles
        img_tile_list, msk_tile_list = process_slide_tiling(img, msk, tile_size)

        for k in range(len(img_tile_list)):
            img = img_tile_list[k]
            msk = msk_tile_list[k]
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
    parser.add_argument('--tile_size', dest='tile_size', type=int, help='The size of the tiles to crop out of the source images, striding across all available pixels in the source images', default=512)

    args = parser.parse_args()
    image_folder = args.image_folder
    mask_folder = args.mask_folder
    output_folder = args.output_folder
    dataset_name = args.dataset_name
    train_fraction = args.train_fraction
    image_format = args.image_format
    annotation_count = args.annotation_count
    annotation_count = np.asarray(annotation_count, dtype=np.float32)
    tile_size = args.tile_size

    assert tile_size % 16 == 0, 'UNet requires tiles with shapes that are multiples of 16'

    if image_format.startswith('.'):
        # remove leading period
        image_format = image_format[1:]

    # ****************************************************

    image_folder = os.path.abspath(image_folder)
    mask_folder = os.path.abspath(mask_folder)
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
    generate_database(train_img_files, database_name, image_folder, mask_folder, output_folder, tile_size)

    print('building test database')
    database_name = 'test-{}.lmdb'.format(dataset_name)
    generate_database(test_img_files, database_name, image_folder, mask_folder, output_folder, tile_size)






