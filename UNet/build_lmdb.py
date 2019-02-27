import skimage.io
import numpy as np
import os
import skimage
import skimage.transform
import csv
from isg_ai_pb2 import ImageMaskPair
import shutil
import lmdb
import random


# Define the inputs
# ****************************************************

output_filepath = '/scratch/small-data-cnns/data/semantic-segmentation/'
database_common_name = 'HES'

image_filepath = '/scratch/small-data-cnns/source_data/HES_raw/' # contains the filtered HES with foreground
mask_filepath = '/scratch/small-data-cnns/source_data/hes_orig/validate/mask/'


# ****************************************************

def read_image(fp):
    img = skimage.io.imread(fp, as_gray=True)
    return img


def write_img_to_db(txn, img, msk, key_str):
    if type(img) is not np.ndarray:
        raise Exception("Img must be numpy array to store into db")
    if type(msk) is not np.ndarray:
        raise Exception("Img must be numpy array to store into db")

    datum = ImageMaskPair()
    datum.channels = 1
    datum.img_height = img.shape[0]
    datum.img_width = img.shape[1]

    datum.img_type = img.dtype.str
    datum.mask_type = msk.dtype.str

    datum.image = img.tobytes()
    datum.mask = msk.tobytes()

    txn.put(key_str.encode('ascii'), datum.SerializeToString())
    return


def generate_database(img_list, database_name):
    output_image_lmdb_file = os.path.join(output_filepath, database_name)

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
        img = img.astype(np.uint16)
        msk = read_image(os.path.join(mask_filepath, img_file_name))
        msk[msk > 0] = 1 # make binary (from labeled mask)

        key_str = '{}_{:08d}'.format(block_key, txn_nb)
        txn_nb += 1
        write_img_to_db(image_txn, img, msk, key_str)

        if txn_nb % 1000 == 0:
            image_txn.commit()
            image_txn = image_env.begin(write=True)

    image_txn.commit()
    image_env.close()


if __name__ == "__main__":
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    # find the image files for which annotations exist
    img_files = [f for f in os.listdir(image_filepath) if f.endswith('.tif')]

    # in place shuffle
    random.shuffle(img_files)

    number_examples = 100

    idx = int(0.5 * len(img_files))
    train_img_files = img_files[0:idx]
    train_img_files = train_img_files[0:number_examples]
    test_img_files = img_files[idx:]
    test_img_files = test_img_files[0:number_examples]

    print('building train database')
    database_name = 'train-{}-{}.lmdb'.format(database_common_name, number_examples)
    generate_database(train_img_files, database_name)

    print('building test database')
    database_name = 'test-{}-{}.lmdb'.format(database_common_name, number_examples)
    generate_database(test_img_files, database_name)


