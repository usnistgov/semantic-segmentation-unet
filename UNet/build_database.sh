#!/bin/bash

# ************************************
# MODIFY THESE OPTIONS

#image_folder="/path/to/your/image/folder"
#mask_folder="/path/to/your/mask/folder"

#output_folder="/path/to/output/directory/where/results/are/saved"

image_folder='/home/mmajursk/Downloads/tf2.0-data/images'
mask_folder='/home/mmajursk/Downloads/tf2.0-data/masks'

output_folder='/home/mmajursk/Downloads/tf2.0-data/'

# what common name to use in saving the lmdb dataset
dataset_name="my_dataset"

# what fraction of your data to use for training. Test is 1.0 - train_fraction of the data
train_fraction=0.8 # (0.0, 1.0)

# what format are the images in your image/mask folder
image_format="tif"

use_tiling=1 #{0, 1}
tile_size=256

# END OF MODIFY THESE OPTIONS
# ************************************


python3 build_lmdb.py --image_folder=${image_folder} --mask_folder=${mask_folder} --output_folder=${output_folder} --dataset_name=${dataset_name} --train_fraction=${train_fraction} --image_format=${image_format} --tile_size=${tile_size} --use_tiling=${use_tiling}

