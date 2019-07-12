#!/bin/bash

# ************************************
# MODIFY THESE OPTIONS
# Modify this: which gpu (according to nvidia-smi) do you want to use for training
# this can be a single number, or a list. E.g "3" or "0,1" "0,2,3"
# the training script will use all gpus you list
GPU="0"

# how large is an epoch, or sub/super epoch test dataset evaluation
test_every_n_step=1000
batch_size=8

# where is your training lmdb database
train_database="path/to/the/training/database.lmdb"
test_database="path/to/the/training/database.lmdb"

output_folder="/path/to/output/directory/where/results/are/saved"

# how many classes exist in your training dataset (e.g. 2 for binary segmentation)
number_classes=4

# what learning rate should the network use
learning_rate=3e-4 # Karpathy Constant

use_augmentation=1 # {0, 1}
balance_classes=1 # {0, 1}

# END MODIFY THESE OPTIONS
# ************************************


# limit the script to only the GPUs you selected above
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${GPU}


python train_unet.py --test_every_n_steps=${test_every_n_step} --batch_size=${batch_size} --train_database=${train_database} --test_database=${test_database} --output_dir=${output_folder} --number_classes=${number_classes} --learning_rate=${learning_rate}  --use_augmentation=${use_augmentation} --balance_classes=${balance_classes}