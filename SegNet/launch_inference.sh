#!/bin/bash

# ************************************
# MODIFY THESE OPTIONS
# Modify this: which gpu (according to nvidia-smi) do you want to use for training
# this must be a single number. E.g "2"
GPU="0"

# where is your image data is for inferencing
input_data_directory="/path/to/your/images"

# where to save your results
output_directory="/path/to/your/results"

# which model checkpoint to use for inferencing
checkpoint_filepath="/path/to/your/model/checkpoint/model.ckpt"

# how many classes exist in your training dataset (e.g. 2 for binary segmentation)
number_classes=4

# how big are your images, this is required to load the model for your image size
image_width=512
image_height=512

# this script assumes you have enough gpu memory to run a full sized image through the network in a single pass. It will not handle tiling of the input image through the network

# MODIFY THESE OPTIONS
# ************************************


# DO NOT MODIFY ANYTHING BELOW

# limit the script to only the GPUs you selected above
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${GPU}


python inference.py --checkpoint_filepath=${checkpoint_filepath} --image_folder=${input_data_directory} --output_folder=${output_directory} --number_classes=${number_classes} --image_height=${image_height} --image_width=${image_width}