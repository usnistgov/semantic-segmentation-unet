#!/bin/bash

# ************************************
# MODIFY THESE OPTIONS
# Modify this: which gpu (according to nvidia-smi) do you want to use for training
# this must be a single number. E.g "2"
GPU="0"

# where is your image data is for inferencing
input_data_directory="/path/to/your/images"
input_data_directory="/scratch/small-data-cnns/source_data/Concrete_raw"

# where to save your results
output_directory="/path/to/your/results"
output_directory="/home/mmajursk/Gitlab/Semantic-Segmentation/SegNet/inference"

# which model checkpoint to use for inferencing
checkpoint_filepath="/path/to/your/model/checkpoint/model.ckpt"
checkpoint_filepath="/home/mmajursk/Gitlab/Semantic-Segmentation/SegNet/model/checkpoint/model.ckpt"

# how many classes exist in your training dataset (e.g. 2 for binary segmentation)
number_classes=2

use_tiling=1 # {0, 1} # use tiling when the images being inferenced are too small to fit in GPU memory
tile_size=256
image_format='tif'

# MODIFY THESE OPTIONS
# ************************************


# DO NOT MODIFY ANYTHING BELOW

# limit the script to only the GPUs you selected above
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${GPU}


if [ ${use_tiling} -eq 0 ]
then
	python3 inference.py --checkpoint_filepath=${checkpoint_filepath} --image_folder=${input_data_directory} --output_folder=${output_directory} --number_classes=${number_classes} --image_format=${image_format}
else
	python3 inference_tiling.py --checkpoint_filepath=${checkpoint_filepath} --image_folder=${input_data_directory} --output_folder=${output_directory} --number_classes=${number_classes} --tile_size=${tile_size} --image_format=${image_format}
fi