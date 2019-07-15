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
saved_model_filepath="/path/to/your/model/checkpoint/ckpt"

use_tiling=1 # {0, 1} # use tiling when the images being inferenced are too small to fit in GPU memory
tile_size=256
image_format='tif'

# END MODIFY THESE OPTIONS
# ************************************


# DO NOT MODIFY ANYTHING BELOW

# limit the script to only the GPUs you selected above
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${GPU}


python3 inference.py --saved_model_filepath=${saved_model_filepath} --image_folder=${input_data_directory} --output_folder=${output_directory} --image_format=${image_format} --use_tiling=${use_tiling} -tile_size=${tile_size}
