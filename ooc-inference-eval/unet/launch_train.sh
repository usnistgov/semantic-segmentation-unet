#!/bin/bash

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


# ************************************
# MODIFY THESE OPTIONS
# Modify this: which gpu (according to nvidia-smi) do you want to use for training
# this can be a single number, or a list. E.g "3" or "0,1" "0,2,3"
# the training script will use all gpus you list
GPU="0,1,2,3"

# how large is an epoch, or sub/super epoch test dataset evaluation
test_every_n_step=1000
batch_size=8

# where is your training lmdb database
train_database="/mnt/m2/mmajursk/ooc/train-hes.lmdb"
test_database="/mnt/m2/mmajursk/ooc/test-hes.lmdb"

output_folder="/mnt/m2/mmajursk/ooc/model/"

# how many classes exist in your training dataset (e.g. 2 for binary segmentation)
number_classes=2

# what learning rate should the network use
learning_rate=1e-4 # Karpathy Constant

use_augmentation=1 # {0, 1}
balance_classes=1 # {0, 1}

# END MODIFY THESE OPTIONS
# ************************************


# limit the script to only the GPUs you selected above
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${GPU}


python train_unet.py --test_every_n_steps=${test_every_n_step} --batch_size=${batch_size} --train_database=${train_database} --test_database=${test_database} --output_dir=${output_folder} --number_classes=${number_classes} --learning_rate=${learning_rate}  --use_augmentation=${use_augmentation} --balance_classes=${balance_classes}
