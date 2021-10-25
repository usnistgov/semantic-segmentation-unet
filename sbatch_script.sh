#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=isg
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name=unet
#SBATCH -o log-%N.%j.out
#SBATCH --time=24:0:0

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

# export CUDA_DEVICE_ORDER=PCI_BUS_ID  # for the 3090s

source /mnt/isgnas/home/mmajursk/anaconda3/etc/profile.d/conda.sh
conda activate unet


train_img_fp=/home/mmajurski/usnistgov/semantic-segmentation-unet/data/rpe2d/train_images
train_msk_fp=/home/mmajurski/usnistgov/semantic-segmentation-unet/data/rpe2d/train_masks
val_img_fp=/home/mmajurski/usnistgov/semantic-segmentation-unet/data/rpe2d/test_images
val_msk_fp=/home/mmajurski/usnistgov/semantic-segmentation-unet/data/rpe2d/test_masks
test_img_fp=/home/mmajurski/usnistgov/semantic-segmentation-unet/data/rpe2d/test_images
test_msk_fp=/home/mmajurski/usnistgov/semantic-segmentation-unet/data/rpe2d/test_masks
ext=tif
output_filepath=/home/mmajurski/usnistgov/semantic-segmentation-unet/test-model


python main.py --train-image-filepath=${train_img_fp} --train-mask-filepath=${train_msk_fp} --val-image-filepath=${val_img_fp} --val-mask-filepath=${val_msk_fp} --test-image-filepath=${test_img_fp} --test-mask-filepath=${test_msk_fp} --file-extension=${ext} --output-filepath=${output_filepath}
