
# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import build_lmdb

# ************************************
# MODIFY THESE OPTIONS

image_folder="D:/Programming/Car_T_cell/Data/CAR-T_training_dataset_20190823/Raw_training_images"
mask_folder="D:/Programming/Car_T_cell/Data/CAR-T_training_dataset_20190823/Segmented_images"

output_folder="D:/Programming/Car_T_cell/Data/CAR-T_training_dataset_20190823/lmdb_folders"

# what common name to use in saving the lmdb dataset
dataset_name="GFP_"

# what fraction of your data to use for training. Test is 1.0 - train_fraction of the data
train_fraction=0.8 # (0.0, 1.0)

# what format are the images in your image/mask folder
image_format="tif"

use_tiling=1 #{0, 1}
tile_size=256

# END OF MODIFY THESE OPTIONS
# ************************************

build_lmdb.main(image_folder,mask_folder,output_folder,dataset_name,train_fraction,image_format,use_tiling,tile_size)

