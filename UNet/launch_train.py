# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import train

# Which gpu do you want to use for training
# this can be a single number, or a list. E.g "3" or "0,1" "0,2,3"
gpu_ids="0"

# how large is an epoch, or sub/super epoch test dataset evaluation
test_every_n_steps=1000
batch_size=8

# where is your training lmdb database
train_lmdb_filepath="path/to/the/training/database.lmdb"
test_lmdb_filepath="path/to/the/training/database.lmdb"

output_folder="/path/to/output/directory/where/results/are/saved"

# how many classes exist in your training dataset (e.g. 2 for binary segmentation)
number_classes=2

# what learning rate should the network use
learning_rate=3e-4

use_augmentation=1 # {0, 1}
balance_classes=1 # {0, 1}
early_stopping_count=10 # Perform early stopping when the test loss does not improve for N epochs.
reader_count = 1 # how many threads to use for disk I/O and augmentation per gpu


# Launch Training
train.train_model(output_folder, batch_size, reader_count, train_lmdb_filepath, test_lmdb_filepath, use_augmentation, number_classes, balance_classes, learning_rate, test_every_n_steps, early_stopping_count, gpu_ids)

