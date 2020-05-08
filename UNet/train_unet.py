# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi

import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch import nn

#import unet_model
import imagereader
from peterpy import peter
#from utils.dataset import BasicDataset
from data_gen import Dataset
from torch.utils.data import DataLoader, random_split
from unet.unet_model import UNet
from torch.utils import data

def train_model(output_folder, batch_size, reader_count, train_lmdb_filepath, test_lmdb_filepath, use_augmentation, number_classes, balance_classes, learning_rate, test_every_n_steps, early_stopping_count, dir_img, dir_mask):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # TODO add ability to reload a checkpoint or saved model to resume training
    training_checkpoint_filepath = None

    # uses all available devices
    global_batch_size = batch_size
    # scale the number of I/O readers based on the GPU count

    # Tensor type to use, select CUDA or not
    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device('cuda')
    # if args.cuda else device_cpu

    # Set seeds
    np.random.seed(1)
    torch.manual_seed(1)
    #if args.cuda:
    torch.cuda.manual_seed_all(1)

    #print('Setting up test image reader')
    #test_reader = imagereader.ImageReader(test_lmdb_filepath, use_augmentation=False, shuffle=False, num_workers=reader_count, balance_classes=False, number_classes=number_classes)
    #print('Test Reader has {} images'.format(test_reader.get_image_count()))


    # set up the Model
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    #logging.info(f'Network:\n'
    #             f'\t{net.n_channels} input channels\n'
    #             f'\t{net.n_classes} output channels (classes)\n'
    #             f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    #if args.load:
    #    net.load_state_dict(
    #        torch.load(args.load, map_location=device)
    #    )
    #    logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.CrossEntropyLoss()

    # create and training and validation sets
    print(dir_img,dir_mask)
    partition = []
    labels = []
    ids = os.listdir(dir_img)
    print('total images',len(ids))
    for j in range(0,len(ids)):
        partition.append('image'+str(j)+'.tif')
        labels.append('image'+str(j)+'.tif')

    val_percent = 0.2
    n_val = int(len(partition) * val_percent)
    n_train = len(partition) - n_val
    train, val = random_split(partition, [n_train, n_val])
    print('split',len(train),len(val))
    #train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #valset_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    params = {batch_size:8,
          'shuffle': True,
          'num_workers': 6}
    train_loader = Dataset(train,labels,dir_img,dir_mask)
    training_generator = data.DataLoader(train_loader, batch_size=4,num_workers=4)
    val_loader = Dataset(val,labels,dir_img,dir_mask)
    val_generator = data.DataLoader(val_loader, batch_size=4,num_workers=4)

    try:
        for epoch in range(5):
            #net.train()

            epoch_loss = 0
            for local_batch, local_labels in training_generator:
                print('sizes',local_batch.shape,local_labels.shape)
            #    break
                # Transfer to GPU
                #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            epoch += 1

    finally: # if any erros happened during training, shut down the disk readers
        print('Shutting down train_reader')
        #train_reader.shutdown()
        print('Shutting down test_reader')
        #test_reader.shutdown()

    # convert training checkpoint to the saved model format
    if training_checkpoint_filepath is not None:
        # restore the checkpoint and generate a saved model
        model = unet_model.UNet(number_classes, global_batch_size, train_reader.get_image_size(), learning_rate)
        checkpoint = tf.train.Checkpoint(optimizer=model.get_optimizer(), model=model.get_keras_model())
        checkpoint.restore(training_checkpoint_filepath)
        tf.saved_model.save(model.get_keras_model(), os.path.join(output_folder, 'saved_model'))


def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='train_unet', description='Script which trains a unet model')

    parser.add_argument('--batch_size', dest='batch_size', type=int, help='training batch size', default=4)
    parser.add_argument('--number_classes', dest='number_classes', type=int, default=2)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=3e-4)
    parser.add_argument('--output_dir', dest='output_folder', type=str, help='Folder where outputs will be saved (Required)', required=True)
    parser.add_argument('--test_every_n_steps', dest='test_every_n_steps', type=int, help='number of gradient update steps to take between test epochs', default=1000)
    parser.add_argument('--balance_classes', dest='balance_classes', type=int, help='whether to balance classes [0 = false, 1 = true]', default=0)
    parser.add_argument('--use_augmentation', dest='use_augmentation', type=int, help='whether to use data augmentation [0 = false, 1 = true]', default=1)

    parser.add_argument('--train_database', dest='train_database_filepath', type=str, help='lmdb database to use for (Required)', required=True)
    parser.add_argument('--test_database', dest='test_database_filepath', type=str, help='lmdb database to use for testing (Required)', required=True)
    parser.add_argument('--early_stopping', dest='early_stopping_count', type=int, help='Perform early stopping when the test loss does not improve for N epochs.', default=10)
    parser.add_argument('--reader_count', dest='reader_count', type=int, help='how many threads to use for disk I/O and augmentation per gpu', default=1)

    parser.add_argument('-i', '--dir_img', dest='dir_img', type=str, default='/home/peskin/unet_milesial/images/',
                        help='directory with images')
    parser.add_argument('-m', '--dir_mask', dest='dir_mask', type=str, default='/home/peskin/unet_milesial/masks/',
                        help='directory with images')
    # TODO add parameter to specify the devices to use for training

    args = parser.parse_args()
    batch_size = args.batch_size
    output_folder = args.output_folder
    number_classes = args.number_classes
    early_stopping_count = args.early_stopping_count
    train_lmdb_filepath = args.train_database_filepath
    test_lmdb_filepath = args.test_database_filepath
    learning_rate = args.learning_rate
    test_every_n_steps = args.test_every_n_steps
    balance_classes = args.balance_classes
    use_augmentation = args.use_augmentation
    reader_count = args.reader_count
    dir_img = args.dir_img
    dir_mask = args.dir_mask

    print('Arguments:')
    print('batch_size = {}'.format(batch_size))
    print('number_classes = {}'.format(number_classes))
    print('learning_rate = {}'.format(learning_rate))
    print('test_every_n_steps = {}'.format(test_every_n_steps))
    print('balance_classes = {}'.format(balance_classes))
    print('use_augmentation = {}'.format(use_augmentation))

    #print('train_database = {}'.format(train_lmdb_filepath))
    #print('test_database = {}'.format(test_lmdb_filepath))
    print('output folder = {}'.format(output_folder))

    print('early_stopping count = {}'.format(early_stopping_count))
    print('reader_count = {}'.format(reader_count))

    train_model(output_folder, batch_size, reader_count, train_lmdb_filepath, test_lmdb_filepath, use_augmentation, number_classes, balance_classes, learning_rate, test_every_n_steps, early_stopping_count, dir_img, dir_mask)


if __name__ == "__main__":
    main()
