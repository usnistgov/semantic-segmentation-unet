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
# gpus_to_use must bs comma separated list of gpu ids, e.g. "1,3,4"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # "0, 1" for multiple

import argparse
import datetime
import numpy as np

import unet_model
#import imagereader
#from unet.unet_model import UNet
import unet_dataset
import torch
import torch.optim as optim
from torch import nn
from peterpy import peter

def train_model(output_folder, batch_size, reader_count, train_lmdb_filepath, test_lmdb_filepath, use_augmentation, number_classes, balance_classes, learning_rate, test_every_n_steps, early_stopping_count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    use_gpu = torch.cuda.is_available()
    print('is gpu available',use_gpu)
    num_workers = 1 #int(config['batch_size'] / 2)

    torch_model_ofp = os.path.join(output_folder, 'checkpoint')
    if os.path.exists(torch_model_ofp):
        import shutil
        shutil.rmtree(torch_model_ofp)
    os.makedirs(torch_model_ofp)

    print('batch_size',batch_size)
    pin_dataloader_memory = True
    train_dataset = unet_dataset.UnetDataset(train_lmdb_filepath, augment=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_dataloader_memory, drop_last=True)

    test_dataset = unet_dataset.UnetDataset(test_lmdb_filepath, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_dataloader_memory, drop_last=True)


    try:  # if any errors happen we want to catch them and shut down the multiprocess readers
        print('Starting Readers')

        print('Creating model')
        model = unet_model.UNet(1, number_classes)
        if use_gpu:
            model = torch.nn.DataParallel(model)
            # move model to GPU
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        train_epoch_size = test_every_n_steps
        test_epoch_size = test_dataset.get_image_count() / batch_size

        test_loss = list()

        # Prepare the metrics.

        current_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

        epoch = 0
        epoch_loss = 0
        loss_regress = nn.SmoothL1Loss()
        criterion = torch.nn.CrossEntropyLoss()
        train_loss = list()
        test_loss_avg = list()
        ('Running Network')
        while True:  # loop until early stopping
            print('---- Epoch: {} ----'.format(epoch))
            model.train()  # put the model in training mode
            batch_count = 0
            for i, (images, target) in enumerate(train_loader):
                target = target.reshape([target.shape[0],images.shape[2],images.shape[3]])
                if use_gpu:
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                optimizer.zero_grad()
                batch_count = batch_count + 1

                pred = model.forward(images)
                if use_gpu:
                    pred = pred.cpu()
                    target = target.cpu()
                #pred = pred.type(torch.int32)
                #print('size of target',target.shape,pred.shape,target.dtype,pred.dtype)
                target = target.type(torch.LongTensor)
                loss = criterion(pred, target)
                epoch_loss += np.sum(loss.item())
                sum1 = np.sum(loss.item())
                train_loss.append(sum1)
                print("Epoch: {} Batch {}/{} loss {}".format(epoch, i, len(train_loader), sum1))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

            print('running test epoch')
            test_loss = list()
            with torch.no_grad():
                for i, (images, target) in enumerate(test_loader):
                    target = target.reshape([target.shape[0],images.shape[2],images.shape[3]])
                    if use_gpu:
                        images = images.cuda(non_blocking=True)
                        target = target.cuda(non_blocking=True)

                        optimizer.zero_grad()
                        pred = model.forward(images)
                        if use_gpu:
                            pred = pred.cpu()
                            target = target.cpu()
                        target = target.type(torch.LongTensor)
                        loss = criterion(pred, target)
                        sum1 = np.sum(loss.item())
                        test_loss.append(sum1)
                        # loss is [1, 5]
            testavg = np.mean(test_loss)
            test_loss_avg.append(testavg)
            print('this test loss',test_loss_avg)
            CONVERGENCE_TOLERANCE = 1e-4
            min_test_loss = np.min(test_loss_avg)
            error_from_best = np.abs(test_loss_avg - min_test_loss)
            error_from_best[error_from_best < CONVERGENCE_TOLERANCE] = 0
            best_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened
            print('Best epoch: {}'.format(best_epoch))

            # keep this model if it is the best so far
            if (len(test_loss_avg) - 1) == np.argmin(test_loss_avg):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(output_folder, 'checkpoint/unet.ckpt'))
                print("Saved best checkpoint so far in %s " % args.save)

            if len(test_loss_avg) - best_epoch > 5:
                break  # break the epoch loop

            epoch = epoch + 1

    finally: # if any erros happened during training, shut down the disk readers
        print('Shutting down train_reader')


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

    print('Arguments:')
    print('batch_size = {}'.format(batch_size))
    print('number_classes = {}'.format(number_classes))
    print('learning_rate = {}'.format(learning_rate))
    print('test_every_n_steps = {}'.format(test_every_n_steps))
    print('balance_classes = {}'.format(balance_classes))
    print('use_augmentation = {}'.format(use_augmentation))

    print('train_database = {}'.format(train_lmdb_filepath))
    print('test_database = {}'.format(test_lmdb_filepath))
    print('output folder = {}'.format(output_folder))

    print('early_stopping count = {}'.format(early_stopping_count))
    print('reader_count = {}'.format(reader_count))

    train_model(output_folder, batch_size, reader_count, train_lmdb_filepath, test_lmdb_filepath, use_augmentation, number_classes, balance_classes, learning_rate, test_every_n_steps, early_stopping_count)


if __name__ == "__main__":
    main()
