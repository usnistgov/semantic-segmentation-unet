import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import argparse
import os
import datetime


# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
# define the number of disk readers (which are each single threaded) to match the number of GPUs, so we have one single threaded reader per gpu
READER_COUNT = 4


# Setup the Argument parsing
parser = argparse.ArgumentParser(prog='train_unet', description='Script which trains a unet model')

parser.add_argument('--batch_size', dest='batch_size', type=int, help='training batch size', default=4)
parser.add_argument('--number_classes', dest='number_classes', type=int, default=2)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=3e-4)
parser.add_argument('--output_dir', dest='output_folder', type=str, help='Folder where outputs will be saved (Required)', required=True)
parser.add_argument('--test_every_n_steps', dest='test_every_n_steps', type=int, help='number of gradient update steps to take between test epochs', default=100)
parser.add_argument('--balance_classes', dest='balance_classes', type=int, help='whether to balance classes [0 = false, 1 = true]', default=0)
parser.add_argument('--use_augmentation', dest='use_augmentation', type=int, help='whether to use data augmentation [0 = false, 1 = true]', default=1)

parser.add_argument('--train_database', dest='train_database_filepath', type=str, help='lmdb database to use for (Required)', required=True)
parser.add_argument('--test_database', dest='test_database_filepath', type=str, help='lmdb database to use for testing (Required)', required=True)
parser.add_argument('--early_stopping', dest='terminate_after_num_epochs_without_test_loss_improvement', type=int, help='Perform early stopping when the test loss does not improve for N epochs.', default=10)


args = parser.parse_args()
batch_size = args.batch_size
output_folder = args.output_folder
number_classes = args.number_classes
terminate_after_num_epochs_without_test_loss_improvement = args.terminate_after_num_epochs_without_test_loss_improvement
train_lmdb_filepath = args.train_database_filepath
test_lmdb_filepath = args.test_database_filepath
learning_rate = args.learning_rate
test_every_n_steps = args.test_every_n_steps
balance_classes = args.balance_classes
use_augmentation = args.use_augmentation


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

print('early_stopping count = {}'.format(terminate_after_num_epochs_without_test_loss_improvement))


import numpy as np
import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    import warnings
    warnings.warn('Codebase designed for Tensorflow 2.x.x')

import unet_model
import imagereader


def train_model():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('Setting up test image reader')
    test_reader = imagereader.ImageReader(test_lmdb_filepath, use_augmentation=False, shuffle=False, num_workers=READER_COUNT, balance_classes=False, number_classes=number_classes)
    print('Test Reader has {} images'.format(test_reader.get_image_count()))
    test_dataset = test_reader.get_tf_dataset()
    test_dataset = test_dataset.batch(batch_size).prefetch(READER_COUNT)

    print('Setting up training image reader')
    train_reader = imagereader.ImageReader(train_lmdb_filepath, use_augmentation=use_augmentation, shuffle=True, num_workers=READER_COUNT, balance_classes=balance_classes, number_classes=number_classes)
    print('Train Reader has {} images'.format(train_reader.get_image_count()))
    train_dataset = train_reader.get_tf_dataset()
    train_dataset = train_dataset.batch(batch_size).prefetch(READER_COUNT)

    try: # if any errors happen we want to catch them and shut down the multiprocess readers
        print('Starting Readers')
        train_reader.startup()
        test_reader.startup()

        print('Creating model')
        model = unet_model.UNet(number_classes, learning_rate)

        # (NCHW)
        model.build(input_shape=(batch_size, 1, train_reader.get_image_size()[0], train_reader.get_image_size()[1]))
        model.summary()

        tf.keras.utils.plot_model(model, os.path.join(output_folder, 'model.png'), show_shapes=True)
        tf.keras.utils.plot_model(model, os.path.join(output_folder, 'model.dot'), show_shapes=True)

        # train_epoch_size = train_reader.get_image_count()/batch_size
        train_epoch_size = test_every_n_steps
        test_epoch_size = test_reader.get_image_count() / batch_size

        test_loss = list()

        # epoch = 0
        # print('Running Network')
        # while True:  # loop until early stopping
        #     print('---- Epoch: {} ----'.format(epoch))
        #
        #     # Iterate over the batches of the train dataset.
        #     for step, (batch_images, batch_labels) in enumerate(train_dataset):
        #         if step > train_epoch_size:
        #             break
        #         loss_value = model.train_step(batch_images, batch_labels)
        #         loss_value = loss_value.numpy() # convert tensor to numpy array
        #         print('Train Epoch {}: Batch {}/{}: Loss {}'.format(epoch, step, train_epoch_size, loss_value))
        #
        #
        #     # Iterate over the batches of the test dataset.
        #     epoch_test_loss = list()
        #     for step, (batch_images, batch_labels) in enumerate(test_dataset):
        #         if step > test_epoch_size:
        #             break
        #         loss_value = model.test_step(batch_images, batch_labels)
        #         loss_value = loss_value.numpy() # convert tensor to numpy array
        #         epoch_test_loss.append(loss_value)
        #         print('Test Epoch {}: Batch {}/{}: Loss {}'.format(epoch, step, test_epoch_size, loss_value))
        #     test_loss.append(np.mean(epoch_test_loss))


        # Prepare the metrics.
        train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
        test_loss_metric = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        test_acc_metric = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

        current_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

        train_log_dir = os.path.join(output_folder, current_time, 'train')
        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)
        test_log_dir = os.path.join(output_folder, current_time, 'test')
        if not os.path.exists(test_log_dir):
            os.makedirs(test_log_dir)

        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        epoch = 0
        print('Running Network')
        while True:  # loop until early stopping
            print('---- Epoch: {} ----'.format(epoch))

            # Iterate over the batches of the train dataset.
            for step, (batch_images, batch_labels) in enumerate(train_dataset):
                if step > train_epoch_size:
                    break
                loss_value, softmax_value = model.train_step(batch_images, batch_labels)
                # update the metrics
                train_loss_metric(loss_value)
                train_acc_metric(batch_labels, softmax_value)

                # print('Train Epoch {}: Batch {}/{}: Loss {}'.format(epoch, step, train_epoch_size, loss_value.numpy()))

                print('Train Epoch {}: Batch {}/{}: Loss {} Accuracy = {}'.format(epoch, step, train_epoch_size, train_loss_metric.result(), train_acc_metric.result()))
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss_metric.result(), step=int(epoch * train_epoch_size + step))
                    tf.summary.scalar('accuracy', train_acc_metric.result(), step=int(epoch * train_epoch_size + step))
                train_loss_metric.reset_states()
                train_acc_metric.reset_states()

            # Iterate over the batches of the test dataset.
            epoch_test_loss = list()
            for step, (batch_images, batch_labels) in enumerate(test_dataset):
                if step > test_epoch_size:
                    break
                loss_value, softmax_value = model.test_step(batch_images, batch_labels)
                # update the metrics
                test_loss_metric(loss_value)
                test_acc_metric(batch_labels, softmax_value)

                epoch_test_loss.append(loss_value.numpy())
                # print('Test Epoch {}: Batch {}/{}: Loss {}'.format(epoch, step, test_epoch_size, loss_value))
            test_loss.append(np.mean(epoch_test_loss))

            print('Test Epoch: {}: Loss = {} Accuracy = {}'.format(epoch, test_loss_metric.result(), test_acc_metric.result()))
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss_metric.result(), step=int((epoch+1) * train_epoch_size))
                tf.summary.scalar('accuracy', test_acc_metric.result(), step=int((epoch+1) * train_epoch_size))
            test_loss_metric.reset_states()
            test_acc_metric.reset_states()


            with open(os.path.join(output_folder, 'test_loss.csv'), 'w') as csvfile:
                for i in range(len(test_loss)):
                    csvfile.write(str(test_loss[i]))
                    csvfile.write('\n')

            # determine if to record a new checkpoint based on best test loss
            if (len(test_loss) - 1) == np.argmin(test_loss):
                # save tf checkpoint
                print('Test loss improved: {}, saving checkpoint'.format(np.min(test_loss)))
                # tf.keras.experimental.export_saved_model(model, os.path.join(output_folder, 'saved_model'), serving_only=True)
                # tf.saved_model.save(model, os.path.join(output_folder, 'checkpoint'))

            # determine early stopping
            CONVERGENCE_TOLERANCE = 1e-4
            print('Best Current Epoch Selection:')
            print('Test Loss:')
            print(test_loss)
            min_test_loss = np.min(test_loss)
            error_from_best = np.abs(test_loss - min_test_loss)
            error_from_best[error_from_best < CONVERGENCE_TOLERANCE] = 0
            best_epoch = np.where(error_from_best == 0)[0][0] # unpack numpy array, select first time since that value has happened
            print('Best epoch: {}'.format(best_epoch))

            if len(test_loss) - best_epoch > terminate_after_num_epochs_without_test_loss_improvement:
                break  # break the epoch loop
            epoch = epoch + 1

    finally: # if any erros happened during training, shut down the disk readers
        print('Shutting down train_reader')
        train_reader.shutdown()
        print('Shutting down test_reader')
        test_reader.shutdown()


if __name__ == "__main__":
    train_model()
    import gc
    gc.collect() # https://github.com/tensorflow/tensorflow/issues/21277
