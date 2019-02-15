import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os, sys


def get_available_gpu_count():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpus_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpus_names)

NUM_GPUS = get_available_gpu_count()
print('Found {} GPUS'.format(NUM_GPUS))
GPU_IDS = list(range(NUM_GPUS))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
str_gpu_ids = ','.join(str(g) for g in GPU_IDS)
os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu_ids  # "0, 1" for multiple
LEARNING_RATE = 1e-5
READER_COUNT = NUM_GPUS
print('Reader Count: {}'.format(READER_COUNT))

EPOCH_SIZE_MULTIPLIER = 100


parser = argparse.ArgumentParser(prog='train_unet', description='Script which trains a unet model')

parser.add_argument('--batch_size', dest='batch_size', type=int,
                    help='training batch size', default=8)
parser.add_argument('--number_classes', dest='number_classes', type=int, default=2)
parser.add_argument('--epoch_count', dest='epoch_count', type=int,
                    help='number of epochs to train the model for', default=100)
parser.add_argument('--train_data_folder', dest='train_data_folder', type=str, required=True)
parser.add_argument('--test_data_folder', dest='test_data_folder', type=str, required=True)
parser.add_argument('--number_training_examples', dest='number_training_examples', type=int, default=None)
parser.add_argument('--output_dir', dest='output_folder', type=str,
                    help='Folder where outputs will be saved (Required)', required=True)
parser.add_argument('--gradient_update_location', dest='gradient_update_location', type=str,
                    help="Where to perform gradient averaging and update. Options: ['cpu', 'gpu:#']. Use the GPU if you have a fully connected topology, cpu otherwise.", default='gpu:0')


args = parser.parse_args()

batch_size = args.batch_size
epoch_count = args.epoch_count
train_data_folder = args.train_data_folder
test_data_folder = args.test_data_folder
output_folder = args.output_folder
gradient_update_location = args.gradient_update_location
number_training_examples = args.number_training_examples
number_classes = args.number_classes



# verify gradient_update_location is valid
valid_location = False
if gradient_update_location == 'cpu':
    valid_location = True
    gradient_update_location = gradient_update_location + ':0' # append the useless id number
for id in GPU_IDS:
    if gradient_update_location == 'gpu:{}'.format(id):
        valid_location = True
if not valid_location:
    print("Invalid option for 'gradient_update_location': {}".format(gradient_update_location))
    exit(1)


print('Arguments:')
print('batch_size = {}'.format(batch_size))
print('epoch_count = {}'.format(epoch_count))
print('train_data_folder = {}'.format(train_data_folder))
print('test_data_folder = {}'.format(test_data_folder))
print('output folder = {}'.format(output_folder))
print('gradient_update_location = {}'.format(gradient_update_location))
print('number_training_examples = {}'.format(number_training_examples))
print('number_classes = {}'.format(number_classes))


import numpy as np
import tensorflow as tf
import unet_model
import shutil
import imagereader
import csv
import time




def load_block_list(filepath):
    A = []
    if os.path.exists(filepath):
        with open(filepath) as csvfile:
            A = csvfile.readlines()
        A = [x.strip().replace('.csv', '') for x in A]
    return A


def save_csv_file(output_folder, data, filename):
    np.savetxt(os.path.join(output_folder, filename), np.asarray(data), fmt='%.6g', delimiter=",")


def save_conf_csv_file(output_folder, TP, TN, FP, FN, filename):
    a = np.reshape(np.asarray(TP), (len(TP), 1))
    b = np.reshape(np.asarray(TN), (len(TN), 1))
    c = np.reshape(np.asarray(FP), (len(FP), 1))
    d = np.reshape(np.asarray(FN), (len(FN), 1))
    dat = np.hstack((a, b, c, d))
    np.savetxt(os.path.join(output_folder, filename), dat, fmt='%.6g', delimiter=",", header='TP, TN, FP, FN')


def save_text_csv_file(output_folder, data, filename):
    with open(os.path.join(output_folder, filename), 'w') as csvfile:
        for i in range(len(data)):
            csvfile.write(data[i])
            csvfile.write('\n')


def generate_plots(output_folder, train_loss, train_accuracy, test_loss, test_loss_sigma, test_accuracy, test_accuracy_sigma, nb_batches):
    mpl.rcParams['agg.path.chunksize'] = 10000  # fix for error in plotting large numbers of points

    # generate the loss and accuracy plots
    train_loss = np.array(train_loss)
    train_accuracy = np.array(train_accuracy)
    test_loss = np.array(test_loss)
    test_loss_sigma = np.array(test_loss_sigma) * 1.96 # convert to 95% CI
    test_accuracy = np.array(test_accuracy)
    if test_accuracy_sigma is None:
        test_accuracy_sigma = np.zeros((test_accuracy.shape))
    test_accuracy_sigma = np.array(test_accuracy_sigma) * 1.96 # convert to 95% CI
    nb_batches = np.array(nb_batches, dtype=np.float32)

    iterations = np.arange(0, len(train_loss))
    test_iterations = np.arange(0, len(test_accuracy))

    dot_size = 4
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = plt.gca()
    ax.scatter(iterations / nb_batches, train_accuracy, c='b', s=dot_size)
    ax.errorbar(test_iterations, test_accuracy, yerr=test_accuracy_sigma, fmt='r--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    fig.savefig(os.path.join(output_folder, 'accuracy.png'))

    fig.clf()
    ax = plt.gca()
    ax.scatter(iterations / nb_batches, train_loss, c='b', s=dot_size)
    ax.errorbar(test_iterations, test_loss, yerr=test_loss_sigma, fmt='r--')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    fig.savefig(os.path.join(output_folder, 'loss.png'))

    fig.clf()
    ax = plt.gca()
    ax.scatter(iterations / nb_batches, train_loss, c='b', s=dot_size)
    ax.errorbar(test_iterations, test_loss, yerr=test_loss_sigma, fmt='r--')
    ax.set_yscale('log')
    plt.ylim((np.min(train_loss), np.max(train_loss)))
    plt.ylabel('Loss (log scale)')
    plt.xlabel('Epoch')
    ax.set_yscale('log')
    fig.savefig(os.path.join(output_folder, 'loss_logscale.png'))

    plt.close(fig)


def train_model():

    print('Setting up test image reader')
    test_img_folder = os.path.join(test_data_folder, 'images')
    test_msk_folder = os.path.join(test_data_folder, 'masks')
    test_reader = imagereader.ImageReader(test_img_folder, test_msk_folder, batch_size=batch_size,
                                          use_augmentation=False, shuffle=False, num_workers=READER_COUNT)
    print('Test Reader has {} batches'.format(test_reader.get_epoch_size()))

    print('Setting up training image reader')
    train_img_folder = os.path.join(train_data_folder, 'images')
    train_msk_folder = os.path.join(train_data_folder, 'masks')
    train_reader = imagereader.ImageReader(train_img_folder, train_msk_folder, batch_size=batch_size,
                                           use_augmentation=True, shuffle=True, num_workers=READER_COUNT, sample_to_N_images=number_training_examples)
    if len(train_reader.get_fn_list()) > number_training_examples:
        print('Required number of examples do not exist')
        sys.exit(1)
    print('Train Reader has {} batches'.format(train_reader.get_epoch_size()))

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # write csv list of filenames used for training
    save_text_csv_file(output_folder, train_reader.get_fn_list(), 'training_images.csv')

    # start the input queues
    try:
        print('Starting Readers')
        train_reader.startup()
        test_reader.startup()

        print('Creating model')
        with tf.Graph().as_default(), tf.device('/' + gradient_update_location):

            is_training_placeholder = tf.placeholder(tf.bool, name='is_training')

            # wrap the input queues into a Dataset
            image_shape = tf.TensorShape((batch_size, train_reader.imageHeight, train_reader.imageWidth, 1))
            label_shape = tf.TensorShape((batch_size, train_reader.imageHeight, train_reader.imageWidth))
            train_dataset = tf.data.Dataset.from_generator(train_reader.generator, output_types=(tf.float32, tf.int32), output_shapes=(image_shape, label_shape))
            train_dataset = train_dataset.prefetch(2*NUM_GPUS) # prefetch N batches
            # train_iterator = train_dataset.make_initializable_iterator()

            test_dataset = tf.data.Dataset.from_generator(test_reader.generator, output_types=(tf.float32, tf.int32), output_shapes=(image_shape, label_shape))
            test_dataset = test_dataset.prefetch(2*NUM_GPUS)  # prefetch N batches
            # test_iterator = test_dataset.make_initializable_iterator()

            # create a iterator of the correct shape and type
            iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            train_init_op = iter.make_initializer(train_dataset)
            test_init_op = iter.make_initializer(test_dataset)

            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

            # Calculate the gradients for each model tower.
            tower_grads = []
            ops_per_gpu = {}
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(NUM_GPUS):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % (unet_model.TOWER_NAME, i)) as scope:
                            # Dequeues one batch for the GPU
                            image_batch, label_batch = iter.get_next()

                            # Calculate the loss for one tower of the CIFAR model. This function
                            # constructs the entire CIFAR model but shares the variables across
                            # all towers.
                            loss_op, accuracy_op = unet_model.tower_loss(image_batch, label_batch, number_classes, is_training_placeholder)
                            ops_per_gpu['gpu{}-loss'.format(i)] = loss_op
                            ops_per_gpu['gpu{}-accuracy'.format(i)] = accuracy_op

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Calculate the gradients for the batch of data on this tower.
                            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                                grads = optimizer.compute_gradients(loss_op)

                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = unet_model.average_gradients(tower_grads)

            # create merged accuracy stats
            all_loss_sum = tf.constant(0, dtype=tf.float32)
            all_accuracy_sum = tf.constant(0, dtype=tf.float32)

            for i in range(NUM_GPUS):
                all_loss_sum = tf.add(all_loss_sum, ops_per_gpu['gpu{}-loss'.format(i)])
                all_accuracy_sum = tf.add(all_accuracy_sum, ops_per_gpu['gpu{}-accuracy'.format(i)])

            all_loss = tf.divide(all_loss_sum, tf.constant(NUM_GPUS, dtype=tf.float32))
            all_accuracy = tf.divide(all_accuracy_sum, tf.constant(NUM_GPUS, dtype=tf.float32))

            # Apply the gradients to adjust the shared variables.
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.apply_gradients(grads)

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                train_loss = list()
                train_accuracy = list()
                test_loss = list()
                test_loss_sigma = list()
                test_accuracy = list()
                test_accuracy_sigma = list()
                test_loss.append(1)
                test_loss_sigma.append(0)
                test_accuracy.append(0)
                test_accuracy_sigma.append(0)

                train_epoch_size = train_reader.get_epoch_size() * EPOCH_SIZE_MULTIPLIER
                test_epoch_size = test_reader.get_epoch_size()

                print('Running Network')
                for epoch in range(epoch_count):

                    sess.run(train_init_op)
                    adj_batch_count = int(np.ceil(train_epoch_size / NUM_GPUS))
                    for step in range(adj_batch_count):
                        _, loss_val, accuracy_val = sess.run([train_op, all_loss, all_accuracy], feed_dict={is_training_placeholder: True})
                        train_loss.append(loss_val)
                        train_accuracy.append(accuracy_val)
                    print('Train Epoch: {} : Accuracy = {}'.format(epoch, np.mean(train_accuracy[-train_epoch_size:])))

                    sess.run(test_init_op)
                    adj_batch_count = int(np.ceil(test_epoch_size / NUM_GPUS))
                    epoch_test_loss = list()
                    epoch_test_accuracy = list()
                    for step in range(adj_batch_count):
                        loss_val, accuracy_val = sess.run([all_loss, all_accuracy], feed_dict={is_training_placeholder: True})
                        epoch_test_loss.append(loss_val)
                        epoch_test_accuracy.append(accuracy_val)
                    test_loss.append(np.mean(epoch_test_loss))
                    test_loss_sigma.append(np.std(epoch_test_loss))
                    test_accuracy.append(np.mean(epoch_test_accuracy))
                    test_accuracy_sigma.append(np.std(epoch_test_accuracy))
                    print('Test Epoch: {} : Accuracy = {}'.format(epoch, test_accuracy[-1]))

                generate_plots(output_folder, train_loss, train_accuracy, test_loss, test_loss_sigma, test_accuracy, test_accuracy_sigma, train_epoch_size/NUM_GPUS)
                save_csv_file(output_folder, train_accuracy, 'train_accuracy.csv')
                save_csv_file(output_folder, train_loss, 'train_loss.csv')
                save_csv_file(output_folder, test_loss, 'test_loss.csv')
                save_csv_file(output_folder, test_loss_sigma, 'test_loss_sigma.csv')
                save_csv_file(output_folder, test_accuracy, 'test_accuracy.csv')
                save_csv_file(output_folder, test_accuracy_sigma, 'test_accuracy_sigma.csv')

                # save tf checkpoint
                saver = tf.train.Saver(tf.global_variables())
                checkpoint_filepath = os.path.join(output_folder, 'checkpoint', 'model.ckpt')
                saver.save(sess, checkpoint_filepath)

    finally:
        print('Shutting down train_reader')
        train_reader.shutdown()
        print('Shutting down test_reader')
        test_reader.shutdown()


if __name__ == "__main__":
    train_model()
    import gc
    gc.collect() # https://github.com/tensorflow/tensorflow/issues/21277
