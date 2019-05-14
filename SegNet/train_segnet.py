import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os



# Function to query tensorflow to obtain the number of GPUs the system has
def get_available_gpu_count():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpus_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpus_names)

# Get the number of system gpus, so we know later how many towers we need to build, 1 per gpu
NUM_GPUS = get_available_gpu_count()
print('Found {} GPUS'.format(NUM_GPUS))
# build a list of GPU_IDs which are numbered 0 through NUM_GPUS-1
GPU_IDS = list(range(NUM_GPUS))

# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
# define the number of disk readers (which are each single threaded) to match the number of GPUs, so we have one single threaded reader per gpu
READER_COUNT = NUM_GPUS
if len(GPU_IDS) == 0:
    exit(1)
print('Reader Count: {}'.format(READER_COUNT))


# Setup the Argument parsing
parser = argparse.ArgumentParser(prog='train_segnet', description='Script which trains a segnet model')

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
parser.add_argument('--gradient_update_location', dest='gradient_update_location', type=str, help="Where to perform gradient averaging and update. Options: ['cpu', 'gpu:#']. Use the GPU if you have a fully connected topology, cpu otherwise.", default='gpu:0')
parser.add_argument('--restore_checkpoint_filepath', dest='restore_checkpoint_filepath', type=str, help='checkpoint to resume from', default=None)
parser.add_argument('--restore_var_common_name', dest='restore_var_common_name', type=str, default=None)

args = parser.parse_args()
batch_size = args.batch_size
output_folder = args.output_folder
gradient_update_location = args.gradient_update_location
number_classes = args.number_classes
terminate_after_num_epochs_without_test_loss_improvement = args.terminate_after_num_epochs_without_test_loss_improvement
train_lmdb_filepath = args.train_database_filepath
test_lmdb_filepath = args.test_database_filepath
learning_rate = args.learning_rate
restore_checkpoint_filepath = args.restore_checkpoint_filepath
test_every_n_steps = args.test_every_n_steps
balance_classes = args.balance_classes
use_augmentation = args.use_augmentation
restore_var_common_name = args.restore_var_common_name

# verify gradient_update_location is valid

valid_location = False
if gradient_update_location == 'cpu':
    # if the GPUs do not have a fully connected topology (e.g. NVLink), its faster to perform gradient averaging on the CPU
    valid_location = True
    gradient_update_location = gradient_update_location + ':0' # append the useless id number
for id in GPU_IDS:
    if gradient_update_location == 'gpu:{}'.format(id):
        valid_location = True
if not valid_location:
    raise Exception("Invalid option for 'gradient_update_location': {}".format(gradient_update_location))


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

print('gradient_update_location = {}'.format(gradient_update_location))
print('early_stopping count = {}'.format(terminate_after_num_epochs_without_test_loss_improvement))
print('restore_checkpoint_filepath = {}'.format(restore_checkpoint_filepath))
print('restore_var_common_name = {}'.format(restore_var_common_name))


import numpy as np
import tensorflow as tf
tf_version = tf.__version__.split('.')
if tf_version[0] is not '1' or tf_version[1] is not '12':
    import warnings
    warnings.warn('Codebase only tested using Tensorflow version 1.12.x')
import segnet_model
import imagereader



def save_csv_file(output_folder, data, filename):
    np.savetxt(os.path.join(output_folder, filename), np.asarray(data), fmt='%.6g', delimiter=",")


def save_text_csv_file(output_folder, data, filename):
    with open(os.path.join(output_folder, filename), 'w') as csvfile:
        for i in range(len(data)):
            csvfile.write(data[i])
            csvfile.write('\n')


def plot(output_folder, name, train_val, test_val, epoch_size, log_scale=True):
    mpl.rcParams['agg.path.chunksize'] = 10000  # fix for error in plotting large numbers of points

    train_val = np.asarray(train_val)
    test_val = np.asarray(test_val)
    epoch_size = np.array(epoch_size, dtype=np.float32)
    iterations = np.arange(0, len(train_val))
    test_iterations = np.arange(0, len(test_val)) * epoch_size

    dot_size = 4
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = plt.gca()
    ax.scatter(iterations, train_val, c='b', s=dot_size)
    ax.plot(test_iterations, test_val, 'r-', marker='o', markersize=12)

    min_x = np.min(train_val)
    max_x = np.max(train_val)
    tmp = test_val[np.isfinite(test_val)]
    if tmp.size > 0:
        min_x = min(min_x, np.min(tmp))
        max_x = max(max_x, np.max(tmp))

    plt.ylim((min_x, max_x))
    if log_scale:
        ax.set_yscale('log')
    plt.ylabel('{}'.format(name))
    plt.xlabel('Iterations')
    fig.savefig(os.path.join(output_folder, '{}.png'.format(name)))

    plt.close(fig)


def train_model():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('Setting up test image reader')
    test_reader = imagereader.ImageReader(test_lmdb_filepath, batch_size=batch_size, use_augmentation=False, shuffle=False, num_workers=READER_COUNT, balance_classes=False, number_classes=number_classes)
    print('Test Reader has {} batches'.format(test_reader.get_epoch_size()))

    print('Setting up training image reader')
    train_reader = imagereader.ImageReader(train_lmdb_filepath, batch_size=batch_size, use_augmentation=use_augmentation, shuffle=True, num_workers=READER_COUNT, balance_classes=balance_classes, number_classes=number_classes)
    print('Train Reader has {} batches'.format(train_reader.get_epoch_size()))

    try: # if any errors happen we want to catch them and shut down the multiprocess readers
        print('Starting Readers')
        train_reader.startup()
        test_reader.startup()

        print('Creating model')
        with tf.Graph().as_default(), tf.device('/' + gradient_update_location):
            train_init_op, test_init_op, train_op, loss_op, accuracy_op, is_training_placeholder = segnet_model.build_towered_model(train_reader, test_reader, GPU_IDS, learning_rate, number_classes)

            print('Starting Session')
            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            init = tf.global_variables_initializer()
            sess.run(init)

            if restore_checkpoint_filepath is not None:
                print('Loading checkpoint weights')
                # build list of variables to restore
                vars = tf.global_variables()
                vars = [v for v in vars if 'Adam' not in v._shared_name]  # remove Adam variables
                vars = [v for v in vars if 'logits' not in v._shared_name]  # remove output layer variables
                vars = [v for v in vars if 'batch_normalization' not in v._shared_name]  # remove output layer variables

                if restore_var_common_name is not None:
                    vars = [v for v in vars if restore_var_common_name in v._shared_name]  # remove output layer variables
                print('*********************************')
                print('Restoring Vars')
                for v in vars:
                    print('{}   {}'.format(v._shared_name, v.shape))
                print('*********************************')

                saver = tf.train.Saver(vars)
                saver.restore(sess, restore_checkpoint_filepath)

            # setup network accuracy tracking variables
            train_loss = list()
            train_accuracy = list()
            test_loss = list()
            test_accuracy = list()
            test_loss.append(np.inf)
            test_accuracy.append(np.nan)

            train_epoch_size = train_reader.get_epoch_size()
            train_epoch_size = test_every_n_steps
            test_epoch_size = test_reader.get_epoch_size()

            epoch = 0
            print('Running Network')
            while True: # loop until early stopping
                print('---- Epoch: {} ----'.format(epoch))

                print('initializing training data iterator')
                sess.run(train_init_op)
                print('   iterator init complete')

                adj_batch_count = train_epoch_size
                for step in range(adj_batch_count):
                    _, loss_val, accuracy_val = sess.run([train_op, loss_op, accuracy_op], feed_dict={is_training_placeholder: True})
                    train_loss.append(loss_val)
                    train_accuracy.append(accuracy_val)
                    print('Train Epoch: {} Batch {}/{}: loss = {}'.format(epoch, step, adj_batch_count, loss_val))
                print('Train Epoch: {} : Accuracy = {}'.format(epoch, np.mean(train_accuracy[-train_epoch_size:])))

                sess.run(test_init_op)
                adj_batch_count = int(np.ceil(test_epoch_size / NUM_GPUS))
                epoch_test_loss = list()
                epoch_test_accuracy = list()
                for step in range(adj_batch_count):
                    loss_val, accuracy_val = sess.run([loss_op, accuracy_op], feed_dict={is_training_placeholder: False})
                    epoch_test_loss.append(loss_val)
                    epoch_test_accuracy.append(accuracy_val)
                    print('Test Epoch: {} Batch {}/{}: loss = {}'.format(epoch, step, adj_batch_count, loss_val))
                test_loss.append(np.mean(epoch_test_loss))
                test_accuracy.append(np.mean(epoch_test_accuracy))
                print('Test Epoch: {} : Accuracy = {}'.format(epoch, test_accuracy[-1]))

                plot(output_folder, 'loss', train_loss, test_loss, train_epoch_size)
                plot(output_folder, 'accuracy', train_accuracy, test_accuracy, train_epoch_size, log_scale=False)

                save_csv_file(output_folder, train_accuracy, 'train_accuracy.csv')
                save_csv_file(output_folder, train_loss, 'train_loss.csv')
                save_csv_file(output_folder, test_loss, 'test_loss.csv')
                save_csv_file(output_folder, test_accuracy, 'test_accuracy.csv')

                # determine if to record a new checkpoint based on best test loss
                if (len(test_loss) - 1) == np.argmin(test_loss):
                    # save tf checkpoint
                    print('Test loss improved: {}, saving checkpoint'.format(np.min(test_loss)))
                    saver = tf.train.Saver(tf.global_variables())
                    checkpoint_filepath = os.path.join(output_folder, 'checkpoint', 'model.ckpt')
                    saver.save(sess, checkpoint_filepath)

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
