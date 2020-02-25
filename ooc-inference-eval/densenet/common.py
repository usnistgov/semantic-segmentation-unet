import os
import shutil
import datetime
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

def backup_tf_checkpoint(checkpoint_dir, checkpoint_step, target_dir, checkpoint_file=True):
    delete_dir(target_dir)
    os.makedirs(target_dir)

    os.link(os.path.join(checkpoint_dir, 'model.ckpt-%s.data-00000-of-00001' % checkpoint_step), 
            os.path.join(target_dir, 'model.ckpt-%s.data-00000-of-00001' % checkpoint_step))
    os.link(os.path.join(checkpoint_dir, 'model.ckpt-%s.index' % checkpoint_step), 
            os.path.join(target_dir, 'model.ckpt-%s.index' % checkpoint_step))
    os.link(os.path.join(checkpoint_dir, 'model.ckpt-%s.meta' % checkpoint_step), 
            os.path.join(target_dir, 'model.ckpt-%s.meta' % checkpoint_step))

    if checkpoint_file:
        with open(os.path.join(target_dir, 'checkpoint'), 'w') as f:
            f.write('model_checkpoint_path: "model.ckpt-"' % checkpoint_step)

def pretty_tf_logging():
    tf_logger = logging.getLogger('tensorflow')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in tf_logger.handlers:
        handler.formatter = formatter

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

def distribution_strategy():
    num_gpus = len(get_available_gpus())
    tf.logging.info('Number of GPUs: %d' % num_gpus)
    if num_gpus == 0:
        tf.logging.info('Use OneDeviceStrategy with cpu:0')
        return tf.contrib.distribute.OneDeviceStrategy(device='/cpu:0')
    elif num_gpus == 1:
        tf.logging.info('Use OneDeviceStrategy with gpu:0')
        return tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
    else:
        tf.logging.info('Use MirroredStrategy')
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def delete_dir(path):
    shutil.rmtree(path, ignore_errors=True)

def ts_rand():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    random_num = random.randint(1e6, 1e7-1)
    return '%s_%d' % (ts, random_num)


def open_image(image_path, image_size): 
    """Load an image and resize it to `image_size`.
       Args:
           image_path: path of image file
           image_size: a tuple (width, height)
       Return:
           array of shape (height, width, 3)
    """
    return np.array(Image.open(image_path).resize(image_size, Image.NEAREST))

def resize_image_array(img_arr, image_size):
    """Resize an array that contains image data.
       Args:
           img_arr: image array
           image_size: a tuple (width, height)
       Return:
           array of shape (height, width)
    """
    # convert img_arr to an image and apply the same resize operation
    # that we use in `open_image()`
    img = Image.fromarray(img_arr).resize(image_size, Image.NEAREST)
    return np.array(img)

def split_dataset(images, labels, test_fraction):
    nb_images = images.shape[0]
    idxs = np.arange(nb_images)
    np.random.shuffle(idxs)
    test_idxs = idxs[0:int(idxs.shape[0] * test_fraction)]
    train_idxs = idxs[int(idxs.shape[0] * test_fraction):]
    return images[train_idxs,:,:,:], labels[train_idxs,:], images[test_idxs,:,:,:], labels[test_idxs,:]

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tf_records(tfrecords_path, images, labels):
    """Write images and labels as TFRecords to a file.
    Args:
    tfrecords_path: output path
    images: array with images
    labels: array with labels
    """
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for index in range(images.shape[0]):
            feature = { 
                        'height': _int64_feature(images[index].shape[0]),
                        'width': _int64_feature(images[index].shape[1]),
                        'label': _bytes_feature(tf.compat.as_bytes(labels[index].tostring())), 
                        'image': _bytes_feature(tf.compat.as_bytes(images[index].tostring()))
                      }
            example = tf.train.Example(features=tf.train.Features(feature=feature))    
            writer.write(example.SerializeToString())

def show_image_row(images, figsize=(15, 15), show_axis=True):
    fig = plt.figure(figsize=figsize)
    number_of_images = len(images)
    for i in range(number_of_images):
        a = fig.add_subplot(1, number_of_images, i+1)
        plt.imshow(images[i])
        if not show_axis:
            plt.axis('off')