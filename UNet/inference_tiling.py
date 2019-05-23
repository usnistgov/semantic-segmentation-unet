import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import argparse
import os
import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 1 or int(tf_version[1]) != 12:
    import warnings
    warnings.warn('Codebase only tested using Tensorflow version 1.12.x')

import unet_model
import numpy as np
import imagereader


# image size must be a factor of 16 to allow UNet conv and deconv layer dimension to line up
SIZE_FACTOR = 16


def load_model(checkpoint_filepath, gpu_id, number_classes):
    print('Creating model')
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_op = tf.placeholder(tf.float32, shape=(1, 1, None, None))

        # Calculate the gradients for each model tower.
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('%s_%d' % (unet_model.TOWER_NAME, gpu_id)) as scope:
                    logits_op = unet_model.add_inference_ops(input_op, is_training=False, number_classes=number_classes)

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        print('Starting Session')
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)

        # build list of variables to restore
        vars = tf.global_variables()
        print('Loading variables:')
        for v in vars:
            print('{}   {}'.format(v._shared_name, v.shape))

        # restore only inference ops from checkpoint
        saver = tf.train.Saver(vars)
        saver.restore(sess, checkpoint_filepath)

    return sess, input_op, logits_op


def _inference(img_filepath, sess, input_op, logits_op, tile_size):

    # TODO test this
    print('Loading image: {}'.format(img_filepath))
    img = imagereader.imread(img_filepath)
    img = img.astype(np.float32)

    # normalize with whole image stats
    img = imagereader.zscore_normalize(img)
    height, width, = img.shape
    mask = np.zeros(img.shape)
    print('  img.shape={}'.format(img.shape))

    radius = SIZE_FACTOR
    zone_of_responsibility_size = tile_size - 2 * radius
    for i in range(0, height, zone_of_responsibility_size):
        for j in range(0, width, zone_of_responsibility_size):

            x_st_z = j
            y_st_z = i
            x_end_z = x_st_z + zone_of_responsibility_size
            y_end_z = y_st_z + zone_of_responsibility_size

            # pad zone of responsibility by radius
            x_st = x_st_z - radius
            y_st = y_st_z - radius
            x_end = x_end_z + radius
            y_end = y_end_z + radius

            pre_pad_x = 0
            if x_st < 0:
                pre_pad_x = -x_st
                x_st = 0
            pre_pad_y = 0
            if y_st < 0:
                pre_pad_y = -y_st
                y_st = 0
            post_pad_x = 0
            if x_end > width:
                post_pad_x = x_end - width
                x_end = width
            post_pad_y = 0
            if y_end > height:
                post_pad_y = y_end - height
                y_end = height

            # crop out the tile
            tile = img[y_st:y_end, x_st:x_end]

            if pre_pad_x > 0 or pre_pad_y > 0 or post_pad_x > 0 or post_pad_y > 0:
                # ensure its correct size (if tile exists at the edge of the image
                tile = np.pad(tile, pad_width=((pre_pad_y, post_pad_y), (pre_pad_x, post_pad_x)), mode='reflect')

            batch_data = tile.reshape((1, 1, tile.shape[0], tile.shape[1]))

            [logits] = sess.run([logits_op], feed_dict={input_op: batch_data})
            pred = np.squeeze(np.argmax(logits, axis=-1).astype(np.int32))

            pred = pred[pre_pad_y:-post_pad_y, pre_pad_x:-post_pad_x]

            mask[y_st_z:y_end_z, x_st_z:x_end_z] = pred

    return mask


def inference(gpu_id, checkpoint_filepath, image_folder, output_folder, number_classes, image_format, tile_size):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    sess, input_op, logits_op = load_model(checkpoint_filepath, gpu_id, number_classes)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, slide_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), slide_name))

        segmented_mask = _inference(img_filepath, sess, input_op, logits_op, tile_size)

        if 0 <= np.max(segmented_mask) <= 255:
            segmented_mask = segmented_mask.astype(np.uint8)
        if 255 < np.max(segmented_mask) < 65536:
            segmented_mask = segmented_mask.astype(np.uint16)
        if np.max(segmented_mask) > 65536:
            segmented_mask = segmented_mask.astype(np.int32)
        imagereader.imwrite(segmented_mask, os.path.join(output_folder, slide_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference_tiling',
                                     description='Script to detect stars with the selected unet model')

    parser.add_argument('--gpu', dest='gpu_id', type=int,
                        help='which gpu to use for training (can only use a single gpu)', default=0)
    parser.add_argument('--checkpoint_filepath', dest='checkpoint_filepath', type=str,
                        help='Checkpoint filepath to the  model to use', required=True)
    parser.add_argument('--image_folder', dest='image_folder', type=str,
                        help='filepath to the folder containing tif images to inference (Required)', required=True)
    parser.add_argument('--output_folder', dest='output_folder', type=str, required=True)
    parser.add_argument('--number_classes', dest='number_classes', type=int, default=2)
    parser.add_argument('--image_format', dest='image_format', type=str, help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')
    parser.add_argument('--tile_size', dest='tile_size',type=int, help='tile size to break input images down to with overlap to fit onto gpu memory', default=256)

    args = parser.parse_args()

    gpu_id = args.gpu_id
    checkpoint_filepath = args.checkpoint_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    number_classes = args.number_classes
    image_format = args.image_format
    tile_size = args.tile_size

    print('Arguments:')
    print('number_classes = {}'.format(number_classes))
    print('gpu_id = {}'.format(gpu_id))
    print('checkpoint_filepath = {}'.format(checkpoint_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))
    print('image_format = {}'.format(image_format))
    print('tile_size = {}'.format(tile_size))

    inference(gpu_id, checkpoint_filepath, image_folder, output_folder, number_classes, image_format, tile_size)

