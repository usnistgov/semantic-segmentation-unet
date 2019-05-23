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
import segnet_model
import numpy as np
import imagereader

# image size must be a factor of 32 to allow Segnet conv and deconv layer dimension to line up
SIZE_FACTOR = 32


def translate_image_size(img_size):
    h = img_size[0]
    w = img_size[1]

    pad_x = 0
    pad_y = 0
    if h % SIZE_FACTOR != 0:
        pad_y = (SIZE_FACTOR - h % SIZE_FACTOR)
        print('image height needs to be a multiple of {}, padding with reflect'.format(SIZE_FACTOR))
    if w % SIZE_FACTOR != 0:
        pad_x = (SIZE_FACTOR - w % SIZE_FACTOR)
        print('image width needs to be a multiple of {}, padding with reflect'.format(SIZE_FACTOR))

    tgt_h = int(h + pad_y)
    tgt_w = int(w + pad_x)

    return tgt_h, tgt_w, pad_y, pad_x


def load_model(checkpoint_filepath, gpu_id, number_classes, model_input_h, model_input_w):
    print('Creating model')
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_op = tf.placeholder(tf.float32, shape=(1, model_input_h, model_input_w, 1))

        # Calculate the gradients for each model tower.
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('%s_%d' % (segnet_model.TOWER_NAME, gpu_id)) as scope:
                    logits_op = segnet_model.add_inference_ops(input_op, is_training=False, number_classes=number_classes)

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        print('Starting Session')
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)

        # build list of variables to restore
        vars = tf.global_variables()
        for v in vars:
            print('{}   {}'.format(v._shared_name, v.shape))

        # restore only inference ops from checkpoint
        saver = tf.train.Saver(vars)
        saver.restore(sess, checkpoint_filepath)

    return sess, input_op, logits_op


def _inference(img_filepath, sess, input_op, logits_op):

    print('Loading image: {}'.format(img_filepath))
    img = imagereader.imread(img_filepath)
    img = img.astype(np.float32)

    # normalize with whole image stats
    img = imagereader.zscore_normalize(img)

    print('  img.shape={}'.format(img.shape))
    _, _, pad_y, pad_x = translate_image_size(img.shape)

    if pad_x > 0 or pad_y > 0:
        img = np.pad(img, pad_width=((0, pad_y), (0, pad_x)), mode='reflect')

    batch_data = img.reshape((1, img.shape[0], img.shape[1], 1))

    [logits] = sess.run([logits_op], feed_dict={input_op: batch_data})
    pred = np.squeeze(np.argmax(logits, axis=-1).astype(np.int32))

    if pad_x > 0:
        pred = pred[:, 0:-pad_x]
    if pad_y > 0:
        pred = pred[0:-pad_y, :]

    return pred


def inference(gpu_id, checkpoint_filepath, image_folder, output_folder, number_classes, image_format):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    if gpu_id != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    image_height, image_width = imagereader.imread(img_filepath_list[0]).shape
    model_input_h, model_input_w, _, _ = translate_image_size([image_height, image_width])
    print('model input image size: ({}, {})'.format(model_input_h, model_input_w))


    sess, input_op, logits_op = load_model(checkpoint_filepath, gpu_id, number_classes, model_input_h, model_input_w)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, slide_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), slide_name))

        segmented_mask = _inference(img_filepath, sess, input_op, logits_op)

        if 0 <= np.max(segmented_mask) <= 255:
            segmented_mask = segmented_mask.astype(np.uint8)
        if 255 < np.max(segmented_mask) < 65536:
            segmented_mask = segmented_mask.astype(np.uint16)
        if np.max(segmented_mask) > 65536:
            segmented_mask = segmented_mask.astype(np.int32)
        imagereader.imwrite(segmented_mask, os.path.join(output_folder, slide_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference',
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

    args = parser.parse_args()

    gpu_id = args.gpu_id
    checkpoint_filepath = args.checkpoint_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    number_classes = args.number_classes
    image_format = args.image_format

    print('Arguments:')
    print('number_classes = {}'.format(number_classes))
    print('gpu_id = {}'.format(gpu_id))
    print('checkpoint_filepath = {}'.format(checkpoint_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))
    print('image_format = {}'.format(image_format))

    inference(gpu_id, checkpoint_filepath, image_folder, output_folder, number_classes, image_format)

