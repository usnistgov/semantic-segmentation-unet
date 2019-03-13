import argparse
import os
import tensorflow as tf
import unet_model
import numpy as np
import imagereader


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
    pad_x = 0
    pad_y = 0

    if img.shape[0] % 16 != 0:
        pad_y = (16 - img.shape[0] % 16)
        print('image height needs to be a multiple of 16, padding with reflect')
    if img.shape[1] % 16 != 0:
        pad_x = (16 - img.shape[1] % 16)
        print('image width needs to be a multiple of 16, padding with reflect')
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


def main():
    parser = argparse.ArgumentParser(prog='inference', description='Script to detect stars with the selected unet model')

    parser.add_argument('--gpu', dest='gpu_id', type=int, help='which gpu to use for training (can only use a single gpu)', default=0)
    parser.add_argument('--checkpoint_filepath', dest='checkpoint_filepath', type=str, help='Checkpoint filepath to the  model to use', required=True)
    parser.add_argument('--image_folder', dest='image_folder', type=str, help='filepath to the folder containing tif images to inference (Required)', required=True)
    parser.add_argument('--output_folder', dest='output_folder', type=str, required=True)
    parser.add_argument('--number_classes', dest='number_classes', type=int, default=2)

    args = parser.parse_args()

    gpu_id = args.gpu_id
    checkpoint_filepath = args.checkpoint_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    number_classes = args.number_classes

    print('Arguments:')
    print('number_classes = {}'.format(number_classes))
    print('gpu_id = {}'.format(gpu_id))
    print('checkpoint_filepath = {}'.format(checkpoint_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    if gpu_id != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.tif')]

    sess, input_op, logits_op = load_model(checkpoint_filepath, gpu_id, number_classes)

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
    main()
