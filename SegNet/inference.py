import argparse
import os
import tensorflow as tf
import segnet_model
import numpy as np
import imagereader


def load_model(checkpoint_filepath, gpu_id, number_classes, tile_size):
    print('Creating model')
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_op = tf.placeholder(tf.float32, shape=(1, tile_size, tile_size, 1))

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


def _inference(img_filepath, sess, input_op, logits_op, tile_size):

    img = imagereader.imread(img_filepath)

    if img.shape[0] != tile_size or img.shape[1] != tile_size:
        print('Image Size: {}, {}'.format(img.shape[0], img.shape[1]))
        print('Expected Size: {}, {}'.format(tile_size, tile_size))
        raise Exception('Invalid input shape, does not match specified network tile size.')

    batch_data = img.astype(np.float32)
    batch_data = batch_data.reshape((1, img.shape[0], img.shape[1], 1))

    # normalize with whole image stats
    batch_data = imagereader.zscore_normalize(batch_data)

    [logits] = sess.run([logits_op], feed_dict={input_op: batch_data})
    pred = np.squeeze(np.argmax(logits, axis=3).astype(np.int32))

    return pred


def main():
    parser = argparse.ArgumentParser(prog='inference', description='Script to detect stars with the selected model')

    parser.add_argument('--gpu', dest='gpu_id', type=int, help='which gpu to use for training (can only use a single gpu)', default=0)
    parser.add_argument('--tile_size', dest='tile_size', type=int, help='image tile size the network is expecting', default=256)
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
    tile_size = args.tile_size

    print('Arguments:')
    print('number_classes = {}'.format(number_classes))
    print('tile_size = {}'.format(tile_size))
    print('gpu_id = {}'.format(gpu_id))
    print('checkpoint_filepath = {}'.format(checkpoint_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    str_gpu_ids = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu_ids

    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.tif')]

    sess, input_op, logits_op = load_model(checkpoint_filepath, gpu_id, number_classes, tile_size)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, slide_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), slide_name))

        segmented_mask = _inference(img_filepath, sess, input_op, logits_op, tile_size)
        if 0 < np.max(segmented_mask) <= 255:
            segmented_mask = segmented_mask.astype(np.uint8)
        if 255 < np.max(segmented_mask) < 65536:
            segmented_mask = segmented_mask.astype(np.uint16)
        if np.max(segmented_mask) > 65536:
            segmented_mask = segmented_mask.astype(np.int32)
        imagereader.imwrite(segmented_mask, os.path.join(output_folder, slide_name))


if __name__ == "__main__":
    main()
