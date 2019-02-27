import argparse
import os
import tensorflow as tf
import segnet_model
import numpy as np
import imagereader

TILE_SIZE = 256

def load_model(checkpoint_filepath, gpu_id, number_classes):
    print('Creating model')
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_op = tf.placeholder(tf.float32, shape=(1, TILE_SIZE, TILE_SIZE, 1))

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
        # for v in vars:
        #     print('{}   {}'.format(v._shared_name, v.shape))

        # restore only inference ops from checkpoint
        saver = tf.train.Saver(vars)
        saver.restore(sess, checkpoint_filepath)

        # # Run inference on the trained model:
        # graph = tf.get_default_graph()
        #
        # input_op = graph.get_tensor_by_name(segnet_model.INPUT_NODE_NAME + ':0')
        # logits_op = graph.get_tensor_by_name(segnet_model.OUTPUT_NODE_NAME + ':0')

    return sess, input_op, logits_op


def _inference(img_filepath, sess, input_op, logits_op):

    img = imagereader.imread(img_filepath)

    assert (img.shape[0] == TILE_SIZE and img.shape[1] == TILE_SIZE), 'Invalid input shape'

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
    str_gpu_ids = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu_ids

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
        if 0 < np.max(segmented_mask) <= 255:
            segmented_mask = segmented_mask.astype(np.uint8)
        if 255 < np.max(segmented_mask) < 65536:
            segmented_mask = segmented_mask.astype(np.uint16)
        if np.max(segmented_mask) > 65536:
            segmented_mask = segmented_mask.astype(np.int32)
        imagereader.imwrite(segmented_mask, os.path.join(output_folder, slide_name))


if __name__ == "__main__":
    main()
