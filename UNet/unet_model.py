import numpy as np
import tensorflow as tf


NUMBER_CHANNELS = 1  # grayscale

BASELINE_FEATURE_DEPTH = 64

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def conv_layer(layer_in, fc_out, kernel, is_training, stride=1, name=None):
    conv_output = tf.layers.conv2d(
        layer_in,
        filters=fc_out,
        kernel_size=kernel,
        padding='same',
        activation=tf.nn.relu,
        strides=stride,
        data_format='channels_first',  # translates to NCHW
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name=name)
    layer_output = tf.layers.batch_normalization(conv_output, training=is_training, axis=3)  # axis to normalize (the channels dimension)
    return layer_output


def deconv_layer(layer_in, fc_out, kernel, is_training, stride=1, name=None):
    conv_output = tf.layers.conv2d_transpose(
        layer_in,
        filters=fc_out,
        kernel_size=kernel,
        padding='same',
        activation=None,
        strides=stride,
        data_format='channels_first',  # translates to NCHW
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name=name)
    layer_output = tf.layers.batch_normalization(conv_output, training=is_training, axis=3)  # axis to normalize (the channels dimension)
    return layer_output


def add_inference_ops(inputs, is_training, number_classes=2):
    kernel = 3
    deconv_kernel = 2
    pooling_stride = 2

    with tf.variable_scope('unet'):
        with tf.variable_scope("encoder"):
            conv_1 = conv_layer(inputs, BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv_1 = conv_layer(conv_1, BASELINE_FEATURE_DEPTH, kernel, is_training)

            pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=pooling_stride, strides=pooling_stride)

            conv_2 = conv_layer(pool_1, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv_2 = conv_layer(conv_2, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=pooling_stride, strides=pooling_stride)

            conv_3 = conv_layer(pool_2, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv_3 = conv_layer(conv_3, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            pool_3 = tf.layers.max_pooling2d(conv_3, pool_size=pooling_stride, strides=pooling_stride)

            conv_4 = conv_layer(pool_3, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv_4 = conv_layer(conv_4, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            pool_4 = tf.layers.max_pooling2d(conv_4, pool_size=pooling_stride, strides=pooling_stride)

            # bottleneck
            bottleneck = conv_layer(pool_4, 16 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            bottleneck = conv_layer(bottleneck, 16 * BASELINE_FEATURE_DEPTH, kernel, is_training)

        with tf.variable_scope("decoder"):
            # up-conv which reduces the number of feature channels by 2
            deconv_4 = deconv_layer(bottleneck, 8 * BASELINE_FEATURE_DEPTH, deconv_kernel, is_training, stride=pooling_stride)
            deconv_4 = tf.concat([conv_4, deconv_4], axis=3)
            deconv_4 = conv_layer(deconv_4, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            deconv_4 = conv_layer(deconv_4, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            deconv_3 = deconv_layer(deconv_4, 4 * BASELINE_FEATURE_DEPTH, deconv_kernel, is_training, stride=pooling_stride)
            deconv_3 = tf.concat([conv_3, deconv_3], axis=3)
            deconv_3 = conv_layer(deconv_3, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            deconv_3 = conv_layer(deconv_3, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            deconv_2 = deconv_layer(deconv_3, 2 * BASELINE_FEATURE_DEPTH, deconv_kernel, is_training, stride=pooling_stride)
            deconv_2 = tf.concat([conv_2, deconv_2], axis=3)
            deconv_2 = conv_layer(deconv_2, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            deconv_2 = conv_layer(deconv_2, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            deconv_1 = deconv_layer(deconv_2, BASELINE_FEATURE_DEPTH, deconv_kernel, is_training, stride=pooling_stride)
            deconv_1 = tf.concat([conv_1, deconv_1], axis=3)
            deconv_1 = conv_layer(deconv_1, BASELINE_FEATURE_DEPTH, kernel, is_training)
            deconv_1 = conv_layer(deconv_1, BASELINE_FEATURE_DEPTH, kernel, is_training)

        logits = conv_layer(deconv_1, number_classes, 1, is_training, name='logits') # 1x1 kernel to convert feature map into class map
    return logits


def add_loss_ops(logits, labels, number_classes=2):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int32)
    labels_one_hot = tf.one_hot(labels, depth=number_classes)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits))

    return loss


def tower_loss(images, labels, number_classes, is_training):
    # Build inference Graph.
    logits = add_inference_ops(images, is_training,  number_classes=number_classes)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    loss = add_loss_ops(logits, labels, number_classes=number_classes)

    # compute accuracy
    prediction = tf.cast(tf.argmax(logits, axis=3), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

    return loss, accuracy


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads




