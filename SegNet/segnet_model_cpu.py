import tensorflow as tf

NUMBER_CHANNELS = 1  # grayscale

BASELINE_FEATURE_DEPTH = 64

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


#############################################################################################################
#############################################################################################################


def max_pool(net, stride):
    """
    Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
    Therefore, we use max_pool_with_argmax to extract mask and
    plain max_pool for, eeem... max_pooling.
    """
    with tf.name_scope('MaxPoolArgMax'):
        _, mask = tf.nn.max_pool_with_argmax(
            net,
            ksize=[1, stride, stride, 1],
            strides=[1, stride, stride, 1],
            padding='SAME')
        mask = tf.stop_gradient(mask)
        # net = slim.max_pool2d(net, kernel_size=[stride, stride])
        net = tf.layers.max_pooling2d(net, pool_size=stride, strides=stride)
        return net, mask


# Thank you, @https://github.com/Pepslee
def max_unpool(net, mask, stride):
    assert mask is not None
    with tf.name_scope('UnPool2D'):
        ksize = [1, stride, stride, 1]
        input_shape = net.get_shape().as_list()
        #  calculation new shape
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int64)
        f = one_like_mask * feature_range
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(net)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(net, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret


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

#############################################################################################################
#############################################################################################################


def add_inference_ops(inputs, is_training, number_classes):
    with tf.variable_scope('segnet'):
        kernel = 3
        pooling_stride = 2

        with tf.variable_scope('encoder'):
            conv1_1 = conv_layer(inputs, BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv1_2 = conv_layer(conv1_1, BASELINE_FEATURE_DEPTH, kernel, is_training)
            hidden_pool, maxp1_argmax_mask = max_pool(conv1_2, pooling_stride)
            # print('maxp1_argmax_mask shape = {}'.format(maxp1_argmax_mask.shape))

            conv2_1 = conv_layer(hidden_pool, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv2_2 = conv_layer(conv2_1, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            hidden_pool, maxp2_argmax_mask = max_pool(conv2_2, pooling_stride)
            # print('maxp2_argmax_mask shape = {}'.format(maxp2_argmax_mask.shape))

            conv3_1 = conv_layer(hidden_pool, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv3_2 = conv_layer(conv3_1, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv3_3 = conv_layer(conv3_2, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            hidden_pool, maxp3_argmax_mask = max_pool(conv3_3, pooling_stride)
            # print('maxp3_argmax_mask shape = {}'.format(maxp3_argmax_mask.shape))

            conv4_1 = conv_layer(hidden_pool, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv4_2 = conv_layer(conv4_1, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv4_3 = conv_layer(conv4_2, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            hidden_pool, maxp4_argmax_mask = max_pool(conv4_3, pooling_stride)
            # print('maxp4_argmax_mask shape = {}'.format(maxp4_argmax_mask.shape))

            conv5_1 = conv_layer(hidden_pool, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv5_2 = conv_layer(conv5_1, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv5_3 = conv_layer(conv5_2, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            bottleneck, maxp5_argmax_mask = max_pool(conv5_3, pooling_stride)
            # print('maxp5_argmax_mask shape = {}'.format(maxp5_argmax_mask.shape))

        # bottleneck

        with tf.variable_scope('decoder'):
            # print('bottleneck shape = {}'.format(bottleneck.shape))
            # print('maxp5_argmax_mask shape = {}'.format(maxp5_argmax_mask.shape))
            unpooled = max_unpool(bottleneck, maxp5_argmax_mask, pooling_stride)
            # print('unpooled shape = {}'.format(unpooled.shape))
            conv6_1 = conv_layer(unpooled, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv6_2 = conv_layer(conv6_1, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv6_3 = conv_layer(conv6_2, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            # print('conv6_3 shape = {}'.format(conv6_3.shape))
            # print('maxp4_argmax_mask shape = {}'.format(maxp4_argmax_mask.shape))
            unpooled = max_unpool(conv6_3, maxp4_argmax_mask, pooling_stride)
            # print('unpooled shape = {}'.format(unpooled.shape))
            conv7_1 = conv_layer(unpooled, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv7_2 = conv_layer(conv7_1, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv7_3 = conv_layer(conv7_2, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training) # adjust feature dimensionality to match next layer after upsampling

            # print('conv7_3 shape = {}'.format(conv7_3.shape))
            # print('maxp3_argmax_mask shape = {}'.format(maxp3_argmax_mask.shape))
            unpooled = max_unpool(conv7_3, maxp3_argmax_mask, pooling_stride)
            # print('unpooled shape = {}'.format(unpooled.shape))
            conv8_1 = conv_layer(unpooled, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv8_2 = conv_layer(conv8_1, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv8_3 = conv_layer(conv8_2, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training) # adjust feature dimensionality to match next layer after upsampling

            # print('conv8_3 shape = {}'.format(conv8_3.shape))
            # print('maxp2_argmax_mask shape = {}'.format(maxp2_argmax_mask.shape))
            unpooled = max_unpool(conv8_3, maxp2_argmax_mask, pooling_stride)
            # print('unpooled shape = {}'.format(unpooled.shape))
            conv9_1 = conv_layer(unpooled, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv9_2 = conv_layer(conv9_1, BASELINE_FEATURE_DEPTH, kernel, is_training) # adjust feature dimensionality to match next layer after upsampling

            # print('conv9_2 shape = {}'.format(conv9_2.shape))
            # print('maxp1_argmax_mask shape = {}'.format(maxp1_argmax_mask.shape))
            unpooled = max_unpool(conv9_2, maxp1_argmax_mask, pooling_stride)
            # print('unpooled shape = {}'.format(unpooled.shape))
            conv10_1 = conv_layer(unpooled, BASELINE_FEATURE_DEPTH, kernel, is_training)

        logits = conv_layer(conv10_1, number_classes, 1, is_training, name='logits') # 1x1 kernel to convert feature map into class map

    return logits


def add_loss_ops(logits, labels, number_classes=2):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int32)
    labels_one_hot = tf.one_hot(labels, depth=number_classes)

    # compute loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits))

    return loss


def tower_loss(images, labels, number_classes, is_training):
    # Build inference Graph.
    logits = add_inference_ops(images, is_training, number_classes)

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



