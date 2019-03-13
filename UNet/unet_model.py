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
    conv_output = tf.layers.batch_normalization(conv_output, training=is_training, axis=1)  # axis to normalize (the channels dimension)
    return conv_output


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
    conv_output = tf.layers.batch_normalization(conv_output, training=is_training, axis=1)  # axis to normalize (the channels dimension)
    return conv_output


def add_inference_ops(inputs, is_training, number_classes=2):
    kernel = 3
    deconv_kernel = 2
    pooling_stride = 2

    with tf.variable_scope('unet'):
        with tf.variable_scope("encoder"):
            conv_1 = conv_layer(inputs, BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv_1 = conv_layer(conv_1, BASELINE_FEATURE_DEPTH, kernel, is_training)

            pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=pooling_stride, strides=pooling_stride, data_format='channels_first')

            conv_2 = conv_layer(pool_1, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv_2 = conv_layer(conv_2, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=pooling_stride, strides=pooling_stride, data_format='channels_first')

            conv_3 = conv_layer(pool_2, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv_3 = conv_layer(conv_3, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            pool_3 = tf.layers.max_pooling2d(conv_3, pool_size=pooling_stride, strides=pooling_stride, data_format='channels_first')

            conv_4 = conv_layer(pool_3, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            conv_4 = conv_layer(conv_4, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            pool_4 = tf.layers.max_pooling2d(conv_4, pool_size=pooling_stride, strides=pooling_stride, data_format='channels_first')

            # bottleneck
            bottleneck = conv_layer(pool_4, 16 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            bottleneck = conv_layer(bottleneck, 16 * BASELINE_FEATURE_DEPTH, kernel, is_training)

        with tf.variable_scope("decoder"):
            # up-conv which reduces the number of feature channels by 2
            deconv_4 = deconv_layer(bottleneck, 8 * BASELINE_FEATURE_DEPTH, deconv_kernel, is_training, stride=pooling_stride)
            deconv_4 = tf.concat([conv_4, deconv_4], axis=1)
            deconv_4 = conv_layer(deconv_4, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            deconv_4 = conv_layer(deconv_4, 8 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            deconv_3 = deconv_layer(deconv_4, 4 * BASELINE_FEATURE_DEPTH, deconv_kernel, is_training, stride=pooling_stride)
            deconv_3 = tf.concat([conv_3, deconv_3], axis=1)
            deconv_3 = conv_layer(deconv_3, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            deconv_3 = conv_layer(deconv_3, 4 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            deconv_2 = deconv_layer(deconv_3, 2 * BASELINE_FEATURE_DEPTH, deconv_kernel, is_training, stride=pooling_stride)
            deconv_2 = tf.concat([conv_2, deconv_2], axis=1)
            deconv_2 = conv_layer(deconv_2, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)
            deconv_2 = conv_layer(deconv_2, 2 * BASELINE_FEATURE_DEPTH, kernel, is_training)

            deconv_1 = deconv_layer(deconv_2, BASELINE_FEATURE_DEPTH, deconv_kernel, is_training, stride=pooling_stride)
            deconv_1 = tf.concat([conv_1, deconv_1], axis=1)
            deconv_1 = conv_layer(deconv_1, BASELINE_FEATURE_DEPTH, kernel, is_training)
            deconv_1 = conv_layer(deconv_1, BASELINE_FEATURE_DEPTH, kernel, is_training)

        logits = conv_layer(deconv_1, number_classes, 1, is_training, name='logits') # 1x1 kernel to convert feature map into class map
        # convert NCHW to NHWC so that softmax axis is the last dimension
        logits = tf.transpose(logits, [0, 2, 3, 1])
        # logits is [NHWC]
    return logits


def add_loss_ops(logits, labels, number_classes=2):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int32)
    labels_one_hot = tf.one_hot(labels, depth=number_classes) # creates NHWC
    labels_one_hot = tf.stop_gradient(labels_one_hot)

    # compute loss, channels is expected to be the last dimension by softmax_cross_entropy_with_logits_v2
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


def build_input_iterators(train_reader, test_reader, batch_prefetch_count, tile_size):
    img_size = train_reader.get_image_size()
    batch_size = train_reader.get_batch_size()

    if img_size[0] != tile_size or img_size[1] != tile_size:
        print('Image Size: {}, {}'.format(img_size[0], img_size[1]))
        print('Expected Size: {}, {}'.format(tile_size, tile_size))
        raise Exception('Invalid input shape, does not match specified network tile size.')

    print('Creating Input Train Dataset')
    # wrap the input queues into a Dataset
    # this sets up the imagereader class as a Python generator
    image_shape = tf.TensorShape((batch_size, 1, img_size[0], img_size[1]))
    label_shape = tf.TensorShape((batch_size, img_size[0], img_size[1]))
    train_dataset = tf.data.Dataset.from_generator(train_reader.generator, output_types=(tf.float32, tf.int32), output_shapes=(image_shape, label_shape))
    train_dataset = train_dataset.prefetch(batch_prefetch_count)  # prefetch N batches

    print('Creating Input Test Dataset')
    test_dataset = tf.data.Dataset.from_generator(test_reader.generator, output_types=(tf.float32, tf.int32), output_shapes=(image_shape, label_shape))
    test_dataset = test_dataset.prefetch(batch_prefetch_count)  # prefetch N batches

    print('Converting Datasets to Iterator')
    # create a iterator of the correct shape and type
    # the input iterator (feeding from one of the two imagereader generators) can be switched as needed by running the appropriate init_op
    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iter.make_initializer(train_dataset)
    test_init_op = iter.make_initializer(test_dataset)

    return train_init_op, test_init_op, iter


def build_towered_model(train_reader, test_reader, gpu_ids, learning_rate, number_classes, tile_size):
    is_training_placeholder = tf.placeholder(tf.bool, name='is_training')

    if gpu_ids == -1:
        num_gpus = 1
    else:
        num_gpus = len(gpu_ids)
    batch_prefetch_count = num_gpus
    train_init_op, test_init_op, iter = build_input_iterators(train_reader, test_reader, batch_prefetch_count, tile_size)

    if tile_size % 16 != 0:
        raise IOError('Input Image tile size needs to be a multiple of 16 to allow integer sized downscaled feature maps')

    # configure the Adam optimizer for network training with the specified learning reate
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Calculate the gradients for each model tower.
    tower_grads = []
    ops_per_gpu = {}
    with tf.variable_scope(tf.get_variable_scope()):
        if gpu_ids == -1:
            print('Building tower for CPU:0')
            with tf.device('/cpu:0'):
                with tf.name_scope('%s_%d' % (TOWER_NAME, 0)) as scope:
                    # Dequeues one batch for the GPU
                    image_batch, label_batch = iter.get_next()

                    # Calculate the loss for one tower of the model. This function
                    # constructs the entire model but shares the variables across
                    # all towers.
                    loss_op, accuracy_op = tower_loss(image_batch, label_batch, number_classes,
                                                      is_training_placeholder)
                    ops_per_gpu['gpu{}-loss'.format(0)] = loss_op
                    ops_per_gpu['gpu{}-accuracy'.format(0)] = accuracy_op

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Calculate the gradients for the batch of data on this tower.
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        grads = optimizer.compute_gradients(loss_op)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)
                    gpu_ids = [0] # to correctly handle loss and accuracy averaging
        else:
            for i in gpu_ids:
                print('Building tower for GPU:{}'.format(i))
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        # Dequeues one batch for the GPU
                        image_batch, label_batch = iter.get_next()

                        # Calculate the loss for one tower of the model. This function
                        # constructs the entire model but shares the variables across
                        # all towers.
                        loss_op, accuracy_op = tower_loss(image_batch, label_batch, number_classes, is_training_placeholder)
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
    print('Setting up Average Gradient')
    grads = average_gradients(tower_grads)

    # create merged accuracy stats
    print('Setting up Averaged Accuracy')
    all_loss_sum = tf.constant(0, dtype=tf.float32)
    all_accuracy_sum = tf.constant(0, dtype=tf.float32)

    for i in gpu_ids:
        all_loss_sum = tf.add(all_loss_sum, ops_per_gpu['gpu{}-loss'.format(i)])
        all_accuracy_sum = tf.add(all_accuracy_sum, ops_per_gpu['gpu{}-accuracy'.format(i)])

    loss_op = tf.divide(all_loss_sum, tf.constant(num_gpus, dtype=tf.float32))
    accuracy_op = tf.divide(all_accuracy_sum, tf.constant(num_gpus, dtype=tf.float32))

    # Apply the gradients to adjust the shared variables.
    print('Setting up Optimizer')
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.apply_gradients(grads)

    return train_init_op, test_init_op, train_op, loss_op, accuracy_op, is_training_placeholder
