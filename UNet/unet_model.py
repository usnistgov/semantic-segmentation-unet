import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    import warnings
    warnings.warn('Codebase designed for Tensorflow 2.x.x')

NUMBER_CHANNELS = 1  # grayscale

BASELINE_FEATURE_DEPTH = 64




def conv_layer(input, filter_count, kernel, stride=1):
    output = tf.keras.layers.Conv2D(filters=filter_count,
                                         kernel_size=kernel,
                                         strides=stride,
                                        padding='same',
                                         activation=tf.keras.activations.relu, # 'relu'
                                         data_format='channels_first')(input)
    output = tf.keras.layers.BatchNormalization(axis=1)(output)
    return output


def deconv_layer(input, filter_count, kernel, stride=1):
    output = tf.keras.layers.Conv2DTranspose(filters=filter_count,
                                         kernel_size=kernel,
                                         strides=stride,
                                         activation=None,
                                        padding='same',
                                         data_format='channels_first')(input)
    output = tf.keras.layers.BatchNormalization(axis=1)(output)
    return output


def pool(input, size):
    pool = tf.keras.layers.MaxPool2D(pool_size=size, data_format='channels_first')(input)
    return pool


def concat(input1, input2, axis):
    output = tf.keras.layers.Concatenate(axis=axis)([input1, input2])
    return output


def get_model(input, number_classes=2):
    kernel = 3
    deconv_kernel = 2
    pooling_stride = 2

    # Encoder
    conv_1 = conv_layer(input, BASELINE_FEATURE_DEPTH, kernel)
    conv_1 = conv_layer(conv_1, BASELINE_FEATURE_DEPTH, kernel)

    pool_1 = pool(conv_1, pooling_stride)

    conv_2 = conv_layer(pool_1, 2 * BASELINE_FEATURE_DEPTH, kernel)
    conv_2 = conv_layer(conv_2, 2 * BASELINE_FEATURE_DEPTH, kernel)

    pool_2 = pool(conv_2, pooling_stride)

    conv_3 = conv_layer(pool_2, 4 * BASELINE_FEATURE_DEPTH, kernel)
    conv_3 = conv_layer(conv_3, 4 * BASELINE_FEATURE_DEPTH, kernel)

    pool_3 = pool(conv_3, pooling_stride)

    conv_4 = conv_layer(pool_3, 8 * BASELINE_FEATURE_DEPTH, kernel)
    conv_4 = conv_layer(conv_4, 8 * BASELINE_FEATURE_DEPTH, kernel)

    pool_4 = pool(conv_4, pooling_stride)

    # bottleneck
    bottleneck = conv_layer(pool_4, 16 * BASELINE_FEATURE_DEPTH, kernel)
    bottleneck = conv_layer(bottleneck, 16 * BASELINE_FEATURE_DEPTH, kernel)


    # Decoder
    # up-conv which reduces the number of feature channels by 2
    deconv_4 = deconv_layer(bottleneck, 8 * BASELINE_FEATURE_DEPTH, deconv_kernel, stride=pooling_stride)
    deconv_4 = concat(conv_4, deconv_4, axis=1)
    deconv_4 = conv_layer(deconv_4, 8 * BASELINE_FEATURE_DEPTH, kernel)
    deconv_4 = conv_layer(deconv_4, 8 * BASELINE_FEATURE_DEPTH, kernel)

    deconv_3 = deconv_layer(deconv_4, 4 * BASELINE_FEATURE_DEPTH, deconv_kernel, stride=pooling_stride)
    deconv_3 = concat(conv_3, deconv_3, axis=1)
    deconv_3 = conv_layer(deconv_3, 4 * BASELINE_FEATURE_DEPTH, kernel)
    deconv_3 = conv_layer(deconv_3, 4 * BASELINE_FEATURE_DEPTH, kernel)

    deconv_2 = deconv_layer(deconv_3, 2 * BASELINE_FEATURE_DEPTH, deconv_kernel, stride=pooling_stride)
    deconv_2 = concat(conv_2, deconv_2, axis=1)
    deconv_2 = conv_layer(deconv_2, 2 * BASELINE_FEATURE_DEPTH, kernel)
    deconv_2 = conv_layer(deconv_2, 2 * BASELINE_FEATURE_DEPTH, kernel)

    deconv_1 = deconv_layer(deconv_2, BASELINE_FEATURE_DEPTH, deconv_kernel, stride=pooling_stride)
    deconv_1 = concat(conv_1, deconv_1, axis=1)
    deconv_1 = conv_layer(deconv_1, BASELINE_FEATURE_DEPTH, kernel)
    deconv_1 = conv_layer(deconv_1, BASELINE_FEATURE_DEPTH, kernel)

    logits = conv_layer(deconv_1, number_classes, 1) # 1x1 kernel to convert feature map into class map
    # convert NCHW to NHWC so that softmax axis is the last dimension
    logits = tf.transpose(logits, [0, 2, 3, 1], name='logits')
    # logits is [NHWC]

    softmax = tf.keras.layers.Softmax(axis=-1, name='softmax')(logits)

    unet = tf.keras.Model(input, softmax, name='unet')
    unet.summary()

    return unet


#
#
# @tf.function
# def train_step(images, labels, loss_fn, optimier):
#     # Open a GradientTape to record the operations run
#     # during the forward pass, which enables autodifferentiation.
#     with tf.GradientTape() as tape:
#         softmax = model(images)
#
#         loss_value = loss_fn(batch_labels, softmax)
#
#     # Use the gradient tape to automatically retrieve
#     # the gradients of the trainable variables with respect to the loss.
#     grads = tape.gradient(loss_value, model.trainable_weights)
#
#     # Run one step of gradient descent by updating
#     # the value of the variables to minimize the loss.
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#
#     # update the metrics
#     train_loss_metric(loss_value)
#     train_acc_metric(batch_labels, softmax)