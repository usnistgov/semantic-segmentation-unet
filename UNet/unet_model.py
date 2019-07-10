import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    import warnings
    warnings.warn('Codebase designed for Tensorflow 2.x.x')


class UNet(tf.keras.Model):
    _BASELINE_FEATURE_DEPTH = 64

    def __init__(self, number_classes, learning_rate=1e-4):
        super(UNet, self).__init__()

        self.learning_rate = learning_rate
        self.number_classes = number_classes

        self.kernel_size = 3
        self.deconv_kernel_size = 2
        self.pooling_stride = 2

        # encoder layer 1 *************************************************************************************
        filter_count = self._BASELINE_FEATURE_DEPTH
        self.enc_layer1_conv1 = tf.keras.layers.Conv2D(filters=filter_count,
                                                 kernel_size=self.kernel_size,
                                                 strides=1,
                                                 activation=tf.keras.activations.relu,
                                                 padding='same',
                                                 data_format='channels_first')
        self.enc_layer1_bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.enc_layer1_conv2 = tf.keras.layers.Conv2D(filters=filter_count,
                                                  kernel_size=self.kernel_size,
                                                  strides=1,
                                                  activation=tf.keras.activations.relu,
                                                  padding='same',
                                                  data_format='channels_first')
        self.enc_layer1_bn2 = tf.keras.layers.BatchNormalization(axis=1)

        self.enc_layer1_pool = tf.keras.layers.MaxPool2D(pool_size=self.pooling_stride, data_format='channels_first')

        # encoder layer 2 *************************************************************************************
        filter_count = 2 * self._BASELINE_FEATURE_DEPTH
        self.enc_layer2_conv1 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                   data_format='channels_first')
        self.enc_layer2_bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.enc_layer2_conv2 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                   data_format='channels_first')
        self.enc_layer2_bn2 = tf.keras.layers.BatchNormalization(axis=1)

        self.enc_layer2_pool = tf.keras.layers.MaxPool2D(pool_size=self.pooling_stride, data_format='channels_first')

        # encoder layer 3 *************************************************************************************
        filter_count = 4 * self._BASELINE_FEATURE_DEPTH
        self.enc_layer3_conv1 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                   data_format='channels_first')
        self.enc_layer3_bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.enc_layer3_conv2 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                   data_format='channels_first')
        self.enc_layer3_bn2 = tf.keras.layers.BatchNormalization(axis=1)

        self.enc_layer3_pool = tf.keras.layers.MaxPool2D(pool_size=self.pooling_stride, data_format='channels_first')

        # encoder layer 4 *************************************************************************************
        filter_count = 8 * self._BASELINE_FEATURE_DEPTH
        self.enc_layer4_conv1 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                   data_format='channels_first')
        self.enc_layer4_bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.enc_layer4_conv2 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                   data_format='channels_first')
        self.enc_layer4_bn2 = tf.keras.layers.BatchNormalization(axis=1)

        self.enc_layer4_pool = tf.keras.layers.MaxPool2D(pool_size=self.pooling_stride, data_format='channels_first')

        # encoder layer 5 *************************************************************************************
        filter_count = 16 * self._BASELINE_FEATURE_DEPTH
        self.enc_layer5_conv1 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                   data_format='channels_first')
        self.enc_layer5_bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.enc_layer5_conv2 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                   data_format='channels_first')
        self.enc_layer5_bn2 = tf.keras.layers.BatchNormalization(axis=1)

        # decoder layer 4 *************************************************************************************
        filter_count = 8 * self._BASELINE_FEATURE_DEPTH
        # up-conv which reduces the number of feature channels by 2
        self.dec_layer4_deconv0 = tf.keras.layers.Conv2DTranspose(filters=filter_count,
                                                 kernel_size=self.kernel_size,
                                                 strides=self.pooling_stride,
                                                 activation=None,
                                                 padding='same',
                                                 data_format='channels_first')
        self.dec_layer4_bn0 = tf.keras.layers.BatchNormalization(axis=1)
        self.dec_layer4_concat = tf.keras.layers.Concatenate(axis=1)

        self.dec_layer4_conv1 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                    data_format='channels_first')
        self.dec_layer4_bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.dec_layer4_conv2 = tf.keras.layers.Conv2D(filters=filter_count,
                                                   kernel_size=self.kernel_size,
                                                   strides=1,
                                                   activation=tf.keras.activations.relu,
                                                   padding='same',
                                                   data_format='channels_first')
        self.dec_layer4_bn2 = tf.keras.layers.BatchNormalization(axis=1)

        # decoder layer 3 *************************************************************************************
        filter_count = 4 * self._BASELINE_FEATURE_DEPTH
        # up-conv which reduces the number of feature channels by 2
        self.dec_layer3_deconv0 = tf.keras.layers.Conv2DTranspose(filters=filter_count,
                                                                  kernel_size=self.kernel_size,
                                                                  strides=self.pooling_stride,
                                                                  activation=None,
                                                                  padding='same',
                                                                  data_format='channels_first')
        self.dec_layer3_bn0 = tf.keras.layers.BatchNormalization(axis=1)
        self.dec_layer3_concat = tf.keras.layers.Concatenate(axis=1)

        self.dec_layer3_conv1 = tf.keras.layers.Conv2D(filters=filter_count,
                                                       kernel_size=self.kernel_size,
                                                       strides=1,
                                                       activation=tf.keras.activations.relu,
                                                       padding='same',
                                                       data_format='channels_first')
        self.dec_layer3_bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.dec_layer3_conv2 = tf.keras.layers.Conv2D(filters=filter_count,
                                                       kernel_size=self.kernel_size,
                                                       strides=1,
                                                       activation=tf.keras.activations.relu,
                                                       padding='same',
                                                       data_format='channels_first')
        self.dec_layer3_bn2 = tf.keras.layers.BatchNormalization(axis=1)

        # decoder layer 2 *************************************************************************************
        filter_count = 2  * self._BASELINE_FEATURE_DEPTH
        # up-conv which reduces the number of feature channels by 2
        self.dec_layer2_deconv0 = tf.keras.layers.Conv2DTranspose(filters=filter_count,
                                                                  kernel_size=self.kernel_size,
                                                                  strides=self.pooling_stride,
                                                                  activation=None,
                                                                  padding='same',
                                                                  data_format='channels_first')
        self.dec_layer2_bn0 = tf.keras.layers.BatchNormalization(axis=1)
        self.dec_layer2_concat = tf.keras.layers.Concatenate(axis=1)

        self.dec_layer2_conv1 = tf.keras.layers.Conv2D(filters=filter_count,
                                                       kernel_size=self.kernel_size,
                                                       strides=1,
                                                       activation=tf.keras.activations.relu,
                                                       padding='same',
                                                       data_format='channels_first')
        self.dec_layer2_bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.dec_layer2_conv2 = tf.keras.layers.Conv2D(filters=filter_count,
                                                       kernel_size=self.kernel_size,
                                                       strides=1,
                                                       activation=tf.keras.activations.relu,
                                                       padding='same',
                                                       data_format='channels_first')
        self.dec_layer2_bn2 = tf.keras.layers.BatchNormalization(axis=1)

        # decoder layer 1 *************************************************************************************
        filter_count = self._BASELINE_FEATURE_DEPTH
        # up-conv which reduces the number of feature channels by 2
        self.dec_layer1_deconv0 = tf.keras.layers.Conv2DTranspose(filters=filter_count,
                                                                  kernel_size=self.kernel_size,
                                                                  strides=self.pooling_stride,
                                                                  activation=None,
                                                                  padding='same',
                                                                  data_format='channels_first')
        self.dec_layer1_bn0 = tf.keras.layers.BatchNormalization(axis=1)
        self.dec_layer1_concat = tf.keras.layers.Concatenate(axis=1)

        self.dec_layer1_conv1 = tf.keras.layers.Conv2D(filters=filter_count,
                                                       kernel_size=self.kernel_size,
                                                       strides=1,
                                                       activation=tf.keras.activations.relu,
                                                       padding='same',
                                                       data_format='channels_first')
        self.dec_layer1_bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.dec_layer1_conv2 = tf.keras.layers.Conv2D(filters=filter_count,
                                                       kernel_size=self.kernel_size,
                                                       strides=1,
                                                       activation=tf.keras.activations.relu,
                                                       padding='same',
                                                       data_format='channels_first')
        self.dec_layer1_bn2 = tf.keras.layers.BatchNormalization(axis=1)

        # logit layer *************************************************************************************
        self.logit_layer = tf.keras.layers.Conv2D(filters=self.number_classes,
                                                                  kernel_size=1,
                                                                  strides=1,
                                                                  activation=tf.keras.activations.relu,
                                                                  padding='same',
                                                                  data_format='channels_first')

        self.permute_layer = tf.keras.layers.Permute((2, 3, 1)) # does not include samples dim, starts at 1
        self.softmax_layer = tf.keras.layers.Softmax(axis=-1, name='softmax') # NHWC

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    @tf.function
    def call(self, inputs, training=False):
        # Forward pass

        # Encoder

        # layer 1
        conv_1 = self.enc_layer1_conv1(inputs)
        conv_1 = self.enc_layer1_bn1(conv_1, training)
        conv_1 = self.enc_layer1_conv2(conv_1)
        conv_1 = self.enc_layer1_bn2(conv_1, training)
        pool_1 = self.enc_layer1_pool(conv_1)

        # layer 2
        conv_2 = self.enc_layer2_conv1(pool_1)
        conv_2 = self.enc_layer2_bn1(conv_2, training)
        conv_2 = self.enc_layer2_conv2(conv_2)
        conv_2 = self.enc_layer2_bn2(conv_2, training)
        pool_2 = self.enc_layer2_pool(conv_2)

        # layer 3
        conv_3 = self.enc_layer3_conv1(pool_2)
        conv_3 = self.enc_layer3_bn1(conv_3, training)
        conv_3 = self.enc_layer3_conv2(conv_3)
        conv_3 = self.enc_layer3_bn2(conv_3, training)
        pool_3 = self.enc_layer3_pool(conv_3)

        # layer 4
        conv_4 = self.enc_layer4_conv1(pool_3)
        conv_4 = self.enc_layer4_bn1(conv_4, training)
        conv_4 = self.enc_layer4_conv2(conv_4)
        conv_4 = self.enc_layer4_bn2(conv_4, training)
        pool_4 = self.enc_layer4_pool(conv_4)

        # layer 5
        bottleneck = self.enc_layer5_conv1(pool_4)
        bottleneck = self.enc_layer5_bn1(bottleneck, training)
        bottleneck = self.enc_layer5_conv2(bottleneck)
        bottleneck = self.enc_layer5_bn2(bottleneck, training)

        # Decoder

        # layer 4
        deconv_4 = self.dec_layer4_deconv0(bottleneck)
        deconv_4 = self.dec_layer4_bn0(deconv_4, training)
        deconv_4 = self.dec_layer4_concat([conv_4, deconv_4])
        deconv_4 = self.dec_layer4_conv1(deconv_4)
        deconv_4 = self.dec_layer4_bn1(deconv_4, training)
        deconv_4 = self.dec_layer4_conv2(deconv_4)
        deconv_4 = self.dec_layer4_bn2(deconv_4, training)

        # layer 3
        deconv_3 = self.dec_layer3_deconv0(deconv_4)
        deconv_3 = self.dec_layer3_bn0(deconv_3, training)
        deconv_3 = self.dec_layer3_concat([conv_3, deconv_3])
        deconv_3 = self.dec_layer3_conv1(deconv_3)
        deconv_3 = self.dec_layer3_bn1(deconv_3, training)
        deconv_3 = self.dec_layer3_conv2(deconv_3)
        deconv_3 = self.dec_layer3_bn2(deconv_3, training)

        # layer 2
        deconv_2 = self.dec_layer2_deconv0(deconv_3)
        deconv_2 = self.dec_layer2_bn0(deconv_2, training)
        deconv_2 = self.dec_layer2_concat([conv_2, deconv_2])
        deconv_2 = self.dec_layer2_conv1(deconv_2)
        deconv_2 = self.dec_layer2_bn1(deconv_2, training)
        deconv_2 = self.dec_layer2_conv2(deconv_2)
        deconv_2 = self.dec_layer2_bn2(deconv_2, training)

        # layer 1
        deconv_1 = self.dec_layer1_deconv0(deconv_2)
        deconv_1 = self.dec_layer1_bn0(deconv_1, training)
        deconv_1 = self.dec_layer1_concat([conv_1, deconv_1])
        deconv_1 = self.dec_layer1_conv1(deconv_1)
        deconv_1 = self.dec_layer1_bn1(deconv_1, training)
        deconv_1 = self.dec_layer1_conv2(deconv_1)
        deconv_1 = self.dec_layer1_bn2(deconv_1, training)

        # output logit

        logits = self.logit_layer(deconv_1)
        logits_nhwc = self.permute_layer(logits)
        softmax_nhwc = self.softmax_layer(logits_nhwc)

        return softmax_nhwc

    @tf.function
    def train_step(self, images, labels):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            softmax = self.call(images)

            loss_value = self.loss_fn(labels, softmax)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss_value, softmax

    @tf.function
    def test_step(self, images, labels):
        softmax = self.call(images)

        loss_value = self.loss_fn(labels, softmax)

        return loss_value, softmax


