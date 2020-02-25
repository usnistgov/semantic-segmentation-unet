# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')


class FCDensenet():

    __BN_MOMENTUM = 0.9
    __WEIGHT_DECAY = 1E-4
    __DROPOUT_RATE = 0.2

    SIZE_FACTOR = 16

    @staticmethod
    def __conv_block(x, nb_filters):
        x = tf.keras.layers.BatchNormalization(axis=1, momentum=FCDensenet.__BN_MOMENTUM)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters=nb_filters,
                                        kernel_size=3,
                                        padding='same',
                                        activation=None,
                                        use_bias=False,
                                        data_format='channels_first')(x)
        x = tf.keras.layers.Dropout(FCDensenet.__DROPOUT_RATE)(x)
        return x

    @staticmethod
    def __dense_block(x, nb_layers, nb_filters, growth_rate, grow_nb_filters=True):
        x_list = [x]
        for i in range(nb_layers):
            cb = FCDensenet.__conv_block(x, growth_rate)
            x_list.append(cb)
            x = tf.keras.layers.concatenate([x, cb], axis=1)
            if grow_nb_filters:
                nb_filters += growth_rate

        return x, nb_filters, x_list

    @staticmethod
    def __transition_down_block(x, nb_filters):
        x = tf.keras.layers.BatchNormalization(momentum=FCDensenet.__BN_MOMENTUM, axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters=nb_filters,
                                   kernel_size=1,
                                   padding='same',
                                   activation=None,
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(FCDensenet.__WEIGHT_DECAY),
                                   data_format='channels_first')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, data_format='channels_first')(x)
        return x

    @staticmethod
    def __transition_up_block(x, nb_filters):
        return tf.keras.layers.Conv2DTranspose(filters=nb_filters,
                                   kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   strides=2,
                                   kernel_regularizer=tf.keras.regularizers.l2(FCDensenet.__WEIGHT_DECAY),
                                    data_format='channels_first')(x)

    def __init__(self, number_classes, global_batch_size, img_size, learning_rate=1e-4, label_smoothing=0):

        self.global_batch_size = global_batch_size
        self.learning_rate = learning_rate
        self.nb_classes = number_classes
        self.img_size = img_size
        self.initial_kernel_size = (3, 3)

        self.name_scope = 'DenseNetFCN'
        self.nb_dense_block = 3
        self.nb_layers_per_block = 4
        self.init_conv_filters = 48
        self.growth_rate = 12

        # self.name_scope = 'FCDenseNet56'
        # self.nb_dense_block = 5
        # self.nb_layers_per_block = 4
        # self.init_conv_filters = 48
        # self.growth_rate = 12

        # self.name_scope = 'FCDenseNet67'
        # self.nb_dense_block = 5
        # self.nb_layers_per_block = 5
        # self.init_conv_filters = 48
        # self.growth_rate = 16

        # self.name_scope = 'FCDenseNet103'
        # self.nb_dense_block = 5
        # self.nb_layers_per_block = [4, 5, 7, 10, 12, 15]
        # self.init_conv_filters = 48
        # self.growth_rate = 16

        # image is HWC (normally e.g. RGB image) however data needs to be NCHW for network
        self.inputs = tf.keras.Input(shape=(img_size[2], None, None))

        self.model = self.build_model()

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing, reduction=tf.keras.losses.Reduction.NONE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_model(self):

        with tf.name_scope(self.name_scope):

            row, col, channel = self.img_size

            # `nb_layers` is a list with the number of layers in each dense block
            if type(self.nb_layers_per_block) is list or type(self.nb_layers_per_block) is tuple:
                nb_layers = list(self.nb_layers_per_block)  # Convert tuple to list

                if len(nb_layers) != (self.nb_dense_block + 1):
                    raise RuntimeError('If `nb_layers_per_block` is a list, its length must be (`nb_dense_block` + 1)')

                bottleneck_nb_layers = nb_layers[-1]
                rev_layers = nb_layers[::-1]
                nb_layers.extend(rev_layers[1:])
            else:
                bottleneck_nb_layers = self.nb_layers_per_block
                nb_layers = [self.nb_layers_per_block] * (2 * self.nb_dense_block + 1)

            print('Layers in each dense block: {}'.format(nb_layers))

            # make sure we can concatenate the skip connections with the upsampled
            # images on the upsampling path
            img_downsize_factor = 2**self.nb_dense_block
            if row % img_downsize_factor != 0:
                raise RuntimeError('Invalid image height {}. Image height must be a multiple of '
                                 '2^nb_dense_block={}'.format(row, img_downsize_factor))
            if col % img_downsize_factor != 0:
                raise RuntimeError('Invalid image width {}. Image width must be a multiple of '
                                 '2^nb_dense_block={}'.format(col, img_downsize_factor))

            # Initial convolution
            x = tf.keras.layers.Conv2D(filters=self.init_conv_filters, kernel_size=self.initial_kernel_size, padding='same',
                       use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(FCDensenet.__WEIGHT_DECAY), data_format='channels_first')(self.inputs)
            x = tf.keras.layers.BatchNormalization(momentum=FCDensenet.__BN_MOMENTUM, axis=1)(x)
            x = tf.keras.layers.Activation('relu')(x)

            # keeps track of the current number of feature maps
            nb_filter = self.init_conv_filters

            # collect skip connections on the downsampling path so that
            # they can be concatenated with outputs on the upsampling path
            skip_list = []

            # Build the downsampling path by adding dense blocks and transition down blocks
            for block_idx in range(self.nb_dense_block):
                x, nb_filter, _ = FCDensenet.__dense_block(x, nb_layers[block_idx], nb_filter, self.growth_rate)

                skip_list.append(x)
                x = FCDensenet.__transition_down_block(x, nb_filter)

            # Add the bottleneck dense block.
            _, nb_filter, concat_list = FCDensenet.__dense_block(x, bottleneck_nb_layers, nb_filter, self.growth_rate)

            print('Number of skip connections: {}'.format(len(skip_list)))

            # reverse the list of skip connections
            skip_list = skip_list[::-1]

            # Build the upsampling path by adding dense blocks and transition up blocks
            for block_idx in range(self.nb_dense_block):
                n_filters_keep = self.growth_rate * nb_layers[self.nb_dense_block + block_idx]

                # upsampling block must upsample only the feature maps (concat_list[1:]),
                # not the concatenation of the input with the feature maps
                l = tf.keras.layers.concatenate(concat_list[1:], axis=1, name='Concat_DenseBlock_out_{}'.format(block_idx))

                t = FCDensenet.__transition_up_block(l, nb_filters=n_filters_keep)

                # concatenate the skip connection with the transition block output
                x = tf.keras.layers.concatenate([t, skip_list[block_idx]], axis=1, name='Concat_SkipCon_{}'.format(block_idx))

                # Dont allow the feature map size to grow in upsampling dense blocks
                x_up, nb_filter, concat_list = FCDensenet.__dense_block(x, nb_layers[self.nb_dense_block + block_idx + 1], nb_filters=self.growth_rate,
                                                            growth_rate=self.growth_rate, grow_nb_filters=False)

            # final convolution
            x = tf.keras.layers.concatenate(concat_list[1:], axis=1)
            x = tf.keras.layers.Conv2D(filters=self.nb_classes, kernel_size=1, activation='linear', padding='same', use_bias=False, data_format='channels_first', name='logit')(x)

            # convert NCHW to NHWC so that softmax axis is the last dimension
            x = tf.keras.layers.Permute((2, 3, 1))(x)
            x = tf.keras.layers.Softmax(axis=-1, name='softmax')(x)

        fc_densenet = tf.keras.Model(self.inputs, x, name='fcd')

        return fc_densenet

    def get_keras_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.optimizer.learning_rate

    def train_step(self, inputs):
        (images, labels, loss_metric, accuracy_metric) = inputs
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            softmax = self.model(images, training=True)

            loss_value = self.loss_fn(labels, softmax) # [NxHxWx1]
            # average across the batch (N) with the approprite global batch size
            loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size
            # reduce down to a scalar (reduce H, W)
            loss_value = tf.reduce_mean(loss_value)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        loss_metric.update_state(loss_value)
        accuracy_metric.update_state(labels, softmax)

        return loss_value

    @tf.function
    def dist_train_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.train_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)

        return loss_value

    def test_step(self, inputs):
        (images, labels, loss_metric, accuracy_metric) = inputs
        softmax = self.model(images, training=False)

        loss_value = self.loss_fn(labels, softmax)
        # average across the batch (N) with the approprite global batch size
        loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size
        # reduce down to a scalar (reduce H, W)
        loss_value = tf.reduce_mean(loss_value)

        loss_metric.update_state(loss_value)
        accuracy_metric.update_state(labels, softmax)

        return loss_value

    @tf.function
    def dist_test_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.test_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)
        return loss_value
