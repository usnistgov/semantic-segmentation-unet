# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    print('Tensorflow 2.x.x required')
    sys.exit(1)


class UNet():
    _BASELINE_FEATURE_DEPTH = 64
    _KERNEL_SIZE = 3
    _DECONV_KERNEL_SIZE = 2
    _POOLING_STRIDE = 2

    SIZE_FACTOR = 16

    @staticmethod
    def _conv_layer(input, filter_count, kernel, stride=1):
        output = tf.keras.layers.Conv2D(filters=filter_count,
                                        kernel_size=kernel,
                                        strides=stride,
                                        padding='same',
                                        activation=tf.keras.activations.relu,  # 'relu'
                                        data_format='channels_first')(input)
        output = tf.keras.layers.BatchNormalization(axis=1)(output)
        return output

    @staticmethod
    def _deconv_layer(input, filter_count, kernel, stride=1):
        output = tf.keras.layers.Conv2DTranspose(filters=filter_count,
                                                 kernel_size=kernel,
                                                 strides=stride,
                                                 activation=None,
                                                 padding='same',
                                                 data_format='channels_first')(input)
        output = tf.keras.layers.BatchNormalization(axis=1)(output)
        return output

    @staticmethod
    def _pool(input, size):
        pool = tf.keras.layers.MaxPool2D(pool_size=size, data_format='channels_first')(input)
        return pool

    @staticmethod
    def _concat(input1, input2, axis):
        output = tf.keras.layers.Concatenate(axis=axis)([input1, input2])
        return output

    @staticmethod
    def _dropout(input):
        output = tf.keras.layers.Dropout(rate=0.5)(input)
        return output

    def __init__(self, number_classes, global_batch_size, img_size, learning_rate=3e-4):

        self.img_size = img_size
        self.learning_rate = learning_rate
        self.number_classes = number_classes
        self.global_batch_size = global_batch_size
        self.is_training = False

        # image is HWC (normally e.g. RGB image) however data needs to be NCHW for network
        self.inputs = tf.keras.Input(shape=(img_size[2], None, None))
        # self.inputs = tf.keras.Input(shape=(img_size[2], img_size[0], img_size[1]))
        self.model = self._build_model()

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):

        # Encoder
        conv_1 = UNet._conv_layer(self.inputs, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_1 = UNet._conv_layer(conv_1, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        pool_1 = UNet._pool(conv_1, UNet._POOLING_STRIDE)

        conv_2 = UNet._conv_layer(pool_1, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_2 = UNet._conv_layer(conv_2, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        pool_2 = UNet._pool(conv_2, UNet._POOLING_STRIDE)

        conv_3 = UNet._conv_layer(pool_2, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_3 = UNet._conv_layer(conv_3, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        pool_3 = UNet._pool(conv_3, UNet._POOLING_STRIDE)

        conv_4 = UNet._conv_layer(pool_3, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_4 = UNet._conv_layer(conv_4, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_4 = UNet._dropout(conv_4)

        pool_4 = UNet._pool(conv_4, UNet._POOLING_STRIDE)

        # bottleneck
        bottleneck = UNet._conv_layer(pool_4, 16 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        bottleneck = UNet._conv_layer(bottleneck, 16 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        bottleneck = UNet._dropout(bottleneck)

        # Decoder
        # up-conv which reduces the number of feature channels by 2
        deconv_4 = UNet._deconv_layer(bottleneck, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        deconv_4 = UNet._concat(conv_4, deconv_4, axis=1)
        deconv_4 = UNet._conv_layer(deconv_4, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_4 = UNet._conv_layer(deconv_4, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        deconv_3 = UNet._deconv_layer(deconv_4, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        deconv_3 = UNet._concat(conv_3, deconv_3, axis=1)
        deconv_3 = UNet._conv_layer(deconv_3, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_3 = UNet._conv_layer(deconv_3, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        deconv_2 = UNet._deconv_layer(deconv_3, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        deconv_2 = UNet._concat(conv_2, deconv_2, axis=1)
        deconv_2 = UNet._conv_layer(deconv_2, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_2 = UNet._conv_layer(deconv_2, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        deconv_1 = UNet._deconv_layer(deconv_2, UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        deconv_1 = UNet._concat(conv_1, deconv_1, axis=1)
        deconv_1 = UNet._conv_layer(deconv_1, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_1 = UNet._conv_layer(deconv_1, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        logits = UNet._conv_layer(deconv_1, self.number_classes, 1)  # 1x1 kernel to convert feature map into class map
        # convert NCHW to NHWC so that softmax axis is the last dimension
        logits = tf.keras.layers.Permute((2, 3, 1))(logits)
        # logits is [NHWC]

        softmax = tf.keras.layers.Softmax(axis=-1, name='softmax')(logits)

        unet = tf.keras.Model(self.inputs, softmax, name='unet')

        return unet

    def get_keras_model(self):
        return self.model

    def set_is_training(self, val):
        self.is_training = val

    def get_optimizer(self):
        return self.optimizer

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

