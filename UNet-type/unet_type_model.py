# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise Exception('Python3 required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise Exception('Tensorflow 2.x.x required')
import numpy as np
import os


class UNet():
    _DECONV_KERNEL_SIZE = 2
    _POOLING_STRIDE = 2
    _CONFIG_FILE_NAME = 'unet_config.txt'

    @staticmethod
    def _compute_radius(M, nl, k):
        seq = list()
        for i in range(0, M + 1):
            seq.append(i)
        for i in range(M - 1, -1, -1):
            seq.append(i)

        radius = 0
        for i in seq:
            radius = radius + ((nl * np.floor(k / 2)) * np.power(2, i))
        return radius

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

    @staticmethod
    def convert_checkpoint_to_saved_model(training_checkpoint_folder, saved_model_folder):
        # create empty UNet
        model = UNet()
        # load the configuration for this UNet
        model._read_model_config(training_checkpoint_folder)
        model._build()

        checkpoint = tf.train.Checkpoint(optimizer=model.get_optimizer(), model=model.get_keras_model())
        checkpoint.restore(os.path.join(training_checkpoint_folder, 'ckpt'))
        tf.saved_model.save(model.get_keras_model(), saved_model_folder)

        model._write_model_config(saved_model_folder)

    def __init__(self, restore_saved_model_filepath=None):
        self.learning_rate = 3e-4
        self.number_classes = 2
        self.global_batch_size = 1
        self.input_channel_count = 1
        self.label_smoothing = 0
        self.M = 4
        self.nl = 2
        self.KERNEL_SIZE = 3
        self.BASELINE_FEATURE_DEPTH = 64

        self.inputs = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.checkpoint = None
        self.SIZE_FACTOR = np.power(2, self.M)
        self.RADIUS = int(self.SIZE_FACTOR * (np.floor(UNet._compute_radius(self.M, self.nl, self.KERNEL_SIZE) / self.SIZE_FACTOR) + 1))

        self.checkpoint_folder = None
        if restore_saved_model_filepath is not None:
            # perform tensorflow model load
            self.model = tf.saved_model.load(restore_saved_model_filepath)
            self._read_model_config(restore_saved_model_filepath)

    def configure(self, number_classes=2, global_batch_size=1, input_channel_count=1, learning_rate=3e-4, M=4, nl=2, k=3, b=64, label_smoothing=0):
        self.learning_rate = learning_rate
        self.number_classes = number_classes
        self.global_batch_size = global_batch_size
        self.input_channel_count = input_channel_count
        self.label_smoothing = label_smoothing

        if M < 1:
            raise Exception('Invalid number of UNet levels. Specified M = {}, must be >=1'.format(M))
        if nl < 1:
            raise Exception('Invalid number of UNet conv layers per level. Specified nl = {}, must be >=1'.format(nl))
        if k < 3 or k % 2 == 0:
            raise Exception('Invalid kerel size. Specified k = {}, must be >=1 and odd'.format(k))
        if b < 1:
            raise Exception('Invalid baseline feature map depth for UNet. Specified base_feature_depth = {}, must be >=1'.format(b))

        self.M = M
        self.nl = nl
        self.KERNEL_SIZE = k
        self.BASELINE_FEATURE_DEPTH = b

        self.SIZE_FACTOR = np.power(2, M)
        self.RADIUS = int(self.SIZE_FACTOR * (np.floor(UNet._compute_radius(M, nl, k) / self.SIZE_FACTOR) + 1))

        print('Building U-Net model with:')
        print('  Max Level Number: {}'.format(M))
        print('  {} Conv Layers per Level'.format(nl))
        print('  Kernel size: {}'.format(k))
        print('  Baseline Feature Depth: {}'.format(b))

        self._build()

    def _build(self):
        self._build_forward()
        self._build_backward()

    def _build_forward(self):
        # image is HWC (normally e.g. RGB image) however data needs to be NCHW for network
        self.inputs = tf.keras.Input(shape=(self.input_channel_count, None, None))
        self.model = self._build_layers()

    def _build_backward(self):
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=self.label_smoothing, reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    def _build_layers(self):

        encoder = list()
        input = self.inputs

        # Encoder
        for i in range(self.M):
            for j in range(self.nl):
                input = UNet._conv_layer(input, np.power(2, i) * self.BASELINE_FEATURE_DEPTH, self.KERNEL_SIZE)
            if i == (self.M-1):
                input = UNet._dropout(input)
            encoder.append(input)
            input = UNet._pool(input, UNet._POOLING_STRIDE)


        # bottleneck
        bottleneck = UNet._conv_layer(input, self.M * self.BASELINE_FEATURE_DEPTH, self.KERNEL_SIZE)
        bottleneck = UNet._conv_layer(bottleneck, self.M * self.BASELINE_FEATURE_DEPTH, self.KERNEL_SIZE)
        bottleneck = UNet._dropout(bottleneck)

        # Decoder
        input = bottleneck
        for i in range((self.M-1), -1, -1):
            input = UNet._deconv_layer(input, np.power(2, i) * self.BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
            input = UNet._concat(encoder[i], input, axis=1)
            for j in range(self.nl):
                input = UNet._conv_layer(input, np.power(2, i) * self.BASELINE_FEATURE_DEPTH, self.KERNEL_SIZE)

        logits = UNet._conv_layer(input, self.number_classes, 1)  # 1x1 kernel to convert feature map into class map
        # convert NCHW to NHWC so that softmax axis is the last dimension
        logits = tf.keras.layers.Permute((2, 3, 1))(logits)
        # logits is [NHWC]

        softmax = tf.keras.layers.Softmax(axis=-1, name='softmax')(logits)

        unet = tf.keras.Model(self.inputs, softmax, name='unet')

        return unet

    def get_keras_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.optimizer.learning_rate

    def _write_model_config(self, fldr):
        with open(os.path.join(fldr, UNet._CONFIG_FILE_NAME), 'w') as fh:
            fh.write('M:{}\n'.format(self.M))
            fh.write('nl:{}\n'.format(self.nl))
            fh.write('k:{}\n'.format(self.KERNEL_SIZE))
            fh.write('b:{}\n'.format(self.BASELINE_FEATURE_DEPTH))
            fh.write('number_classes:{}\n'.format(self.number_classes))
            fh.write('input_channel_count:{}\n'.format(self.input_channel_count))
            fh.write('label_smoothing:{}\n'.format(self.label_smoothing))

    def _read_model_config(self, fldr):
        # load model specifications from save_model_folder
        with open(os.path.join(fldr, UNet._CONFIG_FILE_NAME), 'r') as fh:
            data = fh.read().replace(' ', '').lower()
            toks = data.split('\n')
            for tok in toks:
                tok = tok.split(':')
                if tok[0] == 'm':
                    self.M = int(tok[1])
                    print('Loaded: M: {}'.format(self.M))
                if tok[0] == 'nl':
                    self.nl = int(tok[1])
                    print('Loaded: nl: {}'.format(self.nl))
                if tok[0] == 'k':
                    self.KERNEL_SIZE = int(tok[1])
                    print('Loaded: k: {}'.format(self.KERNEL_SIZE))
                if tok[0] == 'b':
                    self.BASELINE_FEATURE_DEPTH = int(tok[1])
                    print('Loaded: b: {}'.format(self.BASELINE_FEATURE_DEPTH))
                if tok[0] == 'number_classes':
                    self.number_classes = int(tok[1])
                    print('Loaded: number_classes: {}'.format(self.number_classes))
                if tok[0] == 'input_channel_count':
                    self.input_channel_count = int(tok[1])
                    print('Loaded: input_channel_count: {}'.format(self.input_channel_count))
                if tok[0] == 'label_smoothing':
                    self.label_smoothing = int(tok[1])
                    print('Loaded: label_smoothing: {}'.format(self.label_smoothing))

        self.SIZE_FACTOR = np.power(2, self.M)
        self.RADIUS = int(self.SIZE_FACTOR * (np.floor(UNet._compute_radius(self.M, self.nl, self.KERNEL_SIZE) / self.SIZE_FACTOR) + 1))
        print('Loaded: SIZE_FACTOR = {}'.format(self.SIZE_FACTOR))
        print('Loaded: RADIUS = {}'.format(self.RADIUS))

    def save_checkpoint(self, checkpoint_folder):
        self.checkpoint_folder = checkpoint_folder
        training_checkpoint_filepath = self.checkpoint.write(os.path.join(self.checkpoint_folder, 'ckpt'))
        self._write_model_config(checkpoint_folder)
        return training_checkpoint_filepath

    def save_model(self, saved_model_folder):
        tf.saved_model.save(self.model, saved_model_folder)
        self._write_model_config(saved_model_folder)

    def write_model_summary(self, output_folder):
        # print the model summary to file
        with open(os.path.join(output_folder, 'model.txt'), 'w') as summary_fh:
            print_fn = lambda x: print(x, file=summary_fh)
            self.model.summary(print_fn=print_fn)
        tf.keras.utils.plot_model(self.model, os.path.join(output_folder, 'model.png'), show_shapes=True)
        tf.keras.utils.plot_model(self.model, os.path.join(output_folder, 'model.dot'), show_shapes=True)

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

