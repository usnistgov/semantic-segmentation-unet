import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')

import numpy as np
import os
import skimage.io

import fcd_model

# load model for inference
number_classes = 2
global_batch_size = 1
img_size = [512, 512, 1]
learning_rate = 1e-4

# model = fcd_model.FCDensenet(number_classes, global_batch_size, img_size, learning_rate)
# checkpoint_filepath = '/home/mmajursk/Downloads/todo/ooc/fcd-model/checkpoint/ckpt'
# model.load_checkpoint(checkpoint_filepath)
#
# erf = model.estimate_radius()
# print('estiamted radius : "{}"'.format(erf))


model = fcd_model.FCDensenet(number_classes, global_batch_size, img_size, learning_rate)
erf = model.estimate_radius()
print('estiamted radius : "{}"'.format(erf))
