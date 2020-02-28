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

import unet_model

# load model for inference
number_classes = 2
global_batch_size = 1
img_size = [512, 512, 1]
learning_rate = 1e-4

model = unet_model.UNet(number_classes, global_batch_size, img_size, learning_rate)
checkpoint_filepath = '/home/mmajursk/Downloads/todo/ooc/unet-model/checkpoint/ckpt'
model.load_checkpoint(checkpoint_filepath)

erf = model.estimate_radius()
print('estiamted radius : "{}"'.format(erf))

# keras_model = model.get_keras_model()
#
# N = 2 * unet_model.UNet.RADIUS
#
# def round_erf(x):
#     f = np.ceil(float(x) / unet_model.UNet.SIZE_FACTOR)
#     return unet_model.UNet.SIZE_FACTOR * f
#
# img = tf.Variable(np.random.normal(size=(1, 1, N, N)), dtype=tf.float32)
#
# mid_idx = int(N/2)
# loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
#
# with tf.GradientTape() as tape:
#     softmax = keras_model(img)
#     msk = softmax.numpy()
#     msk[0, mid_idx, mid_idx, :] = 1.0 - msk[0, mid_idx, mid_idx, :]
#     loss_value = loss_fn(msk, softmax)
#
# # Use the gradient tape to automatically retrieve
# # the gradients of the input image with respect to the loss.
# grads = tape.gradient(loss_value, img)
# grad_img = grads[0].numpy().squeeze()
# grad_img = np.abs(grad_img)
#
# # sum_grad_img = sum_grad_img / reps
# skimage.io.imsave(os.path.join('/home/mmajursk/Downloads/todo/ooc/', 'input_grad.tif'), grad_img)
#
# print('Theoretical RF: {}'.format(unet_model.UNet.RADIUS))
# eps = 1e-8
# vec = np.maximum(np.max(grad_img, axis=0).squeeze(), np.max(grad_img, axis=1).squeeze())
# idx = np.nonzero(vec > eps)
# erf = int((np.max(idx) - np.min(idx)) / 2)
# print('erf : "{}"'.format(erf))
# print('  becomes {}'.format(round_erf(erf)))











# for thres in (0.01, 0.02, 0.04, 0.05):
#     eps = thres * np.max(vec)
#     idx = np.nonzero(vec > eps)
#     erf = int((np.max(idx) - np.min(idx)) / 2)
#     print('erf 2 (thres: {}%) : "{}"'.format(thres, erf))
#     print('  becomes {}'.format(round_erf(erf)))
#
# import matplotlib.pyplot as plt
# from scipy import optimize
#
# def gaussian(x, amplitude, mean, stddev):
#     return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
#
# vec = np.maximum(np.max(grad_img, axis=0).squeeze(), np.max(grad_img, axis=1).squeeze())
# x = np.arange(0, len(vec)) - mid_idx
# popt_h, _ = optimize.curve_fit(gaussian, x, vec)
#
# plt.plot(x, vec)
# plt.plot(x, gaussian(x, *popt_h))



# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import skimage.transform
#
# img = grad_img + 1
# # img = np.log(img)
# img = skimage.transform.rescale(img, 0.5)
#
# # create the x and y coordinate arrays (here we just use pixel indices)
# xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
#
# # create the figure
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)
#
# # show it
# plt.show()
#
