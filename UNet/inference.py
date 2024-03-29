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

import argparse
import os
import numpy as np
import imagereader
import skimage.io
import model

TILE_SIZE = 1024


def _inference_tiling(img, unet_model, tile_size):

    # Pad the input image in CPU memory to ensure its dimensions are multiples of the U-Net Size Factor
    pad_x = 0
    pad_y = 0
    if img.shape[0] % model.UNet.SIZE_FACTOR != 0:
        pad_y = (model.UNet.SIZE_FACTOR - img.shape[0] % model.UNet.SIZE_FACTOR)
        print('image height needs to be a multiple of {}, padding with reflect'.format(model.UNet.SIZE_FACTOR))
    if img.shape[1] % model.UNet.SIZE_FACTOR != 0:
        pad_x = (model.UNet.SIZE_FACTOR - img.shape[1] % model.UNet.SIZE_FACTOR)
        print('image width needs to be a multiple of {}, padding with reflect'.format(model.UNet.SIZE_FACTOR))

    if len(img.shape) != 2 and len(img.shape) != 3:
        raise IOError('Invalid number of dimensions for input image. Expecting HW or HWC dimension ordering.')

    if len(img.shape) == 2:
        # add a channel dimension
        img = img.reshape((img.shape[0], img.shape[1], 1))
    if pad_x > 0 or pad_y > 0:
        img = np.pad(img, pad_width=((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
        print('Padded Image Size: {}'.format(img.shape))

    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros((height, width), dtype=np.int32)

    # radius = model.UNet.RADIUS  # theoretical radius
    radius = unet_model.estimate_radius()
    print('Estimated radius based on ERF : "{}"'.format(radius))
    assert tile_size % model.UNet.SIZE_FACTOR == 0
    assert radius % model.UNet.SIZE_FACTOR == 0
    zone_of_responsibility_size = tile_size - 2 * radius
    assert zone_of_responsibility_size >= radius

    for i in range(0, height, zone_of_responsibility_size):
        for j in range(0, width, zone_of_responsibility_size):

            x_st_z = j
            y_st_z = i
            x_end_z = x_st_z + zone_of_responsibility_size
            y_end_z = y_st_z + zone_of_responsibility_size

            # pad zone of responsibility by radius
            x_st = x_st_z - radius
            y_st = y_st_z - radius
            x_end = x_end_z + radius
            y_end = y_end_z + radius

            radius_pre_x = radius
            if x_st < 0:
                x_st = 0
                radius_pre_x = 0

            radius_pre_y = radius
            if y_st < 0:
                radius_pre_y = 0
                y_st = 0

            radius_post_x = radius
            if x_end > width:
                radius_post_x = 0
                x_end = width
                x_end_z = width

            radius_post_y = radius
            if y_end > height:
                radius_post_y = 0
                y_end = height
                y_end_z = height

            # crop out the tile
            tile = img[y_st:y_end, x_st:x_end]

            # convert HWC to CHW
            batch_data = tile.transpose((2, 0, 1))
            # convert CHW to NCHW
            batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))

            sm = unet_model.get_keras_model()(batch_data)  # model output defined in unet_model is softmax
            sm = np.squeeze(sm)
            pred = np.squeeze(np.argmax(sm, axis=-1).astype(np.int32))

            # radius_pre_x
            if radius_pre_x > 0:
                pred = pred[:, radius_pre_x:]
                sm = sm[:, radius_pre_x:]

            # radius_pre_y
            if radius_pre_y > 0:
                pred = pred[radius_pre_y:, :]
                sm = sm[radius_pre_y:, :]

            # radius_post_x
            if radius_post_x > 0:
                pred = pred[:, :-radius_post_x]
                sm = sm[:, :-radius_post_x]

            # radius_post_y
            if radius_post_y > 0:
                pred = pred[:-radius_post_y, :]
                sm = sm[:-radius_post_y, :]

            mask[y_st_z:y_end_z, x_st_z:x_end_z] = pred

    # undo and CPU side image padding to make the image a multiple of U-Net Size Factor
    if pad_x > 0:
        mask = mask[:, 0:-pad_x]
    if pad_y > 0:
        mask = mask[0:-pad_y, :]
    return mask


def _inference(img, unet_model):
    pad_x = 0
    pad_y = 0

    if img.shape[0] % model.UNet.SIZE_FACTOR != 0:
        pad_y = (model.UNet.SIZE_FACTOR - img.shape[0] % model.UNet.SIZE_FACTOR)
        print('image height needs to be a multiple of {}, padding with reflect'.format(model.UNet.SIZE_FACTOR))
    if img.shape[1] % model.UNet.SIZE_FACTOR != 0:
        pad_x = (model.UNet.SIZE_FACTOR - img.shape[1] % model.UNet.SIZE_FACTOR)
        print('image width needs to be a multiple of {}, padding with reflect'.format(model.UNet.SIZE_FACTOR))

    if len(img.shape) != 2 and len(img.shape) != 3:
        raise IOError('Invalid number of dimensions for input image. Expecting HW or HWC dimension ordering.')

    if len(img.shape) == 2:
        # add a channel dimension
        img = img.reshape((img.shape[0], img.shape[1], 1))
    img = np.pad(img, pad_width=((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')

    # convert HWC to CHW
    batch_data = img.transpose((2, 0, 1))
    # convert CHW to NCHW
    batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))
    batch_data = tf.convert_to_tensor(batch_data)

    softmax = unet_model.get_keras_model()(batch_data) # model output defined in unet_model is softmax
    softmax = np.squeeze(softmax)
    pred = np.squeeze(np.argmax(softmax, axis=-1).astype(np.int32))

    if pad_x > 0:
        pred = pred[:, 0:-pad_x]
    if pad_y > 0:
        pred = pred[0:-pad_y, :]

    return pred


def inference(checkpoint_filepath, image_folder, output_folder, number_classes, number_channels, image_format):
    print('Arguments:')
    print('checkpoint_filepath = {}'.format(checkpoint_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))
    print('image_format = {}'.format(image_format))
    print('number_classes = {}'.format(number_classes))
    print('number_channels = {}'.format(number_channels))

    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    unet_model = model.UNet(number_classes, 1, number_channels, 1e-4)
    unet_model.load_checkpoint(checkpoint_filepath)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, slide_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), slide_name))

        print('Loading image: {}'.format(img_filepath))
        img = imagereader.imread(img_filepath)
        img = img.astype(np.float32)

        # normalize with whole image stats
        # image from imagereader.imread(img_filepath) is in channels last format since the reader uses skimage.io
        img = imagereader.zscore_normalize(img, channels_first=False)
        print('  img.shape={}'.format(img.shape))

        # inference function expects channels last (i.e. an image just read from disk)
        if img.shape[0] > TILE_SIZE or img.shape[1] > TILE_SIZE:
            segmented_mask = _inference_tiling(img, unet_model, TILE_SIZE)
        else:
            segmented_mask = _inference(img, unet_model)

        if 0 <= np.max(segmented_mask) <= 255:
            segmented_mask = segmented_mask.astype(np.uint8)
        if 255 < np.max(segmented_mask) < 65536:
            segmented_mask = segmented_mask.astype(np.uint16)
        if np.max(segmented_mask) > 65536:
            segmented_mask = segmented_mask.astype(np.int32)
        if 'tif' in image_format:
            skimage.io.imsave(os.path.join(output_folder, slide_name), segmented_mask, compress=6, bigtiff=True, tile=(1024,1024))
        else:
            try:
                skimage.io.imsave(os.path.join(output_folder, slide_name), segmented_mask, compress=6)
            except TypeError:  # compress option not valid
                skimage.io.imsave(os.path.join(output_folder, slide_name), segmented_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference',
                                     description='Script to detect stars with the selected unet model')

    parser.add_argument('--checkpoint_filepath', dest='checkpoint_filepath', type=str,
                        help='Checkpoint filepath to the  model to use', required=True)
    parser.add_argument('--image_folder', dest='image_folder', type=str,
                        help='filepath to the folder containing tif images to inference (Required)', required=True)
    parser.add_argument('--output_folder', dest='output_folder', type=str, required=True)
    parser.add_argument('--number_classes', dest='number_classes', type=int, required=True)
    parser.add_argument('--number_channels', dest='number_channels', type=int, required=True)
    parser.add_argument('--image_format', dest='image_format', type=str, help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')

    args = parser.parse_args()

    checkpoint_filepath = args.checkpoint_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    image_format = args.image_format
    number_classes = args.number_classes
    number_channels = args.number_channels

    inference(checkpoint_filepath, image_folder, output_folder, number_classes, number_channels, image_format)


