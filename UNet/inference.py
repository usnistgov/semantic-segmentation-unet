# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import argparse
import os
import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    print('Tensorflow 2.x.x required')
    sys.exit(1)

import unet_model
import numpy as np
import imagereader


def _inference_tiling(img, model, tile_size):

    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros(img.shape, dtype=np.int32)

    # radius = unet_model.UNet.SIZE_FACTOR
    assert tile_size % unet_model.UNet.SIZE_FACTOR == 0
    radius = unet_model.UNet.RADIUS
    assert radius % unet_model.UNet.SIZE_FACTOR == 0
    zone_of_responsibility_size = tile_size - 2 * radius

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
                dist_from_edge = x_st_z
                radius_pre_x = int(np.ceil(dist_from_edge / unet_model.UNet.SIZE_FACTOR) * unet_model.UNet.SIZE_FACTOR)
                x_st = 0

            radius_pre_y = radius
            if y_st < 0:
                dist_from_edge = y_st_z
                radius_pre_y = int(np.ceil(dist_from_edge / unet_model.UNet.SIZE_FACTOR) * unet_model.UNet.SIZE_FACTOR)
                y_st = 0

            post_pad_x = 0
            radius_post_x = radius
            if x_end > width:
                if x_end_z > width:
                    tmp_w = width - x_st_z
                    w_plus_radius = int(np.ceil(tmp_w / unet_model.UNet.SIZE_FACTOR) * unet_model.UNet.SIZE_FACTOR)
                    radius_post_x = w_plus_radius - tmp_w
                    post_pad_x = radius_post_x
                    x_end = width
                    x_end_z = width
                else:
                    dist_from_edge = width - x_end_z
                    radius_post_x = int(np.ceil(dist_from_edge / unet_model.UNet.SIZE_FACTOR) * unet_model.UNet.SIZE_FACTOR)
                    post_pad_x = radius_post_x - dist_from_edge
                    x_end = width

            post_pad_y = 0
            radius_post_y = radius
            if y_end > height:
                if y_end_z > height:
                    tmp_h = height - y_st_z
                    h_plus_radius = int(np.ceil(tmp_h / unet_model.UNet.SIZE_FACTOR) * unet_model.UNet.SIZE_FACTOR)
                    radius_post_y = h_plus_radius - tmp_h
                    post_pad_y = radius_post_y
                    y_end = height
                    y_end_z = height
                else:
                    dist_from_edge = height - y_end_z
                    radius_post_y = int(np.ceil(dist_from_edge / unet_model.UNet.SIZE_FACTOR) * unet_model.UNet.SIZE_FACTOR)
                    post_pad_y = radius_post_y - dist_from_edge
                    y_end = height

            # crop out the tile
            tile = img[y_st:y_end, x_st:x_end]

            if len(tile.shape) != 2 and len(tile.shape) != 3:
                raise IOError('Invalid number of dimensions for input image. Expecting HW or HWC dimension ordering.')

            if len(tile.shape) == 2:
                # add a channel dimension
                tile = tile.reshape((tile.shape[0], tile.shape[1], 1))

            if post_pad_x > 0 or post_pad_y > 0:
                # ensure its correct size (if tile exists at the edge of the image
                tile = np.pad(tile, pad_width=((0, post_pad_y, 0), (0, post_pad_x, 0)), mode='reflect')

            # convert HWC to CHW
            batch_data = tile.transpose((2, 0, 1))
            # convert CHW to NCHW
            batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))

            softmax = model(batch_data) # model output defined in unet_model is softmax
            softmax = np.squeeze(softmax)
            pred = np.squeeze(np.argmax(softmax, axis=-1).astype(np.int32))

            # radius_pre_x
            if radius_pre_x > 0:
                pred = pred[:, radius_pre_x:]

            # radius_pre_y
            if radius_pre_y > 0:
                pred = pred[radius_pre_y:, :]

            # radius_post_x
            if radius_post_x > 0:
                pred = pred[:, :-radius_post_x]

            # radius_post_y
            if radius_post_y > 0:
                pred = pred[:-radius_post_y, :]

            mask[y_st_z:y_end_z, x_st_z:x_end_z] = pred

    return mask


def _inference(img, model):
    pad_x = 0
    pad_y = 0

    if img.shape[0] % unet_model.UNet.SIZE_FACTOR != 0:
        pad_y = (unet_model.UNet.SIZE_FACTOR - img.shape[0] % unet_model.UNet.SIZE_FACTOR)
        print('image height needs to be a multiple of {}, padding with reflect'.format(unet_model.UNet.SIZE_FACTOR))
    if img.shape[1] % unet_model.UNet.SIZE_FACTOR != 0:
        pad_x = (unet_model.UNet.SIZE_FACTOR - img.shape[1] % unet_model.UNet.SIZE_FACTOR)
        print('image width needs to be a multiple of {}, padding with reflect'.format(unet_model.UNet.SIZE_FACTOR))

    if len(img.shape) != 2 and len(img.shape) != 3:
        raise IOError('Invalid number of dimensions for input image. Expecting HW or HWC dimension ordering.')

    if len(img.shape) == 2:
        # add a channel dimension
        img = img.reshape((img.shape[0], img.shape[1], 1))
    img = np.pad(img, pad_width=((0, pad_y, 0), (0, pad_x, 0)), mode='reflect')

    # convert HWC to CHW
    batch_data = img.transpose((2, 0, 1))
    # convert CHW to NCHW
    batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))

    softmax = model(batch_data) # model output defined in unet_model is softmax
    softmax = np.squeeze(softmax)
    pred = np.squeeze(np.argmax(softmax, axis=-1).astype(np.int32))

    if pad_x > 0:
        pred = pred[:, 0:-pad_x]
    if pad_y > 0:
        pred = pred[0:-pad_y, :]

    return pred


def inference(saved_model_filepath, image_folder, output_folder, image_format):
    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    prob_folder = output_folder.replace('_mask','_prob')
    if not os.path.exists(prob_folder):
        os.mkdir(prob_folder)

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    model = tf.saved_model.load(saved_model_filepath)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, slide_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), slide_name))

        print('Loading image: {}'.format(img_filepath))
        img = imagereader.imread(img_filepath)
        img = img.astype(np.float32)

        # normalize with whole image stats
        img = imagereader.zscore_normalize(img)
        print('  img.shape={}'.format(img.shape))

        if img.shape[0] > 1024 or img.shape[1] > 1024:
            tile_size = 512
            segmented_mask = _inference_tiling(img, model, tile_size)
        else:
            segmented_mask = _inference(img, model)

        if 0 <= np.max(segmented_mask) <= 255:
            segmented_mask = segmented_mask.astype(np.uint8)
        if 255 < np.max(segmented_mask) < 65536:
            segmented_mask = segmented_mask.astype(np.uint16)
        if np.max(segmented_mask) > 65536:
            segmented_mask = segmented_mask.astype(np.int32)
        imagereader.imwrite(segmented_mask, os.path.join(output_folder, slide_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference',
                                     description='Script to detect stars with the selected unet model')

    parser.add_argument('--saved_model_filepath', dest='saved_model_filepath', type=str,
                        help='SavedModel filepath to the  model to use', required=True)
    parser.add_argument('--image_folder', dest='image_folder', type=str,
                        help='filepath to the folder containing tif images to inference (Required)', required=True)
    parser.add_argument('--output_folder', dest='output_folder', type=str, required=True)
    parser.add_argument('--image_format', dest='image_format', type=str, help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')

    args = parser.parse_args()

    saved_model_filepath = args.saved_model_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    image_format = args.image_format

    print('Arguments:')
    print('saved_model_filepath = {}'.format(saved_model_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))
    print('image_format = {}'.format(image_format))

    inference(saved_model_filepath, image_folder, output_folder, image_format)

