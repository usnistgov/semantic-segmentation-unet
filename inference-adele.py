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
import skimage.io
import warnings
import torch

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def zscore_normalize(image_data):
    image_data = image_data.astype(np.float32)

    if len(image_data.shape) == 3:
        # input is CHW
        for c in range(image_data.shape[0]):
            std = np.std(image_data[c, :, :])
            mv = np.mean(image_data[c, :, :])
            if std <= 1.0:
                # normalize (but dont divide by zero)
                image_data[c, :, :] = (image_data[c, :, :] - mv)
            else:
                # z-score normalize
                image_data[c, :, :] = (image_data[c, :, :] - mv) / std
    elif len(image_data.shape) == 2:
        # input is HW
        std = np.std(image_data)
        mv = np.mean(image_data)
        if std <= 1.0:
            # normalize (but dont divide by zero)
            image_data = (image_data - mv)
        else:
            # z-score normalize
            image_data = (image_data - mv) / std
    else:
        raise IOError("Input to Z-Score normalization needs to be either a 2D or 3D image [HW, or CHW]")

    return image_data

def normalize(image_data):
    # data coming in is int16
    image_data = image_data.astype(np.float32)
    std = np.std(image_data)
    mv = np.mean(image_data)
    #print('in norm',mv,std)
    image_data = (image_data - mv) / std
    #print('after norm',np.min(image_data),np.max(image_data))
    # limit range to +-5
    rangeMin = max(np.min(image_data),-5)
    rangeMax = min(np.max(image_data),5)
    #print('range',rangeMin,rangeMax)
    image_data[image_data>rangeMax] = rangeMax
    image_data[image_data<rangeMin] = rangeMin

    # make data 0-1
    pmin = np.min(image_data)
    pmax = np.max(image_data)
    #print('first pmin',pmin,pmax)
    image_data = (image_data - pmin)/(pmax - pmin)
    pmin = np.min(image_data)
    pmax = np.max(image_data)
    #print('second pmin',pmin,pmax)

    # create in int32 from this
    image_data = image_data*255.0
    image_data = image_data.astype(np.float32)
    return image_data

def _inference_tiling(img, model, device, tile_size):
    print('start of inference tiling')
    radius = 96
    SIZE_FACTOR = 16
    # Pad the input image in CPU memory to ensure its dimensions are multiples of the U-Net Size Factor
    pad_x = 0
    pad_y = 0
    if img.shape[0] % SIZE_FACTOR != 0:
        pad_y = (SIZE_FACTOR - img.shape[0] % SIZE_FACTOR)
        print('image height needs to be a multiple of {}, padding with reflect'.format(SIZE_FACTOR))
    if img.shape[1] % SIZE_FACTOR != 0:
        pad_x = (SIZE_FACTOR - img.shape[1] % SIZE_FACTOR)
        print('image width needs to be a multiple of {}, padding with reflect'.format(SIZE_FACTOR))

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

    assert tile_size % SIZE_FACTOR == 0
    assert radius % SIZE_FACTOR == 0
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
            print('next tile',y_st,y_end,x_st,x_end)
            tile = img[y_st:y_end, x_st:x_end]
            #tile = zscore_normalize(tile)

            # convert HWC to CHW
            batch_data = tile.transpose((2, 0, 1))
            # convert CHW to NCHW
            batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))
            batch_data = torch.Tensor(batch_data)
            sm = []
            with torch.no_grad():
                batch_data = batch_data.to(device)
                sm = model(batch_data)
                sm = sm.cpu()
            #sm = model(batch_data)  # model output defined in unet_model is softmax

            sm = sm.numpy()
            sm = np.squeeze(sm)
            sm = np.squeeze(np.argmax(sm, axis=-1).astype(np.int32))
            #print('pred',np.count_nonzero(pred))
            pred = sm
            # radius_pre_x
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

            #mask[y_st_z:y_end_z, x_st_z:x_end_z] = pred

    # undo and CPU side image padding to make the image a multiple of U-Net Size Factor
    if pad_x > 0:
        mask = mask[:, 0:-pad_x]
    if pad_y > 0:
        mask = mask[0:-pad_y, :]
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
    img = np.pad(img, pad_width=((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')

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


def inference(saved_model_filepath, image_folder, output_folder, image_format, use_intensity_scaling):
    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    prob_folder = output_folder.replace('_mask','_prob')
    if not os.path.exists(prob_folder):
        os.mkdir(prob_folder)

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    print('Loading saved model')
    print('  found files:')
    #unet_model.UNet(n_channels=number_channels, n_classes=num_classes)for fn in os.listdir(saved_model_filepath):
    #    print(fn)
    #model = tf.saved_model.load(saved_model_filepath)
    model = unet_model.UNet(n_channels=1, n_classes=2)
    model = torch.load(saved_model_filepath)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, slide_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), slide_name))

        # print('Loading image: {}'.format(img_filepath))
        img = skimage.io.imread(img_filepath) # HW or HWC format
        img = img.astype(np.float32)

        # enable or disable intensity scaling - added to support concrete project
        img = normalize(img) # normalize with whole image stats
        print('  img.shape={}'.format(img.shape))

        if img.shape[0] > 1024 or img.shape[1] > 1024:
            tile_size = 1024 # in theory UNet takes about 420x the amount of memory of the input image
            # to a tile size of 1024 should require 1.7 GB of GPU memory
            segmented_mask = _inference_tiling(img, model, device, tile_size)
        else:
            segmented_mask = _inference(img, model)

        if 0 <= np.max(segmented_mask) <= 255:
            segmented_mask = segmented_mask.astype(np.uint8)
        if 255 < np.max(segmented_mask) < 65536:
            segmented_mask = segmented_mask.astype(np.uint16)
        if np.max(segmented_mask) > 65536:
            segmented_mask = segmented_mask.astype(np.int32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if 'tif' in image_format:
                skimage.io.imsave(os.path.join(output_folder, slide_name), segmented_mask, compress=6, bigtiff=True, tile=(1024,1024))
            else:
                skimage.io.imsave(os.path.join(output_folder, slide_name), segmented_mask, compress=6)


def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='inference', description='Script which inferences a folder of images using a unet model')

    parser.add_argument('--model', dest='saved_model_filepath', type=str,
                        help='SavedModel filepath to the  model to use', required=True)
    parser.add_argument('--imageDir', dest='image_dir', type=str, help='filepath to the directory containing the images', required=True)
    parser.add_argument('--outputDir', dest='output_dir', type=str, help='Folder where outputs will be saved (Required)', required=True)
    # added for the concrete project
    parser.add_argument('--useIntensityScaling', dest='use_intensity_scaling', type=str,
                        help='whether to use intensity scaling when inferring [YES, NO]', default="YES")

    print('Arguments:')
    args = parser.parse_args()

    saved_model_filepath = args.saved_model_filepath
    output_dir = args.output_dir
    image_dir = args.image_dir
    use_intensity_scaling = args.use_intensity_scaling
    use_intensity_scaling = use_intensity_scaling.upper() == "YES"

    print('model = {}'.format(saved_model_filepath))
    print('imageDir = {}'.format(image_dir))
    print('outputDir = {}'.format(output_dir))
    print('use_intensity_scaling = {}'.format(use_intensity_scaling))

    image_format = 'tif'

    inference(saved_model_filepath, image_dir, output_dir, image_format, use_intensity_scaling)


if __name__ == "__main__":
    main()
