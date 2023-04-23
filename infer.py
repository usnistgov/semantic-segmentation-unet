import os
import argparse
import shutil
import numpy as np
import skimage.io
import torch

# local imports
import dataset


def infer(args):
    if os.path.exists(args.output_dirpath):
        print("output directory exists, deleting")
        shutil.rmtree(args.output_dirpath)
    os.makedirs(args.output_dirpath)

    # Load the image file list
    img_filepath_list = [os.path.join(args.image_dirpath, fn) for fn in os.listdir(args.image_dirpath) if fn.endswith(args.image_extension)]
    img_filepath_list.sort()
    print("Found {} images to inference".format(len(img_filepath_list)))

    # load the model
    model = torch.load('./model/model.pt')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # move the model to device
    model.to(device)
    # put the model in eval mode to prevent gradients from being computed
    model.eval()

    # break the images into tiles
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]

        # load the images
        img = skimage.io.imread(img_filepath)

        # create an empty image the same shape as the input
        output_mask = np.zeros((img.shape[0], img.shape[1]), dtype=int)

        # break the image into tiles
        i_tile_list, _, location_list = dataset.SemanticSegmentationDataset.tile_image_mask_pair(img, msk=None, tile_size=args.tile_size)

        for k in range(len(i_tile_list)):
            # get the tiles image data
            tile = i_tile_list[k]
            # get the tiles location within the larger image
            x_st, y_st = location_list[k]

            # use the augmentation test time transforms to convert the image into the appropriate format
            results = dataset.SemanticSegmentationDataset.TRANSFORM_TEST(image=tile)
            tile = results['image']
            # add a degenerate batch (N) dimension to appease the model
            tile = torch.unsqueeze(tile, 0)
            # move the data to device
            tile = tile.to(device)

            # inference the tile (data is expected to be in NCHW format)
            logits = model(tile)
            # .detach().cpu().numpy() detaches the tensor from the graph, moves it to the CPU, and converts it to a numpy array
            logits = logits.detach().cpu().numpy()
            # get rid of the batch dimension
            logits = logits[0,]
            # convert the one-hot output into a class prediction
            pred = np.argmax(logits, axis=0).astype(int)

            # copy the prediction results into the output mask
            y_end = y_st + pred.shape[0]
            x_end = x_st + pred.shape[1]
            output_mask[y_st:y_end, x_st:x_end] = pred

        # save the output mask
        fn = os.path.basename(img_filepath)
        print("saving output for {}".format(fn))
        _, ext = os.path.splitext(fn)
        fn = fn.replace(ext, '.png')
        print("  into file: {}".format(fn))
        skimage.io.imsave(os.path.join(args.output_dirpath, fn), output_mask)



def main():
    parser = argparse.ArgumentParser(description='PyTorch UNet Model Inference')
    parser.add_argument('--image-dirpath', default='./data/imgs', type=str, help='filepath to where the training images are saved.')
    parser.add_argument('--image-extension', default="jpg", type=str, help='file extension of the training images.')
    parser.add_argument('--output-dirpath', default='./infer_results', type=str, help='filepath to where the outputs will be saved.')
    parser.add_argument('--tile-size', default=256, type=int, help='tile size')

    args = parser.parse_args()
    infer(args)

if __name__ == '__main__':
    main()

