import os
import numpy as np
import time
import torch
import copy

import albumentations
import albumentations.pytorch

import dataset
import train_model
import unet_model




























if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inference a single model')
    parser.add_argument('--model-filepath', type=str, required=True,
                        help='Filepath to the model to use for inference.')
    parser.add_argument('--image-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the image data exists')
    parser.add_argument('--file-extension', type=str, required=True,
                        help='Image/mask file extension. I.e. tif, png, jpg')
    parser.add_argument('--output-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('--cast-grayscale-to-color', action='store_true')

    args = parser.parse_args()
    model_filepath = args.model_filepath
    image_filepath = args.image_filepath
    file_extension = args.file_extension

    output_filepath = args.output_filepath
    cast_grayscale_to_color = args.cast_grayscale_to_color


    # define the data augmentation transformations
    # if you normalize the images within these transforms, disable zscore norm in the SemanticSegmentationDataset
    train_transform = albumentations.Compose(
        [
            albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=360, p=0.5),
            albumentations.pytorch.ToTensorV2()  # This transform needs to be present, and last in the composed list
        ]
    )

    val_test_transform = albumentations.Compose(
        [
            albumentations.pytorch.ToTensorV2()  # This transform needs to be present, and last in the composed list
        ]
    )


    # define the datasets
    train_dataset = dataset.SemanticSegmentationDataset(image_filepath=train_image_filepath, mask_filepath=train_mask_filepath, extension=file_extension, transform=train_transform, preload_into_memory=cache_images_in_memory, cast_grayscale_to_color=cast_grayscale_to_color)
    val_dataset = dataset.SemanticSegmentationDataset(image_filepath=val_image_filepath, mask_filepath=val_mask_filepath, extension=file_extension, transform=val_test_transform, preload_into_memory=cache_images_in_memory, cast_grayscale_to_color=cast_grayscale_to_color)
    test_dataset = None
    if test_image_filepath is not None:
        test_dataset = dataset.SemanticSegmentationDataset(image_filepath=test_image_filepath, mask_filepath=test_mask_filepath, extension=file_extension, transform=val_test_transform, preload_into_memory=cache_images_in_memory, cast_grayscale_to_color=cast_grayscale_to_color)

    number_channels = train_dataset.get_number_channels()
    if number_channels is None:
        raise RuntimeError("could not compute the number of input image channels")

    # define the model
    model = unet_model.UNet(n_channels=number_channels, n_classes=num_classes)

    # import pytorch_models
    # pretrained = True
    # model_name = pytorch_models.SUPPORTED_MODELS[0]
    # model = pytorch_models.initializeModel(outputchannels=num_classes, pretrained=pretrained, name=model_name)  # Calls function to create the deeplabv3 model from torchvision.

    # train the model end to end
    train_model.train(train_dataset, val_dataset, test_dataset, model, output_filepath, learning_rate, num_io_workers, early_stopping_count, early_stopping_loss_eps, adv_train_prob, adv_train_eps, use_cycle_lr)

