import albumentations.pytorch

import dataset
import train_model
import unet_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a single model')
    parser.add_argument('--train-image-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the input training image data exists')
    parser.add_argument('--train-mask-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the input training mask data exists. Masks must have the same filename as their corresponding image.')
    parser.add_argument('--val-image-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the input validation image data exists')
    parser.add_argument('--val-mask-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the input validation mask data exists. Masks must have the same filename as their corresponding image.')
    parser.add_argument('--test-image-filepath', type=str, default=None,
                        help='Filepath to the folder/directory where the input test image data exists')
    parser.add_argument('--test-mask-filepath', type=str, default=None,
                        help='Filepath to the folder/directory where the input test mask data exists. Masks must have the same filename as their corresponding image.')
    parser.add_argument('--file-extension', type=str, required=True,
                        help='Image/mask file extension. I.e. tif, png, jpg')
    parser.add_argument('--output-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--early-stopping-count', type=int, default=10, help='The number of epochs past the current best loss to continue training in search of a better final loss.')
    parser.add_argument('--early-stopping-loss-eps', type=float, default=1e-3, help="The loss epsilon (delta below which loss values are considered equal) when handling early stopping.")
    parser.add_argument('--adv-train-prob', type=float, default=0.1)
    parser.add_argument('--adv-train-eps', type=float, default=(4.0/255.0))
    parser.add_argument('--disable-cycle-lr', action='store_false')
    parser.add_argument('--cache-images-in-memory', action='store_true')
    parser.add_argument('--cast-grayscale-to-color', action='store_true')
    parser.add_argument('--num-io-workers', type=int, default=0, help='The number of parallel threads doing I/O loading, preprocessing, and augmentation. Set this to 0 for debugging so all I/O happens on the master thread.')
    parser.add_argument('--num-classes', type=int, default=2, help='The number of classes the model is to predict.')

    args = parser.parse_args()
    train_image_filepath = args.train_image_filepath
    train_mask_filepath = args.train_mask_filepath
    val_image_filepath = args.val_image_filepath
    val_mask_filepath = args.val_mask_filepath
    test_image_filepath = args.test_image_filepath
    test_mask_filepath = args.test_mask_filepath
    file_extension = args.file_extension

    output_filepath = args.output_filepath
    learning_rate = args.learning_rate
    early_stopping_count = args.early_stopping_count
    early_stopping_loss_eps = args.early_stopping_loss_eps
    adv_train_prob = args.adv_train_prob
    adv_train_eps = args.adv_train_eps
    use_cycle_lr = args.disable_cycle_lr
    num_io_workers = args.num_io_workers
    cache_images_in_memory = args.cache_images_in_memory
    num_classes = args.num_classes
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




