# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

# local imports
import train_model
import utils


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train a single model')

    # image data params
    parser.add_argument('--train-image-dirpath', type=str, required=True,
                        help='Filepath to the folder/directory where the input training image data exists')
    parser.add_argument('--train-mask-dirpath', type=str, required=True,
                        help='Filepath to the folder/directory where the input training mask data exists. Masks must have the same filename as their corresponding image.')
    parser.add_argument('--val-image-dirpath', type=str, default=None,
                        help='Filepath to the folder/directory where the input validation image data exists')
    parser.add_argument('--val-mask-dirpath', type=str, default=None,
                        help='Filepath to the folder/directory where the input validation mask data exists. Masks must have the same filename as their corresponding image.')
    parser.add_argument('--test-image-dirpath', type=str, default=None,
                        help='Filepath to the folder/directory where the input test image data exists')
    parser.add_argument('--test-mask-dirpath', type=str, default=None,
                        help='Filepath to the folder/directory where the input test mask data exists. Masks must have the same filename as their corresponding image.')
    parser.add_argument('--image-extension', type=str, required=True,
                        help='Image/mask file extension. I.e. tif, png, jpg')
    parser.add_argument('--mask-extension', type=str, required=True,
                        help='Image/mask file extension. I.e. tif, png, jpg')

    # output location
    parser.add_argument('--output-dirpath', type=str, required=True,
                        help='Filepath to the folder/directory where the results should be stored')

    # training parameters
    parser.add_argument('--num-classes', type=int, default=2, help='The number of classes the model is to predict.')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--tile-size', type=int, default=None, help='Tile size, default value of None will not use tiling')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--val-fraction', default=0.1, type=float, help='Fraction of the training data to use for validation. This can only be used if val-image-filepath and val-mask-filepath are None. Otherwise it will be ignored.')
    parser.add_argument('--test-every-n-steps', default=None, help='Run the validation data every N trainig steps. If None, the normal epoch size will be used.', type=int)

    # early stopping configuration
    parser.add_argument('--patience', default=10, type=int, help='number of epochs past optimal to explore before early stopping terminates training.')
    parser.add_argument('--loss-eps', default=1e-4, type=float, help='loss value eps for determining early stopping loss equivalence.')
    parser.add_argument('--num-lr-reductions', default=2, type=int)
    parser.add_argument('--lr-reduction-factor', default=0.2, type=float)
    parser.add_argument('--cycle-factor', default=2.0, type=float, help='Cycle factor for cyclic learning rate scheduler.')

    # adversarial training
    parser.add_argument('--adv-prob', type=float, default=None, help='Float value [0.0, 1.0] determining what percentage of the batches have FastIsBetterThanFree adversarial training turned on')
    parser.add_argument('--adv-eps', type=float, default=(4.0/255.0), help='Adversarial training perturbation budget')

    # parallel data I/O and augmentation
    parser.add_argument('--num-workers', type=int, default=4, help='The number of parallel threads doing I/O loading, preprocessing, and augmentation. Set this to 0 for debugging so all I/O happens on the master thread.')


    args = parser.parse_args()
    # check if IDE is in debug mode, and set the num parallel workers to 0
    utils.check_for_ide_debug_mode(args)

    try:
        train_model.train(args)
        return 0
    except:
        import traceback
        tb = traceback.format_exc()
        logging.warning(tb)
        return 1


if __name__ == "__main__":
    main()

