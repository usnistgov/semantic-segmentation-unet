# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import shutil
import inference_softmax
import metrics

# which model checkpoint to use for inferencing
fp = '/home/mmajursk/Downloads/todo/ooc/unet-model/'
saved_model_filepath= os.path.join(fp, "saved_model")

image_format='tif'

N = 3584
ifp = '/home/mmajursk/USNISTGOV/semantic-segmentation-unet/data/images_{}'.format(N)
ifpAm = os.path.join(fp, 'inference_whole_{}_mask'.format(N))
ifpAs = os.path.join(fp, 'inference_whole_{}_softmax'.format(N))
if os.path.exists(ifpAm):
    shutil.rmtree(ifpAm)
if os.path.exists(ifpAs):
    shutil.rmtree(ifpAs)

number_classes = 2
inference_softmax.inference(saved_model_filepath, ifp, ifpAm, ifpAs, image_format, 0, 0, number_classes)

ifpBm = os.path.join(fp, 'inference_tiling_{}_mask'.format(N))
ifpBs = os.path.join(fp, 'inference_tiling_{}_softmax'.format(N))
if os.path.exists(ifpBm):
    shutil.rmtree(ifpBm)
if os.path.exists(ifpBs):
    shutil.rmtree(ifpBs)


tile_size = 512
with open(os.path.join(fp, 'tile_overlap_impact.csv'), 'a') as fh:
    fh.write('Image_Size, Tile_Size, ZoR, Radius, RMSE, ME, NME\n')
for Radius in range(0,97,16):
    inference_softmax.inference(saved_model_filepath, ifp, ifpBm, ifpBs, image_format, tile_size, Radius, number_classes)

    rmse = metrics.avg_rmse(ifpAs, ifpBs)
    me = metrics.avg_me(ifpAm, ifpBm)
    nme = metrics.avg_nme(ifpAm, ifpBm)

    with open(os.path.join(fp, 'tile_overlap_impact.csv'), 'a') as fh:
        fh.write('{}, {}, {}, {}, {:.2e}, {:.1f}, {:.1e}\n'.format(N, tile_size, tile_size - 2*Radius, Radius, rmse, me, nme))

    with open(os.path.join(fp, 'tile_overlap_impact.tex'), 'a') as fh:
        fh.write('{} & {} & {} & {:.2e} & {:.1f} & {:.1e} \\\\\n'.format(tile_size, tile_size - 2*Radius, Radius, rmse, me, nme))





# which model checkpoint to use for inferencing
fp = '/home/mmajursk/Downloads/todo/ooc/unet-model-random/'
saved_model_filepath= os.path.join(fp, "saved_model")

image_format='tif'

N = 3584
ifp = '/home/mmajursk/USNISTGOV/semantic-segmentation-unet/data/images_{}'.format(N)
ifpAm = os.path.join(fp, 'inference_whole_{}_mask'.format(N))
ifpAs = os.path.join(fp, 'inference_whole_{}_softmax'.format(N))
if os.path.exists(ifpAm):
    shutil.rmtree(ifpAm)
if os.path.exists(ifpAs):
    shutil.rmtree(ifpAs)

number_classes = 4
inference_softmax.inference(saved_model_filepath, ifp, ifpAm, ifpAs, image_format, 0, 0, number_classes)

ifpBm = os.path.join(fp, 'inference_tiling_{}_mask'.format(N))
ifpBs = os.path.join(fp, 'inference_tiling_{}_softmax'.format(N))
if os.path.exists(ifpBm):
    shutil.rmtree(ifpBm)
if os.path.exists(ifpBs):
    shutil.rmtree(ifpBs)


tile_size = 1024
with open(os.path.join(fp, 'tile_overlap_impact.csv'), 'a') as fh:
    fh.write('Image_Size, Tile_Size, ZoR, Radius, RMSE, ME, NME\n')
for Radius in range(0,97,16):
    inference_softmax.inference(saved_model_filepath, ifp, ifpBm, ifpBs, image_format, tile_size, Radius, number_classes)

    rmse = metrics.avg_rmse(ifpAs, ifpBs)
    me = metrics.avg_me(ifpAm, ifpBm)
    nme = metrics.avg_nme(ifpAm, ifpBm)

    with open(os.path.join(fp, 'tile_overlap_impact.csv'), 'a') as fh:
        fh.write('{}, {}, {}, {}, {:.2e}, {:.1f}, {:.1e}\n'.format(N, tile_size, tile_size - 2 * Radius, Radius, rmse, me, nme))

    with open(os.path.join(fp, 'tile_overlap_impact.tex'), 'a') as fh:
        fh.write('{} & {} & {} & {:.2e} & {:.1f} & {:.1e} \\\\\n'.format(tile_size, tile_size - 2 * Radius, Radius, rmse, me, nme))
