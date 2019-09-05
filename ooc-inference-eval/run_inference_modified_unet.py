
import os
import shutil
import inference_unet_type_model_softmax
import metrics


# which model checkpoint to use for inferencing
saved_model_filepath="./unet-model-modified/saved_model/"

image_format='tif'

N = 3824
ifp = '../data/images_{}'.format(N)
ifpAm = './inference_whole_{}_mask'.format(N)
ifpAs = './inference_whole_{}_softmax'.format(N)
if os.path.exists(ifpAm):
    shutil.rmtree(ifpAm)
if os.path.exists(ifpAs):
    shutil.rmtree(ifpAs)

inference_unet_type_model_softmax.inference(saved_model_filepath, ifp, ifpAm, ifpAs, image_format, 0, 0)

ifpBm = './inference_tiling_{}_mask'.format(N)
ifpBs = './inference_tiling_{}_softmax'.format(N)
if os.path.exists(ifpBm):
    shutil.rmtree(ifpBm)
if os.path.exists(ifpBs):
    shutil.rmtree(ifpBs)


for tile_size in (512, 512+8, 1024, 1024+8):
    with open('tile_overlap_impact_mod.csv', 'a') as fh:
        fh.write('Image_Size, Tile_Size, ZoR, Radius, RMSE, ME\n')
    for Radius in range(0,97,8):
        inference_unet_type_model_softmax.inference(saved_model_filepath, ifp, ifpBm, ifpBs, image_format, tile_size, Radius)

        rmse = metrics.avg_rmse(ifpAs, ifpBs)
        me = metrics.avg_me(ifpAm, ifpBm)

        with open('tile_overlap_impact_mod.csv', 'a') as fh:
            fh.write('{}, {}, {}, {}, {:.2e}, {:.1f}\n'.format(N, tile_size, tile_size - 2*Radius, Radius, rmse, me))

        with open('tile_overlap_impact_mod.tex', 'a') as fh:
            fh.write('{} & {} & {} & {:.2e} & {:.1f} \\\\\n'.format(tile_size, tile_size - 2*Radius, Radius, rmse, me))

