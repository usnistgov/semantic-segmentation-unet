import os
import shutil
import metrics
import inference_unet_type_model_softmax


saved_model_filepath = './unet-model-random/saved_model'

image_format='tif'

N = 3824
ifp = '../data/images_random_{}'.format(N)
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


for tile_size in (1024, 1024 + 16):
    with open('tile_overlap_impact_rand.csv', 'a') as fh:
        fh.write('Image_Size, Tile_Size, ZoR, Radius, RMSE, ME\n')
    for Radius in range(0,161,16):
        inference_unet_type_model_softmax.inference(saved_model_filepath, ifp, ifpBm, ifpBs, image_format, tile_size, Radius)

        rmse = metrics.avg_rmse(ifpAs, ifpBs)
        me = metrics.avg_me(ifpAm, ifpBm)
        with open('tile_overlap_impact_rand.csv', 'a') as fh:
            fh.write('{}, {}, {}, {}, {:.2e}, {}\n'.format(N, tile_size, tile_size - 2*Radius, Radius, rmse, me))

        with open('tile_overlap_impact_rand.tex', 'a') as fh:
            fh.write('{} & {} & {} & {:.2e}  & {:.1f} \\\\\n'.format(tile_size, tile_size - 2*Radius, Radius, rmse, me))

