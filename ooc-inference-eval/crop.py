
import os
import skimage.io

N = 2432
ifp = '/home/mmajursk/USNISTGOV/semantic-segmentation-unet/data/images_3824'
ofp = '/home/mmajursk/USNISTGOV/semantic-segmentation-unet/data/images_{}'.format(N)
if not os.path.exists(ofp):
    os.makedirs(ofp)

fns = [fn for fn in os.listdir(ifp) if fn.endswith('.tif')]

for fn in fns:
    img = skimage.io.imread(os.path.join(ifp, fn))
    img = img[0:N, 0:N]
    skimage.io.imsave(os.path.join(ofp, fn), img)

