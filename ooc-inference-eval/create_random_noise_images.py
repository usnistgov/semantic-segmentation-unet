import os
import skimage.io
import numpy as np


N = 3824
ofp = '../data/data/rand_images_{}'.format(N)
if not os.path.exists(ofp):
    os.makedirs(ofp)

for i in range(20):
    subI = np.random.randn(N, N).astype(np.float32)
    skimage.io.imsave(os.path.join(ofp, '{}.tif'.format(i)), subI)
