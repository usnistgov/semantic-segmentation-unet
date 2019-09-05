import os
import numpy as np
import skimage.io

def avg_rmse(ifpA, ifpB):
    fns = [fn for fn in os.listdir(ifpA) if fn.endswith('.tif')]
    rmse = 0
    for fn in fns:
        A = skimage.io.imread(os.path.join(ifpA, fn)).astype(dtype=np.float64)
        B = skimage.io.imread(os.path.join(ifpB, fn)).astype(dtype=np.float64)

        e = np.abs(A - B)
        e = np.square(e)
        rmse = rmse + np.sqrt(np.mean(e))

    rmse = rmse / float(len(fns))
    return rmse


def avg_me(ifpA, ifpB):
    fns = [fn for fn in os.listdir(ifpA) if fn.endswith('.tif')]
    me = 0.0
    for fn in fns:
        A = skimage.io.imread(os.path.join(ifpA, fn))
        B = skimage.io.imread(os.path.join(ifpB, fn))

        c = np.count_nonzero(A - B)
        me = me + c

    me = me / float(len(fns))
    return me