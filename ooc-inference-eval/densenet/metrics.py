# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

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


def avg_nme(ifpA, ifpB):
    fns = [fn for fn in os.listdir(ifpA) if fn.endswith('.tif')]
    me = 0.0
    for fn in fns:
        A = skimage.io.imread(os.path.join(ifpA, fn))
        B = skimage.io.imread(os.path.join(ifpB, fn))

        c = np.count_nonzero(A - B)
        c = float(c) / float(A.size)
        me = me + c

    me = me / float(len(fns))
    return me