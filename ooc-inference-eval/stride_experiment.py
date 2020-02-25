# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import skimage.io
import numpy as np
import inference_unet_type_model_softmax
from unet import imagereader
import unet_type_model


def compute_rmse(A, B):
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    e = np.abs(A - B)
    e = np.square(e)
    rmse = np.sqrt(np.mean(e))
    return rmse


ifp = '../data/images_3824/'
fns = [fn for fn in os.listdir(ifp) if fn.endswith('.tif')]
fn = fns[0]

# load model for inference
unet = unet_type_model.UNet('./unet-model/saved_model/')

N = 2048

whole_img = skimage.io.imread(os.path.join(ifp, fn))
whole_img = whole_img[0:0+N, 0:0+32+N]
whole_img = whole_img.astype(np.float32)
whole_img = imagereader.zscore_normalize(whole_img)

_, whole_softmax = inference_unet_type_model_softmax._inference(whole_img, unet)

rmse_whole = list()

for i in range(0, 33):
    subI = whole_img[:, i:i+N]
    ref_output = whole_softmax[96:-96, 128:-128]

    _, sm = inference_unet_type_model_softmax._inference(subI, unet)
    # crop to common area without radius
    sm = sm[96:-96, 96 + 32 - i:-(96 + i)]

    rmse_whole.append(compute_rmse(ref_output, sm))

with open('stride_impact.csv', 'w') as fh:
    fh.write('Offset, RMSE\n')

for i in range(len(rmse_whole)):
    rmse = rmse_whole[i]
    with open('stride_impact.csv', 'a') as fh:
        fh.write('{}, {:.2e}\n'.format(i, rmse))


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

x_vals = np.asarray(list(range(33)))
rmse_whole = np.asarray(rmse_whole)
dot_size = 4
fig = plt.figure(figsize=(16, 9), dpi=200)
ax = plt.gca()
ax.grid()
ax.plot(x_vals, rmse_whole, 'k-', marker='o', markersize=20, linewidth=2)
plt.xticks(np.arange(0, 33, 4))

plt.ylim((np.min(rmse_whole) - 0.0002, np.max(rmse_whole) + 0.0002))
fig.suptitle('Softmax RMSE as a Function of Offset', fontsize=38)
plt.ylabel('Softmax RMSE', fontsize=38)
plt.xlabel('Stride Offset (pixels)', fontsize=38)
ax.tick_params(axis='both', which='major', labelsize=28)

fig.savefig('stride_impact.png')
plt.close(fig)




