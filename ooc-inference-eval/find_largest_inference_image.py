import numpy as np
import unet_type_model
import inference_unet_type_model

# load model for inference
unet = unet_type_model.UNet('./random-unet/saved_model/')

for N in range(1024, 4096, 16):
    img = np.random.randn(N, N).astype(np.float32)
    print('inference N = {}'.format(N))
    segmented_mask = inference_unet_type_model._inference(img, unet)

