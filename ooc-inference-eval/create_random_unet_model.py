import unet_type_model

# Build and save random model
unet = unet_type_model.UNet()
M = 4
nl = 2
k = 3
unet.configure(number_classes=2, global_batch_size=1, input_channel_count=1, M=M, nl=nl, k=k)
unet.save_model('./unet-model-random/saved_model/')