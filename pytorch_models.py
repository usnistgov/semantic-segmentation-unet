# IEEE Format Citation:
# M. S. Minhas, “Transfer Learning for Semantic Segmentation using PyTorch DeepLab v3,” GitHub.com/msminhas93, 12-Sep-2019. [Online]. Available: https://github.com/msminhas93/DeepLabv3FineTuning.
# Link: https://towardsdatascience.com/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch-f770863d6a42

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision import models


SUPPORTED_MODELS = ['Deeplab50', 'Deeplab101', 'Resnet50', 'Resnet101']


def initializeModel(outputchannels, pretrained, name):
    if name == 'Deeplab50' and pretrained is False:
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=outputchannels, progress=True)
        return model
    if name == 'Deeplab50' and pretrained is True:
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        model.classifier = DeepLabHead(2048, outputchannels)
        return model
    if name == 'Deeplab101' and pretrained is True:
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        model.classifier = DeepLabHead(2048, outputchannels)
        return model
    if name == 'Deeplab101' and pretrained is False:
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=outputchannels, progress=True)
        return model
    if name == 'Resnet50' and pretrained is False:
        model = models.segmentation.fcn_resnet50(pretrained=pretrained, num_classes=outputchannels, progress=True)
        return model
    if name == 'Resnet50' and pretrained is True:
        model = models.segmentation.fcn_resnet50(pretrained=pretrained, progress=True)
        model.classifier = FCNHead(2048, outputchannels)
        return model
    if name == 'Resnet101' and pretrained is False:
        model = models.segmentation.fcn_resnet101(pretrained=pretrained, num_classes=outputchannels, progress=True)
        return model
    if name == 'Resnet101' and pretrained is True:
        model = models.segmentation.fcn_resnet101(pretrained=pretrained, progress=True)
        model.classifier = FCNHead(2048, outputchannels)
        return model

