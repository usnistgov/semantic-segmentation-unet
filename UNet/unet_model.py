""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn

def double_conv(fc_in,fc_out,kernel,stride=1):
    stage = torch.nn.Sequential()
    pad = (kernel - 1) // 2
    stage.add_module('conv', torch.nn.Conv2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('conv', torch.nn.Conv2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('pool',torch.nn.MaxPool2d(2))
    return stage

def double_deconv(fc_in,fc_out,kernel,stride=1):
    stage = torch.nn.Sequential()
    pad = (kernel - 1) // 2
    stage.add_module('upsample', torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))
    stage.add_module('deconv', torch.nn.ConvTranspose2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('deconv', torch.nn.ConvTranspose2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    return stage

def bottleneck(fc_in,fc_out,kernel,stride=1):
    stage = torch.nn.Sequential()
    pad = (kernel - 1) // 2
    stage.add_module('conv', torch.nn.Conv2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('conv', torch.nn.Conv2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('dropout',torch.nn.Dropout(0.5))
    return stage

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.mdict = self.build_net_dict(self.n_channels)

    def build_net_dict(self,n_channels):
        mdict = torch.nn.ModuleDict()
        out_channels = 64
        mdict['down_1'] = double_conv(n_channels, out_channels, 3, stride=1)
        mdict['down_2'] = double_conv(out_channels, 2*out_channels, 3, stride=1)
        mdict['down_3'] = double_conv(2*out_channels, 4*out_channels, 3, stride=1)
        mdict['down_4'] = double_conv(4*out_channels, 8*out_channels, 3, stride=1)
        mdict['down_5'] = bottleneck(8*out_channels,16*out_channels, 3, stride=1)
        mdict['up_4'] = double_deconv(16*out_channels, 8*out_channels, 3, stride=1)
        mdict['up_3'] = double_deconv(8*out_channels, 4*out_channels, 3, stride=1)
        mdict['up_2'] = double_deconv(4*out_channels, 2*out_channels, 3, stride=1)
        mdict['up_1'] = double_deconv(2*out_channels, out_channels, 3, stride=1)
        mdict['outconv'] = torch.nn.Conv2d(out_channels, self.n_classes, 1)
        return mdict

    def forward(self, x):
        x = self.mdict['down_1'](x)
        x = self.mdict['down_2'](x)
        x = self.mdict['down_3'](x)
        x = self.mdict['down_4'](x)
        x = self.mdict['down_5'](x)
        x = self.mdict['up_4'](x)
        x = self.mdict['up_3'](x)
        x = self.mdict['up_2'](x)
        x = self.mdict['up_1'](x)
        x = self.mdict['outconv'](x)
        return x
