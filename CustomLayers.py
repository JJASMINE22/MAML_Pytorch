# -*- coding: UTF-8 -*-
'''
@Project ：MAML_Pytorch
@File    ：CustomLayers.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class ResidueBlock(nn.Module):
    def __init__(self,
                 short_cut: bool,
                 drop_rate: float,
                 in_channels: int,
                 out_channels: int,
                 padding_mode: str,
                 stride: int or tuple,
                 negative_slope: float,
                 kernel_size: int or tuple,
                 padding: str or tuple or int):
        super(ResidueBlock, self).__init__()
        self.stride = stride
        self.padding = padding
        self.drop_rate = drop_rate
        self.short_cut = short_cut
        self.drop_rate = drop_rate
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.negative_slope = negative_slope

        if np.not_equal(self.in_channels, self.out_channels):
            self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   padding_mode=self.padding_mode)

        self.conv2 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.out_channels//2 if self.short_cut else self.out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding,
                               padding_mode=self.padding_mode)
        self.bn1 = nn.BatchNorm2d(num_features=self.out_channels//2 if self.short_cut else self.out_channels)

        self.conv3 = nn.Conv2d(in_channels=self.out_channels//2 if self.short_cut else self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               stride=(1, 1),
                               padding='same',
                               padding_mode=self.padding_mode)
        self.bn2 = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, init):

        x = init
        if np.not_equal(self.in_channels, self.out_channels):
            init = self.conv1(init)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)

        x = self.conv3(x)
        x = torch.dropout(x, p=self.drop_rate, train=True)

        x = torch.add(x, init)
        x = self.bn2(x)

        output = F.leaky_relu(x, negative_slope=self.negative_slope)

        return output

class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 padding_mode: str,
                 depthwise_stride: int or tuple,
                 depthwise_ksize: int or tuple,
                 padding: str or tuple or int):
        super(SeparableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_padding = padding
        self.depthwise_stride = depthwise_stride
        self.depthwise_ksize = depthwise_ksize
        self.pointwise_stride = (1, 1)
        self.pointwise_ksize = (1, 1)
        self.padding_mode = padding_mode

        self.depthwise_conv = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.in_channels,
                                        kernel_size=self.depthwise_ksize,
                                        stride=self.depthwise_stride,
                                        padding=self.depthwise_padding,
                                        padding_mode=self.padding_mode,
                                        groups=self.in_channels)

        self.pointwise_cov = nn.Conv2d(in_channels=self.in_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.pointwise_ksize,
                                       stride=self.pointwise_stride)

    def forward(self, input):

        x = self.depthwise_conv(input)

        output = self.pointwise_cov(x)

        return output


class MobileBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 padding_mode: str,
                 depthwise_stride: int,
                 padding: str or tuple or int,
                 expansion_factor: float):
        super(MobileBlock, self).__init__()
        assert padding in [0, (0, 0), 'valid']
        self.depthwise_ksize = (3, 3)
        self.pointwise_stride = (1, 1)
        self.pointwise_ksize = (1, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.depthwise_padding = padding
        self.depthwise_stride = depthwise_stride
        self.expansion_factor = expansion_factor

        self.depthwise_conv = nn.Conv2d(in_channels=int(self.in_channels*self.expansion_factor),
                                        out_channels=int(self.in_channels*self.expansion_factor),
                                        kernel_size=self.depthwise_ksize,
                                        stride=(self.depthwise_stride,)*2,
                                        padding=self.depthwise_padding,
                                        padding_mode=self.padding_mode,
                                        groups=int(self.in_channels*self.expansion_factor))

        self.pointwise_conv1 = nn.Conv2d(in_channels=self.in_channels,
                                         out_channels=int(self.in_channels*self.expansion_factor),
                                         kernel_size=self.pointwise_ksize,
                                         stride=self.pointwise_stride)

        self.pointwise_conv2 = nn.Conv2d(in_channels=int(self.in_channels*self.expansion_factor),
                                         out_channels=self.out_channels,
                                         kernel_size=self.pointwise_ksize,
                                         stride=self.pointwise_stride)

        self.bn1 = nn.BatchNorm2d(num_features=int(self.in_channels*self.expansion_factor))

        self.bn2 = nn.BatchNorm2d(num_features=int(self.in_channels*self.expansion_factor))

        self.bn3 = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, init):
        x = init

        x = self.pointwise_conv1(x)
        x = self.bn1(x)
        x = F.relu6(x)

        if np.greater(self.depthwise_stride, 1):
            x = F.pad(x, pad=(1,)*4, mode=self.padding_mode)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.relu6(x)

        x = self.pointwise_conv2(x)

        if np.logical_and(np.equal(self.in_channels, self.out_channels),
                          np.equal(self.depthwise_stride, 1)):
            x = torch.add(x, init)

        output = self.bn3(x)

        return output

class BatchNormal2d(nn.Module):
    def __init__(self,
                 eps: float=1e-5,
                 training: bool=True,
                 num_features: int = None):
        super(BatchNormal2d, self).__init__()
        self.eps = eps
        self.training = training
        self.num_features = num_features
        self.weight = nn.Parameter(data=torch.ones(size=(self.num_features,),
                                   requires_grad=True))
        self.bias = nn.Parameter(data=torch.zeros(size=(self.num_features,),
                                 requires_grad=True))

    def forward(self, x):

        assert x.size(1) == self.num_features

        x_mean = x.mean(dim=[0, -2, -1], keepdim=True)
        x_var = x.var(dim=[0, -2, -1], keepdim=True)
        x = (x - x_mean) / torch.sqrt(x_var + self.eps)
        output = torch.add(self.weight.view([1, -1, 1, 1]) * x, self.bias.view([1, -1, 1, 1]))

        if not self.training:
            return output.detach()
        return output
