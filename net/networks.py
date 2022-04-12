# -*- coding: UTF-8 -*-
'''
@Project ：MAML_Pytorch
@File    ：networks.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import math
import torch
from torch import nn
import torch.nn.functional as F
from CustomLayers import BatchNormal2d

class CreateModel(nn.Module):
    def __init__(self):
        super(CreateModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=3)
        self.bn1 = BatchNormal2d(num_features=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3)
        self.bn2 = BatchNormal2d(num_features=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3)
        self.bn3 = BatchNormal2d(num_features=256)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(in_features=256, out_features=20)

        self.init_params()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = torch.dropout(x, p=0.3, train=True)
        x = torch.max_pool2d(x, kernel_size=3,
                             stride=2, padding=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = torch.dropout(x, p=0.3, train=True)
        x = torch.max_pool2d(x, kernel_size=3,
                             stride=2, padding=1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = torch.dropout(x, p=0.3, train=True)
        x = torch.max_pool2d(x, kernel_size=x.size(-1),
                             stride=1)

        x = x.squeeze()
        x = self.linear(x)

        output = F.log_softmax(x, dim=-1)

        return output

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    stddev = 1 / math.sqrt(param.size(0))
                    torch.nn.init.normal_(param, std=stddev)
                else:
                    torch.nn.init.zeros_(param)

    def get_weights(self):

        weights = []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    weights.append(param)
                else:
                    continue
        return weights
